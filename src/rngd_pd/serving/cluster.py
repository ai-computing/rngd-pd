#!/usr/bin/env python3
"""Real PD-disaggregated serving over HTTP — the A1/A2/A3 wiring, no mocks.

Three processes, one file:

  --role prefill  RNGD worker. Loads the 32L EDF from the artifact cache (0 s,
                  not a 30-min compile), runs prefill, and pushes the KV out
                  through RngdKVConnector.save_kv_layer() per layer.
  --role decode   Consumes that KV through start_load_kv()/wait_for_layer_load()
                  and continues generation. --backend rngd keeps it on a second
                  Furiosa card; --backend atom runs the fp16 module on a
                  Rebellions card, which makes the pair heterogeneous.
  --role gateway  OpenAI-compatible front (POST /v1/completions). Tokenizes,
                  calls prefill, then decode, and reports what the KV handoff
                  actually cost.

The KV crosses as bf16 bytes over /dev/shm, so prefill and decode pods must
share a host /dev/shm mount. Nothing here re-prefills: the decode side starts
from the prefill side's KV, and that is the whole point.

  python3 pd_cluster.py --role prefill --chip 2 --port 8101
  python3 pd_cluster.py --role decode --backend atom --port 8102
  python3 pd_cluster.py --role gateway --port 8100 \
      --prefill http://127.0.0.1:8101 --decode http://127.0.0.1:8102
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from rngd_pd.paths import ARTIFACT_CACHE



MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SHM_ROOT = os.environ.get("PD_SHM_ROOT", "/dev/shm/rngd_kv_cluster")

DEVICE = threading.Lock()      # NPU runtimes are not reentrant
STATE = {"ready": False, "detail": "starting"}
METRICS = {"requests": 0, "prefill_ms": 0.0, "decode_ms": 0.0, "kv_bytes": 0,
           "save_ms": 0.0, "load_ms": 0.0, "tokens": 0}
LOCK = threading.Lock()


def resolve_chip(want):
    """--chip -1 means "whichever card we were actually given".

    The Furiosa device plugin passes the physical device nodes through with
    their original index (a pod granted one card can see /dev/rngd/npu2*), so a
    hardcoded rngd:0 dies with "device index out of range" inside k8s.
    """
    if want >= 0:
        return want
    import glob
    import re
    seen = sorted({int(m.group(1)) for p in glob.glob("/dev/rngd/npu*mgmt")
                   for m in [re.search(r"npu(\d+)mgmt", p)] if m})
    if not seen:
        raise RuntimeError("no RNGD device visible under /dev/rngd")
    print(f"[chip] devices visible: {seen} -> using rngd:{seen[0]}", flush=True)
    return seen[0]


class _CfgFromNpz:
    """Just the fields the decode path reads, rebuilt from the saved tables."""

    def __init__(self, z):
        self.n_layers = int(z["nl"]); self.hidden = int(z["hidden"])
        self.n_kv = int(z["n_kv"]); self.head_dim = int(z["head_dim"])
        self._inv = z["inv"]

    def inv_freq(self):
        return self._inv


def _conn(n_layers):
    from rngd_pd.connector import transport as kv_transport
    kv_transport.WIRE = "bf16"                    # bit-exact for bf16-valued KV
    from rngd_pd.connector.transport import ShmTransport
    from rngd_pd.connector.rngd_kv import RngdKVConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    os.makedirs(SHM_ROOT, exist_ok=True)
    return RngdKVConnector, KVConnectorRole, ShmTransport


# --------------------------------------------------------------------------
# prefill worker (Furiosa RNGD, cached EDF)
# --------------------------------------------------------------------------
class PrefillWorker:
    def __init__(self, chip, S):
        chip = resolve_chip(chip)
        import torch
        torch.set_grad_enabled(False)
        from rngd_pd.kernels import decode_resident as LER
        from rngd_pd.engines.decode_resident import build_engine, EngineSession
        from rngd_pd.engines.model import rmsnorm
        self.LER, self.rmsnorm = LER, rmsnorm
        t0 = time.time()
        self.cfg, self.rt = build_engine(MODEL, S, 32, chip=chip)
        self.sess = EngineSession(self.rt, LER)
        self.S, self.nl = S, self.rt.nl
        STATE.update(ready=True,
                     detail=f"prefill rngd:{chip} {self.nl}L S={S} "
                            f"loaded in {time.time()-t0:.0f}s")
        print(f"[prefill] {STATE['detail']}", flush=True)

    def run(self, req_id, tokens):
        RngdKVConnector, KVConnectorRole, ShmTransport = _conn(self.nl)
        with DEVICE:
            t0 = time.perf_counter()
            Kp, Vp = self.sess.fresh_kv()
            for pos, t in enumerate(tokens):
                hid, Kp, Vp = self._step(t, pos, Kp, Vp)
            nxt = int(np.argmax(self.rt.lm @ self.rmsnorm(hid, self.rt.norm,
                                                          self.rt.c.eps)))
            prefill_ms = (time.perf_counter() - t0) * 1e3
            Kh, Vh = self.sess.export_kv(Kp, Vp)
        t1 = time.perf_counter()
        conn = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id,
                               transport=ShmTransport(SHM_ROOT), n_layers=self.nl)
        for i in range(self.nl):
            conn.save_kv_layer(f"layer.{i}", (Kh[i], Vh[i]))
        conn.wait_for_save()
        save_ms = (time.perf_counter() - t1) * 1e3
        with LOCK:
            METRICS["prefill_ms"] += prefill_ms
            METRICS["save_ms"] += save_ms
            METRICS["kv_bytes"] += Kh.nbytes + Vh.nbytes
        return {"next": nxt, "pos": len(tokens), "layers": self.nl,
                "kv_mb": round((Kh.nbytes + Vh.nbytes) / 2 / 1e6, 1),
                "prefill_ms": round(prefill_ms, 1), "save_ms": round(save_ms, 1)}

    def _step(self, tok, pos, Kp, Vp):
        L, rt = self.LER, self.rt
        an1, aq, ak, av, ao, an2, awgu, awd = rt.W
        ang = pos * rt.inv
        mask = np.full(rt.S, -1e4, np.float32); mask[:pos + 1] = 0.0
        oh = np.zeros(rt.S, np.float32); oh[pos] = 1.0
        x = L._d(rt.embed[tok].astype(np.float32))
        hid, Kp, Vp = rt.m(x, rt.eps_d, rt.nh_d, an1, aq, ak, av, ao, an2, awgu,
                           awd, Kp, Vp, L._d(np.cos(ang)), L._d(np.sin(ang)),
                           L._d(mask), L._d(oh))
        return hid.detach().float().cpu().numpy(), Kp, Vp


class FullPrefillWorker:
    """Prefill the whole prompt in one pass per layer group — the kv-free graph
    that closed the gap with the vendor engine (0.14 ms/token against 0.10).
    KV comes out in the connector's layout, so the decode side is untouched."""

    def __init__(self, chip, S, C=256, group=2):
        chip = resolve_chip(chip)
        import torch
        torch.set_grad_enabled(False)
        from transformers import AutoModelForCausalLM
        from rngd_pd.engines.model import ModelConfig
        from rngd_pd.engines.prefill_full import FullPrefillEngine
        t0 = time.time()
        hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        cfg = ModelConfig(hf.config)
        sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
        del hf
        cache = str(ARTIFACT_CACHE)
        self.eng = FullPrefillEngine(cfg, sd, C=C, group=group, chip=chip,
                                     cache_dir=os.path.abspath(cache))
        del sd
        self.S, self.nl, self.C = C, cfg.n_layers, C
        STATE.update(ready=True,
                     detail=f"prefill rngd:{chip} kv-free C={C} {self.nl}L "
                            f"ready in {time.time()-t0:.0f}s")
        print(f"[prefill] {STATE['detail']}", flush=True)

    def run(self, req_id, tokens):
        RngdKVConnector, KVConnectorRole, ShmTransport = _conn(self.nl)
        with DEVICE:
            t0 = time.perf_counter()
            Kh, Vh, nxt = self.eng.prefill(tokens)
            prefill_ms = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        conn = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id,
                               transport=ShmTransport(SHM_ROOT), n_layers=self.nl)
        for i in range(self.nl):
            conn.save_kv_layer(f"layer.{i}", (Kh[i], Vh[i]))
        conn.wait_for_save()
        save_ms = (time.perf_counter() - t1) * 1e3
        with LOCK:
            METRICS["prefill_ms"] += prefill_ms
            METRICS["save_ms"] += save_ms
            METRICS["kv_bytes"] += Kh.nbytes + Vh.nbytes
        return {"next": nxt, "pos": len(tokens), "layers": self.nl,
                "kv_mb": round((Kh.nbytes + Vh.nbytes) / 2 / 1e6, 1),
                "prefill_ms": round(prefill_ms, 1), "save_ms": round(save_ms, 1),
                "ms_per_token": round(prefill_ms / max(len(tokens), 1), 3)}


class PrefillATOM:
    """Chunked prefill on a Rebellions card — the other half of pd_reverse.

    ATOM compiles a multi-position graph where the RNGD eager path could not,
    so prefill here walks the prompt C positions at a time against the cache it
    is filling. Same connector layout out, so any decode side can take it.
    """

    def __init__(self, S, C=16, device=0):
        import torch
        torch.set_grad_enabled(False)
        from rngd_pd.vendors.rbln_prefill import build_runtime
        art = os.environ["RBLN_PREFILL_ARTIFACT"]
        z = np.load(os.environ["RBLN_TABLES"])
        self.embed, self.lm = z["embed"].astype(np.float32), z["lm"].astype(np.float32)
        self.cfg = _CfgFromNpz(z)
        t0 = time.time()
        self.rt = build_runtime(None, self.cfg, S, self.cfg.n_layers, C,
                                f"prefill {self.cfg.n_layers}L C={C}",
                                artifact=art, device=device)
        self.S, self.C, self.nl = S, C, self.cfg.n_layers
        self.NKV, self.HD = self.cfg.n_kv, self.cfg.head_dim
        self.inv = self.cfg.inv_freq()
        STATE.update(ready=True,
                     detail=f"prefill atom {self.nl}L C={C} S={S} ready in "
                            f"{time.time()-t0:.0f}s")
        print(f"[prefill] {STATE['detail']}", flush=True)

    def _chunk(self, base, toks):
        C, S = self.C, self.S
        ang = np.arange(base, base + C)[:, None] * self.inv[None, :]
        mask = np.full((C, S + C), -1e4, np.float32)
        mask[:, :base] = 0.0
        for r in range(C):
            mask[r, S:S + r + 1] = 0.0
        return (self.embed[toks].astype(np.float32),
                np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32),
                mask)

    def run(self, req_id, tokens):
        RngdKVConnector, KVConnectorRole, ShmTransport = _conn(self.nl)
        C, P = self.C, len(tokens)
        pad = list(tokens) + [tokens[-1]] * (-P % C)     # whole chunks only
        K = np.zeros((self.nl, self.NKV, self.HD, self.S), np.float32)
        V = np.zeros((self.nl, self.NKV, self.S, self.HD), np.float32)
        t0 = time.perf_counter()
        with DEVICE:
            for base in range(0, len(pad), C):
                x, cos, sin, mask = self._chunk(base, pad[base:base + C])
                out = self.rt.run(x, K, V, cos, sin, mask)
                hid = np.asarray(out[0])
                K[:, :, :, base:base + C] = np.asarray(out[1])
                V[:, :, base:base + C, :] = np.asarray(out[2])
        prefill_ms = (time.perf_counter() - t0) * 1e3
        # Chunking writes whole C-wide blocks, so the slots past the prompt hold
        # the padding tokens' K/V. They have to go: a decode step injects its
        # own k/v by adding at a one-hot slot, which is only equal to a write if
        # that slot starts at zero.
        K[:, :, :, P:] = 0.0          # [nl,NKV,HD,S]
        V[:, :, P:, :] = 0.0          # [nl,NKV,S,HD]
        nxt = int(np.argmax(self.lm @ hid[(P - 1) % C]))
        t1 = time.perf_counter()
        conn = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id,
                               transport=ShmTransport(SHM_ROOT), n_layers=self.nl)
        for i in range(self.nl):
            conn.save_kv_layer(f"layer.{i}", (K[i], V[i]))
        conn.wait_for_save()
        save_ms = (time.perf_counter() - t1) * 1e3
        with LOCK:
            METRICS["prefill_ms"] += prefill_ms
            METRICS["save_ms"] += save_ms
            METRICS["kv_bytes"] += K.nbytes + V.nbytes
        return {"next": nxt, "pos": P, "layers": self.nl,
                "kv_mb": round((K.nbytes + V.nbytes) / 2 / 1e6, 1),
                "prefill_ms": round(prefill_ms, 1), "save_ms": round(save_ms, 1),
                "ms_per_token": round(prefill_ms / max(P, 1), 3)}


class ChunkPrefillWorker:
    """Prefill on RNGD with the chunked low-level graph instead of one token
    per pass. Emits the same KV layout, so the connector and every decode
    backend stay unchanged."""

    def __init__(self, chip, S, C=16, group=2):
        chip = resolve_chip(chip)
        import torch
        torch.set_grad_enabled(False)
        from transformers import AutoModelForCausalLM
        from rngd_pd.engines.model import ModelConfig
        from rngd_pd.engines.prefill_chunk import ChunkPrefillEngine
        t0 = time.time()
        hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        cfg = ModelConfig(hf.config)
        sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
        del hf
        self.eng = ChunkPrefillEngine(cfg, sd, S=S, C=C, group=group, chip=chip)
        del sd
        self.S, self.nl, self.C = S, cfg.n_layers, C
        STATE.update(ready=True,
                     detail=f"prefill rngd:{chip} chunked C={C} {self.nl}L S={S} "
                            f"ready in {time.time()-t0:.0f}s")
        print(f"[prefill] {STATE['detail']}", flush=True)

    def run(self, req_id, tokens):
        RngdKVConnector, KVConnectorRole, ShmTransport = _conn(self.nl)
        with DEVICE:
            t0 = time.perf_counter()
            Kh, Vh, nxt = self.eng.prefill(tokens)
            prefill_ms = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        conn = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id,
                               transport=ShmTransport(SHM_ROOT), n_layers=self.nl)
        for i in range(self.nl):
            conn.save_kv_layer(f"layer.{i}", (Kh[i], Vh[i]))
        conn.wait_for_save()
        save_ms = (time.perf_counter() - t1) * 1e3
        with LOCK:
            METRICS["prefill_ms"] += prefill_ms
            METRICS["save_ms"] += save_ms
            METRICS["kv_bytes"] += Kh.nbytes + Vh.nbytes
        return {"next": nxt, "pos": len(tokens), "layers": self.nl,
                "kv_mb": round((Kh.nbytes + Vh.nbytes) / 2 / 1e6, 1),
                "prefill_ms": round(prefill_ms, 1), "save_ms": round(save_ms, 1),
                "ms_per_token": round(prefill_ms / max(len(tokens), 1), 2)}


# --------------------------------------------------------------------------
# decode workers
# --------------------------------------------------------------------------
class DecodeWorkerBase:
    def _load_kv(self, req_id):
        RngdKVConnector, KVConnectorRole, ShmTransport = _conn(self.nl)
        dk = [np.zeros((self.NKV, self.HD, self.S), np.float32) for _ in range(self.nl)]
        dv = [np.zeros((self.NKV, self.S, self.HD), np.float32) for _ in range(self.nl)]
        conn = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id,
                               transport=ShmTransport(SHM_ROOT), n_layers=self.nl)
        conn.register_kv_caches((dk, dv))
        conn.bind_connector_metadata(conn.build_connector_meta(None))
        t0 = time.perf_counter()
        conn.start_load_kv()
        for i in range(self.nl):
            conn.wait_for_layer_load(f"layer.{i}")
        load_ms = (time.perf_counter() - t0) * 1e3
        for f in os.listdir(SHM_ROOT):                # the handoff is consumed;
            if f.startswith(req_id + "__"):           # leaving it would grow
                try:                                  # /dev/shm by 34 MB/request
                    os.unlink(os.path.join(SHM_ROOT, f))
                except OSError:
                    pass
        return np.stack(dk), np.stack(dv), load_ms


class DecodeATOM(DecodeWorkerBase):
    """Rebellions ATOM: 32 layers in fp16 on-card, embedding+lm_head on host.

    Startup takes one of two paths. With RBLN_ARTIFACT pointing at a compiled
    model (plus its embedding/lm_head tables) it loads those and never touches
    the 32 GB fp32 checkpoint — the vendor's documented container story is
    runtime inference, compilation happens on the host. Otherwise it builds and
    compiles, and writes both artifacts out for next time.
    """

    def __init__(self, S):
        import torch
        torch.set_grad_enabled(False)
        from rngd_pd.vendors.rbln_decode import LlamaDecodeRBLN, rbln_compile, load_hf
        from rngd_pd.engines.model import ModelConfig
        art = os.environ.get("RBLN_ARTIFACT", "")
        tab = os.environ.get("RBLN_TABLES", "")
        t0 = time.time()
        if art and tab and os.path.exists(art) and os.path.exists(tab):
            z = np.load(tab)
            self.embed, self.lm = z["embed"].astype(np.float32), z["lm"].astype(np.float32)
            cfg = _CfgFromNpz(z)
            self.rt = rbln_compile(None, cfg, S, int(z["nl"]),
                                   f"llama {int(z['nl'])}L S={S} fp16", artifact=art)
        else:
            from transformers import AutoConfig
            _tk, cfg, sd = load_hf()
            self.embed = sd["model.embed_tokens.weight"]
            self.lm = sd["lm_head.weight"]
            mod = LlamaDecodeRBLN(cfg, sd, S, cfg.n_layers).eval()
            del sd
            self.rt = rbln_compile(mod, cfg, S, cfg.n_layers,
                                   f"llama {cfg.n_layers}L S={S} fp16", artifact=art or None)
            del mod
            if tab:
                np.savez(tab, embed=self.embed.astype(np.float16),
                         lm=self.lm.astype(np.float16), nl=cfg.n_layers,
                         hidden=cfg.hidden, n_kv=cfg.n_kv, head_dim=cfg.head_dim,
                         inv=cfg.inv_freq())
                print(f"[decode] saved tables -> {tab}", flush=True)
        self.cfg, self.S, self.nl = cfg, S, cfg.n_layers
        self.NKV, self.HD = cfg.n_kv, cfg.head_dim
        self.inv = cfg.inv_freq()
        STATE.update(ready=True,
                     detail=f"decode atom {self.nl}L S={S} fp16 ready in "
                            f"{time.time()-t0:.0f}s")
        print(f"[decode] {STATE['detail']}", flush=True)

    def run(self, req_id, first, pos, max_new):
        K, V, load_ms = self._load_kv(req_id)
        outs, ts = [first], []
        cur, p = first, pos
        t0 = time.perf_counter()
        with DEVICE:                    # one generation owns the card end to end
            for _ in range(max_new - 1):
                ang = p * self.inv
                mask = np.concatenate([np.zeros(p + 1, np.float32),
                                       np.full(self.S - p - 1, -1e4, np.float32)])
                t1 = time.perf_counter()
                out = self.rt.run(self.embed[cur].astype(np.float32).reshape(1, -1),
                                  K, V, np.cos(ang).astype(np.float32),
                                  np.sin(ang).astype(np.float32), mask,
                                  np.eye(self.S, dtype=np.float32)[p])
                ts.append((time.perf_counter() - t1) * 1e3)
                K[:, :, :, p] = np.asarray(out[1])      # host-side KV scatter
                V[:, :, p, :] = np.asarray(out[2])
                cur = int(np.argmax(self.lm @ np.asarray(out[0]).reshape(-1)))
                outs.append(cur)
                p += 1
        return self._done(outs, ts, load_ms, time.perf_counter() - t0)

    def _done(self, outs, ts, load_ms, wall):
        with LOCK:
            METRICS["decode_ms"] += wall * 1e3
            METRICS["load_ms"] += load_ms
            METRICS["tokens"] += len(outs)
        ts.sort()
        return {"tokens": outs, "load_ms": round(load_ms, 1),
                "tpot_ms": round(ts[len(ts) // 2], 1) if ts else 0.0,
                "decode_ms": round(wall * 1e3, 1)}


class _Batcher:
    """Holds arriving requests briefly so a B-wide graph pass carries B of them.

    One pass reads the whole 14 GB of weights whether it advances one stream or
    four, so the only way batching shows up as throughput is if concurrent
    requests meet inside one call. Callers block in submit() exactly as they
    would on a serial worker; what changes is that B of them wake together.
    """

    def __init__(self, B, wait_ms, runner):
        self.B, self.wait, self.runner = B, wait_ms / 1e3, runner
        self.q, self.cv = [], threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, item):
        slot = {"in": item, "out": None, "err": None, "ev": threading.Event()}
        with self.cv:
            self.q.append(slot)
            self.cv.notify()
        slot["ev"].wait()
        if slot["err"]:
            raise slot["err"]
        return slot["out"]

    def _loop(self):
        while True:
            with self.cv:
                while not self.q:
                    self.cv.wait()
                end = time.time() + self.wait          # let the rest of the batch land
                while len(self.q) < self.B and time.time() < end:
                    self.cv.wait(timeout=max(end - time.time(), 1e-3))
                batch = self.q[:self.B]
                del self.q[:len(batch)]
            try:
                for slot, out in zip(batch, self.runner([s["in"] for s in batch])):
                    slot["out"] = out
            except Exception as e:                     # one bad request fails its batch
                for slot in batch:
                    slot["err"] = e
            for slot in batch:
                slot["ev"].set()


class DecodeATOMBatch(DecodeWorkerBase):
    """B independent streams per graph pass, from the pre-compiled B artifact.

    Each stream brings its own KV out of /dev/shm; they are stacked into the
    [nl, B, ...] cache the batched graph expects and advanced in lockstep. A
    batch that arrives short is padded by repeating its last stream, so the
    fixed-shape graph still runs — the padded lanes are dropped on the way out.
    """

    def __init__(self, S, B, wait_ms=60):
        import torch
        torch.set_grad_enabled(False)
        from rngd_pd.vendors.rbln_decode_batch import build_runtime
        art = os.environ["RBLN_BATCH_ARTIFACT"]
        z = np.load(os.environ["RBLN_TABLES"])
        self.embed, self.lm = z["embed"].astype(np.float32), z["lm"].astype(np.float32)
        self.cfg = _CfgFromNpz(z)
        t0 = time.time()
        self.rt = build_runtime(None, self.cfg, S, self.cfg.n_layers, B,
                                f"decode {self.cfg.n_layers}L B={B}",
                                artifact=art, device=0)
        self.S, self.B, self.nl = S, B, self.cfg.n_layers
        self.NKV, self.HD = self.cfg.n_kv, self.cfg.head_dim
        self.inv = self.cfg.inv_freq()
        self.batcher = _Batcher(B, wait_ms, self._run_batch)
        STATE.update(ready=True,
                     detail=f"decode atom B={B} {self.nl}L S={S} ready in "
                            f"{time.time()-t0:.0f}s")
        print(f"[decode] {STATE['detail']}", flush=True)

    def run(self, req_id, first, pos, max_new):
        return self.batcher.submit({"req_id": req_id, "first": first,
                                    "pos": pos, "max_new": max_new})

    def _run_batch(self, items):
        B, S, n = self.B, self.S, len(items)
        Ks, Vs, loads = [], [], []
        for it in items:
            K, V, load_ms = self._load_kv(it["req_id"])
            Ks.append(K); Vs.append(V); loads.append(load_ms)
        while len(Ks) < B:                              # pad the unused lanes
            Ks.append(Ks[-1]); Vs.append(Vs[-1])
        Kb = np.stack(Ks, 1)                            # [nl,B,NKV,HD,S]
        Vb = np.stack(Vs, 1)                            # [nl,B,NKV,S,HD]
        pos = [it["pos"] for it in items]
        cur = [it["first"] for it in items]
        pos += [pos[-1]] * (B - n)
        cur += [cur[-1]] * (B - n)
        max_new = max(it["max_new"] for it in items)
        outs = [[c] for c in cur]
        ts = []
        t0 = time.perf_counter()
        with DEVICE:
            for _ in range(max_new - 1):
                ang = np.stack([p * self.inv for p in pos]).astype(np.float32)
                mask = np.stack([np.concatenate(
                    [np.zeros(p + 1, np.float32),
                     np.full(S - p - 1, -1e4, np.float32)]) for p in pos])
                oh = np.stack([np.eye(S, dtype=np.float32)[p] for p in pos])
                x = np.stack([self.embed[c].astype(np.float32) for c in cur])
                t1 = time.perf_counter()
                out = self.rt.run(x, Kb, Vb, np.cos(ang).astype(np.float32),
                                  np.sin(ang).astype(np.float32), mask, oh)
                ts.append((time.perf_counter() - t1) * 1e3)
                k, v = np.asarray(out[1]), np.asarray(out[2])   # [nl,B,NKV,HD]
                h = np.asarray(out[0])                          # [B,H]
                for b in range(B):
                    Kb[:, b, :, :, pos[b]] = k[:, b]      # [nl,B,NKV,HD,S]
                    Vb[:, b, :, pos[b], :] = v[:, b]      # [nl,B,NKV,S,HD]
                    cur[b] = int(np.argmax(self.lm @ h[b]))
                    outs[b].append(cur[b])
                    pos[b] += 1
        wall = time.perf_counter() - t0
        ts.sort()
        step = ts[len(ts) // 2] if ts else 0.0
        with LOCK:
            METRICS["decode_ms"] += wall * 1e3
            METRICS["tokens"] += sum(len(o) for o in outs[:n])
        return [{"tokens": outs[i], "load_ms": round(loads[i], 1),
                 "tpot_ms": round(step, 1), "decode_ms": round(wall * 1e3, 1),
                 "batch": n, "batch_width": B} for i in range(n)]


class DecodeRNGDBatch(DecodeWorkerBase):
    """B streams per step on a Furiosa card.

    The eager path has no 32-layer batched graph — tactic search stops at two
    layers — so this walks the stack with one compiled two-layer module, the
    same reuse the kv-free prefill needed. The cache stays on the card between
    steps and between groups; only the initial load crosses from /dev/shm.

    Streams keep their own positions, so a batch can mix a request on its third
    token with one on its hundredth.
    """

    def __init__(self, chip, S, B, group=2, wait_ms=8000):
        chip = resolve_chip(chip)
        import torch
        torch.set_grad_enabled(False)
        from transformers import AutoModelForCausalLM
        from rngd_pd.engines.model import ModelConfig
        from rngd_pd.engines.decode_batch import BatchDecodeEngine
        t0 = time.time()
        hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        cfg = ModelConfig(hf.config)
        sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
        del hf
        cache = str(ARTIFACT_CACHE)
        self.eng = BatchDecodeEngine(cfg, sd, B=B, S=S, group=group, chip=chip,
                                     cache_dir=cache)
        del sd
        self.S, self.B, self.nl = S, B, cfg.n_layers
        self.NKV, self.HD = cfg.n_kv, cfg.head_dim
        self.batcher = _Batcher(B, wait_ms, self._run_batch)
        STATE.update(ready=True,
                     detail=f"decode rngd:{chip} B={B} {self.nl}L S={S} "
                            f"ready in {time.time()-t0:.0f}s")
        print(f"[decode] {STATE['detail']}", flush=True)

    def run(self, req_id, first, pos, max_new):
        return self.batcher.submit({"req_id": req_id, "first": first,
                                    "pos": pos, "max_new": max_new})

    def _run_batch(self, items):
        B, n = self.B, len(items)
        Ks, Vs, loads = [], [], []
        for it in items:
            K, V, load_ms = self._load_kv(it["req_id"])
            Ks.append(K); Vs.append(V); loads.append(load_ms)
        pos = [it["pos"] for it in items]
        cur = [it["first"] for it in items]
        pos += [pos[-1]] * (B - n)                  # padded lanes ride along
        cur += [cur[-1]] * (B - n)
        max_new = max(it["max_new"] for it in items)
        outs = [[c] for c in cur]
        ts = []
        t0 = time.perf_counter()
        with DEVICE:
            Kd, Vd = self.eng.load_kv(Ks, Vs)
            for _ in range(max_new - 1):
                t1 = time.perf_counter()
                cur, Kd, Vd = self.eng.step(cur, pos, Kd, Vd)
                ts.append((time.perf_counter() - t1) * 1e3)
                for b in range(B):
                    outs[b].append(cur[b])
                    pos[b] += 1
        wall = time.perf_counter() - t0
        ts.sort()
        step = ts[len(ts) // 2] if ts else 0.0
        with LOCK:
            METRICS["decode_ms"] += wall * 1e3
            METRICS["tokens"] += sum(len(o) for o in outs[:n])
        return [{"tokens": outs[i], "load_ms": round(loads[i], 1),
                 "tpot_ms": round(step, 1), "decode_ms": round(wall * 1e3, 1),
                 "batch": n, "batch_width": B} for i in range(n)]


class DecodeRNGD(DecodeWorkerBase):
    """Second Furiosa card — the homogeneous control for the same handoff."""

    def __init__(self, chip, S):
        chip = resolve_chip(chip)
        import torch
        torch.set_grad_enabled(False)
        from rngd_pd.kernels import decode_resident as LER
        from rngd_pd.engines.decode_resident import (build_engine, build_lmhead,
                                                     EngineSession)
        from rngd_pd.engines.model import rmsnorm
        self.LER, self.rmsnorm = LER, rmsnorm
        t0 = time.time()
        self.cfg, self.rt = build_engine(MODEL, S, 32, chip=chip)
        # lm_head on the card, not in numpy: a 2.1 GB host read per token is
        # a third of this step, and the batched worker pays it per stream.
        # Keeping both workers on the same placement is also what makes the
        # B=1 and B=4 rows of the sweep comparable.
        self.sess = EngineSession(self.rt, LER, lmhead=build_lmhead(self.rt, LER, chip))
        self.S, self.nl = S, self.rt.nl
        self.NKV, self.HD = LER.NKV, LER.HD
        STATE.update(ready=True,
                     detail=f"decode rngd:{chip} {self.nl}L S={S} device-lmhead "
                            f"loaded in {time.time()-t0:.0f}s")
        print(f"[decode] {STATE['detail']}", flush=True)

    def run(self, req_id, first, pos, max_new):
        K, V, load_ms = self._load_kv(req_id)
        Kp, Vp = self.sess.import_kv(K, V)
        outs, ts = [first], []
        cur, p = first, pos
        t0 = time.perf_counter()
        for _ in range(max_new - 1):
            t1 = time.perf_counter()
            with DEVICE:
                cur, Kp, Vp = self.sess.step(cur, p, Kp, Vp)
            ts.append((time.perf_counter() - t1) * 1e3)
            outs.append(cur)
            p += 1
        return DecodeATOM._done(self, outs, ts, load_ms, time.perf_counter() - t0)


# --------------------------------------------------------------------------
# gateway
# --------------------------------------------------------------------------
class Gateway:
    """Fronts every prefill/decode pairing behind one llm-d route.

    Which two cards serve a request is chosen by the model name, the way any
    other inference gateway picks a backend, and the decode batch width by a
    `batch` field — so all four configurations in the sweep enter through the
    same Envoy → InferencePool → EPP path instead of four parallel ones.
    """

    def __init__(self, prefill, decode, S, targets=None):
        from transformers import AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(MODEL)
        self.prefill, self.decode, self.S = prefill, decode, S
        self.targets = targets or {}
        self.seq = 0
        self.seq_lock = threading.Lock()
        detail = (f"gateway → {len(self.targets)} configs: "
                  f"{', '.join(sorted(self.targets))}" if self.targets
                  else f"gateway → P={prefill} D={decode}")
        STATE.update(ready=True, detail=detail)
        print(f"[gateway] {detail}", flush=True)

    def _route(self, body):
        """(prefill url, decode url, config name) for this request."""
        name = body.get("model", "")
        tgt = self.targets.get(name)
        if tgt is None:
            if self.targets:
                raise ValueError(f"unknown model {name!r}; "
                                 f"have {sorted(self.targets)}")
            return self.prefill, self.decode, "default"
        dec = tgt["decode"]
        if isinstance(dec, dict):                        # one pod per batch width
            B = str(int(body.get("batch", 1)))
            if B not in dec:
                raise ValueError(f"{name}: batch {B} not served; "
                                 f"have {sorted(dec)}")
            dec = dec[B]
        return tgt["prefill"], dec, name

    def complete(self, body):
        prompt = body.get("prompt", "")
        if not prompt and body.get("messages"):          # chat-style callers
            prompt = "\n".join(m.get("content", "") for m in body["messages"])
        max_new = int(body.get("max_tokens", 8))
        toks = self.tk.encode(prompt, add_special_tokens=False)[:self.S - max_new - 1]
        if not toks:
            raise ValueError(f"empty prompt (body keys: {sorted(body)})")
        prefill, decode, cfg_name = self._route(body)
        with self.seq_lock:                              # concurrent callers
            self.seq += 1
            seq = self.seq
        req_id = f"pd-{int(time.time()*1000)}-{seq}"
        t0 = time.perf_counter()
        pf = _post(f"{prefill}/prefill", {"req_id": req_id, "tokens": toks})
        ttft_ms = (time.perf_counter() - t0) * 1e3
        dc = _post(f"{decode}/decode", {"req_id": req_id, "first": pf["next"],
                                        "pos": pf["pos"], "max_new": max_new})
        wall_ms = (time.perf_counter() - t0) * 1e3
        with LOCK:
            METRICS["requests"] += 1
        text = self.tk.decode(dc["tokens"])
        return {"id": req_id, "object": "text_completion", "model": MODEL,
                "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(toks),
                          "completion_tokens": len(dc["tokens"]),
                          "total_tokens": len(toks) + len(dc["tokens"])},
                "pd": {"mode": "disaggregated-kv-transfer", "config": cfg_name,
                       "prefill_ms": pf["prefill_ms"], "ttft_ms": round(ttft_ms, 1),
                       "kv_mb": pf["kv_mb"], "kv_save_ms": pf["save_ms"],
                       "kv_load_ms": dc["load_ms"], "tpot_ms": dc["tpot_ms"],
                       "decode_ms": dc.get("decode_ms", 0.0),
                       "batch": dc.get("batch", 1),
                       "batch_width": dc.get("batch_width", 1),
                       "wall_ms": round(wall_ms, 1)}}


def _post(url, obj, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------
def make_handler(routes):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/healthz"):
                return self._send(200 if STATE["ready"] else 503, STATE)
            if self.path.rstrip("/") == "/metrics":
                return self._send(200, dict(METRICS, **STATE))
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"object": "list", "data": [
                    {"id": MODEL, "object": "model", "owned_by": "rngd-llmd"}]})
            self._send(404, {"error": "not found"})

        def _read_body(self):
            """Envoy forwards chunked, so Content-Length alone loses the body."""
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                out = []
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        self.rfile.readline()
                        break
                    out.append(self.rfile.read(size))
                    self.rfile.readline()
                return b"".join(out)
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def do_POST(self):
            try:
                body = json.loads(self._read_body() or b"{}")
            except Exception as e:
                return self._send(400, {"error": str(e)})
            path = self.path.rstrip("/")
            fn = routes.get(path)
            if fn is None:
                return self._send(404, {"error": f"no route {path}"})
            if not STATE["ready"]:
                return self._send(503, STATE)
            try:
                return self._send(200, fn(body))
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._send(500, {"error": repr(e)})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["prefill", "decode", "gateway"])
    ap.add_argument("--backend", default="atom",
                    choices=["atom", "rngd", "rngd-chunk", "rngd-full"])
    ap.add_argument("--B", type=int, default=1, help="decode batch width")
    ap.add_argument("--batch-wait", type=float, default=60.0,
                    help="ms to hold a partial batch before running it")
    ap.add_argument("--C", type=int, default=16, help="chunked prefill chunk size")
    ap.add_argument("--group", type=int, default=2, help="layers per compiled call")
    ap.add_argument("--chip", type=int, default=-1,  # -1 = whichever card k8s gave us
                    help="RNGD device index; -1 auto-detects from /dev/rngd")
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--prefill", default="http://127.0.0.1:8101")
    ap.add_argument("--decode", default="http://127.0.0.1:8102")
    ap.add_argument("--targets-file", default="",
                    help="JSON map: model name -> {prefill, decode|{B: url}}")
    a = ap.parse_args()

    routes = {}
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(routes))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[{a.role}] listening on {a.host}:{a.port}", flush=True)

    if a.role == "prefill":
        if a.backend == "rngd-full":
            w = FullPrefillWorker(a.chip, a.S, a.C, a.group)
        elif a.backend == "rngd-chunk":
            w = ChunkPrefillWorker(a.chip, a.S, a.C, a.group)
        elif a.backend == "atom":
            w = PrefillATOM(a.S, a.C)
        else:
            w = PrefillWorker(a.chip, a.S)
        routes["/prefill"] = lambda b: w.run(b["req_id"], b["tokens"])
    elif a.role == "decode":
        if a.backend == "atom":
            # the batch artifact is what decides: with one present every width
            # (including B=1) goes through the same batched graph, so the sweep
            # compares like with like.
            w = (DecodeATOMBatch(a.S, a.B, a.batch_wait)
                 if os.environ.get("RBLN_BATCH_ARTIFACT") else DecodeATOM(a.S))
        elif a.B > 1:
            w = DecodeRNGDBatch(a.chip, a.S, a.B, a.group, a.batch_wait)
        else:
            w = DecodeRNGD(a.chip, a.S)
        routes["/decode"] = lambda b: w.run(b["req_id"], b["first"], b["pos"],
                                            b.get("max_new", 8))
    else:
        targets = None
        if a.targets_file:
            with open(a.targets_file) as fh:
                targets = json.load(fh)
        g = Gateway(a.prefill, a.decode, a.S, targets)
        routes["/v1/completions"] = g.complete

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
