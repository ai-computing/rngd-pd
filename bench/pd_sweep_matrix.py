#!/usr/bin/env python3
"""Four ways to spend one RNGD and one ATOM, measured across context and batch.

  rngd_alone   Furiosa does the whole request
  atom_alone   Rebellions does the whole request (chunked prefill, batched decode)
  pd_forward   RNGD prefills, ATOM decodes      <- what the cluster runs today
  pd_reverse   ATOM prefills, RNGD decodes      <- the faster assignment

Each cell reports TTFT, TPOT, output tokens/s, card power, utilisation, and the
weight-stream bandwidth the step implies (weight bytes / step time), which is
the number that explains why chunking and batching pay off at all.

Batch sweeps only where the decoder is ATOM: the RNGD eager path has no
multi-stream decode graph at 32 layers, which is a compiler limit we have
already mapped, not an omission here.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_sweep_matrix.py --out sweep.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

import _paths  # noqa: F401

torch.set_grad_enabled(False)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "connector")))

RNGD_WEIGHT_BYTES = 8.03e9 * 2      # 8.03 B params, bf16, resident on card
ATOM_WEIGHT_BYTES = 6.98e9 * 2      # 32 layers fp16; embed/lm_head stay on host


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------
class Sampler(threading.Thread):
    """Polls both vendors' tools while a workload runs."""

    def __init__(self, rngd_idx, atom_card, period=0.5):
        super().__init__(daemon=True)
        self.rngd_idx, self.atom_card, self.period = rngd_idx, atom_card, period
        self.stop_flag = threading.Event()
        self.rngd_w, self.rngd_u, self.atom_w, self.atom_u, self.atom_mem = ([] for _ in range(5))

    def _rngd(self):
        try:
            out = subprocess.run(["furiosa-smi", "info"], capture_output=True,
                                 text=True, timeout=4).stdout
            for line in out.splitlines():
                if f"npu{self.rngd_idx}" in line:
                    self.rngd_w.append(float(line.split("|")[5].strip().rstrip(" W")))
            out = subprocess.run(["furiosa-smi", "status"], capture_output=True,
                                 text=True, timeout=4).stdout
            # one device per block; blocks are separated by lines starting '+'.
            # Cores read '-' until a process opens the card.
            blocks, cur = [], []
            for line in out.splitlines():
                if line.startswith("+"):
                    if cur:
                        blocks.append(cur)
                    cur = []
                else:
                    cur.append(line)
            if cur:
                blocks.append(cur)
            for b in blocks:
                if f"npu{self.rngd_idx}" not in "\n".join(b):
                    continue
                vals = []
                for line in b:
                    if "Core" in line:
                        v = line.split(":")[-1].replace("|", "").replace("%", "").strip()
                        if v not in ("-", ""):
                            vals.append(float(v))
                if vals:
                    self.rngd_u.append(sum(vals) / len(vals))
                break
        except Exception:
            pass

    def _atom(self):
        try:
            js = json.loads(subprocess.run(["rbln-stat", "--json"], capture_output=True,
                                           text=True, timeout=4).stdout)
            for d in js.get("devices", []):
                if d.get("device") == self.atom_card:
                    self.atom_w.append(float(str(d["card_power"]).rstrip("uW")) / 1e6)
                    self.atom_u.append(float(d.get("util", 0)))
                    self.atom_mem.append(float(d["memory"]["used"]) / 1e9)
        except Exception:
            pass

    def run(self):
        while not self.stop_flag.is_set():
            self._rngd()
            self._atom()
            time.sleep(self.period)

    def result(self):
        f = lambda v: (round(float(np.mean(v)), 1) if v else None)
        g = lambda v: (round(float(np.max(v)), 1) if v else None)
        return {"rngd_power_w": f(self.rngd_w), "rngd_util_pct": g(self.rngd_u),
                "atom_power_w": f(self.atom_w), "atom_util_pct": g(self.atom_u),
                "atom_mem_gb": g(self.atom_mem)}


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------
class Engines:
    def __init__(self, a):
        from pd_hetero_llama import MODEL, load_hf
        from pd_prefill_atom import LlamaPrefillRBLN, build_runtime as prefill_rt
        from pd_decode_atom_batch import build_runtime as decode_rt
        import llama_engine_resident as LER
        from pd_engine import build_engine, EngineSession
        from transformers import AutoTokenizer

        self.a = a
        self.tk = AutoTokenizer.from_pretrained(MODEL)
        print(f"[rngd] engine on chip {a.chip}...", flush=True)
        self.cfg, self.rt = build_engine(MODEL, a.S, 32, chip=a.chip)
        self.sess = EngineSession(self.rt, LER)
        self.nl = self.rt.nl

        _tk, cfg2, sd = load_hf()
        self.embed = sd["model.embed_tokens.weight"]
        self.lm = sd["lm_head.weight"]
        self.cfg2 = cfg2
        del sd
        # One ATOM card holds 15.7 GB and each artifact carries 13.96 GB of
        # weights, so only one can be resident. Runtimes are created on demand
        # and the previous one dropped; swap time is excluded from all timings.
        self._prefill_rt_fn = prefill_rt
        self._decode_rt_fn = decode_rt
        self._atom = None            # (kind, B, runtime)
        self.atom_card = None        # which physical rbln card we actually use
        self._mem_before = self._rbln_mem()
        self.have_decode = {B for B in [int(b) for b in a.batches.split(",")]
                            if os.path.exists(f"{a.artifact_dir}/decode_S{a.S}_B{B}.rbln")}
        for B in [int(b) for b in a.batches.split(",")]:
            if B not in self.have_decode:
                print(f"[warn] missing decode artifact for B={B}", flush=True)

    @staticmethod
    def _rbln_mem():
        try:
            js = json.loads(subprocess.run(["rbln-stat", "--json"], capture_output=True,
                                           text=True, timeout=4).stdout)
            return {d["device"]: float(d["memory"]["used"]) for d in js.get("devices", [])}
        except Exception:
            return {}

    def atom_rt(self, kind, B=1):
        """Swap the resident ATOM artifact if the requested one is not loaded."""
        if self._atom and self._atom[0] == kind and self._atom[1] == B:
            return self._atom[2]
        if self._atom:
            import gc
            del self._atom
            self._atom = None
            gc.collect()          # release card memory before loading the next
        a = self.a
        if kind == "prefill":
            rt = self._prefill_rt_fn(None, self.cfg2, a.S, self.nl, a.C,
                                     f"prefill C={a.C}", artifact=a.prefill_artifact,
                                     device=a.atom_device)
        else:
            rt = self._decode_rt_fn(None, self.cfg2, a.S, self.nl, B,
                                    f"decode B={B}",
                                    artifact=f"{a.artifact_dir}/decode_S{a.S}_B{B}.rbln",
                                    device=a.atom_device)
        self._atom = (kind, B, rt)
        if self.atom_card is None:              # rebel's device index does not
            best, gain = None, 0.0              # follow rbln-stat's ordering, so
            for dev, used in self._rbln_mem().items():   # see which card grew
                d = used - self._mem_before.get(dev, 0.0)
                if d > gain:
                    best, gain = dev, d
            if best and gain > 1e9:
                self.atom_card = best
                print(f"[telemetry] ATOM work landed on {best} (+{gain/1e9:.1f} GB)",
                      flush=True)
        return rt

    # ---- primitives -------------------------------------------------------
    def _rngd_step_raw(self, tok, pos, Kp, Vp):
        """Engine step without the host lm_head — prefill only needs the KV."""
        import llama_engine_resident as LER
        rt = self.rt
        an1, aq, ak, av, ao, an2, awgu, awd = rt.W
        ang = pos * rt.inv
        mask = np.full(rt.S, -1e4, np.float32); mask[:pos + 1] = 0.0
        oh = np.zeros(rt.S, np.float32); oh[pos] = 1.0
        x = LER._d(rt.embed[tok].astype(np.float32))
        hid, Kp, Vp = rt.m(x, rt.eps_d, rt.nh_d, an1, aq, ak, av, ao, an2, awgu,
                           awd, Kp, Vp, LER._d(np.cos(ang)), LER._d(np.sin(ang)),
                           LER._d(mask), LER._d(oh))
        return hid, Kp, Vp

    def rngd_prefill(self, toks):
        from generic_runtime import rmsnorm
        Kp, Vp = self.sess.fresh_kv()
        for pos, t in enumerate(toks):
            hid, Kp, Vp = self._rngd_step_raw(t, pos, Kp, Vp)
        h = hid.detach().float().cpu().numpy()
        nxt = int(np.argmax(self.rt.lm @ rmsnorm(h, self.rt.norm, self.rt.c.eps)))
        return nxt, Kp, Vp

    def rngd_decode(self, first, pos, D, Kp, Vp):
        cur, ts = first, []
        for _ in range(D - 1):
            t0 = time.perf_counter()
            cur, Kp, Vp = self.sess.step(cur, pos, Kp, Vp)
            ts.append((time.perf_counter() - t0) * 1e3)
            pos += 1
        return ts

    def atom_prefill(self, toks):
        from pd_prefill_atom import chunk_inputs
        rt = self.atom_rt("prefill")
        C, S, nl = self.a.C, self.a.S, self.nl
        K = np.zeros((nl, self.cfg2.n_kv, self.cfg2.head_dim, S), np.float32)
        V = np.zeros((nl, self.cfg2.n_kv, S, self.cfg2.head_dim), np.float32)
        hid = None
        for base in range(0, len(toks), C):
            x, cos, sin, mask = chunk_inputs(self.cfg2, S, C, base, self.embed,
                                             toks[base:base + C])
            out = rt.run(x, K, V, cos, sin, mask)
            hid = np.asarray(out[0])
            K[:, :, :, base:base + C] = np.asarray(out[1])
            V[:, :, base:base + C, :] = np.asarray(out[2])
        return int(np.argmax(self.lm @ hid[-1])), K, V

    def atom_decode(self, first, pos0, D, K, V, B):
        """B streams that all start from the same KV — batch throughput, not xPyD."""
        rt = self.atom_rt("decode", B)
        S, nl = self.a.S, self.nl
        NKV, HD = self.cfg2.n_kv, self.cfg2.head_dim
        inv = self.cfg2.inv_freq()
        Kb = np.repeat(K[:, None], B, axis=1).copy()
        Vb = np.repeat(V[:, None], B, axis=1).copy()
        cur = np.full(B, first, np.int64)
        pos = np.full(B, pos0, np.int64)
        ts = []
        for _ in range(D - 1):
            x = self.embed[cur].astype(np.float32)
            ang = pos[:, None] * inv[None, :]
            mask = np.full((B, S), -1e4, np.float32)
            oh = np.zeros((B, S), np.float32)
            for b in range(B):
                mask[b, :pos[b] + 1] = 0.0
                oh[b, pos[b]] = 1.0
            t0 = time.perf_counter()
            out = rt.run(x, Kb, Vb, np.cos(ang).astype(np.float32),
                         np.sin(ang).astype(np.float32), mask, oh)
            ts.append((time.perf_counter() - t0) * 1e3)
            hid, k, v = (np.asarray(o) for o in out[:3])
            for b in range(B):
                Kb[:, b, :, :, pos[b]] = k[:, b]
                Vb[:, b, :, pos[b], :] = v[:, b]
            cur = np.argmax(hid @ self.lm.T, axis=-1)
            pos += 1
        return ts

    def handoff(self, K, V):
        import shutil
        import kv_transport
        kv_transport.WIRE = "bf16"
        from kv_transport import ShmTransport
        from rngd_kv_connector import RngdKVConnector
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        root = "/dev/shm/rngd_kv_sweep"
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        t0 = time.perf_counter()
        pf = RngdKVConnector(KVConnectorRole.WORKER, req_id="sweep",
                             transport=ShmTransport(root), n_layers=self.nl)
        for i in range(self.nl):
            pf.save_kv_layer(f"layer.{i}", (K[i], V[i]))
        pf.wait_for_save()
        dk = [np.zeros_like(K[i]) for i in range(self.nl)]
        dv = [np.zeros_like(V[i]) for i in range(self.nl)]
        dc = RngdKVConnector(KVConnectorRole.WORKER, req_id="sweep",
                             transport=ShmTransport(root), n_layers=self.nl)
        dc.register_kv_caches((dk, dv))
        dc.bind_connector_metadata(dc.build_connector_meta(None))
        dc.start_load_kv()
        for i in range(self.nl):
            dc.wait_for_layer_load(f"layer.{i}")
        return np.stack(dk), np.stack(dv), (time.perf_counter() - t0) * 1e3


# --------------------------------------------------------------------------
def prewarm(eng, config, B):
    """Load the ATOM artifact this case opens with, before the clock starts."""
    if config in ("atom_alone", "pd_reverse"):
        eng.atom_rt("prefill")
    elif config == "pd_forward":
        eng.atom_rt("decode", B)


def run_case(eng, config, P, D, B, toks):
    """One request; returns timings plus which cards were involved."""
    if config == "rngd_alone":
        t0 = time.perf_counter()
        nxt, Kp, Vp = eng.rngd_prefill(toks)
        ttft = (time.perf_counter() - t0) * 1e3
        ts = eng.rngd_decode(nxt, P, D, Kp, Vp)
        return ttft, ts, 1, 0.0

    if config == "atom_alone":
        t0 = time.perf_counter()
        nxt, K, V = eng.atom_prefill(toks)
        ttft = (time.perf_counter() - t0) * 1e3
        ts = eng.atom_decode(nxt, P, D, K, V, B)
        return ttft, ts, B, 0.0

    if config == "pd_forward":
        t0 = time.perf_counter()
        nxt, Kp, Vp = eng.rngd_prefill(toks)
        Kh, Vh = eng.sess.export_kv(Kp, Vp)
        K2, V2, ho = eng.handoff(Kh, Vh)
        ttft = (time.perf_counter() - t0) * 1e3
        ts = eng.atom_decode(nxt, P, D, K2, V2, B)
        return ttft, ts, B, ho

    if config == "pd_reverse":
        t0 = time.perf_counter()
        nxt, K, V = eng.atom_prefill(toks)
        K2, V2, ho = eng.handoff(K, V)
        Kd, Vd = eng.sess.import_kv(K2, V2)
        ttft = (time.perf_counter() - t0) * 1e3
        ts = eng.rngd_decode(nxt, P, D, Kd, Vd)
        return ttft, ts, 1, ho
    raise ValueError(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--contexts", default="32,64,128,224")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--configs", default="rngd_alone,atom_alone,pd_forward,pd_reverse")
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--atom-device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--artifact-dir", default="/home/gspark/rbln-llama8b-pd")
    ap.add_argument("--prefill-artifact",
                    default="/home/gspark/rbln-llama8b-pd/prefill_S256_C16.rbln")
    ap.add_argument("--out", default="/home/gspark/workspace/01_pd_disag/rngd_llmd/bench/sweep_hetero.jsonl")
    a = ap.parse_args()

    eng = Engines(a)
    base = ("The three primary colors are red, blue and yellow. Mixing them yields "
            "the secondary colors orange, green and purple. Painters have relied on "
            "this relationship for centuries and printers still exploit it today. ")
    long_ids = eng.tk.encode(base * 12, add_special_tokens=False)

    rows = []
    with open(a.out, "w") as fh:
        for config in a.configs.split(","):
            for P in [int(p) for p in a.contexts.split(",")]:
                P = (P // a.C) * a.C or a.C
                if P + a.D > a.S:
                    continue
                toks = long_ids[:P]
                batches = ([int(b) for b in a.batches.split(",")]
                           if config in ("atom_alone", "pd_forward") else [1])
                for B in batches:
                    if B not in eng.have_decode and config in ("atom_alone", "pd_forward"):
                        continue
                    prewarm(eng, config, B)     # detect the card on first load
                    smp = Sampler(a.chip, eng.atom_card or "rbln0")
                    smp.start()
                    ttfts, tpots, walls = [], [], []
                    ho_ms = 0.0
                    for _ in range(a.reps):
                        # atom_alone swaps prefill->decode inside a request, so
                        # the next rep starts on the wrong artifact; reload it
                        # here, outside the clock.
                        prewarm(eng, config, B)
                        ttft, ts, streams, ho = run_case(eng, config, P, a.D, B, toks)
                        walls.append(ttft + float(np.sum(ts)))
                        ttfts.append(ttft)
                        tpots.append(float(np.median(ts)) if ts else 0.0)
                        ho_ms = ho
                    smp.stop_flag.set()
                    smp.join(timeout=3)
                    tel = smp.result()
                    ttft = float(np.median(ttfts))
                    tpot = float(np.median(tpots))
                    wall = float(np.median(walls))
                    out_tok = a.D * (B if config in ("atom_alone", "pd_forward") else 1)
                    toks_s = out_tok / (wall / 1000)
                    dec_bytes = (ATOM_WEIGHT_BYTES
                                 if config in ("atom_alone", "pd_forward")
                                 else RNGD_WEIGHT_BYTES)
                    bw = dec_bytes / (tpot / 1000) / 1e9 if tpot else None
                    row = dict(config=config, P=P, D=a.D, B=B, ttft_ms=round(ttft, 1),
                               tpot_ms=round(tpot, 2), tokens_s=round(toks_s, 2),
                               wall_ms=round(wall, 1), handoff_ms=round(ho_ms, 1),
                               decode_bw_gbs=round(bw, 1) if bw else None, **tel)
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"{config:11s} P={P:3d} B={B} | TTFT {ttft:7.0f} ms | "
                          f"TPOT {tpot:6.2f} ms | {toks_s:6.2f} tok/s | "
                          f"BW {bw:5.0f} GB/s | rngd {tel['rngd_power_w']}W/"
                          f"{tel['rngd_util_pct']}% | atom {tel['atom_power_w']}W/"
                          f"{tel['atom_util_pct']}%", flush=True)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
