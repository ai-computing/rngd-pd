#!/usr/bin/env python3
"""HETEROGENEOUS PD at 8B scale — Llama-3.1-8B across two vendors.

  Furiosa RNGD  : prefill on the resident wired engine (32L EDF comes from the
                  artifact cache, so no 30-min recompile)
  llm-d         : KV crosses through the real vllm KVConnectorBase_V1 lifecycle
                  (bf16 wire, bit-exact) over /dev/shm
  Rebellions ATOM: decode step compiled with rebel-compiler

ATOM is 15.7 GiB, so the decode module keeps 32 transformer layers in fp16
(6.98 B params = 13.96 GB) and leaves embedding + lm_head on the host — the
same split the RNGD engine already uses for its reference path.

Activations stay fp32 (Llama-3 hidden states carry outliers that overflow an
fp16 rmsnorm accumulation); only matmul operands are cast to fp16.

  --smoke : 2-layer module vs CPU fp32 reference
  default : RNGD prefill(P) → llm-d handoff → ATOM decode(D), compared against
            the RNGD engine running the same continuation monolithically

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_hetero_llama.py --P 32 --D 8
"""
import argparse
import os
import math
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RBLN_PART = os.environ.get("RBLN_PART", "RBLN-CA22")


class LlamaDecodeRBLN(torch.nn.Module):
    """One decode step, all layers, weights baked in as fp16 constants.

    KV enters in the RNGD engine's layout so exported KV drops straight in:
    K [nl, NKV, HD, S] · V [nl, NKV, S, HD]. It does NOT come back out — the
    module returns only this step's k and v ([nl, NKV, HD] each) and the host
    scatters them into the cache it already owns. Emitting whole KV tensors
    instead makes the graph carry a broadcast slot-write per layer, which
    miscompiles at NKV=8/S=256, and costs 134 MB of transfer per step.

    Returns the RMS-normed hidden state; the host applies lm_head (keeping
    0.5 B params off the card).
    """

    def __init__(self, cfg, sd, S, nl, wdt=torch.float16):
        super().__init__()
        self.S, self.nl, self.wdt = S, nl, wdt
        self.H, self.M = cfg.hidden, cfg.inter
        self.NH, self.NKV, self.HD, self.G = (cfg.n_heads, cfg.n_kv,
                                              cfg.head_dim, cfg.gqa)
        self.eps = cfg.eps
        self.scale = 1.0 / math.sqrt(self.HD)
        g = lambda k: torch.from_numpy(sd[k].astype(np.float32))
        for i in range(nl):
            p = f"model.layers.{i}."
            self.register_buffer(f"n1_{i}", g(p + "input_layernorm.weight"))
            self.register_buffer(f"n2_{i}", g(p + "post_attention_layernorm.weight"))
            for n, k in [("wq", "self_attn.q_proj.weight"),
                         ("wk", "self_attn.k_proj.weight"),
                         ("wv", "self_attn.v_proj.weight"),
                         ("wg", "mlp.gate_proj.weight"),
                         ("wu", "mlp.up_proj.weight"),
                         ("wd", "mlp.down_proj.weight")]:
                self.register_buffer(f"{n}_{i}", g(p + k).to(wdt))
            # o_proj pre-reshaped to [NKV, G*HD, H]: a rank-collapsing reshape
            # of the 3D attention output does not survive graph conversion.
            self.register_buffer(f"wo3_{i}",
                                 g(p + "self_attn.o_proj.weight").t()
                                 .reshape(self.NKV, self.G * self.HD, self.H)
                                 .contiguous().to(wdt))
        self.register_buffer("normw", g("model.norm.weight"))

    def _mm(self, a, w):
        return (a.to(self.wdt) @ w.t()).float()

    def _rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def _rope(self, x, nh, cos, sin):
        half = self.HD // 2
        xr = x.view(nh, self.HD)
        a, b = xr[..., :half], xr[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1) \
                    .reshape(1, nh * self.HD)

    def forward(self, x, Kp, Vp, cos, sin, mask, oh):
        NH, NKV, HD, G, S = self.NH, self.NKV, self.HD, self.G, self.S
        ks, vs = [], []
        for i in range(self.nl):
            h = self._rms(x, getattr(self, f"n1_{i}"))
            q = self._rope(self._mm(h, getattr(self, f"wq_{i}")), NH, cos, sin)
            k = self._rope(self._mm(h, getattr(self, f"wk_{i}")), NKV, cos, sin)
            v = self._mm(h, getattr(self, f"wv_{i}"))
            Kf = Kp[i] + k.view(NKV, HD, 1) * oh.view(1, 1, S)
            sc = torch.matmul(q.view(NKV, G, HD), Kf) * self.scale \
                + mask.view(1, 1, S)
            p = torch.softmax(sc, dim=-1)
            pj = (p * oh.view(1, 1, S)).sum(-1, keepdim=True)
            ao = torch.matmul(p, Vp[i]) + pj * v.view(NKV, 1, HD)
            y = torch.matmul(ao.view(NKV, 1, G * HD).to(self.wdt),
                             getattr(self, f"wo3_{i}")).float()
            x2 = x
            for j in range(NKV):
                x2 = x2 + y[j]
            h2 = self._rms(x2, getattr(self, f"n2_{i}"))
            gg = self._mm(h2, getattr(self, f"wg_{i}"))
            uu = self._mm(h2, getattr(self, f"wu_{i}"))
            x = self._mm(torch.nn.functional.silu(gg) * uu,
                         getattr(self, f"wd_{i}")) + x2
            ks.append(k.view(NKV, HD))
            vs.append(v.view(NKV, HD))
        return self._rms(x, self.normw), torch.stack(ks), torch.stack(vs)


def rbln_compile(mod, cfg, S, nl, tag, artifact=None):
    """Compile for ATOM, or load a previously compiled artifact.

    Compiling needs the vendor toolchain to probe the machine, which does not
    work inside a container — the RBLN docs only cover runtime inference there.
    So the artifact is produced once on the host and loaded everywhere else,
    which also turns a 90 s startup into a few seconds.
    """
    import os
    import rebel
    artifact = artifact or os.environ.get("RBLN_ARTIFACT") or None
    if artifact and os.path.exists(artifact):
        t0 = time.time()
        rt = rebel.Runtime(rebel.RBLNCompiledModel(artifact))
        print(f"[rbln] loaded {tag} from {artifact} in {time.time()-t0:.0f}s",
              flush=True)
        return rt
    H, NKV, HD = cfg.hidden, cfg.n_kv, cfg.head_dim
    half = HD // 2
    ex = [torch.zeros(1, H), torch.zeros(nl, NKV, HD, S), torch.zeros(nl, NKV, S, HD),
          torch.zeros(half), torch.zeros(half), torch.zeros(S), torch.zeros(S)]
    t0 = time.time()
    # The compiler probes the machine for a card to target and gives up
    # when every one is claimed by a pod. Naming the part lets a build run
    # while the fleet is busy, which is the normal case here.
    compiled = rebel.compile_from_torch(mod, example_inputs=ex, npu=RBLN_PART)
    print(f"[rbln] compiled {tag} in {time.time()-t0:.0f}s", flush=True)
    if artifact:
        os.makedirs(os.path.dirname(artifact) or ".", exist_ok=True)
        compiled.save(artifact)
        print(f"[rbln] saved artifact -> {artifact}", flush=True)
    return rebel.Runtime(compiled)


def load_hf(nl_keep=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rngd_pd.engines.model import ModelConfig
    tk = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {}
    for k, v in hf.state_dict().items():
        if nl_keep is not None and k.startswith("model.layers."):
            if int(k.split(".")[2]) >= nl_keep:
                continue
        sd[k] = v.detach().numpy()
    del hf
    return tk, cfg, sd


def smoke(a):
    tk, cfg, sd = load_hf(nl_keep=2)
    nl = 2
    mod = LlamaDecodeRBLN(cfg, sd, a.S, nl).eval()
    rt = rbln_compile(mod, cfg, a.S, nl, f"llama {nl}L S={a.S} fp16")
    inv = cfg.inv_freq()
    Kp = torch.zeros(nl, cfg.n_kv, cfg.head_dim, a.S)
    Vp = torch.zeros(nl, cfg.n_kv, a.S, cfg.head_dim)
    toks = tk.encode("The capital of France is", add_special_tokens=False)
    lm = sd["lm_head.weight"]
    agree = total = 0
    for pos in range(min(len(toks), a.S)):
        t = toks[pos]
        ang = pos * inv
        cos = torch.from_numpy(np.cos(ang).astype(np.float32))
        sin = torch.from_numpy(np.sin(ang).astype(np.float32))
        mask = torch.full((a.S,), -1e4)
        mask[:pos + 1] = 0.0
        oh = torch.zeros(a.S)
        oh[pos] = 1.0
        x = torch.from_numpy(
            sd["model.embed_tokens.weight"][t].astype(np.float32)).view(1, -1)
        hc, Kc, Vc = mod(x, Kp, Vp, cos, sin, mask, oh)
        out = rt.run(x.numpy(), Kp.numpy(), Vp.numpy(), cos.numpy(), sin.numpy(),
                     mask.numpy(), oh.numpy())
        hr, kr, vr = np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2])
        ca = int(np.argmax(lm @ hc.numpy().reshape(-1)))
        ra = int(np.argmax(lm @ hr.reshape(-1)))
        rel = float(np.linalg.norm(hr - hc.numpy()) / np.linalg.norm(hc.numpy()))
        agree += (ca == ra)
        total += 1
        print(f"pos {pos}: cpu={ca} rbln={ra} hidden_rel={rel:.5f}", flush=True)
        Kp[:, :, :, pos] = torch.from_numpy(kr)      # host-side KV scatter
        Vp[:, :, pos, :] = torch.from_numpy(vr)
    print(f"SMOKE {'PASS' if agree == total else 'PARTIAL'}: argmax {agree}/{total}")


def full(a):
    import rebel  # noqa: F401 — fail fast before any expensive setup
    from rngd_pd.kernels import decode_resident as LER
    from rngd_pd.engines.decode_resident import build_engine, EngineSession
    from rngd_pd.engines.model import rmsnorm

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL)
    prompt = tk.encode(a.prompt, add_special_tokens=False)[:a.P]
    P = len(prompt)

    print(f"[rngd] loading engine (chip {a.chip}, S={a.S}, 32L; cached EDF)...",
          flush=True)
    cfg, rt = build_engine(MODEL, a.S, 32, chip=a.chip)
    nl, NKV, HD = rt.nl, LER.NKV, LER.HD
    sess = EngineSession(rt, LER)

    # ---- reference: same engine, monolithic prefill + decode ----------------
    Kp, Vp = sess.fresh_kv()
    for pos, t in enumerate(prompt[:-1]):
        _, Kp, Vp = sess.step(t, pos, Kp, Vp)
    nxt, Kp, Vp = sess.step(prompt[-1], P - 1, Kp, Vp)
    ref = [nxt]
    cur, p = nxt, P
    for _ in range(a.D - 1):
        cur, Kp, Vp = sess.step(cur, p, Kp, Vp)
        ref.append(cur)
        p += 1
    print(f"[rngd] monolithic reference: {ref}", flush=True)

    # ---- PD: prefill only, then hand the KV off ----------------------------
    Kp, Vp = sess.fresh_kv()
    t0 = time.perf_counter()
    for pos, t in enumerate(prompt[:-1]):
        _, Kp, Vp = sess.step(t, pos, Kp, Vp)
    nxt, Kp, Vp = sess.step(prompt[-1], P - 1, Kp, Vp)
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    Kh, Vh = sess.export_kv(Kp, Vp)
    t_export = (time.perf_counter() - t1) * 1e3
    print(f"[rngd] prefill {P} tok in {t_pre*1e3:.0f} ms | KV export "
          f"{Kh.nbytes*2/1e6:.0f} MB in {t_export:.0f} ms | next={nxt}", flush=True)
    np.savez("/dev/shm/hetero_llama_kv.npz", K=Kh, V=Vh, nxt=nxt, P=P,
             t_pre=t_pre, t_export=t_export, ref=np.array(ref))
    del Kp, Vp

    # ---- llm-d protocol handoff -------------------------------------------
    import os as _os, shutil as _sh
    from rngd_pd.connector import transport as kv_transport
    kv_transport.WIRE = "bf16"
    from rngd_pd.connector.transport import ShmTransport
    from rngd_pd.connector.rngd_kv import RngdKVConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1, KVConnectorRole)
    shm_root = "/dev/shm/rngd_kv_llama"
    _sh.rmtree(shm_root, ignore_errors=True)
    req = "hetero-llama-0"
    tS = time.perf_counter()
    pf = RngdKVConnector(KVConnectorRole.WORKER, req_id=req,
                         transport=ShmTransport(shm_root), n_layers=nl)
    for i in range(nl):
        pf.save_kv_layer(f"layer.{i}", (Kh[i], Vh[i]))
    pf.wait_for_save()
    t_save = (time.perf_counter() - tS) * 1e3
    dk = [np.zeros((NKV, HD, a.S), np.float32) for _ in range(nl)]
    dv = [np.zeros((NKV, a.S, HD), np.float32) for _ in range(nl)]
    dc = RngdKVConnector(KVConnectorRole.WORKER, req_id=req,
                         transport=ShmTransport(shm_root), n_layers=nl)
    dc.register_kv_caches((dk, dv))
    dc.bind_connector_metadata(dc.build_connector_meta(None))
    tL = time.perf_counter()
    dc.start_load_kv()
    for i in range(nl):
        dc.wait_for_layer_load(f"layer.{i}")
    t_load = (time.perf_counter() - tL) * 1e3
    Kc, Vc = np.stack(dk), np.stack(dv)
    llmd = (f"KVConnectorBase_V1={isinstance(pf, KVConnectorBase_V1)} | "
            f"save {t_save:.0f} ms / load {t_load:.0f} ms | "
            f"{len(_os.listdir(shm_root))} shm objs (bf16 wire) | "
            f"bit-exact K={np.array_equal(Kc, Kh)} V={np.array_equal(Vc, Vh)}")
    print(f"[llm-d] {llmd}", flush=True)

    # ---- ATOM decode -------------------------------------------------------
    print("[rbln] building 32L fp16 decode module "
          f"(~{32 * 218e6 * 2 / 1e9:.1f} GB of weights)...", flush=True)
    _tk, cfg2, sd = load_hf()
    lm, normw = sd["lm_head.weight"], sd["model.norm.weight"]
    embed = sd["model.embed_tokens.weight"]
    mod = LlamaDecodeRBLN(cfg2, sd, a.S, nl).eval()
    del sd
    rrt = rbln_compile(mod, cfg2, a.S, nl, f"llama {nl}L S={a.S} fp16")
    del mod

    inv = cfg2.inv_freq()
    Kd, Vd = Kc, Vc
    outs, ts, hts = [nxt], [], []
    cur, p = nxt, P
    for _ in range(a.D - 1):
        ang = p * inv
        mask = np.concatenate([np.zeros(p + 1, np.float32),
                               np.full(a.S - p - 1, -1e4, np.float32)])
        t1 = time.perf_counter()
        out = rrt.run(embed[cur].astype(np.float32).reshape(1, -1), Kd, Vd,
                      np.cos(ang).astype(np.float32),
                      np.sin(ang).astype(np.float32),
                      mask, np.eye(a.S, dtype=np.float32)[p])
        ts.append(time.perf_counter() - t1)
        hid = np.asarray(out[0])
        Kd[:, :, :, p] = np.asarray(out[1])          # host-side KV scatter
        Vd[:, :, p, :] = np.asarray(out[2])
        t2 = time.perf_counter()
        cur = int(np.argmax(lm @ hid.reshape(-1)))
        hts.append(time.perf_counter() - t2)
        outs.append(cur)
        p += 1
    dev_ms = sorted(ts)[len(ts) // 2] * 1e3 if ts else 0
    host_ms = sorted(hts)[len(hts) // 2] * 1e3 if hts else 0

    match = sum(x == y for x, y in zip(outs, ref))
    print("\n" + "=" * 74)
    print("HETEROGENEOUS PD @ 8B — Furiosa RNGD prefill → Rebellions ATOM decode")
    print(f"  model {MODEL} ({nl}L, H={cfg2.hidden})  P={P} D={a.D} S={a.S}")
    print(f"  RNGD prefill: {t_pre*1e3:.0f} ms ({t_pre*1e3/P:.1f} ms/tok) | "
          f"KV {Kh.nbytes*2/1e6:.0f} MB, export {t_export:.0f} ms")
    print(f"  llm-d handoff: {llmd}")
    print(f"  ATOM decode: {dev_ms:.1f} ms/token on-card "
          f"(+ host lm_head {host_ms:.1f} ms)")
    print(f"  hetero: {outs}")
    print(f"  RNGD  : {ref}")
    print(f"  text  : {tk.decode(outs)!r}")
    print(f"  argmax agreement: {match}/{len(outs)} "
          f"{'PASS ✅ cross-vendor 8B KV handoff' if match >= len(outs) - 1 else 'CHECK'}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    return smoke(a) if a.smoke else full(a)


if __name__ == "__main__":
    main()
