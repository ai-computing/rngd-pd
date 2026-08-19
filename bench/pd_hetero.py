#!/usr/bin/env python3
"""HETEROGENEOUS PD disaggregation — Furiosa RNGD prefill → Rebellions ATOM decode.

The KV cache is model math, not device state: prefill computes it on RNGD
(resident wired engine), the bf16/fp32 tensors cross via host, and an ATOM-side
decode module (pure torch compiled with rebel-compiler) continues generation
from the injected KV. Model: Qwen2.5-0.5B-Instruct (ATOM is 15.7 GiB; 8B bf16
does not fit a single card).

  --smoke : 2-layer RBLN module vs CPU torch reference (compile sanity)
  default : full pipeline  RNGD prefill(P) → KV → RBLN decode(D)
            vs CPU-oracle & RNGD-monolithic references

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_hetero.py --P 32 --D 16
"""
import argparse
import math
import time

import numpy as np
import torch

import _paths  # noqa: F401

torch.set_grad_enabled(False)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # overridden by --model


class QwenDecodeRBLN(torch.nn.Module):
    """One decode step, all layers, weights baked in. Pure torch ops only —
    compiled for ATOM via rebel.compile_from_torch. KV enters and leaves as
    tensors, so KV computed on ANY device (here: RNGD) can be injected."""

    def __init__(self, cfg, sd, S, nl):
        super().__init__()
        self.S, self.nl = S, nl
        self.H, self.M = cfg.hidden, cfg.inter
        self.NH, self.NKV, self.HD, self.G = (cfg.n_heads, cfg.n_kv,
                                              cfg.head_dim, cfg.gqa)
        self.eps = cfg.eps
        self.scale = 1.0 / math.sqrt(self.HD)
        g = lambda k: torch.from_numpy(sd[k].astype(np.float32))
        zb = torch.zeros(self.NH * self.HD)
        zkv = torch.zeros(self.NKV * self.HD)
        gb = lambda k, z: (g(k) if k in sd else z)
        for i in range(nl):
            p = f"model.layers.{i}."
            self.register_buffer(f"n1_{i}", g(p + "input_layernorm.weight"))
            self.register_buffer(f"n2_{i}", g(p + "post_attention_layernorm.weight"))
            self.register_buffer(f"wq_{i}", g(p + "self_attn.q_proj.weight"))
            self.register_buffer(f"bq_{i}", gb(p + "self_attn.q_proj.bias", zb))
            self.register_buffer(f"wk_{i}", g(p + "self_attn.k_proj.weight"))
            self.register_buffer(f"bk_{i}", gb(p + "self_attn.k_proj.bias", zkv))
            self.register_buffer(f"wv_{i}", g(p + "self_attn.v_proj.weight"))
            self.register_buffer(f"bv_{i}", gb(p + "self_attn.v_proj.bias", zkv))
            # o_proj weight pre-reshaped to [NKV, G*HD, H]: rank-collapsing
            # reshape of the 3D attention output fails rebel's device graph
            # conversion, so the merge is a 3D matmul + per-group index sum.
            self.register_buffer(f"wo3_{i}",
                                 g(p + "self_attn.o_proj.weight").t()
                                 .reshape(self.NKV, self.G * self.HD, self.H)
                                 .contiguous())
            self.register_buffer(f"wg_{i}", g(p + "mlp.gate_proj.weight"))
            self.register_buffer(f"wu_{i}", g(p + "mlp.up_proj.weight"))
            self.register_buffer(f"wd_{i}", g(p + "mlp.down_proj.weight"))
        self.register_buffer("normw", g("model.norm.weight"))
        lm = sd["model.embed_tokens.weight"] if cfg.tie else sd["lm_head.weight"]
        self.register_buffer("lm", torch.from_numpy(lm.astype(np.float32)))

    def _rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def _rope(self, x, nh, cos, sin):
        half = self.HD // 2
        xr = x.view(nh, self.HD)                                      # [1,nh*HD] -> [nh,HD]
        a, b = xr[..., :half], xr[..., half:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1) \
                    .reshape(1, nh * self.HD)

    def forward(self, x, Kp, Vp, cos, sin, mask, oh):
        # x [1,H] f32 (2D throughout: 1-D reductions fail rebel's device
        # graph conversion — everything else, slices included, converts fine)
        NH, NKV, HD, G, M, S = self.NH, self.NKV, self.HD, self.G, self.M, self.S
        ks, vs = [], []
        for i in range(self.nl):
            h = self._rms(x, getattr(self, f"n1_{i}"))                 # [1,H]
            q = self._rope(h @ getattr(self, f"wq_{i}").t() + getattr(self, f"bq_{i}"),
                           NH, cos, sin)
            k = self._rope(h @ getattr(self, f"wk_{i}").t() + getattr(self, f"bk_{i}"),
                           NKV, cos, sin)
            v = h @ getattr(self, f"wv_{i}").t() + getattr(self, f"bv_{i}")
            Kf = Kp[i] + k.view(NKV, HD, 1) * oh.view(1, 1, S)
            # Vf is emitted for the KV cache but NOT consumed by the attention
            # matmul: V-inject fused into a downstream matmul silently
            # miscompiles at HD=128 (fine at 64), so ao uses the algebraic
            # split  p@(Vp + oh⊗v) = p@Vp + (p·oh)⊗v  instead.
            Vf = Vp[i] + v.view(NKV, 1, HD) * oh.view(1, S, 1)
            sc = torch.matmul(q.view(NKV, G, HD), Kf) * self.scale \
                + mask.view(1, 1, S)
            p = torch.softmax(sc, dim=-1)
            pj = (p * oh.view(1, 1, S)).sum(-1, keepdim=True)    # [NKV,G,1]
            ao = torch.matmul(p, Vp[i]) + pj * v.view(NKV, 1, HD)  # [NKV,G,HD]
            y = torch.matmul(ao.view(NKV, 1, G * HD), getattr(self, f"wo3_{i}"))
            x2 = x
            for j in range(NKV):
                x2 = x2 + y[j]
            h2 = self._rms(x2, getattr(self, f"n2_{i}"))
            gg = h2 @ getattr(self, f"wg_{i}").t()
            uu = h2 @ getattr(self, f"wu_{i}").t()
            x = (torch.nn.functional.silu(gg) * uu) @ getattr(self, f"wd_{i}").t() + x2
            ks.append(Kf)
            vs.append(Vf)
        logits = self._rms(x, self.normw) @ self.lm.t()               # [1,V]
        return logits, torch.stack(ks), torch.stack(vs)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from generic_runtime import ModelConfig
    tk = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf
    return tk, cfg, sd


def rbln_compile(mod, cfg, S, nl, tag):
    import rebel
    H, NKV, HD = cfg.hidden, cfg.n_kv, cfg.head_dim
    half = HD // 2
    ex = [torch.zeros(1, H), torch.zeros(nl, NKV, HD, S), torch.zeros(nl, NKV, S, HD),
          torch.zeros(half), torch.zeros(half), torch.zeros(S), torch.zeros(S)]
    t0 = time.time()
    compiled = rebel.compile_from_torch(mod, example_inputs=ex)
    print(f"[rbln] compiled {tag} in {time.time()-t0:.0f}s", flush=True)
    rt = rebel.Runtime(compiled)
    return rt


def smoke(a):
    tk, cfg, sd = load_model()
    nl = 2
    mod = QwenDecodeRBLN(cfg, sd, a.S, nl).eval()
    rt = rbln_compile(mod, cfg, a.S, nl, f"qwen {nl}L S={a.S}")
    inv = cfg.inv_freq()
    Kp = torch.zeros(nl, cfg.n_kv, cfg.head_dim, a.S)
    Vp = torch.zeros(nl, cfg.n_kv, a.S, cfg.head_dim)
    toks = tk.encode("The capital of France is", add_special_tokens=False)
    agree = total = 0
    for pos in range(min(len(toks) + 4, a.S)):
        t = toks[pos] if pos < len(toks) else int(cpu_arg)
        ang = pos * inv
        cos = torch.from_numpy(np.cos(ang).astype(np.float32))
        sin = torch.from_numpy(np.sin(ang).astype(np.float32))
        mask = torch.full((a.S,), -1e4)
        mask[:pos + 1] = 0.0
        oh = torch.zeros(a.S)
        oh[pos] = 1.0
        x = torch.from_numpy(sd["model.embed_tokens.weight"][t].astype(np.float32)).view(1, -1)
        cpu_logits, Kc, Vc = mod(x, Kp, Vp, cos, sin, mask, oh)
        out = rt.run(x.numpy(), Kp.numpy(), Vp.numpy(), cos.numpy(), sin.numpy(),
                     mask.numpy(), oh.numpy())
        rb_logits, Kr, Vr = out[0], out[1], out[2]
        cpu_arg = int(torch.argmax(cpu_logits))
        rb_arg = int(np.argmax(rb_logits))
        rel = float(np.linalg.norm(rb_logits - cpu_logits.numpy())
                    / np.linalg.norm(cpu_logits.numpy()))
        agree += (cpu_arg == rb_arg)
        total += 1
        print(f"pos {pos}: cpu={cpu_arg} rbln={rb_arg} logit_rel={rel:.5f}")
        Kp, Vp = torch.from_numpy(np.array(Kr)), torch.from_numpy(np.array(Vr))
    print(f"SMOKE {'PASS' if agree == total else 'PARTIAL'}: argmax {agree}/{total}")


def full(a):
    import rebel  # noqa: F401 — fail fast: a missing dep must not cost a 25-min RNGD compile
    import engine_wired_generic as ewg
    ewg.DEV = a.dev
    from engine_wired_generic import GenericWiredRuntime, _d, _df
    from generic_runtime import GenericRuntime, rmsnorm

    tk, cfg, sd = load_model()
    nl = cfg.n_layers
    inv = cfg.inv_freq()
    prompt = tk.encode(a.prompt, add_special_tokens=False)[:a.P]
    P = len(prompt)
    S = a.S
    NKV, HD = cfg.n_kv, cfg.head_dim

    print("[oracle] CPU reference...", flush=True)
    oracle = GenericRuntime(cfg, sd)
    ref = oracle.generate(list(prompt), a.D)

    if a.kv_npz:                        # resume: skip the RNGD stage entirely
        z = np.load(a.kv_npz)
        Kh, Vh, nxt = z["K"], z["V"], int(z["nxt"])
        P, t_pre, t_export = int(z["P"]), float(z["t_pre"]), float(z["t_export"])
        print(f"[rngd] (resumed from {a.kv_npz}) P={P} next={nxt}", flush=True)
        return _decode_and_report(a, tk, cfg, sd, nl, inv, S, NKV, HD,
                                  Kh, Vh, nxt, P, t_pre, t_export, ref)

    print("[rngd] compiling prefill engine...", flush=True)
    rt_f = GenericWiredRuntime(cfg, sd, S)
    print(f"  compiled in {rt_f.compile_s:.0f}s", flush=True)
    an1, aq, abq, ak, abk, av, abv, ao, an2, awgu, awd = rt_f.W
    Kp_d = _d(np.zeros((nl, NKV, HD, S), np.float32))
    Vp_d = _d(np.zeros((nl, NKV, S, HD), np.float32))
    t0 = time.perf_counter()
    nxt = None
    for pos, t in enumerate(prompt):
        ang = pos * inv
        cos = _d(np.cos(ang)); sin = _d(np.sin(ang))
        mask = np.full(S, -1e4, np.float32); mask[:pos + 1] = 0.0
        oh = np.zeros(S, np.float32); oh[pos] = 1.0
        x = _d(rt_f.embed[t].astype(np.float32))
        hid, Kp_d, Vp_d = rt_f.m(x, rt_f.eps_d, rt_f.nh_d, an1, aq, abq, ak, abk,
                                 av, abv, ao, an2, awgu, awd, Kp_d, Vp_d,
                                 cos, sin, _d(mask), _d(oh))
    nxt = int(np.argmax(rt_f.lm @ rmsnorm(hid.detach().float().cpu().numpy(),
                                          rt_f.norm, cfg.eps)))
    t_pre = time.perf_counter() - t0
    tk0 = time.perf_counter()
    Kh = Kp_d.detach().float().cpu().numpy()
    Vh = Vp_d.detach().float().cpu().numpy()
    t_export = (time.perf_counter() - tk0) * 1e3
    print(f"[rngd] prefill {P} tok in {t_pre*1e3:.0f} ms | KV export "
          f"{Kh.nbytes*2/1e6:.1f} MB in {t_export:.0f} ms | next={nxt}", flush=True)
    ckpt = "/dev/shm/hetero_kv_ckpt.npz"
    np.savez(ckpt, K=Kh, V=Vh, nxt=nxt, P=P, t_pre=t_pre, t_export=t_export)
    print(f"[rngd] KV checkpoint -> {ckpt} (resume with --kv-npz)", flush=True)
    return _decode_and_report(a, tk, cfg, sd, nl, inv, S, NKV, HD,
                              Kh, Vh, nxt, P, t_pre, t_export, ref)


def _decode_and_report(a, tk, cfg, sd, nl, inv, S, NKV, HD,
                       Kh, Vh, nxt, P, t_pre, t_export, ref):
    print("[rbln] compiling decode engine (full model)...", flush=True)
    mod = QwenDecodeRBLN(cfg, sd, S, nl).eval()
    rrt = rbln_compile(mod, cfg, S, nl, f"qwen {nl}L S={S}")

    # --- llm-d protocol handoff: real vLLM KVConnectorBase_V1 lifecycle over /dev/shm ---
    import os as _os, shutil as _sh, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                      "..", "connector"))
    import kv_transport
    kv_transport.WIRE = "bf16"          # bit-exact for bf16-valued RNGD KV
    from kv_transport import ShmTransport
    from rngd_kv_connector import RngdKVConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1, KVConnectorRole)
    shm_root = "/dev/shm/rngd_kv_hetero"
    _sh.rmtree(shm_root, ignore_errors=True)
    req = "hetero-0"
    tS = time.perf_counter()
    pf = RngdKVConnector(KVConnectorRole.WORKER, req_id=req,
                         transport=ShmTransport(shm_root), n_layers=nl)
    for i in range(nl):
        pf.save_kv_layer(f"layer.{i}", (Kh[i], Vh[i]))
    pf.wait_for_save()
    t_save = (time.perf_counter() - tS) * 1e3
    dk = [np.zeros((NKV, HD, S), np.float32) for _ in range(nl)]
    dv = [np.zeros((NKV, S, HD), np.float32) for _ in range(nl)]
    dc = RngdKVConnector(KVConnectorRole.WORKER, req_id=req,
                         transport=ShmTransport(shm_root), n_layers=nl)
    dc.register_kv_caches((dk, dv))
    dc.bind_connector_metadata(dc.build_connector_meta(None))
    tL = time.perf_counter()
    dc.start_load_kv()
    for i in range(nl):
        dc.wait_for_layer_load(f"layer.{i}")
    t_load = (time.perf_counter() - tL) * 1e3
    Kc2, Vc2 = np.stack(dk), np.stack(dv)
    bit_k = bool(np.array_equal(Kc2, Kh)); bit_v = bool(np.array_equal(Vc2, Vh))
    n_shm = len(_os.listdir(shm_root))
    llmd_line = (f"KVConnectorBase_V1={isinstance(pf, KVConnectorBase_V1)} | "
                 f"save {t_save:.0f} ms / load {t_load:.0f} ms | {n_shm} shm objs "
                 f"(bf16 wire) | bit-exact K={bit_k} V={bit_v}")
    print(f"[llm-d] {llmd_line}", flush=True)

    Kp, Vp = Kc2, Vc2                   # decode consumes CONNECTOR-delivered KV
    outs = [nxt]
    ts = []
    p = P
    cur = nxt
    for _ in range(a.D - 1):
        ang = p * inv
        t1 = time.perf_counter()
        out = rrt.run(sd["model.embed_tokens.weight"][cur].astype(np.float32).reshape(1, -1),
                      Kp, Vp, np.cos(ang).astype(np.float32),
                      np.sin(ang).astype(np.float32),
                      np.concatenate([np.zeros(p + 1, np.float32),
                                      np.full(S - p - 1, -1e4, np.float32)]),
                      np.eye(S, dtype=np.float32)[p])
        ts.append(time.perf_counter() - t1)
        logits, Kp, Vp = out[0], np.array(out[1]), np.array(out[2])
        cur = int(np.argmax(logits))
        outs.append(cur)
        p += 1
    tpot = sorted(ts)[len(ts) // 2] * 1e3 if ts else 0

    match = sum(x == y for x, y in zip(outs, ref))
    print("\n" + "=" * 70)
    print("HETEROGENEOUS PD — Furiosa RNGD prefill → Rebellions ATOM decode")
    print(f"  model {MODEL} ({nl}L)  P={P} D={a.D} S={S}")
    print(f"  RNGD prefill: {t_pre*1e3:.0f} ms | KV handoff: "
          f"{Kh.nbytes*2/1e6:.1f} MB, export {t_export:.0f} ms")
    print(f"  llm-d handoff: {llmd_line}")
    print(f"  RBLN decode TPOT p50: {tpot:.1f} ms/token")
    print(f"  hetero output : {outs}")
    print(f"  CPU oracle ref: {ref}")
    print(f"  argmax agreement vs oracle: {match}/{len(outs)} "
          f"{'PASS ✅ cross-vendor KV handoff works' if match >= len(outs) - 1 else 'CHECK'}")
    print("=" * 70)


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=64)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--kv-npz", default="",
                    help="resume decode from a saved KV checkpoint (skips RNGD stage)")
    ap.add_argument("--dev", default="rngd:2",
                    help="RNGD device for prefill (0/1 may be held by k8s pods; 3 has EDF-loader issue)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    MODEL = a.model
    if a.smoke:
        return smoke(a)
    return full(a)


if __name__ == "__main__":
    main()
