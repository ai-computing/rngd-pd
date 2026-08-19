#!/usr/bin/env python3
"""Batched decode on Rebellions ATOM — B independent streams per graph pass.

Same weight-bandwidth argument as chunked prefill, applied to the other stage:
one pass reads 13.96 GB of fp16 weights whether it advances one sequence or
four, so per-token cost should fall close to 1/B. Chunking shortens a single
request's prefill; batching raises concurrent decode throughput. Different axes,
same underlying reason.

Each stream carries its own position, so cos/sin/mask/oh are per-stream. KV
stays out of the graph outputs — the module returns each stream's k/v for this
step and the host places them, the same defence used in the B=1 module.

  --smoke : 2 layers, B=2, against CPU fp32
  --build : compile 32 layers for each B and save artifacts
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
from pd_hetero_llama import MODEL, load_hf

torch.set_grad_enabled(False)


class LlamaDecodeBatchRBLN(torch.nn.Module):
    def __init__(self, cfg, sd, S, nl, B, wdt=torch.float16):
        super().__init__()
        self.S, self.nl, self.B, self.wdt = S, nl, B, wdt
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
        B, half = self.B, self.HD // 2
        xr = x.view(B, nh, self.HD)
        a, b = xr[..., :half], xr[..., half:]
        cos = cos.view(B, 1, half)
        sin = sin.view(B, 1, half)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

    def forward(self, x, Kp, Vp, cos, sin, mask, oh):
        # x [B,H] · Kp [nl,B,NKV,HD,S] · Vp [nl,B,NKV,S,HD]
        # cos/sin [B,HD/2] · mask/oh [B,S]  (per-stream position)
        B, NH, NKV, HD, G, S = (self.B, self.NH, self.NKV, self.HD,
                                self.G, self.S)
        ks, vs = [], []
        for i in range(self.nl):
            h = self._rms(x, getattr(self, f"n1_{i}"))
            q = self._rope(self._mm(h, getattr(self, f"wq_{i}")), NH, cos, sin)
            k = self._rope(self._mm(h, getattr(self, f"wk_{i}")), NKV, cos, sin)
            v = self._mm(h, getattr(self, f"wv_{i}")).view(B, NKV, HD)
            Kf = Kp[i] + k.unsqueeze(-1) * oh.view(B, 1, 1, S)
            sc = torch.matmul(q.view(B, NKV, G, HD), Kf) * self.scale \
                + mask.view(B, 1, 1, S)
            p = torch.softmax(sc, dim=-1)
            pj = (p * oh.view(B, 1, 1, S)).sum(-1, keepdim=True)
            ao = torch.matmul(p, Vp[i]) + pj * v.unsqueeze(-2)   # [B,NKV,G,HD]
            y = torch.matmul(ao.reshape(B, NKV, 1, G * HD).to(self.wdt),
                             getattr(self, f"wo3_{i}").unsqueeze(0)).float()
            x2 = x
            for j in range(NKV):
                x2 = x2 + y[:, j, 0]
            h2 = self._rms(x2, getattr(self, f"n2_{i}"))
            gg = self._mm(h2, getattr(self, f"wg_{i}"))
            uu = self._mm(h2, getattr(self, f"wu_{i}"))
            x = self._mm(torch.nn.functional.silu(gg) * uu,
                         getattr(self, f"wd_{i}")) + x2
            ks.append(k)
            vs.append(v)
        return self._rms(x, self.normw), torch.stack(ks), torch.stack(vs)


def example_inputs(cfg, S, nl, B):
    H, NKV, HD = cfg.hidden, cfg.n_kv, cfg.head_dim
    return [torch.zeros(B, H), torch.zeros(nl, B, NKV, HD, S),
            torch.zeros(nl, B, NKV, S, HD),
            torch.zeros(B, HD // 2), torch.zeros(B, HD // 2),
            torch.zeros(B, S), torch.zeros(B, S)]


def build_runtime(mod, cfg, S, nl, B, tag, artifact=None, device=0):
    import rebel
    if artifact and os.path.exists(artifact):
        t0 = time.time()
        rt = rebel.RBLNCompiledModel(artifact).create_runtime(device=device)
        print(f"[rbln] loaded {tag} in {time.time()-t0:.0f}s", flush=True)
        return rt
    t0 = time.time()
    compiled = rebel.compile_from_torch(mod, example_inputs=example_inputs(cfg, S, nl, B))
    print(f"[rbln] compiled {tag} in {time.time()-t0:.0f}s", flush=True)
    if artifact:
        os.makedirs(os.path.dirname(artifact) or ".", exist_ok=True)
        compiled.save(artifact)
        print(f"[rbln] saved -> {artifact}", flush=True)
    return compiled.create_runtime(device=device)


def smoke(a):
    tk, cfg, sd = load_hf(nl_keep=2)
    nl, B, S = 2, a.B, a.S
    mod = LlamaDecodeBatchRBLN(cfg, sd, S, nl, B).eval()
    rt = build_runtime(mod, cfg, S, nl, B, f"decode {nl}L B={B}", device=a.device)
    embed = sd["model.embed_tokens.weight"]
    inv = cfg.inv_freq()
    rng = np.random.default_rng(0)
    toks = rng.integers(1000, 20000, size=B)
    pos = np.array([7, 19, 31, 43][:B])
    x = embed[toks].astype(np.float32)
    ang = pos[:, None] * inv[None, :]
    mask = np.full((B, S), -1e4, np.float32)
    oh = np.zeros((B, S), np.float32)
    for b in range(B):
        mask[b, :pos[b] + 1] = 0.0
        oh[b, pos[b]] = 1.0
    Kp = rng.standard_normal((nl, B, cfg.n_kv, cfg.head_dim, S)).astype(np.float32) * .1
    Vp = rng.standard_normal((nl, B, cfg.n_kv, S, cfg.head_dim)).astype(np.float32) * .1
    args = [x, Kp, Vp, np.cos(ang).astype(np.float32),
            np.sin(ang).astype(np.float32), mask, oh]
    out = rt.run(*args)
    hr, kr, vr = (np.asarray(o) for o in out[:3])
    hc, kc, vc = mod(*[torch.from_numpy(v) for v in args])
    rel = lambda p, q: float(np.linalg.norm(p - q) / (np.linalg.norm(q) + 1e-9))
    print(f"\nbatched decode B={B}, {nl} layers — ATOM vs CPU fp32")
    print(f"  hidden rel {rel(hr, hc.numpy()):.5f}")
    print(f"  k      rel {rel(kr, kc.numpy()):.5f}")
    print(f"  v      rel {rel(vr, vc.numpy()):.5f}")
    per = [rel(hr[b], hc.numpy()[b]) for b in range(B)]
    print(f"  per-stream hidden rel: {['%.4f' % p for p in per]}")
    ok = max(per) < 0.05
    print(f"  {'PASS ✅ every stream matches' if ok else 'FAIL ❌'}")


def build(a):
    tk, cfg, sd = load_hf()
    nl = cfg.n_layers
    for B in [int(b) for b in a.batches.split(",")]:
        art = f"{a.artifact_dir}/decode_S{a.S}_B{B}.rbln"
        if os.path.exists(art):
            print(f"[skip] {art} exists", flush=True)
            continue
        mod = LlamaDecodeBatchRBLN(cfg, sd, a.S, nl, B).eval()
        rt = build_runtime(mod, cfg, a.S, nl, B, f"decode {nl}L B={B}",
                           artifact=art, device=a.device)
        args = [v.numpy() for v in example_inputs(cfg, a.S, nl, B)]
        rt.run(*args)
        ts = []
        for _ in range(8):
            t0 = time.perf_counter(); rt.run(*args); ts.append((time.perf_counter()-t0)*1e3)
        ts.sort()
        m = ts[len(ts) // 2]
        print(f"  B={B}: step {m:.1f} ms → {m/B:.1f} ms/token", flush=True)
        del rt, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--artifact-dir", default="/home/gspark/rbln-llama8b-pd")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    return smoke(a) if a.smoke else build(a)


if __name__ == "__main__":
    main()
