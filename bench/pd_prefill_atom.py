#!/usr/bin/env python3
"""Chunked prefill on Rebellions ATOM — the other half of a reversed PD split.

The decode step is weight-bandwidth bound: reading 13.96 GB of fp16 weights
costs ~62 ms whether one token or thirty-two ride along. Prefill knows all its
positions up front, so it can put C of them through one graph pass under a
causal mask and amortize that read — measured 4.72 → 4.47 ms per step going
from C=1 to C=32 at two layers, i.e. 34x less per token.

That inverts the natural PD assignment on this pair. RNGD runs a token at a
time (its eager path cannot compile a multi-position graph), so ATOM chunked
is the faster prefill and RNGD the faster decoder.

KV leaves as this chunk's k/v only; the host places it. Emitting whole KV
tensors makes the graph carry a broadcast slot-write, which miscompiles.

  --smoke : 2 layers vs CPU fp32, numerics first
  --build : compile 32 layers, save the artifact
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


class LlamaPrefillRBLN(torch.nn.Module):
    """C prompt positions per pass. Weights fp16, activations fp32."""

    def __init__(self, cfg, sd, S, nl, C, wdt=torch.float16):
        super().__init__()
        self.S, self.nl, self.C, self.wdt = S, nl, C, wdt
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
                         ("wo", "self_attn.o_proj.weight"),
                         ("wg", "mlp.gate_proj.weight"),
                         ("wu", "mlp.up_proj.weight"),
                         ("wd", "mlp.down_proj.weight")]:
                self.register_buffer(f"{n}_{i}", g(p + k).to(wdt))
        self.register_buffer("normw", g("model.norm.weight"))

    def _mm(self, a, w):
        return (a.to(self.wdt) @ w.t()).float()

    def _rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def _rope(self, x, nh, cos, sin):
        C, half = self.C, self.HD // 2
        xr = x.view(C, nh, self.HD)
        a, b = xr[..., :half], xr[..., half:]
        cos = cos.view(C, 1, half)
        sin = sin.view(C, 1, half)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

    def forward(self, x, Kp, Vp, cos, sin, mask):
        # x [C,H] · Kp [nl,NKV,HD,S] · Vp [nl,NKV,S,HD]
        # cos/sin [C,HD/2] · mask [C,S+C] causal, already offset for this chunk
        C, NH, NKV, HD, G, S = (self.C, self.NH, self.NKV, self.HD,
                                self.G, self.S)
        ks, vs = [], []
        for i in range(self.nl):
            h = self._rms(x, getattr(self, f"n1_{i}"))
            q = self._rope(self._mm(h, getattr(self, f"wq_{i}")), NH, cos, sin)
            k = self._rope(self._mm(h, getattr(self, f"wk_{i}")), NKV, cos, sin)
            v = self._mm(h, getattr(self, f"wv_{i}")).view(C, NKV, HD)
            kT = k.permute(1, 2, 0)                       # [NKV,HD,C]
            vT = v.permute(1, 0, 2)                       # [NKV,C,HD]
            Kall = torch.cat([Kp[i], kT], dim=-1)         # [NKV,HD,S+C]
            Vall = torch.cat([Vp[i], vT], dim=-2)         # [NKV,S+C,HD]
            qh = q.view(C, NKV, G, HD).permute(1, 2, 0, 3)  # [NKV,G,C,HD]
            sc = torch.matmul(qh, Kall.unsqueeze(1)) * self.scale \
                + mask.view(1, 1, C, S + C)
            p = torch.softmax(sc, dim=-1)
            ao = torch.matmul(p, Vall.unsqueeze(1))       # [NKV,G,C,HD]
            ao = ao.permute(2, 0, 1, 3).reshape(C, NH * HD)
            x2 = self._mm(ao, getattr(self, f"wo_{i}")) + x
            h2 = self._rms(x2, getattr(self, f"n2_{i}"))
            gg = self._mm(h2, getattr(self, f"wg_{i}"))
            uu = self._mm(h2, getattr(self, f"wu_{i}"))
            x = self._mm(torch.nn.functional.silu(gg) * uu,
                         getattr(self, f"wd_{i}")) + x2
            ks.append(kT)
            vs.append(vT)
        return self._rms(x, self.normw), torch.stack(ks), torch.stack(vs)


def build_runtime(mod, cfg, S, nl, C, tag, artifact=None, device=0):
    import rebel
    H, NKV, HD = cfg.hidden, cfg.n_kv, cfg.head_dim
    if artifact and os.path.exists(artifact):
        t0 = time.time()
        rt = rebel.RBLNCompiledModel(artifact).create_runtime(device=device)
        print(f"[rbln] loaded {tag} in {time.time()-t0:.0f}s", flush=True)
        return rt
    ex = [torch.zeros(C, H), torch.zeros(nl, NKV, HD, S), torch.zeros(nl, NKV, S, HD),
          torch.zeros(C, HD // 2), torch.zeros(C, HD // 2), torch.zeros(C, S + C)]
    t0 = time.time()
    compiled = rebel.compile_from_torch(mod, example_inputs=ex)
    print(f"[rbln] compiled {tag} in {time.time()-t0:.0f}s", flush=True)
    if artifact:
        os.makedirs(os.path.dirname(artifact) or ".", exist_ok=True)
        compiled.save(artifact)
        print(f"[rbln] saved -> {artifact}", flush=True)
    return compiled.create_runtime(device=device)


def chunk_inputs(cfg, S, C, base, embed, toks):
    """Embeddings, RoPE tables and the causal mask for positions [base, base+C)."""
    inv = cfg.inv_freq()
    pos = np.arange(base, base + C)
    ang = pos[:, None] * inv[None, :]
    x = embed[toks].astype(np.float32)
    # a query at chunk-row r attends to cache slots < base and chunk cols <= r
    mask = np.full((C, S + C), -1e4, np.float32)
    mask[:, :base] = 0.0
    for r in range(C):
        mask[r, S:S + r + 1] = 0.0
    return (x, np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), mask)


def smoke(a):
    tk, cfg, sd = load_hf(nl_keep=2)
    nl, C = 2, a.C
    mod = LlamaPrefillRBLN(cfg, sd, a.S, nl, C).eval()
    rt = build_runtime(mod, cfg, a.S, nl, C, f"prefill {nl}L C={C}", device=a.device)
    embed = sd["model.embed_tokens.weight"]
    toks = tk.encode("The three primary colors are red, blue and yellow. "
                     "The capital of France is", add_special_tokens=False)[:C]
    toks = (toks + [0] * C)[:C]
    Kp = np.zeros((nl, cfg.n_kv, cfg.head_dim, a.S), np.float32)
    Vp = np.zeros((nl, cfg.n_kv, a.S, cfg.head_dim), np.float32)
    x, cos, sin, mask = chunk_inputs(cfg, a.S, C, 0, embed, toks)
    args = [x, Kp, Vp, cos, sin, mask]
    out = rt.run(*args)
    hr, kr, vr = (np.asarray(o) for o in out[:3])
    hc, kc, vc = mod(*[torch.from_numpy(v) for v in args])
    rel = lambda p, q: float(np.linalg.norm(p - q) / (np.linalg.norm(q) + 1e-9))
    print(f"\nchunked prefill C={C}, {nl} layers — ATOM vs CPU fp32")
    print(f"  hidden rel {rel(hr, hc.numpy()):.5f}")
    print(f"  K      rel {rel(kr, kc.numpy()):.5f}")
    print(f"  V      rel {rel(vr, vc.numpy()):.5f}")
    ok = all(rel(p, q.numpy()) < 0.05 for p, q in ((hr, hc), (kr, kc), (vr, vc)))
    print(f"  {'PASS ✅ numerics match' if ok else 'FAIL ❌'}")


def build(a):
    tk, cfg, sd = load_hf()
    nl, C = cfg.n_layers, a.C
    mod = LlamaPrefillRBLN(cfg, sd, a.S, nl, C).eval()
    del sd
    t0 = time.time()
    rt = build_runtime(mod, cfg, a.S, nl, C, f"prefill {nl}L C={C} fp16",
                       artifact=a.artifact, device=a.device)
    del mod
    _tk, cfg2, sd2 = load_hf()
    embed = sd2["model.embed_tokens.weight"]
    lm = sd2["lm_head.weight"]
    del sd2
    prompt = tk.encode(a.prompt, add_special_tokens=False)
    P = (len(prompt) // C) * C or C
    prompt = prompt[:P]
    K = np.zeros((nl, cfg.n_kv, cfg.head_dim, a.S), np.float32)
    V = np.zeros((nl, cfg.n_kv, a.S, cfg.head_dim), np.float32)
    ts = []
    t1 = time.perf_counter()
    for base in range(0, P, C):
        x, cos, sin, mask = chunk_inputs(cfg, a.S, C, base, embed,
                                         prompt[base:base + C])
        t2 = time.perf_counter()
        out = rt.run(x, K, V, cos, sin, mask)
        ts.append((time.perf_counter() - t2) * 1e3)
        hid = np.asarray(out[0])
        K[:, :, :, base:base + C] = np.asarray(out[1])
        V[:, :, base:base + C, :] = np.asarray(out[2])
    wall = (time.perf_counter() - t1) * 1e3
    nxt = int(np.argmax(lm @ hid[-1]))
    np.savez(a.kv_out, K=K, V=V, nxt=nxt, P=P)
    print("\n" + "=" * 66)
    print(f"ATOM chunked prefill — {nl}L, C={C}, {P} tokens")
    print(f"  {wall:.0f} ms total · {wall/P:.2f} ms/token · {len(ts)} chunks "
          f"({np.median(ts):.1f} ms each)")
    print(f"  next token: {nxt} {tk.decode([nxt])!r}")
    print(f"  KV -> {a.kv_out}  ({(K.nbytes+V.nbytes)/1e6:.0f} MB fp32)")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--device", type=int, default=1,
                    help="ATOM device index (0 is held by the decode pod)")
    ap.add_argument("--artifact", default="/home/gspark/rbln-llama8b-pd/prefill_S256_C16.rbln")
    ap.add_argument("--kv-out", default="/dev/shm/atom_prefill_kv.npz")
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    return smoke(a) if a.smoke else build(a)


if __name__ == "__main__":
    main()
