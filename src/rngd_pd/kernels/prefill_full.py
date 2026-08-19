#!/usr/bin/env python3
"""Prefill the whole prompt in one pass, the way the vendor's buckets do.

Reading Furiosa's own artifact settled what our chunked prefill was doing
wrong. Their prefill family is eleven buckets with `kv_cache_size = 0` and
attention sizes 128…1024: the entire prompt goes through one graph call, with
no cache tensor at all. Ours carried the full S=256 cache in and out of every
call — 8.4 MB round-tripped per call, 134 MB per chunk — and computed attention
against all S columns even when the cache was empty, all for chunks of 16 to 64
positions.

This layer drops both. There is no KV input; attention is [C, C] causal within
the prompt; and the K/V the prompt produces leave as outputs for whoever
decodes next. C is the prompt length, so the weights are read once for the
whole prompt instead of once per chunk.

  python3 llama_prefill_full.py --nl 1 --C 256 --chip 1 --validate
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)
nd.set_mesh_kind(nd.MeshKind.Mesh(2))
from furiosa.torch.custom_ops import TacticKernelModule, CompileModule  # noqa: E402
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig  # noqa: E402
from rngd_pd.kernels.prefill_chunk import (BF, H, M, NH, NKV, HD, G, rms_dsl, zipadd_dsl,
                                 swiglu_dsl, softmax_dsl, rot_matrices)  # noqa: E402

HINT = os.environ.get("CHUNK_TACTIC_HINT", "ForLlmModelPrefill")


class PrefillLayer(torch.nn.Module):
    """One layer over C prompt positions, with nothing cached beforehand."""

    def __init__(self, C):
        super().__init__()
        self.C = C
        self.rms1 = TacticKernelModule(rms_dsl(C, H))
        self.rms2 = TacticKernelModule(rms_dsl(C, H))
        self.sm = TacticKernelModule(softmax_dsl(NKV, G * C, C))   # [K, G*C, C]
        self.r1 = TacticKernelModule(zipadd_dsl(C, H))
        self.r2 = TacticKernelModule(zipadd_dsl(C, H))
        self.sw = TacticKernelModule(swiglu_dsl(C, M))

    def _rope(self, x, nh, rot):
        C = self.C
        return torch.matmul(x.view(C, nh, HD), rot).reshape(C, nh * HD).to(BF)

    def forward(self, x, eps, nh, n1, wqT, wkT, wvT, woT, n2, wgT, wuT, wdT,
                rot, mask):
        C = self.C
        h = self.rms1(x, n1, eps, nh)[0]
        q = self._rope(torch.matmul(h, wqT), NH, rot)
        k = self._rope(torch.matmul(h, wkT), NKV, rot)
        v = torch.matmul(h, wvT)

        kT = k.view(C, NKV, HD).permute(1, 2, 0)          # [NKV,HD,C]
        vT = v.view(C, NKV, HD).permute(1, 0, 2)          # [NKV,C,HD]
        qk = q.view(C, NKV, G, HD).permute(1, 2, 0, 3).reshape(NKV, G * C, HD)
        sc = torch.matmul(qk, kT) + mask                  # [NKV,G*C,C] causal
        p = self.sm(sc.to(BF))[0]
        ao = torch.matmul(p, vT)                          # [NKV,G*C,HD]
        ao = ao.view(NKV, G, C, HD).permute(2, 0, 1, 3).reshape(C, NH * HD)

        x2 = self.r1(torch.matmul(ao, woT), x)[0]
        h2 = self.rms2(x2, n2, eps, nh)[0]
        gg = torch.matmul(h2, wgT)
        uu = torch.matmul(h2, wuT)
        return self.r2(torch.matmul(self.sw(gg, uu)[0], wdT), x2)[0], kT, vT


class PrefillModel(torch.nn.Module):
    def __init__(self, C, nl):
        super().__init__()
        self.nl = nl
        self.layers = torch.nn.ModuleList([PrefillLayer(C) for _ in range(nl)])

    def forward(self, x, eps, nh, an1, aq, ak, av, ao, an2, awg, awu, awd,
                rot, mask):
        ks, vs = [], []
        for i in range(self.nl):
            x, k, v = self.layers[i](x, eps, nh, an1[i], aq[i], ak[i], av[i],
                                     ao[i], an2[i], awg[i], awu[i], awd[i],
                                     rot, mask)
            ks.append(k)
            vs.append(v)
        return x, torch.stack(ks), torch.stack(vs)


def cpu_layer_p(x, w, inv, C, eps):
    n1, wq, wk, wv, wo, n2, wgu, wd = w
    rms = lambda v, g: v * (1.0 / np.sqrt((v * v).mean(-1, keepdims=True) + eps)) * g
    h = rms(x, n1)

    def rope(t, nh):
        t = t.reshape(C, nh, HD).copy()
        half = HD // 2
        for c in range(C):
            ang = c * inv
            cs, sn = np.cos(ang), np.sin(ang)
            a, b = t[c, :, :half].copy(), t[c, :, half:].copy()
            t[c, :, :half] = a * cs - b * sn
            t[c, :, half:] = b * cs + a * sn
        return t

    q = rope(h @ wq.T, NH)
    k = rope(h @ wk.T, NKV)
    v = (h @ wv.T).reshape(C, NKV, HD)
    kT, vT = k.transpose(1, 2, 0), v.transpose(1, 0, 2)
    qk = q.reshape(C, NKV, G, HD).transpose(1, 2, 0, 3)
    sc = np.einsum("kgch,khd->kgcd", qk, kT) / math.sqrt(HD)
    m = np.triu(np.full((C, C), -1e4, np.float32), 1)
    sc = sc + m[None, None]
    sc = sc - sc.max(-1, keepdims=True)
    p = np.exp(sc); p /= p.sum(-1, keepdims=True)
    ao = np.einsum("kgcd,kdh->kgch", p, vT).transpose(2, 0, 1, 3).reshape(C, NH * HD)
    x2 = ao @ wo.T + x
    h2 = rms(x2, n2)
    gu = h2 @ wgu.T
    g_, u_ = gu[:, :M], gu[:, M:]
    return (g_ / (1 + np.exp(-g_)) * u_) @ wd.T + x2, kT, vT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", type=int, default=1)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--chip", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    C, nl = a.C, a.nl
    rng = np.random.default_rng(0)
    eps, sq, scale = 1e-5, math.sqrt(H), 1.0 / math.sqrt(HD)

    ws = []
    for _ in range(nl):
        ws.append((rng.standard_normal(H).astype(np.float32) * .05,
                   rng.standard_normal((NH * HD, H)).astype(np.float32) * .02,
                   rng.standard_normal((NKV * HD, H)).astype(np.float32) * .02,
                   rng.standard_normal((NKV * HD, H)).astype(np.float32) * .02,
                   rng.standard_normal((H, NH * HD)).astype(np.float32) * .02,
                   rng.standard_normal(H).astype(np.float32) * .05,
                   rng.standard_normal((2 * M, H)).astype(np.float32) * .02,
                   rng.standard_normal((H, M)).astype(np.float32) * .02))

    d = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(BF).to(dev)
    df = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32).to(dev)
    inv = (1.0 / (500000.0 ** (np.arange(0, HD, 2) / HD))).astype(np.float32)

    an1 = torch.stack([d(np.tile(w[0] * sq, (C, 1))) for w in ws])
    aq = torch.stack([d((w[1] * scale).T.copy()) for w in ws])
    ak = torch.stack([d(w[2].T.copy()) for w in ws])
    av = torch.stack([d(w[3].T.copy()) for w in ws])
    ao_ = torch.stack([d(w[4].T.copy()) for w in ws])
    an2 = torch.stack([d(np.tile(w[5] * sq, (C, 1))) for w in ws])
    awg = torch.stack([d(w[6][:M].T.copy()) for w in ws])
    awu = torch.stack([d(w[6][M:].T.copy()) for w in ws])
    awd = torch.stack([d(w[7].T.copy()) for w in ws])

    sample = (torch.zeros(C, H, dtype=BF),
              torch.zeros(C, H, dtype=torch.float32), torch.zeros(C, H, dtype=torch.float32),
              torch.zeros(nl, C, H, dtype=BF),
              torch.zeros(nl, H, NH * HD, dtype=BF), torch.zeros(nl, H, NKV * HD, dtype=BF),
              torch.zeros(nl, H, NKV * HD, dtype=BF), torch.zeros(nl, NH * HD, H, dtype=BF),
              torch.zeros(nl, C, H, dtype=BF),
              torch.zeros(nl, H, M, dtype=BF), torch.zeros(nl, H, M, dtype=BF),
              torch.zeros(nl, M, H, dtype=BF),
              torch.zeros(C, HD, HD, dtype=BF), torch.zeros(1, G * C, C, dtype=BF))

    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=getattr(TacticHintConfig, HINT))
    print(f"[full] compiling nl={nl} C={C} hint={HINT} (no KV input)...", flush=True)
    t0 = time.time()
    m = CompileModule.from_module(PrefillModel(C, nl), sample, compiler_config=cfg)
    m.to(dev)
    print(f"[full] compiled in {time.time()-t0:.0f}s", flush=True)

    x0 = rng.standard_normal((C, H)).astype(np.float32) * .1
    causal = np.triu(np.full((C, C), -1e4, np.float32), 1)
    args = (d(x0), df(np.full((C, H), eps * H, np.float32)),
            df(np.full((C, H), -0.5, np.float32)), an1, aq, ak, av, ao_, an2,
            awg, awu, awd, d(rot_matrices(inv, 0, C)),
            d(np.tile(causal, (G, 1)).reshape(1, G * C, C)))
    out = m(*args)
    out[0][0, 0].item()
    ts = []
    for _ in range(a.reps):
        t1 = time.perf_counter()
        out = m(*args)
        out[0][0, 0].item()                       # sync before stopping the clock
        ts.append((time.perf_counter() - t1) * 1e3)
    ts.sort()
    step = ts[len(ts) // 2]
    per_layer_bytes = 218e6 * 2
    print(f"[full] {step:.2f} ms for {C} tokens · {nl} layers -> {step/C:.4f} ms/token")
    print(f"[full] extrapolated to 32 layers: {step/nl*32/C:.3f} ms/token "
          f"(vendor engine: 0.10)")
    print(f"[full] weight bandwidth: {per_layer_bytes*nl/(step/1000)/1e9:.0f} GB/s")

    if a.validate:
        hid = out[0].detach().float().cpu().numpy()
        x = x0.copy()
        for i in range(nl):
            x, _k, _v = cpu_layer_p(x, ws[i], inv, C, eps)
        rel = float(np.linalg.norm(hid - x) / np.linalg.norm(x))
        print(f"[full] hidden rel vs CPU fp32: {rel:.5f} "
              f"{'PASS ✅' if rel < 0.05 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
