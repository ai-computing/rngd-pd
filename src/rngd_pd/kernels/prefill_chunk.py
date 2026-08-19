#!/usr/bin/env python3
"""Chunked prefill for RNGD, built on the low-level tactic kernels.

Every RNGD prefill path we have walks the prompt one token per graph pass, so
an 8B prompt pays a full weight read per token — 14 ms each. Prefill knows all
its positions up front, so C of them can ride one weight read. This layer does
that while staying inside the shapes the compiler actually accepts:

  softmax   C is folded into the existing group dimension, [K, G*C, S], so the
            reduce stays on the last axis where the working kernel already has
            it — no new leading axis, which is what the batched decode kernels
            add.
  rms/swiglu/zip_add   these take a leading [C, N] the same way the batched
            decode kernels do; verified to compile at C up to 4 and used here
            at 16.
  RoPE      positions differ across the chunk, so each gets its own [HD, HD]
            block-rotation matrix and the whole chunk rotates with one batched
            matmul. Slicing the head dimension would be the obvious way and is
            exactly what silently miscompiles on 3-D activations.
  KV write  the single-position engine injects with a one-hot outer product;
            for C positions that generalises to a [C, S] one-hot matmul, so the
            cache update stays a matmul instead of a scatter.

  python3 llama_prefill_chunk.py --nl 1 --C 16 --chip 2 --validate
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

BF = torch.bfloat16
H, M, NH, NKV, HD = 4096, 14336, 32, 8, 128
HINT = os.environ.get("CHUNK_TACTIC_HINT", "ForLlmModelPrefill")
G = NH // NKV


def rms_dsl(C, N):
    return f"""#tactic_kernel_dsl
def rms(x: [C={C},N={N}]/bf16, w: [C={C},N={N}]/bf16, e: [C={C},N={N}]/f32, nh: [C={C},N={N}]/f32) -> [C={C},N={N}]/bf16:
    output: [C,N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w)
        %2 = read(e)
        %3 = read(nh)
        %4 = ve.exec(%0 * %0)
        %5 = ve.exec(+ t[N] %4)
        %6 = ve.exec(%5 + %2)
        %7 = ve.exec(log %6)
        %8 = ve.exec(%7 * %3)
        %9 = ve.exec(exp %8)
        %10 = ve.exec(%0 * %9)
        %11 = ve.exec(%10 * %1)
        write.to([C,N]/bf16)
    )
    return output
"""


def zipadd_dsl(C, N):
    return f"""#tactic_kernel_dsl
def zip_add(a: [C={C},N={N}]/bf16, b: [C={C},N={N}]/bf16) -> [C={C},N={N}]/bf16:
    output: [C,N]/bf16 = tk.Interleaving(
        %0 = read(a)
        %1 = read(b)
        %2 = ve.exec(%0 + %1)
        write.to([C,N]/bf16)
    )
    return output
"""


def swiglu_dsl(C, N):
    return f"""#tactic_kernel_dsl
def swiglu(gate: [C={C},N={N}]/bf16, up: [C={C},N={N}]/bf16) -> [C={C},N={N}]/bf16:
    silu: [C,N]/bf16 = tk.Elementwise(
        %0 = read(gate)
        %1 = ve.exec(sigmoid %0)
        %2 = ve.exec(%0 * %1)
        write.to([C,N]/bf16)
    )
    output: [C,N]/bf16 = tk.Interleaving(
        %0 = read(silu)
        %1 = read(up)
        %2 = ve.exec(%0 * %1)
        write.to([C,N]/bf16)
    )
    return output
"""


def softmax_dsl(K, GC, S):
    return f"""#tactic_kernel_dsl
def softmax_attn(sc: [K={K},G={GC},S={S}]/bf16) -> [K={K},G={GC},S={S}]/bf16:
    output: [K,G,S]/bf16 = tk.ReduceByVe(
        %0 = read(sc)
        %1 = ve.exec(max_f t[S] %0)
        %2 = ve.exec(%0 - %1)
        %3 = ve.exec(exp %2)
        %4 = ve.exec(+ t[S] %3)
        %5 = ve.exec(%3 / %4)
        write.to([K,G,S]/bf16)
    )
    return output
"""


class ChunkLayer(torch.nn.Module):
    """One transformer layer advancing C positions of a single sequence."""

    def __init__(self, C, S):
        super().__init__()
        self.C, self.S = C, S
        self.rms1 = TacticKernelModule(rms_dsl(C, H))
        self.rms2 = TacticKernelModule(rms_dsl(C, H))
        self.sm = TacticKernelModule(softmax_dsl(NKV, G * C, S))
        self.r1 = TacticKernelModule(zipadd_dsl(C, H))
        self.r2 = TacticKernelModule(zipadd_dsl(C, H))
        self.sw = TacticKernelModule(swiglu_dsl(C, M))

    def _rope(self, x, nh, rot):
        """x [C, nh*HD] · rot [C, HD, HD] -> [C, nh*HD], one matmul per position."""
        C = self.C
        return torch.matmul(x.view(C, nh, HD), rot).reshape(C, nh * HD).to(BF)

    def forward(self, x, eps, nh, n1, wqT, wkT, wvT, woT, n2, wgT, wuT, wdT,
                Kp, Vp, rot, ohK, ohV, mask):
        # x [C,H] · Kp [NKV,HD,S] · Vp [NKV,S,HD] · rot [C,HD,HD]
        # ohK [C,S] one-hot columns for this chunk · ohV [S,C] its transpose
        C, S = self.C, self.S
        h = self.rms1(x, n1, eps, nh)[0]
        q = self._rope(torch.matmul(h, wqT), NH, rot)
        k = self._rope(torch.matmul(h, wkT), NKV, rot)
        v = torch.matmul(h, wvT)                                   # [C, NKV*HD]

        # write C columns with one matmul instead of a scatter
        kT = k.view(C, NKV, HD).permute(1, 2, 0)                   # [NKV,HD,C]
        vT = v.view(C, NKV, HD).permute(1, 0, 2)                   # [NKV,C,HD]
        Kf = Kp + torch.matmul(kT, ohK)                            # [NKV,HD,S]
        Vf = Vp + torch.matmul(ohV, vT)                            # [NKV,S,HD]

        qk = q.view(C, NKV, G, HD).permute(1, 2, 0, 3).reshape(NKV, G * C, HD)
        sc = torch.matmul(qk, Kf) + mask                           # [NKV,G*C,S]
        p = self.sm(sc.to(BF))[0]
        ao = torch.matmul(p, Vf)                                   # [NKV,G*C,HD]
        ao = ao.view(NKV, G, C, HD).permute(2, 0, 1, 3).reshape(C, NH * HD)

        x2 = self.r1(torch.matmul(ao, woT), x)[0]
        h2 = self.rms2(x2, n2, eps, nh)[0]
        gg = torch.matmul(h2, wgT)
        uu = torch.matmul(h2, wuT)
        return self.r2(torch.matmul(self.sw(gg, uu)[0], wdT), x2)[0], Kf, Vf


class ChunkModel(torch.nn.Module):
    def __init__(self, C, S, nl):
        super().__init__()
        self.nl = nl
        self.layers = torch.nn.ModuleList([ChunkLayer(C, S) for _ in range(nl)])

    def forward(self, x, eps, nh, an1, aq, ak, av, ao, an2, awg, awu, awd,
                Kp, Vp, rot, ohK, ohV, mask):
        ks, vs = [], []
        for i in range(self.nl):
            x, Kf, Vf = self.layers[i](x, eps, nh, an1[i], aq[i], ak[i], av[i],
                                       ao[i], an2[i], awg[i], awu[i], awd[i],
                                       Kp[i], Vp[i], rot, ohK, ohV, mask)
            ks.append(Kf)
            vs.append(Vf)
        return x, torch.stack(ks), torch.stack(vs)


# --------------------------------------------------------------------------
def rot_matrices(inv, base, C):
    """Block-rotation matrix per position: [C, HD, HD]."""
    half = HD // 2
    out = np.zeros((C, HD, HD), np.float32)
    for c in range(C):
        ang = (base + c) * inv
        cs, sn = np.cos(ang), np.sin(ang)
        for i in range(half):
            out[c, i, i] = cs[i]
            out[c, i + half, i] = -sn[i]
            out[c, i, i + half] = sn[i]
            out[c, i + half, i + half] = cs[i]
    return out


def cpu_layer(x, w, Kp, Vp, inv, base, C, S, eps):
    """Same math in fp32 on the host, for the numeric check."""
    n1, wq, wk, wv, wo, n2, wgu, wd = w
    rms = lambda v, g: v * (1.0 / np.sqrt((v * v).mean(-1, keepdims=True) + eps)) * g
    h = rms(x, n1)
    def rope(t, nh):
        t = t.reshape(C, nh, HD).copy()
        half = HD // 2
        for c in range(C):
            ang = (base + c) * inv
            cs, sn = np.cos(ang), np.sin(ang)
            a, b = t[c, :, :half].copy(), t[c, :, half:].copy()
            t[c, :, :half] = a * cs - b * sn
            t[c, :, half:] = b * cs + a * sn
        return t
    q = rope(h @ wq.T, NH)
    k = rope(h @ wk.T, NKV)
    v = (h @ wv.T).reshape(C, NKV, HD)
    Kf, Vf = Kp.copy(), Vp.copy()
    Kf[:, :, base:base + C] = k.transpose(1, 2, 0)
    Vf[:, base:base + C, :] = v.transpose(1, 0, 2)
    qk = q.reshape(C, NKV, G, HD).transpose(1, 2, 0, 3)             # [K,G,C,HD]
    sc = np.einsum("kgch,khs->kgcs", qk, Kf) / math.sqrt(HD)
    m = np.full((C, S), -1e4, np.float32)
    for c in range(C):
        m[c, :base + c + 1] = 0.0
    sc = sc + m[None, None]
    sc = sc - sc.max(-1, keepdims=True)
    p = np.exp(sc); p /= p.sum(-1, keepdims=True)
    ao = np.einsum("kgcs,ksh->kgch", p, Vf).transpose(2, 0, 1, 3).reshape(C, NH * HD)
    x2 = ao @ wo.T + x
    h2 = rms(x2, n2)
    gu = h2 @ wgu.T
    g_, u_ = gu[:, :M], gu[:, M:]
    return (g_ / (1 + np.exp(-g_)) * u_) @ wd.T + x2, Kf, Vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", type=int, default=1)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--base", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    C, S, nl = a.C, a.S, a.nl
    rng = np.random.default_rng(0)
    eps = 1e-5
    sq = math.sqrt(H)
    scale = 1.0 / math.sqrt(HD)

    # random weights: this measures shape/kernel correctness, not model quality
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

    an1 = torch.stack([d(w[0] * sq) for w in ws])
    aq = torch.stack([d((w[1] * scale).T.copy()) for w in ws])
    ak = torch.stack([d(w[2].T.copy()) for w in ws])
    av = torch.stack([d(w[3].T.copy()) for w in ws])
    ao_ = torch.stack([d(w[4].T.copy()) for w in ws])
    an2 = torch.stack([d(w[5] * sq) for w in ws])
    awg = torch.stack([d(w[6][:M].T.copy()) for w in ws])
    awu = torch.stack([d(w[6][M:].T.copy()) for w in ws])
    awd = torch.stack([d(w[7].T.copy()) for w in ws])

    x0 = rng.standard_normal((C, H)).astype(np.float32) * .1
    Kp0 = np.zeros((nl, NKV, HD, S), np.float32)
    Vp0 = np.zeros((nl, NKV, S, HD), np.float32)
    ohK = np.zeros((C, S), np.float32)
    for c in range(C):
        ohK[c, a.base + c] = 1.0
    maskm = np.full((C, S), -1e4, np.float32)
    for c in range(C):
        maskm[c, :a.base + c + 1] = 0.0
    mask = np.tile(maskm, (G, 1)).reshape(1, G * C, S)

    sample = (torch.zeros(C, H, dtype=BF),
              torch.zeros(C, H, dtype=torch.float32), torch.zeros(C, H, dtype=torch.float32),
              torch.zeros(nl, C, H, dtype=BF),
              torch.zeros(nl, H, NH * HD, dtype=BF), torch.zeros(nl, H, NKV * HD, dtype=BF),
              torch.zeros(nl, H, NKV * HD, dtype=BF), torch.zeros(nl, NH * HD, H, dtype=BF),
              torch.zeros(nl, C, H, dtype=BF),
              torch.zeros(nl, H, M, dtype=BF), torch.zeros(nl, H, M, dtype=BF),
              torch.zeros(nl, M, H, dtype=BF),
              torch.zeros(nl, NKV, HD, S, dtype=BF), torch.zeros(nl, NKV, S, HD, dtype=BF),
              torch.zeros(C, HD, HD, dtype=BF),
              torch.zeros(C, S, dtype=BF), torch.zeros(S, C, dtype=BF),
              torch.zeros(1, G * C, S, dtype=BF))

    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=getattr(TacticHintConfig, HINT))
    print(f"[chunk] compiling nl={nl} C={C} S={S} on {dev}...", flush=True)
    t0 = time.time()
    m = CompileModule.from_module(ChunkModel(C, S, nl), sample, compiler_config=cfg)
    m.to(dev)
    print(f"[chunk] compiled in {time.time()-t0:.0f}s", flush=True)

    eps_d = df(np.full((C, H), eps * H, np.float32))
    nh_d = df(np.full((C, H), -0.5, np.float32))
    an1_r = torch.stack([t.unsqueeze(0).expand(C, H).contiguous() for t in an1])
    an2_r = torch.stack([t.unsqueeze(0).expand(C, H).contiguous() for t in an2])
    args = (d(x0), eps_d, nh_d, an1_r, aq, ak, av, ao_, an2_r, awg, awu, awd,
            d(Kp0), d(Vp0), d(rot_matrices(inv, a.base, C)),
            d(ohK), d(ohK.T.copy()), d(mask))
    reps = 5
    m(*args)
    ts = []
    for _ in range(reps):
        t1 = time.perf_counter()
        out = m(*args)
        out[0][0, 0].item() if isinstance(out, (tuple, list)) else out[0, 0].item()
        ts.append((time.perf_counter() - t1) * 1e3)   # after a real sync
    ts.sort()
    step = ts[len(ts) // 2]
    print(f"[chunk] {step:.2f} ms per {C}-token chunk -> {step/C:.3f} ms/token "
          f"({nl} layers)", flush=True)
    if nl:
        print(f"[chunk] extrapolated to 32 layers: {step/nl*32/C:.3f} ms/token",
              flush=True)

    if a.validate:
        hid = out[0].detach().cpu().float().numpy()
        x, Kp, Vp = x0.copy(), Kp0.copy(), Vp0.copy()
        for i in range(nl):
            x, Kp[i], Vp[i] = cpu_layer(x, ws[i], Kp[i], Vp[i], inv, a.base, C, S, eps)
        rel = np.linalg.norm(hid - x) / np.linalg.norm(x)
        Kd = out[1].detach().cpu().float().numpy()
        kr = np.linalg.norm(Kd - Kp) / (np.linalg.norm(Kp) + 1e-9)
        print(f"[chunk] hidden rel {rel:.5f} · K rel {kr:.5f} "
              f"{'PASS ✅' if rel < 0.05 and kr < 0.05 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
