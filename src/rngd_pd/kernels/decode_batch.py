#!/usr/bin/env python3
"""Batched decode on RNGD: B streams per step, with a real KV update.

The existing batched module (llama_engine_batch.py) is a throughput probe — it
has no wk/wv and never writes the cache, so it measures a step that could not
serve a request. This one is the serving graph: K/V projections, RoPE, the
one-hot cache injection, and the updated cache handed back out, all with a
batch dimension.

Two things learned since batched decode was last attempted are what make it
worth retrying. Tactic choice turned out to matter more than expected on the
prefill side, so it is a knob here too; and a 32-layer graph does not have to
be one graph — the weights are inputs, so one compiled NG-layer module walks
the stack 32/NG times, carrying the hidden state and each group's slice of the
cache. The earlier sweep only ever tried S=256 at nl=4 and nl=8, both of which
exploded in tactic search; nl=2 at S=256 was never tried with a batch.

RoPE is a per-stream [HD,HD] rotation matmul rather than a half-split, because
slicing a 3-D activation miscompiles silently on this path — the same defence
the prefill kernel uses.

  python3 llama_decode_batch.py --B 2 --nl 2 --S 256 --chip 2 --validate
  python3 llama_decode_batch.py --B 4 --nl 32 --group 2 --chip 2 --validate
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
from rngd_pd.kernels.dsl_batch import (rms_dsl, zipadd_dsl, swiglu_dsl,
                                softmax_dsl)  # noqa: E402

BF = torch.bfloat16
H, M, NH, NKV, HD = 4096, 14336, 32, 8, 128
G = NH // NKV
HINT = os.environ.get("DECODE_TACTIC_HINT", "ForLlmModelDecode")

# The layer boundary was never the tactic search policy — ByEstimation failed at
# group=4 exactly as Random did, both after ~11 minutes. It was
# max_num_unique_shapes, which FCConfig leaves unbounded and this repo never
# set. Capping it compiles group=4 in 121s where unbounded fails outright.
_MS = os.environ.get("RNGD_MAX_SHAPES", "")
MAX_SHAPES = {"max_num_unique_shapes": int(_MS)} if _MS else {}


class DecodeLayer(torch.nn.Module):
    """One layer, one step, B streams — each with its own position."""

    def __init__(self, B, S):
        super().__init__()
        self.B, self.S = B, S
        self.rms1 = TacticKernelModule(rms_dsl(B))
        self.rms2 = TacticKernelModule(rms_dsl(B))
        self.sm = TacticKernelModule(softmax_dsl(B, S))
        self.r1 = TacticKernelModule(zipadd_dsl(B))
        self.r2 = TacticKernelModule(zipadd_dsl(B))
        self.sw = TacticKernelModule(swiglu_dsl(B))

    def _rope(self, x, nh, rot):
        B = self.B
        return torch.matmul(x.view(B, nh, HD), rot).reshape(B, nh * HD).to(BF)

    def forward(self, x, eps, nh, n1, wqT, wkT, wvT, woT, n2, wgT, wuT, wdT,
                Kp, Vp, rot, mask, oh):
        B, S = self.B, self.S
        h = self.rms1(x, n1, eps, nh)[0]                       # [B,H]
        q = self._rope(torch.matmul(h, wqT), NH, rot)
        k = self._rope(torch.matmul(h, wkT), NKV, rot)
        v = torch.matmul(h, wvT)                               # [B,NKV*HD]

        Kf = Kp + k.view(B, NKV, HD, 1) * oh.view(B, 1, 1, S)  # [B,NKV,HD,S]
        Vf = Vp + v.view(B, NKV, 1, HD) * oh.view(B, 1, S, 1)  # [B,NKV,S,HD]
        sc = torch.matmul(q.view(B, NKV, G, HD), Kf) + mask.view(B, 1, 1, S)
        p = self.sm(sc.to(BF))[0]                              # [B,NKV,G,S]
        ao = torch.matmul(p, Vf).reshape(B, NH * HD)

        x2 = self.r1(torch.matmul(ao, woT), x)[0]
        h2 = self.rms2(x2, n2, eps, nh)[0]
        gg = torch.matmul(h2, wgT)
        uu = torch.matmul(h2, wuT)
        return self.r2(torch.matmul(self.sw(gg, uu)[0], wdT), x2)[0], Kf, Vf


class DecodeModel(torch.nn.Module):
    def __init__(self, B, S, nl):
        super().__init__()
        self.nl = nl
        self.layers = torch.nn.ModuleList([DecodeLayer(B, S) for _ in range(nl)])

    def forward(self, x, eps, nh, an1, aq, ak, av, ao, an2, awg, awu, awd,
                Kp, Vp, rot, mask, oh):
        ks, vs = [], []
        for i in range(self.nl):
            x, Kf, Vf = self.layers[i](x, eps, nh, an1[i], aq[i], ak[i], av[i],
                                       ao[i], an2[i], awg[i], awu[i], awd[i],
                                       Kp[i], Vp[i], rot, mask, oh)
            ks.append(Kf)
            vs.append(Vf)
        return x, torch.stack(ks), torch.stack(vs)


def rot_batch(inv, positions):
    """[B,HD,HD] rotations, one per stream's position."""
    half = HD // 2
    ang = np.asarray(positions)[:, None] * inv[None, :]
    cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    out = np.zeros((len(positions), HD, HD), np.float32)
    i = np.arange(half)
    out[:, i, i] = cs
    out[:, i + half, i + half] = cs
    out[:, i, i + half] = sn
    out[:, i + half, i] = -sn
    return out


def cpu_layer(x, w, Kp, Vp, inv, positions, S, eps):
    """fp32 reference for one layer, B streams."""
    n1, wq, wk, wv, wo, n2, wgu, wd = w
    B = x.shape[0]
    rms = lambda t, g: t * (1.0 / np.sqrt((t * t).mean(-1, keepdims=True) + eps)) * g
    h = rms(x, n1)

    def rope(t, nh):
        t = t.reshape(B, nh, HD).copy()
        half = HD // 2
        for b in range(B):
            ang = positions[b] * inv
            cs, sn = np.cos(ang), np.sin(ang)
            a, bb = t[b, :, :half].copy(), t[b, :, half:].copy()
            t[b, :, :half] = a * cs - bb * sn
            t[b, :, half:] = bb * cs + a * sn
        return t

    q = rope(h @ wq.T, NH)
    k = rope(h @ wk.T, NKV)
    v = (h @ wv.T).reshape(B, NKV, HD)
    K, V = Kp.copy(), Vp.copy()
    for b in range(B):
        K[b, :, :, positions[b]] = k[b]
        V[b, :, positions[b], :] = v[b]
    sc = np.einsum("bkgh,bkhs->bkgs", q.reshape(B, NKV, G, HD), K) / math.sqrt(HD)
    for b in range(B):
        sc[b, :, :, positions[b] + 1:] = -1e4
    sc = sc - sc.max(-1, keepdims=True)
    p = np.exp(sc); p /= p.sum(-1, keepdims=True)
    ao = np.einsum("bkgs,bksh->bkgh", p, V).reshape(B, NH * HD)
    x2 = ao @ wo.T + x
    h2 = rms(x2, n2)
    gu = h2 @ wgu.T
    g_, u_ = gu[:, :M], gu[:, M:]
    return (g_ / (1 + np.exp(-g_)) * u_) @ wd.T + x2, K, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--nl", type=int, default=2, help="total layers to run")
    ap.add_argument("--group", type=int, default=0,
                    help="layers per compiled call (0 = all in one graph)")
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pos", type=int, default=64, help="starting cache position")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    B, S, nl = a.B, a.S, a.nl
    NG = a.group or nl
    assert nl % NG == 0, "layer count must divide into whole groups"
    ngroups = nl // NG
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

    groups = []
    for g in range(ngroups):
        sl = ws[g * NG:(g + 1) * NG]
        groups.append((
            torch.stack([d(np.tile(w[0] * sq, (B, 1))) for w in sl]),
            torch.stack([d((w[1] * scale).T.copy()) for w in sl]),
            torch.stack([d(w[2].T.copy()) for w in sl]),
            torch.stack([d(w[3].T.copy()) for w in sl]),
            torch.stack([d(w[4].T.copy()) for w in sl]),
            torch.stack([d(np.tile(w[5] * sq, (B, 1))) for w in sl]),
            torch.stack([d(w[6][:M].T.copy()) for w in sl]),
            torch.stack([d(w[6][M:].T.copy()) for w in sl]),
            torch.stack([d(w[7].T.copy()) for w in sl]),
        ))

    sample = (torch.zeros(B, H, dtype=BF),
              torch.zeros(B, H, dtype=torch.float32), torch.zeros(B, H, dtype=torch.float32),
              torch.zeros(NG, B, H, dtype=BF),
              torch.zeros(NG, H, NH * HD, dtype=BF), torch.zeros(NG, H, NKV * HD, dtype=BF),
              torch.zeros(NG, H, NKV * HD, dtype=BF), torch.zeros(NG, NH * HD, H, dtype=BF),
              torch.zeros(NG, B, H, dtype=BF),
              torch.zeros(NG, H, M, dtype=BF), torch.zeros(NG, H, M, dtype=BF),
              torch.zeros(NG, M, H, dtype=BF),
              torch.zeros(NG, B, NKV, HD, S, dtype=BF),
              torch.zeros(NG, B, NKV, S, HD, dtype=BF),
              torch.zeros(B, HD, HD, dtype=BF),
              torch.zeros(B, S, dtype=BF), torch.zeros(B, S, dtype=BF))

    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=getattr(TacticHintConfig, HINT),
                   **MAX_SHAPES)
    print(f"[dbatch] compiling B={B} NG={NG} S={S} hint={HINT} "
          f"shapes={MAX_SHAPES.get('max_num_unique_shapes', 'unbounded')} "
          f"(reused {ngroups}x for {nl} layers)...", flush=True)
    t0 = time.time()
    m = CompileModule.from_module(DecodeModel(B, S, NG), sample, compiler_config=cfg)
    m.to(dev)
    print(f"[dbatch] compiled in {time.time()-t0:.0f}s", flush=True)

    eps_d = df(np.full((B, H), eps * H, np.float32))
    nh_d = df(np.full((B, H), -0.5, np.float32))
    pos = [a.pos + i for i in range(B)]          # streams sit at different places
    x0 = rng.standard_normal((B, H)).astype(np.float32) * .1
    K0 = rng.standard_normal((nl, B, NKV, HD, S)).astype(np.float32) * .05
    V0 = rng.standard_normal((nl, B, NKV, S, HD)).astype(np.float32) * .05
    for b in range(B):                            # nothing written past the head
        K0[:, b, :, :, pos[b]:] = 0.0
        V0[:, b, :, pos[b]:, :] = 0.0

    def dev_kv(p):
        Kd = [d(K0[g * NG:(g + 1) * NG]) for g in range(ngroups)]
        Vd = [d(V0[g * NG:(g + 1) * NG]) for g in range(ngroups)]
        return Kd, Vd

    def step(x, Kd, Vd, positions):
        rot = d(rot_batch(inv, positions))
        mk = np.stack([np.concatenate([np.zeros(p + 1, np.float32),
                                       np.full(S - p - 1, -1e4, np.float32)])
                       for p in positions])
        ohn = np.stack([np.eye(S, dtype=np.float32)[p] for p in positions])
        maskd, ohd = d(mk), d(ohn)
        for g in range(ngroups):
            x, Kf, Vf = m(x, eps_d, nh_d, *groups[g], Kd[g], Vd[g],
                          rot, maskd, ohd)
            Kd[g], Vd[g] = Kf, Vf
        return x, Kd, Vd

    Kd, Vd = dev_kv(0)
    step(d(x0), Kd, Vd, pos)                      # warm up
    ts = []
    for i in range(a.steps):
        Kd, Vd = dev_kv(0)
        t1 = time.perf_counter()
        out, Kd, Vd = step(d(x0), Kd, Vd, pos)
        out[0, 0].item()                          # sync before stopping the clock
        ts.append((time.perf_counter() - t1) * 1e3)
    ts.sort()
    stepms = ts[len(ts) // 2]
    weights = 218e6 * 2 * nl
    print(f"\n[dbatch] {nl} layers · B={B} · {ngroups} calls: {stepms:.2f} ms/step "
          f"-> {stepms/B:.2f} ms per token", flush=True)
    print(f"[dbatch] weight bandwidth {weights/(stepms/1000)/1e9:.0f} GB/s", flush=True)
    if nl == 32:
        print(f"[dbatch] vs B=1 resident engine 25.7 ms/token: "
              f"{25.7/(stepms/B):.2f}x · ATOM B={B} 83-91 ms/token", flush=True)

    if a.validate:
        Kd, Vd = dev_kv(0)
        out, Kd, Vd = step(d(x0), Kd, Vd, pos)
        got = out.detach().cpu().float().numpy()
        x, K, V = x0.copy(), K0.copy(), V0.copy()
        for i in range(nl):
            x, K[i], V[i] = cpu_layer(x, ws[i], K[i], V[i], inv, pos, S, eps)
        rel = float(np.linalg.norm(got - x) / np.linalg.norm(x))
        print(f"[dbatch] hidden rel vs CPU fp32: {rel:.5f} "
              f"{'PASS ✅' if rel < 0.08 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
