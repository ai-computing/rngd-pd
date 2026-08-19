#!/usr/bin/env python3
"""32-layer chunked prefill by reusing one small compiled graph.

The chunked layer compiles at two layers and explodes in tactic search at four,
so a single 32-layer graph is out. It does not have to be one graph: the layer
weights are graph *inputs*, not baked constants, so one compiled two-layer
module can walk the whole model sixteen times over, each call handed a
different pair of weight slices and that pair's slice of the KV cache. The
hidden state threads through the calls exactly as it would inside one graph.

What this costs is per-call dispatch, paid sixteen times instead of once; what
it buys is a prefill that reads each weight once per chunk of C positions
instead of once per token.

  python3 llama_prefill_chunk_chained.py --C 16 --group 2 --chip 2 --validate
"""
import argparse
import math
import time

import numpy as np
import torch

import _paths  # noqa: F401
from llama_prefill_chunk import (BF, H, M, NH, NKV, HD, G, ChunkModel,
                                 rot_matrices, cpu_layer)
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--group", type=int, default=2, help="layers per compiled call")
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=4, help="chunks of C to time")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    C, S, nl, NG = a.C, a.S, a.nl, a.group
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

    # per-group weight bundles, each already contiguous on the card
    groups = []
    for g in range(ngroups):
        sl = ws[g * NG:(g + 1) * NG]
        groups.append((
            torch.stack([d(np.tile(w[0] * sq, (C, 1))) for w in sl]),
            torch.stack([d((w[1] * scale).T.copy()) for w in sl]),
            torch.stack([d(w[2].T.copy()) for w in sl]),
            torch.stack([d(w[3].T.copy()) for w in sl]),
            torch.stack([d(w[4].T.copy()) for w in sl]),
            torch.stack([d(np.tile(w[5] * sq, (C, 1))) for w in sl]),
            torch.stack([d(w[6][:M].T.copy()) for w in sl]),
            torch.stack([d(w[6][M:].T.copy()) for w in sl]),
            torch.stack([d(w[7].T.copy()) for w in sl]),
        ))

    sample = (torch.zeros(C, H, dtype=BF),
              torch.zeros(C, H, dtype=torch.float32), torch.zeros(C, H, dtype=torch.float32),
              torch.zeros(NG, C, H, dtype=BF),
              torch.zeros(NG, H, NH * HD, dtype=BF), torch.zeros(NG, H, NKV * HD, dtype=BF),
              torch.zeros(NG, H, NKV * HD, dtype=BF), torch.zeros(NG, NH * HD, H, dtype=BF),
              torch.zeros(NG, C, H, dtype=BF),
              torch.zeros(NG, H, M, dtype=BF), torch.zeros(NG, H, M, dtype=BF),
              torch.zeros(NG, M, H, dtype=BF),
              torch.zeros(NG, NKV, HD, S, dtype=BF), torch.zeros(NG, NKV, S, HD, dtype=BF),
              torch.zeros(C, HD, HD, dtype=BF),
              torch.zeros(C, S, dtype=BF), torch.zeros(S, C, dtype=BF),
              torch.zeros(1, G * C, S, dtype=BF))

    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=TacticHintConfig.ForLlmModelDecode)
    print(f"[chain] compiling one {NG}-layer chunk graph (C={C}, S={S})...", flush=True)
    t0 = time.time()
    m = CompileModule.from_module(ChunkModel(C, S, NG), sample, compiler_config=cfg)
    m.to(dev)
    print(f"[chain] compiled in {time.time()-t0:.0f}s — reused {ngroups}x for {nl} layers",
          flush=True)

    eps_d = df(np.full((C, H), eps * H, np.float32))
    nh_d = df(np.full((C, H), -0.5, np.float32))
    Kdev = [d(np.zeros((NG, NKV, HD, S), np.float32)) for _ in range(ngroups)]
    Vdev = [d(np.zeros((NG, NKV, S, HD), np.float32)) for _ in range(ngroups)]

    def run_chunk(x_np, base):
        ohK = np.zeros((C, S), np.float32)
        mk = np.full((C, S), -1e4, np.float32)
        for c in range(C):
            ohK[c, base + c] = 1.0
            mk[c, :base + c + 1] = 0.0
        rot = d(rot_matrices(inv, base, C))
        ohKd, ohVd = d(ohK), d(ohK.T.copy())
        maskd = d(np.tile(mk, (G, 1)).reshape(1, G * C, S))
        x = d(x_np)
        for g in range(ngroups):
            an1, aq, ak, av, ao, an2, awg, awu, awd = groups[g]
            x, Kf, Vf = m(x, eps_d, nh_d, an1, aq, ak, av, ao, an2, awg, awu, awd,
                          Kdev[g], Vdev[g], rot, ohKd, ohVd, maskd)
            Kdev[g], Vdev[g] = Kf, Vf
        return x

    x0 = rng.standard_normal((C, H)).astype(np.float32) * .1
    run_chunk(x0, 0)                                  # warm up
    Kdev = [d(np.zeros((NG, NKV, HD, S), np.float32)) for _ in range(ngroups)]
    Vdev = [d(np.zeros((NG, NKV, S, HD), np.float32)) for _ in range(ngroups)]

    ts = []
    for i in range(a.chunks):
        t1 = time.perf_counter()
        out = run_chunk(x0, i * C)
        out[0][0, 0].item() if isinstance(out, (tuple, list)) else out[0, 0].item()
        ts.append((time.perf_counter() - t1) * 1e3)   # after a real sync
    ts.sort()
    step = ts[len(ts) // 2]
    print(f"\n[chain] {nl} layers · C={C}: {step:.1f} ms per chunk "
          f"-> {step/C:.3f} ms/token", flush=True)
    print(f"[chain] vs our sequential prefill 14.8 ms/token: "
          f"{14.8/(step/C):.1f}x faster", flush=True)

    if a.validate:
        Kdev = [d(np.zeros((NG, NKV, HD, S), np.float32)) for _ in range(ngroups)]
        Vdev = [d(np.zeros((NG, NKV, S, HD), np.float32)) for _ in range(ngroups)]
        got = run_chunk(x0, 0).detach().float().cpu().numpy()
        x = x0.copy()
        Kp = np.zeros((nl, NKV, HD, S), np.float32)
        Vp = np.zeros((nl, NKV, S, HD), np.float32)
        for i in range(nl):
            x, Kp[i], Vp[i] = cpu_layer(x, ws[i], Kp[i], Vp[i], inv, 0, C, S, eps)
        rel = float(np.linalg.norm(got - x) / np.linalg.norm(x))
        print(f"[chain] hidden rel vs CPU fp32: {rel:.5f} "
              f"{'PASS ✅ chained graph == one graph' if rel < 0.08 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
