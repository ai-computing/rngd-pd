#!/usr/bin/env python3
"""32-layer whole-prompt prefill: the kv-free graph, chained over layer groups.

llama_prefill_full showed one layer of the vendor-shaped prefill (no KV input,
attention causal within the prompt) at C=256. A 32-layer graph still does not
compile, so the same trick applies as before: weights are graph inputs, so one
compiled NG-layer module walks the stack 32/NG times, threading the hidden
state. Unlike the cached-chunk version there is no KV to carry between calls —
each group simply emits the K/V its layers produced, which is exactly what the
decode side needs.

  python3 llama_prefill_full_chained.py --C 256 --group 4 --chip 2 --validate
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
from llama_prefill_full import (BF, H, M, NH, NKV, HD, G, PrefillModel,
                                cpu_layer_p, HINT)
from llama_prefill_chunk import rot_matrices
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    C, nl, NG = a.C, a.nl, a.group
    assert nl % NG == 0
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
              torch.zeros(C, HD, HD, dtype=BF), torch.zeros(1, G * C, C, dtype=BF))

    cfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=getattr(TacticHintConfig, HINT))
    print(f"[fullchain] compiling one {NG}-layer kv-free graph (C={C}, hint={HINT})...",
          flush=True)
    t0 = time.time()
    m = CompileModule.from_module(PrefillModel(C, NG), sample, compiler_config=cfg)
    m.to(dev)
    print(f"[fullchain] compiled in {time.time()-t0:.0f}s — reused {ngroups}x "
          f"for {nl} layers", flush=True)

    eps_d = df(np.full((C, H), eps * H, np.float32))
    nh_d = df(np.full((C, H), -0.5, np.float32))
    rot = d(rot_matrices(inv, 0, C))
    causal = np.triu(np.full((C, C), -1e4, np.float32), 1)
    maskd = d(np.tile(causal, (G, 1)).reshape(1, G * C, C))
    x0 = rng.standard_normal((C, H)).astype(np.float32) * .1

    def run(collect=False):
        x = d(x0)
        Ks, Vs = [], []
        for g in range(ngroups):
            x, k, v = m(x, eps_d, nh_d, *groups[g], rot, maskd)
            if collect:
                Ks.append(k)
                Vs.append(v)
        return x, Ks, Vs

    run()                                    # warm up
    ts = []
    for _ in range(a.reps):
        t1 = time.perf_counter()
        x, _, _ = run()
        x[0, 0].item()                       # sync before stopping the clock
        ts.append((time.perf_counter() - t1) * 1e3)
    ts.sort()
    step = ts[len(ts) // 2]
    weights = 218e6 * 2 * nl
    print(f"\n[fullchain] {nl} layers · C={C} · {ngroups} calls: {step:.1f} ms "
          f"for {C} tokens -> {step/C:.4f} ms/token", flush=True)
    print(f"[fullchain] weight bandwidth {weights/(step/1000)/1e9:.0f} GB/s "
          f"(sequential engine reaches ~946)", flush=True)
    print(f"[fullchain] vs our sequential prefill 14.8 ms/token: "
          f"{14.8/(step/C):.0f}x · vendor engine 0.10 ms/token: "
          f"{(step/C)/0.10:.1f}x away", flush=True)

    if a.validate:
        x, Ks, Vs = run(collect=True)
        got = x.detach().float().cpu().numpy()
        ref = x0.copy()
        for i in range(nl):
            ref, _k, _v = cpu_layer_p(ref, ws[i], inv, C, eps)
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        print(f"[fullchain] hidden rel vs CPU fp32: {rel:.5f} "
              f"{'PASS ✅' if rel < 0.08 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
