#!/usr/bin/env python3
"""Batched decode graph on 2026.2.0 vs 2026.3.0, same kernels, same widths.

lmhead_hint_sweep covers the graph that used to collapse; this covers the one
that did not, so the pair says whether the new backend moved the decode step
itself or only removed the lm_head cliff. Completion is forced by reading the
output back rather than by the stream, because that is the one method both
SDKs answer to and it is what bench/batch_tactic_probe.py used for the numbers
this compares against.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rngd_pd.engines.fsdk import open_device  # noqa: E402

open_device(8, 2)

from rngd_pd.engines.fsdk import (SDK, CompileModule, FCConfig,  # noqa: E402
                                  TacticHintConfig, dev_str, sync)
from rngd_pd.kernels.decode_batch import (DecodeModel, H, M, NH,  # noqa: E402
                                          NKV, HD, BF)

WBYTES = 8.03e9 * 2          # Llama-3.1-8B weights, bf16, read once per step


def sample_for(B, S, NG):
    z = lambda *s, dt=BF: torch.zeros(*s, dtype=dt)
    return (z(B, H),
            z(B, H, dt=torch.float32), z(B, H, dt=torch.float32),
            z(NG, B, H),
            z(NG, H, NH * HD), z(NG, H, NKV * HD),
            z(NG, H, NKV * HD), z(NG, NH * HD, H),
            z(NG, B, H),
            z(NG, H, M), z(NG, H, M),
            z(NG, M, H),
            z(NG, B, NKV, HD, S), z(NG, B, NKV, S, HD),
            z(B, HD, HD),
            z(B, S), z(B, S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="4,8,16")
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--hints", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    default_hints = (["ForLlmModelComputeBound", "ForLlmModelIOBound",
                      "ForLlmModelComputeIOBalanced", "NoConstraint"]
                     if hasattr(TacticHintConfig, "ForLlmModelIOBound") else
                     ["ForLlmModelDecode", "NoConstraint"])
    hints = [h for h in a.hints.split(",") if h] or default_hints

    dev = dev_str(a.chip)
    rng = np.random.default_rng(0)
    rows = []
    print(f"[decode-ab] SDK={SDK} S={a.S} group={a.group} chip={a.chip} "
          f"hints={hints}", flush=True)
    print(f"{'B':>3} {'hint':28s} {'compile s':>10} {'ms/step':>8} "
          f"{'ms/token':>9} {'GB/s':>7}", flush=True)
    for B in [int(x) for x in a.batch.split(",")]:
        host = sample_for(B, a.S, a.group)
        # random, not zero: a kernel fed all zeros can take a different path
        host = tuple(torch.from_numpy(
            rng.standard_normal(tuple(t.shape)).astype(np.float32) * 0.05
        ).to(t.dtype) for t in host)
        dv = [t.to(dev) for t in host]
        for hn in hints:
            cfg = FCConfig(allow_external_operators=True,
                           allow_unlowered_operators=True,
                           tactic_hint=getattr(TacticHintConfig, hn))
            t0 = time.time()
            try:
                m = CompileModule.from_module(DecodeModel(B, a.S, a.group),
                                              host, compiler_config=cfg)
                comp = time.time() - t0
                m.to(dev)
                o = m(*dv)
                _ = o[0].detach().cpu()
                sync(a.chip)
                ts = []
                for _ in range(a.steps):
                    t1 = time.perf_counter()
                    o = m(*dv)
                    _ = o[0].detach().cpu()
                    sync(a.chip)
                    ts.append((time.perf_counter() - t1) * 1e3)
                ts.sort()
                ms = ts[len(ts) // 2]
                row = dict(sdk=SDK, B=B, S=a.S, hint=hn,
                           compile_s=round(comp, 1), ms_step=round(ms, 2),
                           ms_token=round(ms / B, 2),
                           gbs=round(WBYTES / (ms / 1e3) / 1e9))
                print(f"{B:3d} {hn:28s} {comp:10.0f} {ms:8.2f} {ms/B:9.2f} "
                      f"{row['gbs']:7d}", flush=True)
                del m
            except Exception as e:
                row = dict(sdk=SDK, B=B, S=a.S, hint=hn,
                           error=f"{type(e).__name__}: {str(e)[:160]}")
                print(f"{B:3d} {hn:28s} {'--':>10}  FAIL {type(e).__name__}: "
                      f"{str(e)[:90]}", flush=True)
            rows.append(row)
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"-> {a.out}", flush=True)
    for B in sorted({r["B"] for r in rows}):
        ok = [r for r in rows if r["B"] == B and "ms_step" in r]
        if ok:
            b = min(ok, key=lambda r: r["ms_step"])
            print(f"BEST B={B}: {b['hint']} {b['ms_step']:.2f} ms/step "
                  f"({b['ms_token']:.2f} ms/token)", flush=True)


if __name__ == "__main__":
    main()
