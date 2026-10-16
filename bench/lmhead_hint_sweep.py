#!/usr/bin/env python3
"""lm_head tactic-hint sweep, run identically on 2026.2.0 and 2026.3.0.

The lm_head graph -- [NG, Vp/NG, H] x [H, B] after an rms -- is where batched
decode used to collapse: one hint is 27x the other and which one wins flips
with B. 2026.2.0 made us pick by hand (ForLlmModelDecode at B<=4, NoConstraint
above). 2026.3.0 renamed the hints along exactly that axis (IOBound /
ComputeIOBalanced / ComputeBound), so this measures whether the new names pick
correctly on their own and whether either backend is faster at the same B.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from rngd_pd.engines.fsdk import open_device  # noqa: E402

open_device(8, 2)

from rngd_pd.engines.fsdk import (SDK, CompileModule, FCConfig,  # noqa: E402
                                  TacticHintConfig, TacticKernelModule,
                                  dev_str, sync)
from rngd_pd.kernels import dsl_batch as LEB  # noqa: E402

H, V, NG = 4096, 128256, 32
BF = torch.bfloat16


class LmHead(torch.nn.Module):
    def __init__(self, B):
        super().__init__()
        self.rms = TacticKernelModule(LEB.rms_dsl(B))

    def forward(self, x, nw, eps, nh, wlm):
        h = self.rms(x, nw, eps, nh)[0]
        return torch.matmul(wlm, h.t())


def sample_for(B):
    Vp = (V + NG - 1) // NG * NG
    return (torch.zeros(B, H, dtype=BF), torch.zeros(B, H, dtype=BF),
            torch.full((B, H), 1e-5 * H, dtype=torch.float32),
            torch.full((B, H), -0.5, dtype=torch.float32),
            torch.zeros(NG, Vp // NG, H, dtype=BF))


def time_one(B, hint_name, chip, iters):
    hint = getattr(TacticHintConfig, hint_name)
    cfg = FCConfig(allow_external_operators=True,
                   allow_unlowered_operators=True, tactic_hint=hint)
    dev = dev_str(chip)
    sample = sample_for(B)
    t0 = time.time()
    m = CompileModule.from_module(LmHead(B), sample, compiler_config=cfg)
    m.to(dev)
    compile_s = time.time() - t0
    dv = tuple(t.to(dev) for t in sample)
    for _ in range(3):
        m(*dv)
    sync(chip)
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        m(*dv)
        sync(chip)
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3, compile_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="4,8,16")
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--hints", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    hints = ([h for h in a.hints.split(",") if h] or
             [h for h in dir(TacticHintConfig) if not h.startswith("_")])
    rows = []
    print(f"[lmhead-hint] SDK={SDK} chip={a.chip} hints={hints}", flush=True)
    for B in [int(x) for x in a.batch.split(",")]:
        for hn in hints:
            try:
                ms, cs = time_one(B, hn, a.chip, a.iters)
                rows.append(dict(sdk=SDK, B=B, hint=hn, ms=ms, compile_s=cs))
                print(f"  B={B:3d} {hn:28s} {ms:8.2f} ms   (compile {cs:5.0f}s)",
                      flush=True)
            except Exception as ex:
                rows.append(dict(sdk=SDK, B=B, hint=hn, ms=None,
                                 err=str(ex)[:200]))
                print(f"  B={B:3d} {hn:28s}     FAIL  {str(ex)[:120]}",
                      flush=True)
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}", flush=True)
    for B in sorted({r["B"] for r in rows}):
        ok = [r for r in rows if r["B"] == B and r.get("ms")]
        if ok:
            best = min(ok, key=lambda r: r["ms"])
            print(f"BEST B={B}: {best['hint']} {best['ms']:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
