#!/usr/bin/env python3
"""Why does the batched decode kernel collapse between B=4 and B=8?

Standalone, the step goes 19.1 ms at B=4 to 72.3 ms at B=8 and the weight
bandwidth with it, 729 GB/s down to 193. That is not the serving path, not the
KV handoff and not the layer group -- group=2 is just as bad. What is left is
the compiled graph, so this walks the settings that shape it.

The existing tactic probe only asks whether a config compiles, which is the
wrong question here: B=8 compiles fine and then runs slow. This one compiles
and runs, so a variant is judged on the step time it actually produces.

Weights are random rather than the model's: the kernel's speed does not depend
on their values, and loading the real checkpoint for every variant would cost
more than the compiles. Correctness is checked elsewhere -- what is measured
here is time.

    python3 bench/batch_tactic_probe.py --B 8 --S 256 --group 8 --chip 1
"""
import argparse
import itertools
import json
import time

import numpy as np
import torch
import furiosa.torch                                          # noqa: F401
from rngd_pd.engines.fsdk import open_device as _open_device
from rngd_pd.engines.fsdk import CompileModule
from furiosa.torch.compiler import (Config as FCConfig, TacticHintConfig,
                                    TacticSortingPolicy)

from rngd_pd.kernels.decode_batch import DecodeModel, H, M, NH, NKV, HD, BF


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


def variants(hints, shapes, sorts):
    for h, s, p in itertools.product(hints, shapes, sorts):
        cfg = dict(allow_external_operators=True, allow_unlowered_operators=True,
                   tactic_hint=getattr(TacticHintConfig, h),
                   tactic_sorting_policy=getattr(TacticSortingPolicy, p))
        if s is not None:
            cfg["max_num_unique_shapes"] = s
        yield (h, s, p), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--chip", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--hints", nargs="*",
                    default=["ForLlmModelDecode", "ForLlmModelPrefill",
                             "Default", "NoConstraint"])
    ap.add_argument("--shapes", nargs="*", default=["8", "none", "1"])
    ap.add_argument("--sorts", nargs="*", default=["Random"])
    ap.add_argument("--out", default="bench/batch_tactic_probe.jsonl")
    a = ap.parse_args()

    _open_device(8, 2)
    dev = f"rngd:{a.chip}"
    B, S, NG = a.B, a.S, a.group
    shapes = [None if v == "none" else int(v) for v in a.shapes]

    rng = np.random.default_rng(0)
    host = sample_for(B, S, NG)
    # random, not zero: a kernel fed all zeros can take a different path
    host = tuple(torch.from_numpy(
        rng.standard_normal(tuple(t.shape)).astype(np.float32) * 0.05).to(t.dtype)
        for t in host)
    dev_args = [t.to(dev) for t in host]
    wbytes = 8.03e9 * 2

    fh = open(a.out, "w")
    print(f"B={B} S={S} group={NG} chip={a.chip}", flush=True)
    print(f"{'hint':28s} {'shapes':>7} {'sort':>12} {'compile s':>10} "
          f"{'ms/step':>8} {'ms/token':>9} {'GB/s':>7}", flush=True)
    for (h, sh, p), cfg in variants(a.hints, shapes, a.sorts):
        t0 = time.time()
        try:
            m = CompileModule.from_module(DecodeModel(B, S, NG), host,
                                          compiler_config=FCConfig(**cfg))
            comp = time.time() - t0
            m.to(dev)
        except Exception as e:
            print(f"{h:28s} {str(sh):>7} {p:>12} {'--':>10}  compile 실패: "
                  f"{type(e).__name__}", flush=True)
            fh.write(json.dumps({"hint": h, "shapes": sh, "sort": p,
                                 "error": type(e).__name__}) + "\n"); fh.flush()
            continue
        out = m(*dev_args)                                    # warm
        _ = out[0].detach().cpu()                             # force the queue
        ts = []
        for _ in range(a.steps):
            t1 = time.perf_counter()
            o = m(*dev_args)
            _ = o[0].detach().cpu()                           # dispatches are async
            ts.append((time.perf_counter() - t1) * 1e3)
        ts.sort()
        ms = ts[len(ts) // 2]
        row = {"hint": h, "shapes": sh, "sort": p, "compile_s": round(comp, 1),
               "ms_step": round(ms, 2), "ms_token": round(ms / B, 2),
               "gbs": round(wbytes / (ms / 1e3) / 1e9)}
        print(f"{h:28s} {str(sh):>7} {p:>12} {comp:10.0f} {ms:8.2f} "
              f"{ms/B:9.2f} {row['gbs']:7d}", flush=True)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        del m
    fh.close()
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
