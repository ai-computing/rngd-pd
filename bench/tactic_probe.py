"""Does the layer boundary move when the tactic search is guided?

Every FCConfig in this repo passes the same three settings and leaves the rest
at their defaults. Two of those defaults were never questioned:

    tactic_sorting_policy = TacticSortingPolicy.Random
    max_num_unique_shapes = None          # unbounded

The boundary we hit is a memory explosion during tactic search, so a random
walk over an unbounded search space is exactly the thing to suspect. This
compiles the batched decode group at sizes that are known to work (2) and known
to fail (4, 8) under each policy, and reports which combinations survive.

Compile only -- nothing is moved onto a card, so this is safe to run while the
serving pods hold all three.
"""
import argparse
import time

import numpy as np
import torch

from rngd_pd.kernels.decode_batch import DecodeModel, H, M, NH, NKV, HD, BF
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.compiler import (Config as FCConfig, TacticHintConfig,
                                    TacticSortingPolicy)


def sample_for(B, S, NG):
    """The same input tuple decode_batch.main builds, on the host."""
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
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--groups", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--budget", type=int, default=1800,
                    help="seconds to allow one compile before calling it lost")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="run only these variants by name; default runs all")
    a = ap.parse_args()

    # ByEstimation did not move the boundary (group=4 failed under both, 665s
    # and 690s), so the search policy is not what binds. These are the rest of
    # the settings FCConfig can reach and this repo never set -- sweeping them
    # is what lets us say the reachable surface is exhausted rather than
    # untried. The vendor settings we cannot reach at all are
    # tactic_hint=ForLlmModelComputeBound, padding_policy, dma_preference,
    # partial_sum_policy and local_population_threshold.
    base = dict(allow_external_operators=True, allow_unlowered_operators=True,
                tactic_hint=TacticHintConfig.ForLlmModelDecode)
    variants = [
        ("Random(기준)", dict(base, tactic_sorting_policy=TacticSortingPolicy.Random)),
        ("ByEstimation", dict(base, tactic_sorting_policy=TacticSortingPolicy.ByEstimation)),
        ("shapes=1", dict(base, max_num_unique_shapes=1)),
        ("shapes=8", dict(base, max_num_unique_shapes=8)),
        ("all_activation_cast", dict(base, implicit_type_casting_all_activation=True)),
        ("keep_slice_concat", dict(base, reshape_remover_ignores_slice_concat=False)),
    ]
    if a.variants:
        variants = [v for v in variants if v[0] in a.variants]
        assert variants, f"이름이 맞는 변형이 없다: {a.variants}"
    rows = []
    for NG in a.groups:
        for pname, kw in variants:
            cfg = FCConfig(**kw)
            print(f"\n=== group={NG} {pname} B={a.B} S={a.S} ===", flush=True)
            t0 = time.time()
            try:
                CompileModule.from_module(DecodeModel(a.B, a.S, NG),
                                          sample_for(a.B, a.S, NG),
                                          compiler_config=cfg)
                secs = time.time() - t0
                rows.append((NG, pname, "성공", f"{secs:.0f}s"))
                print(f"  성공 {secs:.0f}s", flush=True)
            except Exception as e:
                secs = time.time() - t0
                rows.append((NG, pname, "실패", f"{type(e).__name__}: {str(e)[:60]}"))
                print(f"  실패 {secs:.0f}s {type(e).__name__}: {str(e)[:160]}", flush=True)

    print(f"\n{'group':>6} {'policy':<14} {'결과':<6} 비고")
    for NG, p, r, note in rows:
        print(f"{NG:>6} {p:<14} {r:<6} {note}")


if __name__ == "__main__":
    main()
