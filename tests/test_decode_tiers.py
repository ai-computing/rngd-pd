"""TieredDecode under the continuous scheduler, with fake engines.

Every stream must get the same tokens whether one wide engine or several
widths serve it, including when widths change while streams are live.

    PYTHONPATH=src python -m pytest -q tests/test_decode_tiers.py
"""
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rngd_pd.engines.tiers import TieredDecode  # noqa: E402
from rngd_pd.serving import cluster  # noqa: E402

S, NGROUPS = 64, 2


class FakeEngine:
    """Per-lane cache [layers, B, S]; the next token depends on the lane's whole cache."""

    def __init__(self, B):
        self.B, self.steps = B, 0

    def blank_kv(self):
        return ([np.zeros((2, self.B, S)) for _ in range(NGROUPS)],
                [np.zeros((2, self.B, S)) for _ in range(NGROUPS)])

    def write_slot_kv(self, Kd, Vd, idx, K, V):
        for g in range(NGROUPS):
            Kd[g][:, idx] = K[g]
            Vd[g][:, idx] = V[g]
        return Kd, Vd

    def step(self, toks, pos, Kd, Vd):
        assert len(toks) == self.B
        self.steps += 1
        time.sleep(0.003)                  # long enough for the batch to fill
        out = []
        for b in range(self.B):
            for g in range(NGROUPS):
                Kd[g][:, b, pos[b] % S] += toks[b] + 1
                Vd[g][:, b, pos[b] % S] += 2 * toks[b] + 1
            out.append(int(sum(Kd[g][:, b].sum() + Vd[g][:, b].sum() for g in range(NGROUPS))) % 997)
        return out, Kd, Vd


def produce(item):
    seed = item["seed"]
    K = [np.full((2, S), float(seed + g)) for g in range(NGROUPS)]
    V = [np.full((2, S), float(2 * seed + g)) for g in range(NGROUPS)]
    return K, V, seed % 50, len(item["tokens"]), 1.0


def serve(engine, B, plan):
    """plan: (delay steps before submit, seed, max_new); returns {seed: tokens}."""
    sched = cluster._ContinuousScheduler(B, engine, produce)
    out, ths = {}, []

    def one(seed, n):
        out[seed] = sched.submit({"tokens": [1, 2, 3], "max_new": n, "seed": seed})["tokens"]
    for i, (seed, n) in enumerate(plan):
        th = threading.Thread(target=one, args=(seed, n), daemon=True)
        th.start(); ths.append(th)
        if i == 9:
            ths[0].join(10)                # a second wave after the batch shrank
    for th in ths:
        th.join(20)
    return out


class Tiers(unittest.TestCase):
    plan = [(s, 5 + (s * 7) % 23) for s in range(1, 14)]

    def test_same_tokens_as_one_wide_engine(self):
        wide = serve(FakeEngine(8), 8, self.plan)
        tiers = [FakeEngine(8), FakeEngine(2), FakeEngine(4)]
        td = TieredDecode(tiers)
        tiered = serve(td, 8, self.plan)
        self.assertEqual(len(wide), len(self.plan))
        self.assertEqual(tiered, wide)
        self.assertGreater(tiers[1].steps, 0)          # the narrow widths were used
        self.assertGreater(td.switches, 0)

    def test_single_stream_stays_narrow(self):
        tiers = [FakeEngine(8), FakeEngine(2)]
        serve(TieredDecode(tiers), 8, [(3, 10)])
        self.assertEqual(tiers[0].steps, 0)
        self.assertEqual(tiers[1].steps, 9)


if __name__ == "__main__":
    unittest.main()
