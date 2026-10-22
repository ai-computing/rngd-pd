"""Block admission (RNGD_ADMIT=block, #14): a burst is prefilled back to back,
each stream seated as soon as its own cache is in, and none is staged.

    PYTHONPATH=src python -m pytest -q tests/test_admission.py
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rngd_pd.serving import cluster  # noqa: E402


class FakeEngine:
    def __init__(self, B):
        self.B = B

    def blank_kv(self):
        return object(), object()

    def write_slot_kv(self, Kd, Vd, idx, K, V):
        return Kd, Vd

    def step(self, cur, pos, Kd, Vd):
        time.sleep(0.02)
        return [c + 1 for c in cur], Kd, Vd


class Owner:
    def __init__(self):
        self.staged = 0

    def begin_produce(self, item):
        self.staged += 1
        raise AssertionError("block admission must not stage")


def run(block):
    os.environ["RNGD_ADMIT"] = "block" if block else "stage"
    owner = Owner()
    produced = []

    def produce(item):
        time.sleep(0.005)
        produced.append(time.perf_counter())
        return None, None, item["seed"], 3, 5.0
    sched = cluster._ContinuousScheduler(8, FakeEngine(8), produce, produce_owner=owner)
    firsts, out, ths = {}, {}, []

    def one(i):
        t = []
        out[i] = sched.submit({"tokens": [0, 1, 2], "max_new": 6, "seed": 10 * i,
                               "on_token": lambda tok: t.append(time.perf_counter())})
        firsts[i] = t[0]
    first = threading.Thread(target=one, args=(0,), daemon=True)
    first.start()
    time.sleep(0.03)                         # one stream decoding, then a burst
    for i in range(1, 6):
        th = threading.Thread(target=one, args=(i,), daemon=True)
        th.start(); ths.append(th)
    for th in [first] + ths:
        th.join(10)
    return owner, out, produced, firsts


class BlockAdmission(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("RNGD_ADMIT", None)

    def test_burst_is_not_staged_and_all_finish(self):
        owner, out, produced, firsts = run(block=True)
        self.assertEqual(owner.staged, 0)
        self.assertEqual(len(out), 6)
        for i, o in out.items():
            self.assertEqual(o["tokens"], [10 * i + k for k in range(6)])

    def test_each_stream_starts_when_its_own_cache_is_in(self):
        _owner, _out, produced, firsts = run(block=True)
        # the first token of a burst member goes out before the last prefill
        # of that burst ends, not after all of them
        self.assertLess(min(firsts[i] for i in range(1, 6)), max(produced))


if __name__ == "__main__":
    unittest.main()
