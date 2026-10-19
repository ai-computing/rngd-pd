"""The continuous decode scheduler under a backlog (rngd-pd#4), with a fake engine.

A missing KV must fail only its own request, a slow load must not stop the
streams already decoding, and nothing popped from the queue may be left hanging.

    PYTHONPATH=src python -m pytest -q tests/test_continuous_backlog.py
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rngd_pd.serving import cluster  # noqa: E402


class FakeEngine:
    """Lane writes and steps that only count; every step emits token 7."""

    def __init__(self, B):
        self.B, self.steps = B, 0

    def blank_kv(self):
        return object(), object()

    def write_slot_kv(self, Kd, Vd, idx, K, V):
        return Kd, Vd

    def step(self, cur, pos, Kd, Vd):
        self.steps += 1
        time.sleep(0.002)
        return [7] * self.B, Kd, Vd


def run_all(sched, items, timeout=10):
    out, errs = {}, {}

    def one(it):
        try:
            out[it["req_id"]] = sched.submit(it)
        except Exception as e:  # noqa: BLE001
            errs[it["req_id"]] = e
    ts = [threading.Thread(target=one, args=(it,), daemon=True) for it in items]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout)
    hung = [t for t in ts if t.is_alive()]
    return out, errs, hung


class Backlog(unittest.TestCase):
    def test_missing_kv_fails_only_its_request(self):
        def produce(item):
            if item["req_id"] == "gone":
                raise TimeoutError("pull timeout: gone/layer.0/k")
            return None, None, 1, 0, 1.0
        sched = cluster._ContinuousScheduler(2, FakeEngine(2), produce, produce_on_device=False)
        items = [{"req_id": r, "max_new": 4} for r in ("a", "gone", "b", "c", "d")]
        out, errs, hung = run_all(sched, items)
        self.assertEqual(hung, [])
        self.assertEqual(set(errs), {"gone"})
        self.assertEqual(set(out), {"a", "b", "c", "d"})
        self.assertTrue(all(len(o["tokens"]) == 4 for o in out.values()))

    def test_slow_load_does_not_stop_running_streams(self):
        release = threading.Event()

        def produce(item):
            if item["req_id"] == "slow":
                release.wait(5)
            return None, None, 1, 0, 1.0
        eng = FakeEngine(2)
        sched = cluster._ContinuousScheduler(2, eng, produce, produce_on_device=False)
        res = {}
        t_slow = threading.Thread(target=lambda: res.setdefault(
            "slow", sched.submit({"req_id": "slow", "max_new": 2})), daemon=True)
        t_slow.start()
        time.sleep(0.05)
        t0 = time.perf_counter()
        fast = sched.submit({"req_id": "fast", "max_new": 50})
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(fast["tokens"]), 50)
        self.assertLess(elapsed, 2.0)          # did not wait for the slow load
        self.assertNotIn("slow", res)
        release.set()
        t_slow.join(5)
        self.assertEqual(len(res["slow"]["tokens"]), 2)

    def test_more_requests_than_lanes_all_finish(self):
        sched = cluster._ContinuousScheduler(
            2, FakeEngine(2), lambda it: (None, None, 1, 0, 1.0), produce_on_device=False)
        items = [{"req_id": f"r{i}", "max_new": 3} for i in range(20)]
        out, errs, hung = run_all(sched, items)
        self.assertEqual((len(out), errs, hung), (20, {}, []))

    def test_on_device_producer_still_used_under_lock(self):
        seen = []

        def produce(item):
            seen.append(cluster.DEVICE.locked())
            return None, None, 1, 0, 1.0
        sched = cluster._ContinuousScheduler(2, FakeEngine(2), produce)
        out, errs, hung = run_all(sched, [{"req_id": "x", "max_new": 2}])
        self.assertEqual((set(out), errs, hung), ({"x"}, {}, []))
        self.assertEqual(seen, [True])


class Prefetch(unittest.TestCase):
    def test_full_map_skips_instead_of_evicting(self):
        w = cluster.DecodeWorkerBase()
        w._prefetch = {}
        w._prefetch_lock = threading.Lock()
        consumed = []
        gate = threading.Event()

        def job(req_id):
            gate.wait(5)
            consumed.append(req_id)
            return None, None, 0.0, None
        w._prefetch_job = job
        w._prefetch_cap = 2
        self.assertEqual(w.prefetch("a")["prefetch"], "started")
        self.assertEqual(w.prefetch("b")["prefetch"], "started")
        self.assertEqual(w.prefetch("c")["prefetch"], "skipped (full)")
        self.assertEqual(set(w._prefetch), {"a", "b"})   # nothing dropped
        gate.set()
        for r in ("a", "b"):
            w._load_kv(r)
        self.assertEqual(sorted(consumed), ["a", "b"])

    def test_stale_entries_are_reclaimed(self):
        w = cluster.DecodeWorkerBase()
        w._prefetch = {}
        w._prefetch_lock = threading.Lock()
        w._prefetch_job = lambda r: (None, None, 0.0, None)
        w._prefetch_cap, w._prefetch_ttl_s = 1, 0.0
        w.prefetch("old")
        time.sleep(0.01)
        self.assertEqual(w.prefetch("new")["prefetch"], "started")
        self.assertEqual(set(w._prefetch), {"new"})


if __name__ == "__main__":
    unittest.main()
