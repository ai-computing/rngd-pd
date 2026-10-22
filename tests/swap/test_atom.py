"""ATOM session logic without a card: rebel is replaced by a fake."""
import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import numpy as np

from rngd_pd.swap.atom import AtomSession, HostPool, Tables
from rngd_pd.swap.variants import Variant


class FakeRuntime:
    """Logits favour token (sum of x) % V; k/v rows are filled with the lane index."""

    def __init__(self, nl, nkv, hd, V=7):
        self.nl, self.nkv, self.hd, self.V = nl, nkv, hd, V
        self.shapes = []

    def run(self, x, K, V, cos, sin, mask, oh):
        B = x.shape[0]
        self.shapes.append((B, mask.shape[1]))
        logits = np.zeros((B, self.V), np.float32)
        for b in range(B):
            logits[b, int(abs(x[b].sum())) % self.V] = 1
        k = np.tile(np.arange(B, dtype=np.float32)[None, :, None, None], (self.nl, 1, self.nkv, self.hd))
        return logits, k, k.copy()


class FakeCompiled:
    def __init__(self, path):
        self.path = path

    def create_runtime(self, device=0, tensor_type="np"):
        return FakeRuntime(2, 2, 4)


def fake_rebel():
    m = types.ModuleType("rebel")
    m.RBLNCompiledModel = FakeCompiled
    return m


def tables():
    t = Tables.__new__(Tables)
    t.cfg = SimpleNamespace(n_kv=2, head_dim=4)
    t.inv = np.array([1.0, 0.5], np.float32)
    t.embed = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    return t


class AtomSessionTest(unittest.TestCase):
    def setUp(self):
        self._old = sys.modules.get("rebel")
        sys.modules["rebel"] = fake_rebel()
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "s.rbln")
        open(self.path, "wb").close()
        with open(self.path + ".json", "w") as f:
            json.dump({"layers": 2, "variants": ["B1-S8", "B4-S8"]}, f)

    def tearDown(self):
        if self._old is None:
            sys.modules.pop("rebel", None)
        else:
            sys.modules["rebel"] = self._old

    def test_swap_selects_by_shape_and_refuses_outside(self):
        s = AtomSession(self.path, tables())
        self.assertEqual([v.name for v in s.variants], ["B1-S8", "B4-S8"])
        s.swap(Variant(4, 8, 0))
        K, V = s.blank_kv()
        self.assertEqual(K.shape, (2, 4, 2, 4, 8))
        s.step([1, 2, 3, 4], [0, 1, 2, 3], K, V)
        self.assertEqual(s.rt.shapes[-1], (4, 8))
        self.assertEqual(float(K[0, 2, 0, 0, 2]), 2.0)     # lane 2's row written at its position
        self.assertEqual(float(K[0, 2, 0, 0, 3]), 0.0)
        with self.assertRaises(KeyError):
            s.swap(Variant(2, 16, 0))

    def test_step_inputs(self):
        x, cos, sin, mask, oh = tables().step_inputs([1, 2], [0, 3], 5)
        self.assertEqual(mask[1].tolist(), [0, 0, 0, 0, -1e4])
        self.assertEqual(oh[1].tolist(), [0, 0, 0, 1, 0])
        np.testing.assert_allclose(cos[1], np.cos([3.0, 1.5]), rtol=1e-6)

    def test_host_pool_switch(self):
        s = AtomSession(self.path, tables())
        p2 = os.path.join(self.d, "s2.rbln")
        open(p2, "wb").close()
        with open(p2 + ".json", "w") as f:
            json.dump({"layers": 2, "variants": ["B1-S8", "B4-S8", "B2-S16"]}, f)
        pool = HostPool()
        pool.prepare(p2)
        s2, t = pool.switch(s, tables())
        self.assertIsNone(s.rt)                            # the old session is closed
        self.assertTrue(s2.timings["parsed_before"])
        self.assertTrue(s2.has(Variant(2, 16, 0)))
        self.assertIn("parse_in_background_s", t)


if __name__ == "__main__":
    unittest.main()
