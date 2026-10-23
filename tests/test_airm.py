"""ai-rm v0.1 adapter: the KV path is chosen by name, and "shm" is the path the workers always used.

    PYTHONPATH=src python -m pytest -q tests/test_airm.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rm_core import TransportError, Workload, registry                     # noqa: E402
from rm_core.conformance import check_kv_transport                         # noqa: E402
from rm_core.reference import LoopbackTransport                            # noqa: E402

from rngd_pd import airm                                                    # noqa: E402
from rngd_pd.connector import transport as kv                               # noqa: E402
from connector_support import connector_module

NL, NKV, S, HD = 4, 8, 16, 128


def _kv(seed):
    rng = np.random.default_rng(seed)
    return ([rng.standard_normal((NKV, S, HD)).astype(np.float32) for _ in range(NL)],
            [rng.standard_normal((NKV, S, HD)).astype(np.float32) for _ in range(NL)])


def _handoff(prefill_transport, decode_transport, req_id="r1"):
    """prefill save -> transport -> decode load through the real connector, bf16 wire as in serving."""
    with connector_module() as module:
        return _handoff_with(module, prefill_transport, decode_transport, req_id)


def _handoff_with(module, prefill_transport, decode_transport, req_id):
    RngdKVConnector, KVConnectorRole = module.RngdKVConnector, module.KVConnectorRole
    ck, cv = _kv(0)
    pf = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id, transport=prefill_transport, n_layers=NL)
    for i in range(NL):
        pf.save_kv_layer(f"layer.{i}", (ck[i], cv[i]))
    dk = [np.zeros((NKV, S, HD), np.float32) for _ in range(NL)]
    dv = [np.zeros((NKV, S, HD), np.float32) for _ in range(NL)]
    dc = RngdKVConnector(KVConnectorRole.WORKER, req_id=req_id, transport=decode_transport, n_layers=NL)
    dc.register_kv_caches((dk, dv))
    dc.bind_connector_metadata(dc.build_connector_meta(None))
    dc.start_load_kv()
    for i in range(NL):
        dc.wait_for_layer_load(f"layer.{i}")
    return np.stack(dk), np.stack(dv)


class _Env:
    """A temporary shm root and bf16 wire, restored afterwards."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/dev/shm" if os.access("/dev/shm", os.W_OK) else None)
        self.wire = kv.WIRE
        kv.WIRE = "bf16"
        kv._FENCED.clear(); kv._CLASSES.clear(); kv._SHARED.clear()
        self.env = mock.patch.dict(os.environ, {"PD_SHM_ROOT": self.tmp.name})
        self.env.start()
        os.environ.pop("AIRM_KV_TRANSPORT", None)

    def tearDown(self):
        self.env.stop()
        kv.WIRE = self.wire
        kv._FENCED.clear(); kv._CLASSES.clear(); kv._SHARED.clear()
        self.tmp.cleanup()


class SelectionTests(_Env, unittest.TestCase):
    def test_shm_is_registered_and_default(self):
        self.assertEqual(registry.available("airm.kv_transports").get("shm"), "rngd_pd.airm:ShmTransport")
        t = kv.select_transport()
        self.assertIsInstance(t, airm.ShmTransport)
        self.assertIsInstance(t, kv.ShmTransport)          # the connector's own Transport
        self.assertEqual(t.root, self.tmp.name)

    def test_env_shm_same_output_as_before(self):
        """AIRM_KV_TRANSPORT=shm gives bit-for-bit the KV the hard-coded ShmTransport gave."""
        before = _handoff(kv.ShmTransport(self.tmp.name + "/legacy"), kv.ShmTransport(self.tmp.name + "/legacy"))
        os.environ["AIRM_KV_TRANSPORT"] = "shm"
        after = _handoff(None, None, req_id="r2")           # connector picks the path itself
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        self.assertTrue(np.any(after[0]))

    def test_old_and_new_shm_interoperate(self):
        """A pod still on the old code and one on the new hand KV across the same files."""
        old, new = kv.ShmTransport(self.tmp.name), kv.select_transport("shm")
        old.push("req/layer.0/k", b"abc")
        self.assertEqual(new.pull("req/layer.0/k", timeout=1), b"abc")
        new.push("req/layer.0/v", b"xyz")
        self.assertEqual(old.pull("req/layer.0/v", timeout=1), b"xyz")

    def test_config_alone_selects_another_name(self):
        os.environ["AIRM_KV_TRANSPORT"] = "loopback"        # rm-core's reference transport
        with connector_module() as module:
            conn = module.RngdKVConnector(module.KVConnectorRole.WORKER, req_id="r", n_layers=NL)
        self.assertIsInstance(conn.t, kv.FencedTransport)
        self.assertIsInstance(conn.t.impl, LoopbackTransport)
        dk, dv = _handoff(None, None, req_id="r3")
        ck, cv = _kv(0)
        ref = lambda a: (a.view(np.uint32) >> 16 << 16).view(np.float32)   # bf16 wire
        np.testing.assert_array_equal(dk, ref(np.stack(ck)))
        np.testing.assert_array_equal(dv, ref(np.stack(cv)))

    def test_installed_entry_point_selected_by_name(self):
        """A transport another package registers (as hetero-npu-p2p registers p2p) needs no code here."""
        with tempfile.TemporaryDirectory() as site:
            dist = os.path.join(site, "fake_p2p-0.0.dist-info")
            os.makedirs(dist)
            with open(os.path.join(dist, "METADATA"), "w") as f:
                f.write("Metadata-Version: 2.1\nName: fake-p2p\nVersion: 0.0\n")
            with open(os.path.join(dist, "entry_points.txt"), "w") as f:
                f.write("[airm.kv_transports]\np2p = rm_core.reference:LoopbackTransport\n")
            sys.path.insert(0, site)
            try:
                os.environ["AIRM_KV_TRANSPORT"] = "p2p"
                t = kv.select_transport()
                self.assertIsInstance(t, kv.FencedTransport)
                self.assertEqual(t.name, "loopback")
            finally:
                sys.path.remove(site)

    def test_unknown_name_fails_loud(self):
        os.environ["AIRM_KV_TRANSPORT"] = "no-such-path"
        with self.assertRaises(KeyError):
            kv.select_transport()

    def test_shm_without_rm_core(self):
        """A pod with only src/ on the path still gets the same /dev/shm path; other names fail."""
        with mock.patch.dict(sys.modules, {"rm_core": None}):
            t = kv.select_transport("shm")
            self.assertIsInstance(t, kv.ShmTransport)
            self.assertEqual(t.root, self.tmp.name)
            with self.assertRaises(KeyError):
                kv.select_transport("p2p")

    def test_class_resolved_once(self):
        kv.select_transport("shm")
        with mock.patch("rm_core.registry.load", side_effect=AssertionError("rescanned")):
            self.assertIsInstance(kv.select_transport("shm"), airm.ShmTransport)

    def test_fenced_duplicate_push_and_close(self):
        t = kv.select_transport("loopback")
        t.push("k", b"ab")
        with self.assertRaises(ValueError):
            t.push("k", b"cd")
        self.assertEqual(t.pull("k", timeout=1), b"ab")
        t.close()
        self.assertIsNot(kv.select_transport("loopback"), t)

    def test_fenced_failure_raises(self):
        t = kv.FencedTransport(LoopbackTransport(fail_request_ids=["req/layer.0/k"]))
        t.push("req/layer.0/k", b"abcd")
        with self.assertRaises(TransportError):
            t.pull("req/layer.0/k", timeout=1)


@unittest.skipUnless(hasattr(os, "getuid"), "POSIX shm conformance requires os.getuid")
class ShmTransportTests(_Env, unittest.TestCase):
    def test_conformance(self):
        t = airm.ShmTransport()
        check_kv_transport(t)
        self.assertEqual(os.listdir(self.tmp.name), [])     # conformance used its own directory
        self.assertTrue(t.measurements)
        self.assertEqual(t.measurements[-1].tags["kv_transport"], "shm")

    def test_payload_round_trip(self):
        t = airm.ShmTransport()
        src, dst, layout = t.conformance_case()
        kv_in = np.random.default_rng(1).standard_normal((2, 2, 4, 2, 8)).astype(np.float32)
        plan = t.plan("q", src, dst, layout, layout, (0, 4))
        res = t.wait(t.submit(plan, kv=kv_in), timeout_s=5)
        self.assertTrue(res.ok)
        got = t.take(plan).reshape(kv_in.shape)
        np.testing.assert_array_equal(got.view(np.uint32), kv_in.view(np.uint32) >> 16 << 16)
        t.close()

    def test_rejects_unsupported(self):
        t = airm.ShmTransport()
        src, dst, layout = t.conformance_case()
        from rm_core import KVLayout, MemoryTier
        dev = MemoryTier("rngd0", "device-hbm", 1 << 30, "n/furiosa:0")
        for args in [(dev, dst, layout, layout, (0, 4)),
                     (src, dst, layout, KVLayout(2, 2, 16, "bf16"), (0, 4)),
                     (src, dst, KVLayout(2, 2, 8, "fp8"), KVLayout(2, 2, 8, "fp8"), (0, 4))]:
            with self.assertRaises(TransportError):
                t.plan("bad", *args)
        t.close()


class RunnerAndMetricsTests(unittest.TestCase):
    def test_runner_registered(self):
        self.assertEqual(registry.available("airm.workloads").get("rngd-pd"), "rngd_pd.airm:run")

    def test_runner_passes_transport_by_name(self):
        argv, env = airm.run(Workload("decode", "m", batch_hint=8), kv_transport="loopback", dry_run=True)
        self.assertEqual(env["AIRM_KV_TRANSPORT"], "loopback")
        self.assertEqual(argv[argv.index("--role") + 1], "decode")
        self.assertEqual(argv[argv.index("--B") + 1], "8")
        argv, env = airm.run(Workload("serve", "m"), metrics_port=9400, dry_run=True)
        self.assertEqual(argv[argv.index("--role") + 1], "gateway")
        self.assertEqual(env["AIRM_METRICS_PORT"], "9400")
        with self.assertRaises(KeyError):
            airm.run(Workload("prefill", "m"), kv_transport="no-such-path", dry_run=True)
        _, env = airm.run(Workload("prefill", "m"), env={"MODEL": "x", "AIRM_KV_TRANSPORT": "y"},
                          dry_run=True)
        self.assertEqual((env["MODEL"], env["AIRM_KV_TRANSPORT"]), ("m", "shm"))

    def test_constructing_does_not_create_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "never")
            airm.ShmTransport(root=root)
            self.assertFalse(os.path.exists(root))

    def test_metric_names(self):
        recs = [{"ttft_ms": 100.0 + i, "tpot_ms": 20.0, "out_tokens": 10, "kv_save_ms": 3.0,
                 "kv_load_ms": 2.0, "t0": 0.0, "t1": 2.0} for i in range(10)]
        s = airm.summarize(recs)
        self.assertEqual(set(s), {"ttft_p90_ms", "tpot_p90_ms", "tok_s", "kv_transfer_ms"})
        self.assertEqual(s["tok_s"], 50.0)
        self.assertEqual(s["kv_transfer_ms"], 5.0)
        text = airm.prometheus_text(airm.measurements(s, kv_transport="shm"))
        self.assertIn('airm_kv_transfer_ms{component="rngd-pd",kv_transport="shm"} 5', text)
        self.assertIn('airm_tok_s{component="rngd-pd"} 50', text)

    def test_window_observes_gateway(self):
        w = airm.Window()
        resp = {"usage": {"completion_tokens": 4},
                "pd": {"mode": "disaggregated-kv-transfer", "ttft_ms": 50.0, "tpot_ms": 10.0,
                       "kv_save_ms": 1.5, "kv_load_ms": 0.5}}
        w.observe(lambda body: resp)({})
        self.assertIn("airm_kv_transfer_ms", w.text())
        self.assertIn("airm_ttft_p90_ms", w.text())

    def test_window_tpot_is_what_the_client_sees(self):
        """A request that waited 400 ms for a decode lane: the engine step says 10 ms, the client 140."""
        w = airm.Window()
        resp = {"usage": {"completion_tokens": 4},
                "pd": {"mode": "disaggregated-kv-transfer", "ttft_ms": 50.0, "tpot_ms": 10.0}}
        w.record(0.0, 0.47, resp)                    # wall 470 ms = 50 TTFT + 400 wait + ~2 steps
        self.assertIn("airm_tpot_p90_ms{component=\"rngd-pd\"} 140", w.text())
        w2 = airm.Window()
        w2.record(0.0, 0.1, {"usage": {"completion_tokens": 1}, "pd": {"ttft_ms": 50.0, "tpot_ms": 10.0}})
        self.assertIn("airm_tpot_p90_ms{component=\"rngd-pd\"} 10", w2.text())   # one token: engine value


if __name__ == "__main__":
    unittest.main()
