"""The cross-node `tcp` KV transport: relay server, pooled client, rm-core adapter, connector handoff.

    PYTHONPATH=src python -m pytest -q tests/test_kv_tcp.py
"""
import os
import socket
import struct
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rm_core.conformance import check_kv_transport                         # noqa: E402

from rngd_pd import airm                                                    # noqa: E402
from rngd_pd.connector import tcpkv                                         # noqa: E402
from rngd_pd.connector import transport as kv                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _server():
    srv = tcpkv.TcpKVServer(0, "127.0.0.1")
    srv.start()
    return srv, srv.wait_ready()


class ServerClientTests(unittest.TestCase):
    def setUp(self):
        self.srv, self.port = _server()
        self.c = tcpkv.TcpKVClient("127.0.0.1", self.port)

    def tearDown(self):
        self.c.close()
        self.srv.stop()

    def test_round_trip_is_byte_exact_and_consumed(self):
        blob = np.random.default_rng(0).bytes(3 << 20)
        self.c.push("r/layer.0/k", blob)
        self.assertEqual(bytes(self.c.take("r/layer.0/k", timeout=1)), blob)
        with self.assertRaises(TimeoutError):
            self.c.take("r/layer.0/k", timeout=0.05)          # handed out once

    def test_take_waits_for_a_later_push(self):
        got = {}
        t = threading.Thread(target=lambda: got.update(v=bytes(self.c.take("late", timeout=5))))
        t.start()
        time.sleep(0.2)
        tcpkv.TcpKVClient("127.0.0.1", self.port).push("late", b"xyz")
        t.join(5)
        self.assertEqual(got.get("v"), b"xyz")

    def test_connections_are_reused(self):
        for i in range(20):
            self.c.push(f"k{i}", b"a" * 1000)
            self.c.take(f"k{i}")
        self.assertEqual(self.c._idle.qsize(), 1)

    def test_concurrent_clients(self):
        blobs = {f"q{i}/layer.{j}/k": os.urandom(50_000) for i in range(8) for j in range(4)}
        errs = []

        def worker(keys):
            try:
                for k in keys:
                    self.c.push(k, blobs[k])
                for k in keys:
                    if bytes(self.c.take(k, timeout=5)) != blobs[k]:
                        errs.append(k)
            except Exception as e:                            # noqa: BLE001 - reported below
                errs.append(repr(e))
        keys = sorted(blobs)
        ts = [threading.Thread(target=worker, args=(keys[i::8],)) for i in range(8)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(errs, [])
        st = self.c.stats()
        self.assertEqual((st["pushed"], st["taken"], st["keys"], st["crc_errors"]), (32, 32, 0, 0))

    def test_corrupt_push_is_rejected(self):
        s = socket.create_connection(("127.0.0.1", self.port))
        data = b"payload"
        s.sendall(b"P" + struct.pack("<i", 1) + b"k" + struct.pack("<qI", len(data), 12345) + data)
        self.assertEqual(s.recv(2), b"ER")
        s.close()
        self.assertEqual(self.c.stats()["crc_errors"], 1)
        with self.assertRaises(TimeoutError):
            self.c.take("k", timeout=0.05)

    def test_expired_blobs_are_dropped(self):
        self.srv.ttl_s = 0.0
        self.c.push("old", b"x")
        time.sleep(1.2)                                       # the accept loop expires every 0.5 s
        self.assertEqual(self.c.stats()["expired"], 1)

    def test_parse_addr(self):
        self.assertEqual(tcpkv.parse_addr("relay.ns.svc:6000"), ("relay.ns.svc", 6000))
        self.assertEqual(tcpkv.parse_addr("relay"), ("relay", 5577))


class _Env:
    def setUp(self):
        self.srv, self.port = _server()
        self.wire = kv.WIRE
        kv.WIRE = "bf16"
        kv._FENCED.clear(); kv._CLASSES.clear(); kv._SHARED.clear()
        self.env = mock.patch.dict(os.environ, {"AIRM_KV_TCP_ADDR": f"127.0.0.1:{self.port}"})
        self.env.start()
        os.environ.pop("AIRM_KV_TRANSPORT", None)

    def tearDown(self):
        self.env.stop()
        kv.WIRE = self.wire
        for t in list(kv._SHARED.values()):
            t.close()
        kv._FENCED.clear(); kv._CLASSES.clear(); kv._SHARED.clear()
        self.srv.stop()


class SelectionTests(_Env, unittest.TestCase):
    def test_tcp_by_name_is_one_shared_instance(self):
        os.environ["AIRM_KV_TRANSPORT"] = "tcp"
        a, b = kv.select_transport(), kv.select_transport()
        self.assertIsInstance(a, airm.TcpTransport)
        self.assertIs(a, b)                                   # one connection pool per process
        self.assertEqual(a.remote, ("127.0.0.1", self.port))

    def test_tcp_without_rm_core(self):
        with mock.patch.dict(sys.modules, {"rm_core": None}):
            t = kv.select_transport("tcp")
            self.assertIsInstance(t, kv.TcpTransport)

    def test_needs_an_address(self):
        os.environ.pop("AIRM_KV_TCP_ADDR")
        t = kv.TcpTransport()
        with self.assertRaises(ValueError):
            t.push("k", b"x")

    def test_connector_handoff_bit_exact(self):
        """prefill save -> relay -> decode load through the real connector, bf16 wire as in serving."""
        from test_airm import _handoff, _kv
        os.environ["AIRM_KV_TRANSPORT"] = "tcp"
        dk, dv = _handoff(kv.select_transport(), kv.select_transport())
        ck, cv = _kv(0)
        ref = lambda a: (a.view(np.uint32) >> 16 << 16).view(np.float32)   # noqa: E731 - bf16 wire
        np.testing.assert_array_equal(dk, ref(np.stack(ck)))
        np.testing.assert_array_equal(dv, ref(np.stack(cv)))


class AdapterTests(_Env, unittest.TestCase):
    def test_conformance(self):
        os.environ.pop("AIRM_KV_TCP_ADDR")
        t = airm.TcpTransport()                               # conformance starts its own server
        check_kv_transport(t)
        self.assertEqual(t.measurements[-1].tags["kv_transport"], "tcp")
        t.close()

    def test_payload_round_trip(self):
        t = airm.TcpTransport()
        src, dst, layout = t.conformance_case()
        kv_in = np.random.default_rng(1).standard_normal((2, 2, 4, 2, 8)).astype(np.float32)
        plan = t.plan("q", src, dst, layout, layout, (0, 4))
        self.assertEqual(plan.detail["server"], f"127.0.0.1:{self.port}")
        res = t.wait(t.submit(plan, kv=kv_in), timeout_s=5)
        self.assertTrue(res.ok)
        got = t.take(plan).reshape(kv_in.shape)
        np.testing.assert_array_equal(got.view(np.uint32), kv_in.view(np.uint32) >> 16 << 16)
        t.close()


class ChartTests(unittest.TestCase):
    def test_relay_copy_in_sync(self):
        """The chart ships tcpkv.py for the relay pod; it must be the module the workers run."""
        with open(os.path.join(ROOT, "src/rngd_pd/connector/tcpkv.py"), "rb") as a, \
                open(os.path.join(ROOT, "deploy/helm/rngd-pd/files/tcpkv.py"), "rb") as b:
            self.assertEqual(a.read(), b.read(), "run: cp src/rngd_pd/connector/tcpkv.py deploy/helm/rngd-pd/files/")


if __name__ == "__main__":
    unittest.main()
