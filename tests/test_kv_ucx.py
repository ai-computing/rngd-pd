"""The `nixl-ucx` KV transport: arena, descriptors, READ + free notification, connector handoff, adapter.

    PYTHONPATH=src python -m pytest -q tests/test_kv_ucx.py

FakeFabric stands in for NIXL: agents in one process, a READ is a memmove between their registered
regions (so the bytes really move between two buffers), notifications are delivered to the remote
agent. RealNixlTests run the same paths through the installed NIXL with the UCX backend (UCX_TLS=tcp
unless set), in one process and across two processes; they skip where `nixl` is not importable.
"""
import ctypes
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rm_core.conformance import check_kv_transport                         # noqa: E402

from rngd_pd import airm                                                    # noqa: E402
from rngd_pd.connector import tcpkv, ucxkv                                  # noqa: E402
from rngd_pd.connector import transport as kv                               # noqa: E402


class FakeFabric:
    """Process-local stand-in for NIXL + UCX: agents by name, regions, READ = memmove, notifications."""

    def __init__(self):
        self.agents = {}

    def agent(self, name, **kw):
        a = FakeAgent(self, name, **kw)
        self.agents[name] = a
        return a


class FakeHandle:
    def __init__(self, run, polls):
        self.run, self.polls, self.released = run, polls, False

    def release(self):
        self.released = True


class FakeAgent:
    def __init__(self, fabric, name, polls=1, final="DONE"):
        self.fabric, self.name, self.polls, self.final = fabric, name, polls, final
        self.regions, self.remotes, self.notifs, self.handles = [], set(), {}, []
        self.lock = threading.Lock()

    def register_memory(self, descs, mem, backends):
        assert mem == "DRAM" and backends == [ucxkv.BACKEND]
        self.regions += [(a, n) for a, n, _dev, _meta in descs]
        return ("reg", tuple(descs))

    def deregister_memory(self, reg, backends):
        for a, n, _d, _m in reg[1]:
            self.regions.remove((a, n))

    def get_agent_metadata(self):
        return b"md:" + self.name.encode()

    def add_remote_agent(self, md):
        name = bytes(md)[3:].decode()
        self.remotes.add(name)
        return name

    def remove_remote_agent(self, name):
        self.remotes.discard(name)

    def get_xfer_descs(self, descs, mem):
        return list(descs)

    def _inside(self, regions, a, n):
        return any(ra <= a and a + n <= ra + rn for ra, rn in regions)

    def initialize_xfer(self, op, local, remote, name, notif, backends):
        assert op == "READ" and (name in self.remotes or name == self.name)
        peer = self.fabric.agents[name]
        for (la, n, _), (ra, rn, _) in zip(local, remote):
            assert n == rn and self._inside(self.regions, la, n) and self._inside(peer.regions, ra, n), \
                "descriptor outside a registered region"

        def run():
            for (la, n, _), (ra, _rn, _) in zip(local, remote):
                ctypes.memmove(la, ra, n)
            with peer.lock:
                peer.notifs.setdefault(self.name, []).append(bytes(notif))
        h = FakeHandle(run, self.polls)
        self.handles.append(h)
        return h

    def transfer(self, h):
        return "PROC"

    def check_xfer_state(self, h):
        h.polls -= 1
        if h.polls > 0:
            return "PROC"
        if self.final == "DONE":
            h.run()
        return self.final

    def get_new_notifs(self):
        with self.lock:
            out, self.notifs = self.notifs, {}
        return out


def _server():
    srv = tcpkv.TcpKVServer(0, "127.0.0.1")
    srv.start()
    return srv, srv.wait_ready()


def _wait(cond, t=5.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class ArenaTests(unittest.TestCase):
    def test_alloc_free_coalesce(self):
        a = ucxkv.Arena(1 << 16)
        self.assertEqual(a.address % 4096, 0)
        x = a.alloc(1000); y = a.alloc(5000); z = a.alloc(64)
        self.assertEqual(x, (0, 1024))
        self.assertEqual(y[0], 1024)
        a.free(*y); a.free(*x); a.free(*z)
        self.assertEqual(a._free, [(0, 1 << 16)])

    def test_alloc_waits_then_times_out(self):
        a = ucxkv.Arena(4096)
        x = a.alloc(4096)
        threading.Timer(0.1, a.free, x).start()
        self.assertEqual(a.alloc(4096, timeout=2), (0, 4096))   # freed by the timer
        with self.assertRaises(TimeoutError):
            a.alloc(64, timeout=0.05)
        with self.assertRaises(ValueError):
            a.alloc(8192)

    def test_descriptor_round_trip(self):
        d = ucxkv.pack_desc("agent-a", b"\x00md\xff", 0x7f0000001000, 12345, 0xdeadbeef)
        self.assertEqual(ucxkv.unpack_desc(d), ("agent-a", b"\x00md\xff", 0x7f0000001000, 12345, 0xdeadbeef))
        with self.assertRaises(IOError):
            ucxkv.unpack_desc(d + b"x")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.srv, port = _server()
        self.fab = FakeFabric()
        self.p = ucxkv.UcxKVStore(tcpkv.TcpKVClient("127.0.0.1", port), agent=self.fab.agent("prefill"),
                                  pool_bytes=8 << 20, name="prefill")
        self.d = ucxkv.UcxKVStore(tcpkv.TcpKVClient("127.0.0.1", port), agent=self.fab.agent("decode"),
                                  pool_bytes=8 << 20, name="decode")

    def tearDown(self):
        self.p.close(); self.d.close()
        self.srv.stop()

    def test_round_trip_frees_the_source_slot(self):
        blob = np.random.default_rng(0).bytes(3 << 20)
        self.p.push("r/layer.0/k", blob)
        self.assertEqual(self.p.pending(), ["r/layer.0/k"])
        self.assertEqual(bytes(self.d.pull("r/layer.0/k", timeout=1)), blob)
        self.assertTrue(_wait(lambda: not self.p.pending()), "READ notification did not free the slot")
        self.assertEqual(self.p._tx.free_bytes(), 8 << 20)
        self.assertEqual(self.d._rx.free_bytes(), 8 << 20)
        self.assertEqual(self.fab.agents["decode"].remotes, {"prefill"})
        with self.assertRaises(TimeoutError):
            self.d.pull("r/layer.0/k", timeout=0.05)          # handed out once

    def test_bytes_in_flight_are_not_overwritten(self):
        """Many blobs staged before any read: each reads back its own bytes (no slot reuse in flight)."""
        blobs = {f"q/{i}": os.urandom(100_000 + i) for i in range(40)}
        for k, b in blobs.items():
            self.p.push(k, b)
        for k, b in blobs.items():
            self.assertEqual(bytes(self.d.pull(k, timeout=1)), b)

    def test_concurrent_requests(self):
        blobs = {f"q{i}/layer.{j}/k": os.urandom(200_000) for i in range(8) for j in range(4)}
        errs = []

        def worker(keys):
            try:
                for k in keys:
                    self.p.push(k, blobs[k])
                for k in keys:
                    if bytes(self.d.pull(k, timeout=5)) != blobs[k]:
                        errs.append(k)
            except Exception as e:                            # noqa: BLE001 - reported below
                errs.append(repr(e))
        keys = sorted(blobs)
        ts = [threading.Thread(target=worker, args=(keys[i::8],)) for i in range(8)]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual(errs, [])
        self.assertTrue(_wait(lambda: not self.p.pending()))

    def test_full_arena_waits_for_reads(self):
        """A producer ahead of the reader blocks for room instead of overwriting staged KV."""
        self.p.xfer_timeout_s = 5
        blobs = [os.urandom(3 << 20) for _ in range(4)]       # 12 MB through an 8 MB arena
        t = threading.Thread(target=lambda: [self.p.push(f"b{i}", b) for i, b in enumerate(blobs)])
        t.start()
        got = [bytes(self.d.pull(f"b{i}", timeout=5)) for i in range(4)]
        t.join(5)
        self.assertEqual(got, blobs)

    def test_failed_read_raises(self):
        self.fab.agents["decode"].final = "ERR"
        self.p.push("bad", b"x" * 1000)
        with self.assertRaises(IOError):
            self.d.pull("bad", timeout=1)
        self.assertEqual(self.d._rx.free_bytes(), 8 << 20)    # the reader's slot is back

    def test_corrupt_source_fails_crc(self):
        self.p.push("c", b"y" * 4096)
        off = self.p._slots["c"][0]
        self.p._tx.buf[off] ^= 0xFF                            # the source changed after staging
        with self.assertRaises(IOError):
            self.d.pull("c", timeout=1)

    def test_ttl_frees_unread_slots(self):
        self.p.ttl_s = 0.0
        self.p.push("old", b"z" * 4096)
        self.assertTrue(_wait(lambda: self.p.stats["expired"] == 1))
        self.assertEqual(self.p._tx.free_bytes(), 8 << 20)

    def test_close_deregisters(self):
        self.p.push("k", b"a" * 128)
        self.d.pull("k", timeout=1)
        self.p.close(); self.d.close()
        self.assertEqual(self.fab.agents["prefill"].regions, [])
        self.assertEqual(self.fab.agents["decode"].regions, [])


class _Env:
    def setUp(self):
        self.srv, self.port = _server()
        self.wire = kv.WIRE
        kv.WIRE = "bf16"
        kv._FENCED.clear(); kv._CLASSES.clear(); kv._SHARED.clear()
        self.env = mock.patch.dict(os.environ, {"AIRM_KV_TCP_ADDR": f"127.0.0.1:{self.port}"})
        self.env.start()
        os.environ.pop("AIRM_KV_TRANSPORT", None)
        self.fab = FakeFabric()

    def tearDown(self):
        self.env.stop()
        kv.WIRE = self.wire
        for t in list(kv._SHARED.values()):
            t.close()
        kv._FENCED.clear(); kv._CLASSES.clear(); kv._SHARED.clear()
        self.srv.stop()

    def pair(self, cls=airm.NixlUcxTransport):
        return (cls(agent=self.fab.agent("prefill"), pool_bytes=8 << 20),
                cls(agent=self.fab.agent("decode"), pool_bytes=8 << 20))


class SelectionTests(_Env, unittest.TestCase):
    def test_by_name_is_one_shared_instance(self):
        os.environ["AIRM_KV_TRANSPORT"] = "nixl-ucx"
        with mock.patch.object(ucxkv, "_default_agent", lambda name: self.fab.agent(name)):
            a, b = kv.select_transport(), kv.select_transport()
        self.assertIsInstance(a, airm.NixlUcxTransport)
        self.assertIs(a, b)
        self.assertEqual(a.remote, ("127.0.0.1", self.port))
        self.assertEqual(a.handshake_metadata()["backend"], "UCX")

    def test_without_rm_core(self):
        with mock.patch.dict(sys.modules, {"rm_core": None}):
            t = kv.select_transport("nixl-ucx")
        self.assertIsInstance(t, kv.NixlUcxTransport)
        self.assertNotIsInstance(t, airm.NixlUcxTransport)

    def test_needs_an_address(self):
        os.environ.pop("AIRM_KV_TCP_ADDR")
        with self.assertRaises(ValueError):
            kv.NixlUcxTransport(agent=self.fab.agent("x")).push("k", b"x")

    def test_no_nixl_raises_never_falls_back(self):
        def boom(name):
            raise ImportError("No module named 'nixl'")
        with mock.patch.object(ucxkv, "_default_agent", boom):
            t = kv.NixlUcxTransport()
            with self.assertRaises(RuntimeError):
                t.push("k", b"x")

    def test_connector_handoff_bit_exact(self):
        """prefill save -> NIXL READ -> decode load through the real connector, bf16 wire as in serving."""
        from test_airm import _handoff, _kv
        pf, dc = self.pair(kv.NixlUcxTransport)
        dk, dv = _handoff(pf, dc)
        ck, cv = _kv(0)
        ref = lambda a: (a.view(np.uint32) >> 16 << 16).view(np.float32)   # noqa: E731 - bf16 wire
        np.testing.assert_array_equal(dk, ref(np.stack(ck)))
        np.testing.assert_array_equal(dv, ref(np.stack(cv)))
        pf.close(); dc.close()


class AdapterTests(_Env, unittest.TestCase):
    def test_conformance(self):
        os.environ.pop("AIRM_KV_TCP_ADDR")
        t = airm.NixlUcxTransport(agent=self.fab.agent("solo"), pool_bytes=8 << 20)
        # one process is both ends: the agent reads from itself (UCX loopback)
        check_kv_transport(t)
        self.assertEqual(t.measurements[-1].tags["kv_transport"], "nixl-ucx")
        t.close()

    def test_payload_round_trip_and_failure(self):
        pf, dc = self.pair()
        src, dst, layout = pf.conformance_case()
        kv_in = np.random.default_rng(1).standard_normal((2, 2, 4, 2, 8)).astype(np.float32)
        plan = pf.plan("q", src, dst, layout, layout, (0, 4))
        self.assertEqual(plan.detail["backend"], "UCX")
        h = pf.submit(plan, kv=kv_in)
        # decode's instance waits on the handle prefill gave out (same key namespace in one test)
        dc._pending[h.token] = pf._pending.pop(h.token)
        res = dc.wait(h, timeout_s=5)
        self.assertTrue(res.ok, res.error)
        got = dc.take(plan).reshape(kv_in.shape)
        np.testing.assert_array_equal(got.view(np.uint32), kv_in.view(np.uint32) >> 16 << 16)
        self.fab.agents["decode"].final = "ERR"
        h = pf.submit(plan, kv=kv_in)
        dc._pending[h.token] = pf._pending.pop(h.token)
        res = dc.wait(h, timeout_s=5)
        self.assertFalse(res.ok)
        self.assertIn("NIXL READ", res.error)
        pf.close(); dc.close()


def _nixl_available():
    try:
        import importlib.util
        return importlib.util.find_spec("nixl") is not None
    except Exception:
        return False


_CHILD_HEAD = """
import hashlib, os, sys, time
sys.path.insert(0, {src!r})
from rngd_pd.connector import tcpkv, ucxkv
def store():
    return ucxkv.UcxKVStore(tcpkv.TcpKVClient("127.0.0.1", {port}), pool_bytes=64 << 20)
def blob(i):
    return hashlib.sha256(str(i).encode()).digest() * (1 << 15)      # 1 MiB, distinct per i
def drained(s, t=30):
    end = time.time() + t
    while s.pending() and time.time() < end:
        time.sleep(0.01)
    return not s.pending()
"""


@unittest.skipUnless(_nixl_available(), "nixl (with the UCX plugin) is not importable here")
class RealNixlTests(unittest.TestCase):
    """The installed NIXL + UCX, UCX_TLS=tcp unless the environment says otherwise.

    Every real agent lives in a child process: with the nixl_cu12 wheel, a process that loaded the
    system rdma-core first (vLLM's KV connector imports do) segfaults in libnl's exit handler after a
    UCX agent existed -- at exit only, after the work is done (see README, limits). The children keep
    that out of the test runner's exit code.
    """

    def setUp(self):
        self.srv, self.port = _server()
        self.env = dict(os.environ, UCX_TLS=os.environ.get("UCX_TLS", "tcp"))

    def tearDown(self):
        self.srv.stop()

    def child(self, body):
        code = _CHILD_HEAD.format(src=os.path.join(ROOT, "src"), port=self.port) + textwrap.dedent(body)
        return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=self.env)

    def test_two_agents_one_process(self):
        c = self.child("""
            p, d = store(), store()
            for i in range(16):
                p.push("r/%d" % i, blob(i))
            ok = all(bytes(d.pull("r/%d" % i, timeout=10)) == blob(i) for i in range(16))
            print("exact", ok, "drained", drained(p), flush=True)
            d.close(); p.close()
        """)
        out, _ = c.communicate(timeout=120)
        self.assertIn("exact True drained True", out)

    def test_two_processes(self):
        """prefill in one child, decode in another: the cross-process path the pods take."""
        pf = self.child("""
            s = store()
            for i in range(8):
                s.push("x/%d" % i, blob(i))
            print("drained", drained(s), flush=True)
            s.close()
        """)
        dc = self.child("""
            s = store()
            print("exact", all(bytes(s.pull("x/%d" % i, timeout=60)) == blob(i) for i in range(8)), flush=True)
            s.close()
        """)
        try:
            self.assertIn("exact True", dc.communicate(timeout=120)[0])
            self.assertIn("drained True", pf.communicate(timeout=120)[0])
        finally:
            for c in (pf, dc):
                if c.poll() is None:
                    c.kill()


if __name__ == "__main__":
    unittest.main()
