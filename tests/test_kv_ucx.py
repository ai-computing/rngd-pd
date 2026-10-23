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
        self.completed = False

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
        if self.final == "DONE" and not h.completed:
            h.run()
            h.completed = True
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
        self.assertTrue(_wait(lambda: not self.p.pending()))
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
        self.assertEqual(self.d._rx.free_bytes(), (8 << 20) - 1024)
        self.assertFalse(self.fab.agents["decode"].handles[-1].released)
        self.fab.agents["decode"].final = "DONE"
        self.assertTrue(_wait(lambda: self.d._rx.free_bytes() == 8 << 20))

    def test_duplicate_key_keeps_original_bytes_and_reclaims_all_memory(self):
        self.p.push("same", b"original" * 100)
        before = self.p._tx.free_bytes()
        with self.assertRaisesRegex(ValueError, "outstanding"):
            self.p.push("same", b"replacement" * 100)
        self.assertEqual(self.p._tx.free_bytes(), before)
        self.assertEqual(bytes(self.d.pull("same")), b"original" * 100)
        self.assertTrue(_wait(lambda: not self.p.pending()))
        self.assertEqual(self.p._tx.free_bytes(), 8 << 20)

    def test_duplicate_during_publication_does_not_allocate_or_replace(self):
        entered, resume = threading.Event(), threading.Event()
        errors = []
        push = self.p.control.push

        def paused_push(*args):
            entered.set()
            if not resume.wait(2):
                raise TimeoutError("test publication gate")
            return push(*args)

        def publish():
            try:
                self.p.push("same", b"original")
            except Exception as e:
                errors.append(e)

        with mock.patch.object(self.p.control, "push", paused_push):
            thread = threading.Thread(target=publish)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaisesRegex(ValueError, "outstanding"):
                    self.p.push("same", b"replacement")
            finally:
                resume.set()
                thread.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(bytes(self.d.pull("same")), b"original")
        self.assertTrue(_wait(lambda: not self.p.pending()))
        self.assertEqual(self.p._tx.free_bytes(), 8 << 20)

    def test_uncertain_read_retains_memory_until_actual_completion(self):
        for failure in ("transfer", "check_xfer_state", "timeout", "error_state"):
            with self.subTest(failure=failure):
                agent = self.fab.agents["decode"]
                self.d.xfer_timeout_s = 0.01
                agent.final = "ERR" if failure == "error_state" else "PROC"
                key = "uncertain-" + failure
                self.p.push(key, b"late DMA" * 128)
                method = "transfer" if failure == "transfer" else "check_xfer_state"
                original = getattr(agent, method)
                try:
                    if failure in ("timeout", "error_state"):
                        with self.assertRaisesRegex(IOError, "timed out|state ERR"):
                            self.d.pull(key)
                    else:
                        with mock.patch.object(agent, method, side_effect=RuntimeError("SDK failure")):
                            with self.assertRaisesRegex(RuntimeError, "SDK failure"):
                                self.d.pull(key)
                    handle = agent.handles[-1]
                    self.assertFalse(handle.released)
                    self.assertEqual(self.d._rx.free_bytes(), (8 << 20) - 1024)
                    with self.assertRaisesRegex(RuntimeError, "in.flight|quarantin"):
                        self.d.close()
                    self.assertTrue(agent.regions, "close deregistered an in-flight DMA target")
                    # A later allocation must not receive late DMA from the failed pull.
                    off, span = self.d._rx.alloc(1024)
                    self.d._rx.buf[off:off + span] = 0x5A
                    agent.final = "DONE"
                    original_check = agent.check_xfer_state
                    original_check(handle)  # performs the real memmove, even after pull failed
                    np.testing.assert_array_equal(self.d._rx.buf[off:off + span], 0x5A)
                    self.d._rx.free(off, span)
                    self.assertTrue(_wait(lambda: handle.released))
                    self.assertEqual(self.d._rx.free_bytes(), 8 << 20)
                    self.assertTrue(_wait(lambda: key not in self.p.pending()))
                finally:
                    setattr(agent, method, original)
                    agent.final = "DONE"

    def test_close_cannot_deregister_a_running_read(self):
        agent = self.fab.agents["decode"]
        entered, resume = threading.Event(), threading.Event()
        errors = []
        self.p.push("running", b"live DMA")
        transfer = agent.transfer

        def paused_transfer(handle):
            entered.set()
            if not resume.wait(2):
                raise TimeoutError("test transfer gate")
            return transfer(handle)

        def pull():
            try:
                self.assertEqual(bytes(self.d.pull("running")), b"live DMA")
            except Exception as e:
                errors.append(e)

        # Polling sleeps outside the agent lock, allowing close to race a live READ.
        self.d.poll_s = 0.05
        agent.polls = 3
        with mock.patch.object(agent, "transfer", paused_transfer):
            thread = threading.Thread(target=pull)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                resume.set()
                with self.assertRaisesRegex(RuntimeError, "in.flight|active"):
                    self.d.close()
                self.assertTrue(agent.regions)
            finally:
                resume.set()
                thread.join(3)
        self.assertEqual(errors, [])
        self.d.close()
        self.assertEqual(agent.regions, [])

    def test_corrupt_source_fails_crc(self):
        self.p.push("c", b"y" * 4096)
        off = self.p._slots["c"][0]
        self.p._tx.buf[off] ^= 0xFF                            # the source changed after staging
        with self.assertRaises(IOError):
            self.d.pull("c", timeout=1)

    def test_ttl_cannot_recycle_a_source_during_a_delayed_read(self):
        self.p.pool_bytes = 4096
        self.p.ttl_s = 0.005
        self.p.xfer_timeout_s = 0.05
        self.p.push("old", b"a" * 4096)
        entered, complete = threading.Event(), threading.Event()
        outputs, errors = [], []
        agent = self.fab.agents["decode"]
        check = agent.check_xfer_state

        def delayed_check(handle):
            entered.set()
            return check(handle) if complete.is_set() else "PROC"

        def pull():
            try:
                outputs.append(bytes(self.d.pull("old")))
            except Exception as e:
                errors.append(e)

        with mock.patch.object(agent, "check_xfer_state", delayed_check):
            thread = threading.Thread(target=pull)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                try:
                    self.p.push("replacement", b"b" * 4096)
                    replaced = True
                except TimeoutError:
                    replaced = False
            finally:
                complete.set()
                thread.join(3)
        self.assertEqual(errors, [], "late READ observed a recycled source span")
        self.assertEqual(outputs, [b"a" * 4096])
        self.assertFalse(replaced)
        self.assertTrue(_wait(lambda: not self.p.pending()))
        self.p.push("after-completion", b"c" * 4096)
        self.assertEqual(bytes(self.d.pull("after-completion")), b"c" * 4096)
        self.assertTrue(_wait(lambda: not self.p.pending()))
        self.assertEqual(self.p._tx.free_bytes(), 4096)

    def test_lost_publication_ack_retains_advertised_source(self):
        self.p.pool_bytes = 4096
        push = self.p.control.push

        def published_without_ack(*args):
            push(*args)
            raise TimeoutError("publication ACK lost")

        with mock.patch.object(self.p.control, "push", published_without_ack):
            with self.assertRaisesRegex(TimeoutError, "ACK lost"):
                self.p.push("ambiguous", b"a" * 4096)
        try:
            self.assertEqual(self.p.pending(), ["ambiguous"])
            self.assertEqual(self.p._tx.free_bytes(), 0)
            with self.assertRaisesRegex(ValueError, "outstanding"):
                self.p.push("ambiguous", b"b" * 4096)
        finally:
            self.assertEqual(bytes(self.d.pull("ambiguous")), b"a" * 4096)
        self.assertTrue(_wait(lambda: not self.p.pending()))
        self.assertEqual(self.p._tx.free_bytes(), 4096)

    def test_failure_before_descriptor_publication_reclaims_source(self):
        self.p.pool_bytes = 4096
        with mock.patch.object(ucxkv, "pack_desc", side_effect=ValueError("bad descriptor")):
            with self.assertRaisesRegex(ValueError, "bad descriptor"):
                self.p.push("unpublished", b"a" * 4096)
        self.assertEqual(self.p.pending(), [])
        self.assertEqual(self.p._tx.free_bytes(), 4096)
        self.p.push("unpublished", b"a" * 4096)
        self.assertEqual(bytes(self.d.pull("unpublished")), b"a" * 4096)

    def test_close_retains_advertised_sources_until_notification(self):
        self.p.push("advertised", b"a" * 4096)
        try:
            with self.assertRaisesRegex(RuntimeError, "advertised|outstanding"):
                self.p.close()
            self.assertTrue(self.fab.agents["prefill"].regions)
        finally:
            self.assertEqual(bytes(self.d.pull("advertised")), b"a" * 4096)
        self.assertTrue(_wait(lambda: not self.p.pending()))
        self.p.close()
        self.assertEqual(self.fab.agents["prefill"].regions, [])

    def test_close_deregisters(self):
        self.p.push("k", b"a" * 128)
        self.d.pull("k", timeout=1)
        self.assertTrue(_wait(lambda: not self.p.pending()))
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
        from test_connector_metadata import connector_module
        pf, dc = self.pair(kv.NixlUcxTransport)
        with connector_module():
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
        self.fab.agents["decode"].final = "DONE"
        self.assertTrue(_wait(lambda: all(h.released for h in self.fab.agents["decode"].handles)))
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
