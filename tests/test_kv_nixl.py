"""`nixl` KV transport against a fake NIXL agent: routes, refusals, descriptors, fence, registry.

    PYTHONPATH=src python -m pytest -q tests/test_kv_nixl.py

The hardware path (RNGD -> ATOM DMA-BUF and the host staging legs) is checked on the node by
bench/nixl_kv_check.py.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rm_core import KVLayout, MemoryTier, TransportError, registry     # noqa: E402

from rngd_pd import kv_nixl                                            # noqa: E402
from rngd_pd.kv_nixl import NixlBinding, NixlTransport, dmabuf_region, host_region, rngd_region  # noqa: E402

NL, NKV, HD, S = 2, 8, 128, 16
LAYOUT = KVLayout(num_layers=NL, num_kv_heads=NKV, head_dim=HD, dtype="bf16", order="vendor:rngd-pd")
NBYTES = S * LAYOUT.bytes_per_token()          # 131072
RNGD = MemoryTier("rngd1", "device-dram", 1 << 30, "node-a/furiosa:1")
RNGD2 = MemoryTier("rngd2", "device-dram", 1 << 30, "node-a/furiosa:2")
ATOM = MemoryTier("atom0", "device-hbm", 1 << 30, "node-a/rebellions:0")
ATOM_B = MemoryTier("atom0", "device-hbm", 1 << 30, "node-b/rebellions:0")
HOST = MemoryTier("host", "host-dram", 1 << 30)
DEV = "/dev/rngd/npu1ch0"


class FakeHandle:
    def __init__(self, agent, op, local, remote, notif):
        self.agent, self.op, self.local, self.remote, self.notif = agent, op, local, remote, notif
        self.polls, self.released = 0, False

    def release(self):
        self.released = True


class FakeAgent:
    """Records what the transport asks of NIXL; completes after `polls` checks with `final`."""
    name = "fake"

    def __init__(self, polls=2, final="DONE", refuse=False):
        self.polls, self.final, self.refuse = polls, final, refuse
        self.registered, self.deregistered, self.handles = [], [], []

    def register_memory(self, descs, mem, backends):
        assert backends == [kv_nixl.BACKEND]
        self.registered.append((mem, descs))
        return ("reg", len(self.registered))

    def deregister_memory(self, reg, backends):
        self.deregistered.append(reg)

    def get_xfer_descs(self, descs, mem):
        return (mem, list(descs))

    def initialize_xfer(self, op, local, remote, name, notif, backends):
        if self.refuse:
            raise RuntimeError("NIXL_ERR_NOT_SUPPORTED")
        h = FakeHandle(self, op, local, remote, notif)
        self.handles.append(h)
        return h

    def transfer(self, h):
        return "PROC"

    def check_xfer_state(self, h):
        h.polls += 1
        return self.final if h.polls >= self.polls else "PROC"


def _binding(src, dst):
    return NixlBinding(src=tuple(src), dst=tuple(dst), context=S)


class NixlTransportTest(unittest.TestCase):
    def setUp(self):
        self.agent = FakeAgent()
        self.t = NixlTransport(agent=self.agent)

    def move(self, src_tier, dst_tier, binding, req="r1"):
        self.t.bind(req, binding)
        plan = self.t.plan(req, src_tier, dst_tier, LAYOUT, LAYOUT, (0, 5))
        return plan, self.t.wait(self.t.submit(plan), 5.0)

    def test_rngd_to_atom_direct(self):
        plan, res = self.move(RNGD, ATOM, _binding([rngd_region(0x10000, NBYTES, DEV)],
                                                   [dmabuf_region(7, NBYTES, base=0x400000, offset=4096)]))
        self.assertEqual(plan.detail["route"], "direct")
        self.assertEqual(plan.nbytes, NBYTES)                 # the S-wide cache, not 5 tokens
        self.assertTrue(res.ok and res.nbytes == NBYTES)
        h, = self.agent.handles
        self.assertEqual(h.op, "WRITE")
        self.assertEqual(h.local, ("VRAM", [(0x10000, NBYTES, 0)]))
        self.assertEqual(h.remote, ("VRAM", [(0x400000, NBYTES, 1)]))
        self.assertTrue(h.released)
        metas = [d[0][3] for _m, d in self.agent.registered]
        self.assertEqual(metas, [f"rngd:{DEV}", "dmabuf:7:4096"])
        self.assertEqual([m.name for m in self.t.measurements], ["kv_transfer_ms", "kv_transfer_gbps"])

    def test_staging_legs(self):
        # RNGD -> host is a READ into local host memory; host -> RNGD a WRITE from it.
        _p, res = self.move(RNGD, HOST, _binding([rngd_region(0x10000, NBYTES, DEV)], [host_region(0x7f0000000000, NBYTES)]))
        self.assertTrue(res.ok)
        h = self.agent.handles[-1]
        self.assertEqual((h.op, h.local[0], h.remote[0]), ("READ", "DRAM", "VRAM"))
        _p, res = self.move(HOST, RNGD2, _binding([host_region(0x7f0000000000, NBYTES)],
                                                  [rngd_region(0x20000, NBYTES, "/dev/rngd/npu2ch0")]), req="r2")
        h = self.agent.handles[-1]
        self.assertEqual((h.op, h.local[0], h.remote[0]), ("WRITE", "DRAM", "VRAM"))
        # Same host buffer registered once.
        self.assertEqual(sum(1 for m, _d in self.agent.registered if m == "DRAM"), 1)

    def test_descriptors_split_at_region_boundaries(self):
        half = NBYTES // 2
        self.t.bind("r1", _binding([rngd_region(0x10000, NBYTES, DEV)],
                                   [dmabuf_region(7, half, base=0x400000), dmabuf_region(8, half, base=0x800000)]))
        plan = self.t.plan("r1", RNGD, ATOM, LAYOUT, LAYOUT, (0, S))
        self.assertTrue(self.t.wait(self.t.submit(plan), 1).ok)
        h = self.agent.handles[-1]
        self.assertEqual(h.local[1], [(0x10000, half, 0), (0x10000 + half, half, 0)])
        self.assertEqual(h.remote[1], [(0x400000, half, 1), (0x800000, half, 2)])

    def test_refused_routes(self):
        b = _binding([rngd_region(0x10000, NBYTES, DEV)], [rngd_region(0x20000, NBYTES, "/dev/rngd/npu2ch0")])
        for src, dst in ((RNGD, RNGD2), (ATOM, RNGD), (HOST, HOST), (HOST, ATOM), (RNGD, ATOM_B)):
            self.t.bind("r", b)
            with self.assertRaises(TransportError):
                self.t.plan("r", src, dst, LAYOUT, LAYOUT, (0, 4))
        with self.assertRaisesRegex(TransportError, "rngd -> host and host -> rngd"):
            self.t.bind("r", b)
            self.t.plan("r", RNGD, RNGD2, LAYOUT, LAYOUT, (0, 4))
        self.assertFalse(self.t.supports(RNGD, RNGD2))
        self.assertTrue(self.t.supports(RNGD, ATOM) and self.t.supports(RNGD, HOST) and self.t.supports(HOST, RNGD))

    def test_plan_checks(self):
        good = _binding([rngd_region(0x10000, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)])
        with self.assertRaisesRegex(TransportError, "bind"):
            self.t.plan("nobody", RNGD, ATOM, LAYOUT, LAYOUT, (0, 4))
        cases = [
            (good, LAYOUT, KVLayout(NL, NKV, HD, "fp16"), (0, 4), "layouts differ"),
            (good, KVLayout(NL, NKV, HD, "fp16"), KVLayout(NL, NKV, HD, "fp16"), (0, 4), "bf16"),
            (good, LAYOUT, LAYOUT, (2, 4), "prefix"),
            (good, LAYOUT, LAYOUT, (0, S + 1), "context"),
            (_binding([rngd_region(0x10000, NBYTES // 2, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]),
             LAYOUT, LAYOUT, (0, 4), "hold"),
            (_binding([rngd_region(0x10040, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]),
             LAYOUT, LAYOUT, (0, 4), "aligned"),
            (_binding([rngd_region(0x10000, NBYTES, "/dev/rngd/npu1ch3")], [dmabuf_region(7, NBYTES, base=0x400000)]),
             LAYOUT, LAYOUT, (0, 4), "PE0"),
            (_binding([host_region(0x10000, NBYTES)], [dmabuf_region(7, NBYTES, base=0x400000)]),
             LAYOUT, LAYOUT, (0, 4), "bound region is host"),
        ]
        for binding, sl, dl, toks, msg in cases:
            self.t.bind("r", binding)
            with self.assertRaisesRegex(TransportError, msg):
                self.t.plan("r", RNGD, ATOM, sl, dl, toks)
        # The binding was consumed by the rejected plan.
        with self.assertRaisesRegex(TransportError, "bind"):
            self.t.plan("r", RNGD, ATOM, LAYOUT, LAYOUT, (0, 4))

    def test_fence_failure_and_single_wait(self):
        t = NixlTransport(agent=FakeAgent(final="ERR"))
        t.bind("r1", _binding([rngd_region(0x10000, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]))
        h = t.submit(t.plan("r1", RNGD, ATOM, LAYOUT, LAYOUT, (0, 4)))
        res = t.wait(h, 1)
        self.assertFalse(res.ok)
        self.assertIn("ERR", res.error)
        self.assertEqual(len(t.measurements), 0)
        with self.assertRaises(TransportError):
            t.wait(h, 1)

    def test_timeout_releases_and_fails(self):
        t = NixlTransport(agent=FakeAgent(polls=10 ** 9))
        t.bind("r1", _binding([rngd_region(0x10000, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]))
        res = t.wait(t.submit(t.plan("r1", RNGD, ATOM, LAYOUT, LAYOUT, (0, 4))), 0.01)
        self.assertFalse(res.ok)
        self.assertIn("timed out", res.error)
        self.assertTrue(t._agent.handles[0].released)

    def test_refused_by_backend_is_transport_error(self):
        t = NixlTransport(agent=FakeAgent(refuse=True))
        t.bind("r1", _binding([rngd_region(0x10000, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]))
        plan = t.plan("r1", RNGD, ATOM, LAYOUT, LAYOUT, (0, 4))
        with self.assertRaisesRegex(TransportError, "NOT_SUPPORTED"):
            t.submit(plan)
        with self.assertRaises(TransportError):                 # a plan is submitted at most once
            t.submit(plan)

    def test_close_deregisters(self):
        self.move(RNGD, ATOM, _binding([rngd_region(0x10000, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]))
        self.t.close()
        self.assertEqual(len(self.agent.deregistered), 2)

    def test_no_agent_is_a_transport_error(self):
        t = NixlTransport()
        t.bind("r1", _binding([rngd_region(0x10000, NBYTES, DEV)], [dmabuf_region(7, NBYTES, base=0x400000)]))
        plan = t.plan("r1", RNGD, ATOM, LAYOUT, LAYOUT, (0, 4))   # planning needs no device
        try:
            import nixl  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(TransportError, "NIXL"):
                t.submit(plan)

    def test_registry_and_conformance_skip(self):
        from rm_core.conformance import NoHardware
        from rngd_pd.connector import transport as kv
        self.assertEqual(kv._OWN.get("nixl"), "rngd_pd.kv_nixl:NixlTransport")
        os.environ.pop("RNGD_PD_NIXL_FIXTURE", None)
        with self.assertRaises(NoHardware):
            NixlTransport().conformance_case()
        names = registry.available("airm.kv_transports")
        if "nixl" in names:                                      # installed with `pip install -e .`
            self.assertEqual(names["nixl"], "rngd_pd.kv_nixl:NixlTransport")


if __name__ == "__main__":
    unittest.main()
