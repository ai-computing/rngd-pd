"""Endpoint Picker (EPP) — KV-aware routing for the RNGD gateway (Phase 2).

The gateway's brain: given a request's tokens, pick which worker(s) serve it. Two modes:

  • replica : one worker does prefill+decode. Route to the worker with the LONGEST cached prefix
              (prefix-cache hit → skip recompute); tie-break by least load. This is llm-d's
              KV-cache-aware routing.
  • pd      : disaggregated. Pick a PREFILL worker (least-loaded) to build KV, then a DECODE worker
              (KV-aware among decode-capable) to receive it via the connector. Enables xPyD pools.

Decision is pure over (registry snapshot, KV index) → testable with no NPU. The gateway (server.py)
turns a RouteDecision into an actual prefill→handoff→decode run using rngd_kv_connector.
"""
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import Role
from kv_events import block_hashes


@dataclass
class RouteDecision:
    mode: str                      # "replica" | "pd"
    decode_wid: str                # worker that produces the output tokens (always set)
    prefill_wid: str = ""          # pd: the prefill worker (== decode_wid in replica)
    kv_hit_blocks: int = 0         # cached prefix blocks reused on the decode worker
    reason: str = ""


class EndpointPicker:
    def __init__(self, registry, kv_index, block_size=16):
        self.reg = registry
        self.kv = kv_index
        self.block_size = block_size

    def _least_loaded(self, workers):
        return min(workers, key=lambda w: w.inflight) if workers else None

    def route(self, token_ids, mode="replica") -> RouteDecision:
        hs = block_hashes(token_ids, self.block_size)

        if mode == "replica":
            cand = self.reg.alive(Role.DECODE)            # decode-capable (BOTH included)
            if not cand: raise RuntimeError("no live decode-capable worker")
            elig = {w.wid for w in cand}
            hit_wid, hit_n = self.kv.best_worker(hs, eligible=elig)
            if hit_wid and hit_n > 0:
                return RouteDecision("replica", decode_wid=hit_wid, prefill_wid=hit_wid,
                                     kv_hit_blocks=hit_n, reason=f"kv-hit {hit_n} blk")
            w = self._least_loaded(cand)
            return RouteDecision("replica", decode_wid=w.wid, prefill_wid=w.wid,
                                 kv_hit_blocks=0, reason="least-load (no kv-hit)")

        if mode == "pd":
            pf = self.reg.alive(Role.PREFILL); dc = self.reg.alive(Role.DECODE)
            if not pf or not dc: raise RuntimeError("pd needs both prefill and decode workers")
            # decode: KV-aware (a decode worker may already hold this prefix from a prior turn)
            elig = {w.wid for w in dc}
            hit_wid, hit_n = self.kv.best_worker(hs, eligible=elig)
            dwid = hit_wid if (hit_wid and hit_n > 0) else self._least_loaded(dc).wid
            # prefill: least-loaded prefill worker (skip if decode already has full prefix)
            pwid = self._least_loaded(pf).wid
            reason = f"pd; decode {'kv-hit '+str(hit_n)+'blk' if hit_n else 'least-load'}"
            return RouteDecision("pd", decode_wid=dwid, prefill_wid=pwid,
                                 kv_hit_blocks=hit_n, reason=reason)

        raise ValueError(f"unknown mode: {mode}")


def _selftest():
    from registry import WorkerRegistry
    from kv_events import KVIndex
    reg = WorkerRegistry(); kv = KVIndex()
    reg.register("d0", Role.DECODE, 1); reg.register("d1", Role.DECODE, 2)
    reg.register("p0", Role.PREFILL, 0); reg.register("p1", Role.PREFILL, 3)
    epp = EndpointPicker(reg, kv)

    toks = list(range(64))
    hs = block_hashes(toks)

    # replica, no cache: least-loaded decode
    reg.incr("d0", 5)                                   # d0 busier → expect d1
    d = epp.route(toks, "replica")
    assert d.decode_wid == "d1" and d.kv_hit_blocks == 0, (d.decode_wid, d.kv_hit_blocks)

    # replica, d0 holds the prefix → KV-hit beats load balancing
    kv.block_stored("d0", hs)
    d = epp.route(toks, "replica")
    assert d.decode_wid == "d0" and d.kv_hit_blocks == 4, (d.decode_wid, d.kv_hit_blocks)

    # pd: prefill least-loaded, decode KV-aware (d0 has prefix)
    reg.incr("p0", 2)                                   # p0 busier → expect p1
    d = epp.route(toks, "pd")
    assert d.mode == "pd" and d.prefill_wid == "p1" and d.decode_wid == "d0", (d.prefill_wid, d.decode_wid)
    assert d.kv_hit_blocks == 4

    # pd, cold decode: least-loaded decode
    kv.all_cleared("d0")
    reg.incr("d1", -0)                                  # d1 has 0 inflight, d0 has 5 → expect d1
    d = epp.route(toks, "pd")
    assert d.decode_wid == "d1" and d.kv_hit_blocks == 0, (d.decode_wid, d.kv_hit_blocks)
    print("epp self-test: PASS ✅ (replica kv-hit vs load, pd prefill/decode split, cold fallback)")


if __name__ == "__main__":
    _selftest()
