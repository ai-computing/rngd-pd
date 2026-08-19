"""Worker registry for the RNGD gateway (Phase 2).

Tracks the pool of RNGD workers behind the gateway: each worker's role (prefill / decode / both),
physical chip, transport endpoint, live load, and heartbeat. The EPP (epp.py) queries this to pick
targets. Mirrors the role split llm-d's Inference Gateway maintains over vLLM pods, scoped to RNGD.
"""
import time, threading
from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"          # replica mode: one worker does prefill+decode


@dataclass
class Worker:
    wid: str
    role: Role
    chip: int                          # physical RNGD chip index (rngd:{chip})
    endpoint: str = ""                 # transport addr (shm root / host:port) the connector uses
    inflight: int = 0                  # current in-flight requests (load signal)
    last_beat: float = field(default_factory=time.time)

    def can_prefill(self): return self.role in (Role.PREFILL, Role.BOTH)
    def can_decode(self):  return self.role in (Role.DECODE, Role.BOTH)


class WorkerRegistry:
    def __init__(self, stale_after=15.0):
        self._w = {}
        self._lock = threading.Lock()
        self.stale_after = stale_after

    def register(self, wid, role, chip, endpoint=""):
        with self._lock:
            self._w[wid] = Worker(wid, Role(role) if isinstance(role, str) else role, chip, endpoint)
        return self._w[wid]

    def heartbeat(self, wid, inflight=None):
        with self._lock:
            w = self._w.get(wid)
            if w:
                w.last_beat = time.time()
                if inflight is not None: w.inflight = inflight

    def incr(self, wid, delta):
        with self._lock:
            w = self._w.get(wid)
            if w: w.inflight = max(0, w.inflight + delta)

    def _live(self, w, now): return (now - w.last_beat) <= self.stale_after

    def alive(self, role=None):
        """Live workers, optionally filtered to those that can serve `role`."""
        now = time.time()
        with self._lock:
            ws = [w for w in self._w.values() if self._live(w, now)]
        if role == Role.PREFILL: ws = [w for w in ws if w.can_prefill()]
        elif role == Role.DECODE: ws = [w for w in ws if w.can_decode()]
        return ws

    def get(self, wid):
        with self._lock: return self._w.get(wid)


def _selftest():
    r = WorkerRegistry()
    r.register("w0", Role.PREFILL, chip=0, endpoint="/dev/shm/rngd_kv_w0")
    r.register("w1", Role.DECODE, chip=1, endpoint="127.0.0.1:5570")
    r.register("w2", Role.BOTH, chip=2)
    assert {w.wid for w in r.alive(Role.PREFILL)} == {"w0", "w2"}, "prefill filter"
    assert {w.wid for w in r.alive(Role.DECODE)} == {"w1", "w2"}, "decode filter"
    r.incr("w1", 3); assert r.get("w1").inflight == 3
    r.incr("w1", -1); assert r.get("w1").inflight == 2
    # staleness
    r.stale_after = -1
    assert r.alive() == [], "all stale"
    print("registry self-test: PASS ✅ (role filter, load counter, staleness)")


if __name__ == "__main__":
    _selftest()
