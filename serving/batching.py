"""Continuous batching scheduler (Phase 5).

vLLM-style continuous batching for the decode loop: many sequences decode concurrently; each step
forms a batch of the sequences' current tokens, runs ONE batched forward, appends each output, and
retires finished sequences — while NEW requests join between steps (no waiting for a static batch to
drain). This is what turns the measured batch throughput (llama_engine_batch: B=128 → 28134 tok/s)
into a real server.

Runtime-agnostic: `step_fn(list_of_active_tokens) -> list_of_next_tokens`. Tested here with a mock
step_fn (NPU-free) so the scheduling invariants are verified independent of the NPU runtime.
"""
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Seq:
    sid: str
    tokens: list                      # generated token ids so far
    max_new: int
    prompt_len: int = 0
    done: bool = False
    def last(self): return self.tokens[-1]


class ContinuousBatcher:
    def __init__(self, step_fn, max_batch=128, eos=None):
        self.step_fn = step_fn        # (list[int] current tokens) -> list[int] next tokens
        self.max_batch = max_batch
        self.eos = eos
        self.waiting = deque()
        self.active = []
        self.finished = []

    def submit(self, sid, first_token, max_new, prompt_len=0):
        self.waiting.append(Seq(sid, [first_token], max_new, prompt_len))

    def _admit(self):
        while self.waiting and len(self.active) < self.max_batch:
            self.active.append(self.waiting.popleft())

    def step(self):
        """One decode step over the whole active batch. Returns number of sequences advanced."""
        self._admit()
        if not self.active: return 0
        cur = [s.last() for s in self.active]
        nxt = self.step_fn(cur)                       # ONE batched forward for all active seqs
        assert len(nxt) == len(self.active), "step_fn must return one token per active seq"
        still = []
        for s, t in zip(self.active, nxt):
            s.tokens.append(t)
            fin = (len(s.tokens) >= s.max_new) or (self.eos is not None and t == self.eos)
            if fin:
                s.done = True; self.finished.append(s)
            else:
                still.append(s)
        self.active = still
        return len(nxt)

    def run(self, max_steps=10000):
        steps = 0
        while (self.waiting or self.active) and steps < max_steps:
            self.step(); steps += 1
        return steps

    def pending(self): return len(self.waiting) + len(self.active)


def _selftest():
    # mock runtime: next token = current + 1 (deterministic), eos at 999
    calls = {"n": 0, "max_batch_seen": 0}
    def step_fn(cur):
        calls["n"] += 1; calls["max_batch_seen"] = max(calls["max_batch_seen"], len(cur))
        return [c + 1 for c in cur]

    b = ContinuousBatcher(step_fn, max_batch=4, eos=None)
    b.submit("a", 10, max_new=3)          # a: 10 -> 11,12  (len reaches 3: [10,11,12])
    b.submit("b", 20, max_new=5)
    # admit mid-flight: add c,d,e (e must wait — max_batch=4, a+b+c+d fill it)
    b.submit("c", 30, max_new=2); b.submit("d", 40, max_new=2); b.submit("e", 50, max_new=2)
    b.run()
    res = {s.sid: s.tokens for s in b.finished}
    assert res["a"] == [10, 11, 12], res.get("a")
    assert res["b"] == [20, 21, 22, 23, 24], res.get("b")
    assert res["c"] == [30, 31] and res["d"] == [40, 41] and res["e"] == [50, 51], res
    assert calls["max_batch_seen"] == 4, f"batch cap respected: {calls['max_batch_seen']}"
    assert len(b.finished) == 5 and b.pending() == 0

    # eos early-stop
    b2 = ContinuousBatcher(lambda cur: [999 for _ in cur], max_batch=8, eos=999)
    b2.submit("x", 1, max_new=100); b2.run()
    assert b2.finished[0].tokens == [1, 999], b2.finished[0].tokens
    print("batching self-test: PASS ✅ (continuous admit, batch cap, per-seq max_new, eos stop)")


if __name__ == "__main__":
    _selftest()
