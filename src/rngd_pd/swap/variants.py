"""Compiled variants and the load-driven choice between them.

A variant is one compiled shape of the same model: a batch width B and a
sequence bucket S (plus, on RNGD, the layer group the graph walks the stack
in). Everything here is plain Python so the policy can be tested without a card.
"""
from dataclasses import dataclass
import re
import time


@dataclass(frozen=True, order=True)
class Variant:
    B: int
    S: int
    group: int = 8

    @property
    def name(self):
        return f"B{self.B}-S{self.S}-g{self.group}"

    @classmethod
    def parse(cls, text, group=8):
        """'B8,S512' / 'B8-S512-g8' / 'b8 s512' -> Variant."""
        found = dict((k.upper(), int(v)) for k, v in re.findall(r"([BbSsGg])\s*(\d+)", text))
        if "B" not in found or "S" not in found:
            raise ValueError(f"variant needs B and S: {text!r}")
        return cls(found["B"], found["S"], found.get("G", group))


class BatchSelector:
    """Pick a batch variant from the queue depth, with hysteresis.

    Going up is cheap to justify: requests are already waiting. Going down
    throws away lanes that may be needed again a moment later, so it waits
    until the queue has stayed small for `down_after` observations and the
    current variant has been in use for at least `min_dwell_s`. That keeps a
    queue that oscillates around a boundary from swapping on every step.

    observe(depth) returns the batch width to use; it changes only when the
    rule says so. `depth` counts running plus waiting streams.
    """

    def __init__(self, batches, start=None, up_after=2, down_after=8,
                 down_ratio=0.5, min_dwell_s=0.0, clock=time.monotonic):
        self.batches = sorted(set(int(b) for b in batches))
        if not self.batches:
            raise ValueError("no batch variants")
        self.current = start if start is not None else self.batches[0]
        if self.current not in self.batches:
            raise ValueError(f"start {self.current} not in {self.batches}")
        self.up_after, self.down_after = up_after, down_after
        self.down_ratio, self.min_dwell_s = down_ratio, min_dwell_s
        self.clock = clock
        self._up = self._down = 0
        self._since = clock()

    def fit(self, depth):
        """Smallest variant that holds `depth` streams (the largest if none does)."""
        for b in self.batches:
            if b >= depth:
                return b
        return self.batches[-1]

    def observe(self, depth):
        want = self.fit(depth)
        if want > self.current:
            self._up += 1
            self._down = 0
            if self._up >= self.up_after:
                return self._switch(want)
        elif want < self.current and depth <= self.down_ratio * self.current:
            self._down += 1
            self._up = 0
            if (self._down >= self.down_after
                    and self.clock() - self._since >= self.min_dwell_s):
                return self._switch(want)
        else:
            self._up = self._down = 0
        return self.current

    def _switch(self, b):
        self.current = b
        self._up = self._down = 0
        self._since = self.clock()
        return b
