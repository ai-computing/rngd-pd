"""Decode widths on one card, chosen per step (no SDK imports, so it tests without a card)."""


class TieredDecode:
    """Several decode widths on one card, stepped at the narrowest that fits.

    A step reads every weight once whatever its width, but its cost still grows
    with B: 21.5 ms at B=4 against 35.5 ms at B=16 (S=512, #14 phase 2). So a
    16-wide engine charges a lone request 16 lanes' worth. Here one engine per
    width shares the weights (BatchDecodeEngine(share=...)); each keeps its own
    resident cache, and the step runs on the narrowest width whose lanes cover
    every live lane. Lane i is lane i at every width, so moving between widths
    copies only the live lanes, on the card.

    The scheduler sees one engine of width max(widths). It must call
    set_live() with the live lanes before each step; Kd/Vd it holds are opaque
    tokens, the caches live here.
    """

    def __init__(self, engines):
        self.tiers = sorted(engines, key=lambda e: e.B)
        self.B = self.tiers[-1].B
        self.by_B = {e.B: e for e in self.tiers}
        self.kv = {}                   # width -> (Kd, Vd), allocated on first use
        self.cur = None
        self.live = []
        self.switches = 0
        self.steps_at = {e.B: 0 for e in self.tiers}

    def _kv(self, b):
        if b not in self.kv:
            self.kv[b] = self.by_B[b].blank_kv()
        return self.kv[b]

    def _fit(self, top):
        return next(e.B for e in self.tiers if e.B > top)

    def _move_to(self, b):
        """Make width b current, carrying the live lanes' caches over."""
        if self.cur == b:
            return
        Kn, Vn = self._kv(b)
        if self.cur is not None:
            Ko, Vo = self.kv[self.cur]
            for i in self.live:
                for g in range(len(Kn)):
                    Kn[g][:, i] = Ko[g][:, i]
                    Vn[g][:, i] = Vo[g][:, i]
            self.switches += 1
        self.cur = b

    def blank_kv(self):
        self._move_to(self.tiers[0].B)
        return self, self

    def set_live(self, live):
        self.live = sorted(live)

    def write_slot_kv(self, Kd, Vd, idx, K, V):
        need = self._fit(max(self.live + [idx]))
        if need > self.cur:
            self._move_to(need)
        Kc, Vc = self.kv[self.cur]
        self.kv[self.cur] = self.by_B[self.cur].write_slot_kv(Kc, Vc, idx, K, V)
        if idx not in self.live:
            self.live = sorted(self.live + [idx])
        return Kd, Vd

    def step(self, toks, positions, Kd, Vd):
        if not self.live:
            return [0] * self.B, Kd, Vd
        self._move_to(self._fit(self.live[-1]))
        b = self.cur
        e = self.by_B[b]
        Kc, Vc = self.kv[b]
        nxt, Kc, Vc = e.step(list(toks)[:b], list(positions)[:b], Kc, Vc)
        self.kv[b] = (Kc, Vc)
        self.steps_at[b] += 1
        return list(nxt) + [0] * (self.B - b), Kd, Vd
