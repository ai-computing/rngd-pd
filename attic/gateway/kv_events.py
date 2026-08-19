"""KV-cache event index for KV-aware routing (Phase 2).

Workers publish which prompt-prefix blocks they hold; the gateway indexes block-hash → workers so the
EPP can route a request to a worker that already has the longest matching prefix cached (prefix-cache
hit → skip recompute). This mirrors vLLM's KV-cache events (BlockStored / BlockRemoved /
AllBlocksCleared) + llm-d's KV-cache indexer, scoped to RNGD.

Block hashing matches vLLM's scheme: a CHAINED hash over fixed-size token blocks, so a hash encodes
the entire prefix up to that block (identical prefixes → identical hashes; a divergence flips all
subsequent hashes). That makes "longest cached prefix" a simple contiguous-match count.
"""
import threading, hashlib


def block_hashes(token_ids, block_size=16):
    """Chained per-block hashes over token_ids (vLLM-style prefix hashing)."""
    hs = []
    parent = b"\x00" * 16
    for i in range(0, len(token_ids) - block_size + 1, block_size):
        blk = token_ids[i:i + block_size]
        h = hashlib.blake2b(parent + b"".join(t.to_bytes(4, "little") for t in blk), digest_size=16).digest()
        hs.append(h)
        parent = h
    return hs                          # only FULL blocks (partial tail block is not cacheable)


class KVIndex:
    """block_hash -> set(worker_id). Fed by worker events; queried by the EPP."""
    def __init__(self):
        self._idx = {}
        self._lock = threading.Lock()

    # --- worker-published events (vLLM KVCacheEvent parity) ---
    def block_stored(self, wid, hashes):
        with self._lock:
            for h in hashes: self._idx.setdefault(h, set()).add(wid)

    def block_removed(self, wid, hashes):
        with self._lock:
            for h in hashes:
                s = self._idx.get(h)
                if s: s.discard(wid)
                if s is not None and not s: del self._idx[h]

    def all_cleared(self, wid):
        with self._lock:
            for h in list(self._idx):
                self._idx[h].discard(wid)
                if not self._idx[h]: del self._idx[h]

    # --- gateway query ---
    def longest_prefix(self, hashes):
        """Return {worker_id: num_contiguous_prefix_blocks_cached} for the request's block hashes.
        Contiguous from block 0 — a worker's score is how deep its cached prefix matches."""
        scores = {}
        with self._lock:
            for h in hashes:                       # walk prefix blocks in order
                holders = self._idx.get(h)
                if not holders: break              # first miss ends every worker's contiguous run...
                advanced = False
                for wid in holders:
                    # only workers that matched every previous block stay eligible
                    if scores.get(wid, 0) == 0 and h != hashes[0]:
                        continue
                    scores[wid] = scores.get(wid, 0) + 1
                    advanced = True
                if not advanced: break
        return scores

    def best_worker(self, hashes, eligible=None):
        """(worker_id, hit_blocks) with the longest cached prefix among `eligible` wids, else None."""
        scores = self.longest_prefix(hashes)
        if eligible is not None:
            scores = {w: n for w, n in scores.items() if w in eligible}
        if not scores: return (None, 0)
        w = max(scores, key=scores.get)
        return (w, scores[w])


def _selftest():
    A = list(range(64)); B = list(range(32)) + list(range(1000, 1032))   # shares first 2 blocks
    ha, hb = block_hashes(A), block_hashes(B)
    assert ha[:2] == hb[:2] and ha[2] != hb[2], "chained prefix hashing"
    idx = KVIndex()
    idx.block_stored("w0", ha)                 # w0 has all of A's prefix
    idx.block_stored("w1", hb[:2])             # w1 has first 2 blocks (shared prefix)
    # request identical to A: w0 full match (4 blocks), w1 only 2
    w, n = idx.best_worker(ha)
    assert w == "w0" and n == 4, f"expected w0/4 got {w}/{n}"
    # request = B: w1 and w0 both match first 2, then w1 has b[... nothing more], w0 diverges
    sc = idx.longest_prefix(hb)
    assert sc.get("w0", 0) == 2 and sc.get("w1", 0) == 2, f"prefix scores {sc}"
    # eligibility filter (only decode workers)
    w, n = idx.best_worker(ha, eligible={"w1"})
    assert w == "w1" and n == 2, f"eligible filter {w}/{n}"
    idx.block_removed("w0", ha); assert idx.best_worker(ha)[0] == None or idx.best_worker(ha)[1] <= 2
    print("kv_events self-test: PASS ✅ (chained hash, longest-prefix, eligibility, removal)")


if __name__ == "__main__":
    _selftest()
