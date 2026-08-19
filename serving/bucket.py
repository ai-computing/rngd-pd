"""Multi-bucket runtime manager (Phase 5).

A wired EDF is compiled for a fixed seq-len bucket (the `_bucket()` rounding). Production serves many
prompt lengths, so we precompile a set of buckets and pick the smallest one that fits each request —
short prompts run a small/fast EDF, long prompts a larger one, none pay for the max length always.

Runtime-agnostic: give it a `compile_fn(bucket)->runtime`; it lazily compiles (or eagerly warms) each
bucket and routes by length. Pairs with artifact_cache so a bucket's EDF loads from disk instead of
recompiling.
"""


def bucket_for(n, buckets):
    """Smallest bucket >= n; the largest bucket if n exceeds all (caller must guard truncation)."""
    fit = [b for b in sorted(buckets) if b >= n]
    return fit[0] if fit else max(buckets)


class BucketManager:
    def __init__(self, buckets, compile_fn, warm=()):
        self.buckets = sorted(buckets)
        self.compile_fn = compile_fn
        self._rt = {}
        for b in warm:
            self.get(b)

    def get(self, seq_len):
        """Return (bucket, runtime) for a sequence of length seq_len, compiling lazily."""
        b = bucket_for(seq_len, self.buckets)
        if b not in self._rt:
            self._rt[b] = self.compile_fn(b)
        return b, self._rt[b]

    def resident_buckets(self):
        return sorted(self._rt)


def _selftest():
    compiled = []
    mgr = BucketManager([64, 128, 256], compile_fn=lambda b: compiled.append(b) or f"rt{b}", warm=[64])
    assert compiled == [64], f"eager warm only 64: {compiled}"
    assert mgr.get(50) == (64, "rt64") and compiled == [64], "50→64, no recompile"
    assert mgr.get(100) == (128, "rt128") and compiled == [64, 128], "100→128 lazy compile"
    assert mgr.get(64) == (64, "rt64"), "exact 64"
    assert mgr.get(300) == (256, "rt256"), "over-max → largest bucket"
    assert mgr.resident_buckets() == [64, 128, 256]
    print("bucket self-test: PASS ✅ (smallest-fit, lazy compile, warm, over-max guard)")


if __name__ == "__main__":
    _selftest()
