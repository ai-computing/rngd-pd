#!/usr/bin/env python3
"""Can one KV column per lane be written on the card in place, cheaply?

The batched decode graph is functional: it takes the S-wide cache and returns a
new one with this step's column injected, so every step re-reads and re-writes
the whole cache. Dropping that needs the append to happen outside the graph.
This times the candidate eager ops on a cache of the serving shape
[group, B, NKV, HD, S] (bf16) and checks they write what they should:

  - per-lane slice assignment (B ops per tensor);
  - one index_put_ over all lanes;
  - the full-copy baseline (clone), which is what the graph pays per step.

  python bench/kv_append_probe.py --B 16 --S 1024 --group 8
"""
import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--chip", type=int, default=-1)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    from rngd_pd.devices import resolve_chip
    from rngd_pd.engines.fsdk import dev_str, sync, open_device
    open_device(8)
    chip = resolve_chip(a.chip)
    dev = dev_str(chip)
    G, B, S, NKV, HD = a.group, a.B, a.S, 8, 128
    K = torch.zeros(G, B, NKV, HD, S, dtype=torch.bfloat16).to(dev)
    k = torch.randn(G, B, NKV, HD).to(torch.bfloat16)
    kd = k.to(dev)
    pos = np.arange(B) * 7 % S

    def timed(fn):
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); fn(); sync(chip); ts.append((time.perf_counter() - t0) * 1e3)
        return round(float(np.median(ts)), 3)

    def per_lane():
        for b in range(B):
            K[:, b, :, :, int(pos[b])] = kd[:, b]

    res = {}
    try:
        res["per_lane_slice_ms"] = timed(per_lane)
        got = K.cpu()
        res["per_lane_slice_ok"] = bool(all(torch.equal(got[:, b, :, :, int(pos[b])], k[:, b])
                                            for b in range(B)))
    except Exception as e:  # noqa: BLE001
        res["per_lane_slice_error"] = repr(e)[:200]
    try:
        bi = torch.arange(B).to(dev)
        pi = torch.from_numpy(pos).to(dev)
        Kt = K.permute(1, 4, 0, 2, 3)            # [B, S, G, NKV, HD] view
        vals = kd.permute(1, 0, 2, 3)            # [B, G, NKV, HD]
        res["index_put_ms"] = timed(lambda: Kt.index_put_((bi, pi), vals))
        got = K.cpu()
        res["index_put_ok"] = bool(all(torch.equal(got[:, b, :, :, int(pos[b])], k[:, b])
                                       for b in range(B)))
    except Exception as e:  # noqa: BLE001
        res["index_put_error"] = repr(e)[:200]
    try:
        res["full_clone_ms"] = timed(lambda: K.clone())
    except Exception as e:  # noqa: BLE001
        res["full_clone_error"] = repr(e)[:200]
    res.update(B=B, S=S, group=G, cache_mb=round(K.numel() * 2 / 1e6, 1))
    print(res, flush=True)


if __name__ == "__main__":
    main()
