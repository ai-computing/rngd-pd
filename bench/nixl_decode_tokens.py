"""Token exactness of the `nixl` RNGD -> ATOM path: decode on ATOM from the KV that went through shm and
from the KV that landed in the ATOM DMA-BUF (bench/nixl_kv_check.py saves both per prompt), and compare.

The serving ATOM decode graph (DecodeATOMBatch, B=1, the `_lm` artifact) takes its cache as a host
input every step, so the landed bytes are read back from the ATOM buffer first; what is compared is
that the bytes which crossed RNGD -> ATOM over NIXL decode to the same tokens as the shm handoff.

    RBLN_BATCH_ARTIFACT=.../decode_S512_B1_lm.rbln RBLN_TABLES=.../tables.npz \
        python bench/nixl_decode_tokens.py runs/nixl-check --tokens 32
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json
import os
import sys

import numpy as np


def bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--S", type=int, default=512)
    a = ap.parse_args()
    from rngd_pd.serving import cluster
    w = cluster.DecodeATOMBatch(a.S, 1)

    def decode(first, pos, K, V):
        w._load_kv = lambda _rid: (K, V, 0.0, -1.0, None)
        out, = w._run_batch([{"req_id": "x", "first": first, "pos": pos, "max_new": a.tokens}])
        return out["tokens"]

    rows = []
    for path in sorted(glob.glob(os.path.join(a.run_dir, "kv-*.npz"))):
        z = np.load(path)
        first, pos = int(z["first"]), int(z["pos"])
        shm = decode(first, pos, bits_to_f32(z["shm_k"]), bits_to_f32(z["shm_v"]))
        nixl = decode(first, pos, bits_to_f32(z["atom_k"]), bits_to_f32(z["atom_v"]))
        rows.append(dict(file=os.path.basename(path), prompt_tokens=pos, tokens=len(shm),
                         kv_bytes_equal=bool(np.array_equal(z["shm_k"], z["atom_k"]) and np.array_equal(z["shm_v"], z["atom_v"])),
                         token_exact=shm == nixl, shm=shm[:12], nixl=nixl[:12]))
        print(json.dumps(rows[-1]), flush=True)
    ok = bool(rows) and all(r["token_exact"] for r in rows)
    with open(os.path.join(a.run_dir, "tokens.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("TOKENS", "EXACT" if ok else "MISMATCH", f"{sum(r['token_exact'] for r in rows)}/{len(rows)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
