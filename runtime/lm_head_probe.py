#!/usr/bin/env python3
"""Where should lm_head run once decode is batched?

The resident engine keeps it on the host: `lm @ rmsnorm(hidden)` in numpy. At
B=1 that is a 2.1 GB fp32 read per token, which hides behind the 32-layer step.
At B=4 it is paid per stream while the transformer step stays B-invariant, so
it stops hiding — measured 78.9 ms/step against 23.2 ms for the stack alone.

This times the alternatives on the card: a plain [B,H] x [H,V] matmul, and the
chunked form the older notes preferred after a single [128k,4096] matvec came
out at 23 GB/s.

  python3 lm_head_probe.py --chip 2
"""
import argparse
import time

import numpy as np
import torch

import _paths  # noqa: F401
import furiosa.torch  # noqa: F401
from furiosa.torch import native_device as nd

nd.set_fusion(8)
nd.set_mesh_kind(nd.MeshKind.Mesh(2))

BF = torch.bfloat16
H, V = 4096, 128256
CH, CV = 32, 4008          # 32 * 4008 = 128256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    dev = f"rngd:{a.chip}"
    rng = np.random.default_rng(0)
    d = lambda v: torch.from_numpy(np.ascontiguousarray(v)).to(BF).to(dev)

    lm = (rng.standard_normal((V, H)).astype(np.float32) * .02)
    lmT = d(lm.T.copy())                       # [H,V]
    lmC = d(lm.reshape(CH, CV, H).transpose(0, 2, 1).copy())   # [CH,H,CV]
    host_lm = lm                               # fp32, as the engines hold it
    bytes_bf16 = V * H * 2

    print(f"{'mode':<18s} {'B':>2s} {'ms':>8s} {'GB/s':>8s}")
    for B in (1, 4, 8):
        x = rng.standard_normal((B, H)).astype(np.float32) * .1
        xd = d(x)

        def timed(fn, label):
            fn()                                # warm up
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                r = fn()
                r[0, 0].item() if r.dim() == 2 else r[0, 0, 0].item()
                ts.append((time.perf_counter() - t0) * 1e3)
            ts.sort()
            ms = ts[len(ts) // 2]
            print(f"{label:<18s} {B:>2d} {ms:>8.2f} {bytes_bf16/(ms/1e3)/1e9:>8.0f}")
            return ms

        timed(lambda: torch.matmul(xd, lmT), "device plain")
        # chunked: [CH,B,H] x [CH,H,CV] -> [CH,B,CV]
        xc = xd.unsqueeze(0).expand(CH, B, H).contiguous()
        timed(lambda: torch.matmul(xc, lmC), "device chunked")

        t0 = time.perf_counter()
        for _ in range(a.reps):
            for b in range(B):
                host_lm @ x[b]
        hms = (time.perf_counter() - t0) * 1e3 / a.reps
        print(f"{'host numpy fp32':<18s} {B:>2d} {hms:>8.2f} "
              f"{V*H*4*B/(hms/1e3)/1e9:>8.0f}")


if __name__ == "__main__":
    main()
