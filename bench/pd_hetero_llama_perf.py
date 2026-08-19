#!/usr/bin/env python3
"""What is the 8B ATOM decode step actually spending its 88 ms on?

KV now enters the compiled graph fresh every step (67 MB of fp32 at S=256), so
the step time should carry a term proportional to KV bytes. Sweeping S — which
scales KV linearly while leaving the 14 GB of weights and all the math
untouched — separates transfer from compute, and an fp16 KV wire halves the
transfer term to confirm the reading.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_hetero_llama_perf.py
"""
import argparse
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401
from rngd_pd.vendors.rbln_decode import MODEL, LlamaDecodeRBLN, load_hf

torch.set_grad_enabled(False)


class KVHalf(LlamaDecodeRBLN):
    """Same module, but KV arrives as fp16 and is widened on-card."""

    def forward(self, x, Kp, Vp, cos, sin, mask, oh):
        return super().forward(x, Kp.float(), Vp.float(), cos, sin, mask, oh)


def bench(mod, cfg, S, nl, reps, tag):
    import rebel
    H, NKV, HD = cfg.hidden, cfg.n_kv, cfg.head_dim
    kvdt = torch.float16 if isinstance(mod, KVHalf) else torch.float32
    ex = [torch.zeros(1, H), torch.zeros(nl, NKV, HD, S, dtype=kvdt),
          torch.zeros(nl, NKV, S, HD, dtype=kvdt),
          torch.zeros(HD // 2), torch.zeros(HD // 2), torch.zeros(S), torch.zeros(S)]
    t0 = time.time()
    rt = rebel.Runtime(rebel.compile_from_torch(mod, example_inputs=ex))
    comp = time.time() - t0
    npdt = np.float16 if kvdt is torch.float16 else np.float32
    args = [np.zeros((1, H), np.float32),
            np.zeros((nl, NKV, HD, S), npdt), np.zeros((nl, NKV, S, HD), npdt),
            np.zeros(HD // 2, np.float32), np.zeros(HD // 2, np.float32),
            np.zeros(S, np.float32), np.eye(S, dtype=np.float32)[0]]
    rt.run(*args)                                    # warm up
    ts = []
    for _ in range(reps):
        t1 = time.perf_counter()
        rt.run(*args)
        ts.append((time.perf_counter() - t1) * 1e3)
    ts.sort()
    kv_mb = (args[1].nbytes + args[2].nbytes) / 1e6
    print(f"{tag:>22} {S:>5} {kv_mb:>9.1f} {ts[len(ts)//2]:>10.1f} "
          f"{ts[0]:>9.1f} {comp:>9.0f}", flush=True)
    del rt
    return kv_mb, ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--sweep", default="64,128,256")
    a = ap.parse_args()

    _tk, cfg, sd = load_hf()
    nl = cfg.n_layers
    print(f"\n{'config':>22} {'S':>5} {'KV MB':>9} {'p50 ms':>10} "
          f"{'best ms':>9} {'compile s':>9}")
    print("-" * 70)
    pts = []
    for S in [int(x) for x in a.sweep.split(",")]:
        mod = LlamaDecodeRBLN(cfg, sd, S, nl).eval()
        pts.append(bench(mod, cfg, S, nl, a.reps, "fp32 KV"))
        del mod
    S = int(a.sweep.split(",")[-1])
    mod = KVHalf(cfg, sd, S, nl).eval()
    half = bench(mod, cfg, S, nl, a.reps, "fp16 KV")
    del mod, sd

    # least squares on (KV bytes -> ms) gives the fixed compute term
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    print("-" * 70)
    print(f"fit: step_ms = {intercept:.1f} + {slope:.3f} * KV_MB")
    print(f"  compute-bound floor      : {intercept:.1f} ms/token")
    print(f"  KV upload rate           : {1000/slope/1000:.1f} GB/s "
          f"({slope:.3f} ms per MB)")
    big = max(pts, key=lambda p: p[0])
    print(f"  at S={S}: {big[1]:.1f} ms of which "
          f"{big[0]*slope:.1f} ms ({big[0]*slope/big[1]*100:.0f}%) is KV transfer")
    print(f"  fp16 KV at S={S}: {half[1]:.1f} ms "
          f"({(1 - half[1]/big[1])*100:.0f}% faster, KV {half[0]:.1f} MB)")


if __name__ == "__main__":
    main()
