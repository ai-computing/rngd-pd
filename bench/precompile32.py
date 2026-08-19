#!/usr/bin/env python3
"""Precompile the nl=32 S=576 resident engine into the artifact cache, then
sanity-run a few steps and report the e2e step time."""
import argparse
import time

import _bootstrap  # noqa: F401
from rngd_pd.engines.decode_resident import EngineSession, build_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--chip", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    cfg, rt = build_engine("meta-llama/Llama-3.1-8B-Instruct", a.S, a.nl, chip=a.chip)
    print(f"PRECOMPILE DONE in {time.time()-t0:.0f}s total", flush=True)
    from rngd_pd.kernels import decode_resident as LER
    s = EngineSession(rt, LER)
    Kp, Vp = s.fresh_kv()
    ts = []
    nxt = 791
    for pos in range(6):
        t1 = time.perf_counter()
        nxt, Kp, Vp = s.step(nxt, pos, Kp, Vp)
        ts.append(time.perf_counter() - t1)
    print(f"e2e step (nl={a.nl}, S={a.S}): first {ts[0]*1e3:.0f} ms, "
          f"steady p50 {sorted(ts[1:])[2]*1e3:.1f} ms", flush=True)
    print("SANITY OK")


if __name__ == "__main__":
    main()
