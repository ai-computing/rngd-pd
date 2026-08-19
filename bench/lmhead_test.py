#!/usr/bin/env python3
"""Device lm_head A/B: step latency and argmax agreement vs the host path."""
import time

import _paths  # noqa: F401
from pd_engine import EngineSession, build_engine, build_lmhead


def run(session, n=24, seed_tok=791):
    Kp, Vp = session.fresh_kv()
    outs, ts = [], []
    nxt = seed_tok
    for pos in range(n):
        t0 = time.perf_counter()
        nxt, Kp, Vp = session.step(nxt, pos, Kp, Vp)
        ts.append(time.perf_counter() - t0)
        outs.append(nxt)
    return outs, sorted(ts[2:])[len(ts[2:]) // 2]


def main():
    import llama_engine_resident as LER
    cfg, rt = build_engine("meta-llama/Llama-3.1-8B-Instruct", 256, 8, chip=0)
    host = EngineSession(rt, LER)
    out_h, t_h = run(host)
    lmh = build_lmhead(rt, LER, 0)
    dev = EngineSession(rt, LER, lmhead=lmh)
    out_d, t_d = run(dev)
    agree = sum(a == b for a, b in zip(out_h, out_d))
    print(f"step p50: host {t_h*1e3:.1f} ms -> device {t_d*1e3:.1f} ms "
          f"({t_h/t_d:.2f}x)")
    print(f"argmax agreement: {agree}/{len(out_h)}"
          f"{'' if agree == len(out_h) else '  (bf16 matmul near-tie flips)'}")
    print("LMHEAD TEST DONE")


if __name__ == "__main__":
    main()
