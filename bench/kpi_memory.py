#!/usr/bin/env python3
"""KPI 4 proxy: what keeping the KV cache on the accelerator is worth.

The KPI asks for throughput against "conventional unified memory". We do not
run UM, but the thing UM does for a decode loop is exactly reproducible here:
the cache lives in host memory and is migrated to the device for every access.
So the baseline is the same engine, same weights, same card, same graph, with
one change -- the KV tensors are pulled back to the host after each step and
pushed up again before the next. Everything else about the step is identical,
which is the only way the ratio means anything.

Two details decide whether the comparison is fair rather than flattering.

The round trip stays in bf16. EngineSession.export_kv casts to float32 on the
way down, which doubles the bytes; a real unified-memory page migration would
move the cache as it is stored. Casting would buy the resident path a 2x
advantage it has not earned, so this harness moves the device tensors directly.

Furiosa dispatches are asynchronous. A timing loop that never reads a result
measures enqueue time and comes out three to four times optimistic. Each step
here ends in an argmax over host-side logits, which forces the queue to drain
before the clock is read.

    python3 bench/kpi_memory.py --chip 0 --S 256 --steps 32
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--out", default="bench/kpi_memory.jsonl")
    a = ap.parse_args()

    from rngd_pd.engines.decode_resident import (build_engine, build_lmhead,
                                                 EngineSession)
    from rngd_pd.kernels import decode_resident as LER

    cfg, rt = build_engine(a.model, a.S, a.nl, chip=a.chip)
    sess = EngineSession(rt, LER, build_lmhead(rt, LER, a.chip))
    print(f"[engine] S={a.S} nl={a.nl} chip={a.chip}", flush=True)

    kv_bytes = None

    def run(mode):
        """Time `steps` decode steps with the cache resident or host-staged."""
        nonlocal kv_bytes
        Kp, Vp = sess.fresh_kv()
        kv_bytes = Kp.numel() * 2 + Vp.numel() * 2         # bf16 on the wire
        tok, times = 128000 % rt.lm.shape[0], []
        for i in range(a.warm + a.steps):
            t0 = time.perf_counter()
            tok, Kp, Vp = sess.step(tok, i % (a.S - 1), Kp, Vp)
            if mode == "host":
                # the unified-memory shape: down to host, back up, every step
                Kh = Kp.detach().cpu()
                Vh = Vp.detach().cpu()
                Kp = Kh.to(LER.DEV)
                Vp = Vh.to(LER.DEV)
                tok = int(tok)                              # force the sync
            if i >= a.warm:
                times.append((time.perf_counter() - t0) * 1e3)
        return np.array(times)

    res = {}
    for mode in ("resident", "host"):
        t = run(mode)
        res[mode] = {"ms_p50": round(float(np.percentile(t, 50)), 2),
                     "ms_mean": round(float(t.mean()), 2),
                     "tok_s": round(1000.0 / float(t.mean()), 2)}
        print(f"[{mode:8s}] {res[mode]['ms_mean']:7.2f} ms/token  "
              f"{res[mode]['tok_s']:6.2f} tok/s", flush=True)

    speedup = res["host"]["ms_mean"] / res["resident"]["ms_mean"]
    row = {"model": a.model, "S": a.S, "nl": a.nl, "steps": a.steps,
           "kv_mb": round(kv_bytes / 1e6, 1),
           "host_staged": res["host"], "resident": res["resident"],
           "speedup_x": round(speedup, 2)}
    print(f"\nKV {row['kv_mb']} MB/step 왕복 제거 → {speedup:.2f}배")
    with open(a.out, "w") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
