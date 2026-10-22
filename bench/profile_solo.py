#!/usr/bin/env python3
"""Where a one-card rngd-pd request spends its time, phase by phase.

Builds the same engines the solo worker serves with (FullPrefillEngine with
its buckets, BatchDecodeEngine at B/S/group) and times, with perf_counter
around each piece (2026.2.0 runs every module call synchronously):

- prefill: the layer-group calls, the finish (cache assembly, lm_head, argmax
  download) and the lane write that seats the cache in the batch;
- one batched decode step: host-side input building, uploads, each group
  call, lm_head, the logits download and the host argmax.

  RNGD_MAX_SHAPES=8 python bench/profile_solo.py --S 512 --B 16 --decode-group 8 \
      --buckets 128,256,512 --chip 0 --out bench/profile_solo_S512.json
"""
import argparse
import json
import os
import statistics as st
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

pc = time.perf_counter


def med(xs):
    return round(st.median(xs) * 1e3, 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--decode-group", type=int, default=8)
    ap.add_argument("--buckets", default="128,256,512")
    ap.add_argument("--prompts", default="128,512")
    ap.add_argument("--chip", type=int, default=-1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    buckets = sorted(int(v) for v in a.buckets.split(","))
    # The serving worker itself, so what is timed is what serves.
    os.environ["RNGD_BUCKETS"] = ",".join(str(c) for c in buckets[:-1])
    from rngd_pd.serving import cluster
    w = cluster.SoloWorker(a.chip, a.S, C=buckets[-1], group=2, B=a.B,
                           decode_group=a.decode_group, sched="continuous")
    pre, dec = w.pre, w.dec
    rng = np.random.default_rng(0)
    report = dict(S=a.S, B=a.B, decode_group=dec.NG, buckets=buckets, prefill={}, decode={})

    # ---- prefill, one prompt at a time ---------------------------------
    Kd, Vd = dec.blank_kv()
    for P in [int(v) for v in a.prompts.split(",")]:
        toks = [int(v) for v in rng.integers(1000, 100000, P)]
        calls, fin, lane, tot = [], [], [], []
        for rep in range(6):
            t0 = pc()
            stage = pre.begin_device_prefill(toks)
            t1 = pc()
            while True:
                c0 = pc()
                done = stage.advance(1)
                calls.append(pc() - c0)
                if done:
                    break
            t2 = pc()
            K, V, nxt = stage.result()
            t3 = pc()
            Kd, Vd = dec.write_slot_kv(Kd, Vd, rep % a.B, K, V)
            t4 = pc()
            if rep:                                   # first one warms up
                fin.append(t3 - t2); lane.append(t4 - t3); tot.append(t4 - t0)
        n = pre.ngroups
        report["prefill"][P] = dict(
            bucket=pre.last_C, group_calls=n, call_ms=med(calls[n:]),
            calls_total_ms=round(med(calls[n:]) * n, 2), embed_upload_ms=None,
            finish_ms=med(fin), lane_write_ms=med(lane), total_ms=med(tot))
        print(f"[prefill P={P}] {report['prefill'][P]}", flush=True)

    # ---- one decode step, piece by piece --------------------------------
    B = a.B
    toks = [int(v) for v in rng.integers(1000, 100000, B)]
    pos = np.array([100 + i for i in range(B)])
    parts = {k: [] for k in ("embed_upload", "tables_upload", "groups", "lmhead",
                             "logits_download", "logits_cast", "argmax", "total")}
    group_each = []
    for step in range(a.steps + 2):
        t0 = pc()
        x = dec._d(dec.embed[toks])
        t1 = pc()
        rot = dec._d(dec._rot[pos]); maskd = dec._d(dec._mask[pos]); ohd = dec._d(dec._eye[pos])
        t2 = pc()
        ge = []
        for g in range(dec.ngroups):
            g0 = pc()
            x, Kf, Vf = dec.m(x, dec.eps_d, dec.nh_d, *dec.groups[g], Kd[g], Vd[g], rot, maskd, ohd)
            Kd[g], Vd[g] = Kf, Vf
            ge.append(pc() - g0)
        t3 = pc()
        out = dec.lmh(x, dec.norm_d, dec.eps_d, dec.nh_d, dec.wlm_d)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        t4 = pc()
        lg = logits.detach().cpu()
        t5 = pc()
        lgf = lg.float().numpy().reshape(dec.Vp, B)[:dec.V]
        t6 = pc()
        nxt = [int(np.argmax(lgf[:, b])) for b in range(B)]
        t7 = pc()
        if step >= 2:
            for k, v in zip(parts, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5,
                                    t7 - t6, t7 - t0)):
                parts[k].append(v)
            group_each += ge
        toks = nxt
        pos = pos + 1
    report["decode"] = {k + "_ms": med(v) for k, v in parts.items()}
    report["decode"]["group_call_ms"] = med(group_each)
    report["decode"]["engine_step_ms"] = None
    ts = []
    for _ in range(a.steps):                          # the engine's own step, for reference
        t0 = pc()
        toks, Kd, Vd = dec.step(toks, list(pos), Kd, Vd)
        ts.append(pc() - t0)
        pos = pos + 1
    report["decode"]["engine_step_ms"] = med(ts)

    # ---- the same steps through the scheduler, in-process (no HTTP) ------
    for conc in (1, 4, 16):
        stamps, outs, ths = {}, {}, []
        import threading

        def one(i):
            toks = [int(v) for v in np.random.default_rng(i).integers(1000, 100000, 128)]
            st_ = stamps.setdefault(i, [])
            t0 = pc()
            outs[i] = w.run(toks, 64, on_token=lambda t: st_.append(pc()))
            stamps[i] = [t0] + st_
        for i in range(conc):
            th = threading.Thread(target=one, args=(i,)); th.start(); ths.append(th)
        for th in ths:
            th.join()
        tpot = [(s_[-1] - s_[1]) / (len(s_) - 2) for s_ in stamps.values()]
        ttft = [s_[1] - s_[0] for s_ in stamps.values()]
        sched_step = [o["tpot_ms"] for o in outs.values()]
        report.setdefault("scheduler", {})[conc] = dict(
            ttft_ms=med(ttft), tpot_ms=med(tpot),
            scheduler_step_ms=round(st.median(sched_step), 2))
        print(f"[scheduler conc={conc}] {report['scheduler'][conc]}", flush=True)
    print(f"[decode B={B} S={a.S} g={dec.NG}] {report['decode']}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote -> {a.out}")


if __name__ == "__main__":
    main()
