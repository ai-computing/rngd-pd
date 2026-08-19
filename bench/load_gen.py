#!/usr/bin/env python3
"""Staggered, mixed-length load — the workload that tells the two schedulers apart.

The context/batch sweep fires B identical requests at the same instant, which
is the one arrival pattern where holding a batch until it fills costs nothing.
Real load does not look like that: requests arrive spread out and ask for
different numbers of tokens, and then the difference between the two policies
is the whole point.

  static      a batch runs start to finish. A request that arrives one step
              late waits for the entire generation ahead of it, and a lane
              whose stream finished early keeps being computed to the end.
  continuous  every step admits whatever is waiting into free slots and
              releases whatever finished.

Arrivals are Poisson at --rate requests/s; output length is drawn uniformly
from --dmin..--dmax so streams finish out of step with each other. What is
reported is what a caller feels: queueing included, per-request.

  python3 bench/load_gen.py --model llama8b-rngd-alone --batch 4 \
      --rate 6 --n 48 --out bench/load_static.jsonl
"""
import argparse
import json
import os
import threading
import time
import urllib.request

import numpy as np

import _bootstrap  # noqa: F401

GATEWAY = os.environ.get("PD_GATEWAY",
                         "http://192.0.2.3/rngd/llama8b-pd/completions")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def build_prompts(n, P, seed=0):
    """Length-exact prompts with a random head, so nothing can be cached."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(seed)
    body = ("Paris is the capital and most populous city of France, situated "
            "on the river Seine in the north of the country. ")
    ids = tk.encode(body * 40, add_special_tokens=False)
    out = []
    for _ in range(n):
        head = " ".join(str(int(v)) for v in rng.integers(10 ** 6, 10 ** 7, size=6))
        hlen = len(tk.encode(head, add_special_tokens=False))
        text = head + " " + tk.decode(ids[:max(P - hlen, 1)])
        out.append(tk.decode(tk.encode(text, add_special_tokens=False)[:P]))
    return out


def fire(model, prompt, max_new, batch, rec, i, t_zero):
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_new, "batch": batch}).encode()
    req = urllib.request.Request(GATEWAY, data=body,
                                 headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        done = time.perf_counter()
        pd = d.get("pd", {})
        rec[i] = {"i": i, "arrive_s": round(sent - t_zero, 3),
                  "latency_ms": round((done - sent) * 1e3, 1),
                  "max_new": max_new,
                  "out_tokens": d["usage"]["completion_tokens"],
                  "ttft_ms": pd.get("ttft_ms"), "tpot_ms": pd.get("tpot_ms"),
                  "prefill_ms": pd.get("prefill_ms"),
                  "sched": pd.get("sched", "static"),
                  "admit_ms": pd.get("admit_ms"),
                  "occupancy": pd.get("occupancy")}
    except Exception as e:
        rec[i] = {"i": i, "error": repr(e)[:120]}


def closed_loop(a, prompts, lens):
    """Hold N requests in flight and see what the system settles at.

    Poisson arrivals answer "can it keep up with this rate"; they cannot say
    what a configuration is worth, because the offered concurrency is an
    outcome rather than a setting. A batched decoder only earns its width when
    that many requests genuinely overlap, so the number that matters is
    throughput as a function of how many are in flight.
    """
    done, lock = [], threading.Lock()
    stop = threading.Event()
    t_zero = time.perf_counter()

    def client(cid):
        i = cid
        while not stop.is_set():
            rec = [None]
            fire(a.model, prompts[i % len(prompts)], int(lens[i % len(lens)]),
                 a.batch, rec, 0, t_zero)
            if rec[0] and "error" not in rec[0]:
                with lock:
                    done.append(rec[0])
            elif rec[0]:
                with lock:
                    done.append(rec[0])
            i += a.concurrency

    ts = [threading.Thread(target=client, args=(k,), daemon=True)
          for k in range(a.concurrency)]
    for t in ts:
        t.start()
    time.sleep(a.seconds)
    stop.set()
    for t in ts:
        t.join(timeout=1800)
    wall = time.perf_counter() - t_zero

    ok = [r for r in done if "error" not in r]
    errs = len(done) - len(ok)
    if not ok:
        print(f"  concurrency {a.concurrency}: 전부 실패 ({errs})")
        return
    lat = np.array([r["latency_ms"] for r in ok])
    toks = sum(r["out_tokens"] for r in ok)
    occ = [r["occupancy"] for r in ok if r.get("occupancy")]
    print(f"  N={a.concurrency:<3d} {toks/wall:7.1f} tok/s  {len(ok)/wall:5.2f} req/s  "
          f"p50 {np.percentile(lat,50):7.0f}  p95 {np.percentile(lat,95):7.0f}  "
          f"{len(ok):3d} ok / {errs} fail"
          + (f"  occ {occ[-1]:.2f}" if occ else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b-rngd-alone")
    ap.add_argument("--batch", type=int, default=4, help="the pod's graph width")
    ap.add_argument("--P", type=int, default=128, help="prompt tokens")
    ap.add_argument("--dmin", type=int, default=4)
    ap.add_argument("--dmax", type=int, default=16)
    ap.add_argument("--rate", type=float, default=6.0, help="arrivals per second")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="closed loop: hold this many requests in flight and "
                         "measure saturation instead of firing at --rate")
    ap.add_argument("--seconds", type=float, default=25.0,
                    help="closed-loop measurement window")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    prompts = build_prompts(a.n + a.warm, a.P)
    lens = rng.integers(a.dmin, a.dmax + 1, size=a.n + a.warm)
    print(f"[load] {a.model} B={a.batch} P={a.P} D={a.dmin}..{a.dmax} "
          f"rate={a.rate}/s n={a.n}", flush=True)

    rec = [None] * a.warm
    ts = [threading.Thread(target=fire, args=(a.model, prompts[i], int(lens[i]),
                                              a.batch, rec, i, time.perf_counter()))
          for i in range(a.warm)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"[load] warmed with {a.warm} requests", flush=True)

    if a.concurrency:
        return closed_loop(a, prompts, lens)

    rec = [None] * a.n
    threads = []
    t_zero = time.perf_counter()
    gaps = rng.exponential(1.0 / a.rate, size=a.n)      # Poisson arrivals
    for i in range(a.n):
        time.sleep(float(gaps[i]))
        t = threading.Thread(target=fire, args=(a.model, prompts[a.warm + i],
                                                int(lens[a.warm + i]), a.batch,
                                                rec, i, t_zero))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_zero

    ok = [r for r in rec if r and "error" not in r]
    errs = len(rec) - len(ok)
    lat = np.array([r["latency_ms"] for r in ok])
    toks = sum(r["out_tokens"] for r in ok)
    print("\n" + "=" * 64)
    print(f"{a.model}  B={a.batch}  sched={ok[0]['sched'] if ok else '?'}")
    print(f"  {len(ok)}/{a.n} ok ({errs} failed) in {wall:.1f} s")
    print(f"  throughput  {toks/wall:.1f} output tok/s   "
          f"{len(ok)/wall:.2f} req/s")
    print(f"  latency     p50 {np.percentile(lat,50):7.0f} ms   "
          f"p95 {np.percentile(lat,95):7.0f} ms   max {lat.max():7.0f} ms")
    if ok and ok[-1].get("admit_ms") is not None:
        print(f"  admission   {ok[-1]['admit_ms']:.1f} ms per composition change")
    occ = [r["occupancy"] for r in ok if r.get("occupancy")]
    if occ:
        print(f"  occupancy   {occ[-1]:.2f} of {a.batch} lanes per step")
    # Little's law: how many requests the load actually keeps in flight. If
    # this is below the graph width, the batch cannot be full however it is
    # scheduled, and a step run half empty is half wasted.
    print(f"  offered     ~{a.rate*np.mean(lat)/1000:.1f} requests in flight "
          f"(graph width {a.batch})")
    print("=" * 64)
    if a.out:
        with open(a.out, "w") as fh:
            for r in rec:
                fh.write(json.dumps(r) + "\n")
        print(f"[load] wrote {a.out}")


if __name__ == "__main__":
    main()
