#!/usr/bin/env python3
"""SLO attainment over a public request trace, per the project's KPI 3.

The KPI is not "is p99 TPOT low". It is: replay a public trace at several
scales, and count how many of those runs kept the 99th-percentile time per
output token under 150 ms. So the unit of measurement is a *case* -- one
configuration at one decode width at one level of offered concurrency -- and
the reported rate is the share of cases that held the bar.

Three things this harness is careful about.

Concurrency is set, not offered. Poisson arrivals make the number of requests
actually in flight an outcome rather than a control, and a batched decoder only
earns its width when that many requests genuinely overlap. Each case therefore
holds exactly N requests in flight and measures what the system settles at.

The trace is filtered, not truncated. Every graph here is compiled for a fixed
context S, and the gateway quietly clips a prompt to S - max_tokens - 1 rather
than refusing it -- which would turn a 1,600-token request into a short one and
report a flattering latency for work that was never done. Requests that do not
fit are dropped before they are sent, and the share dropped is recorded with
the result, because a SLO measured on the short two thirds of a trace is only
honest if it says so.

TPOT comes from the worker, which times the decode loop itself, so queueing
ahead of the first token lands in TTFT where it belongs. A static batcher can
hold a request for its whole batch-wait window; that is a TTFT cost, and
hiding it inside TPOT would make a starved batch look fast.

    python3 bench/kpi_slo.py --S 512 --N 1,2,4,8 --out bench/kpi_slo.jsonl
"""
import argparse
import json
import threading
import time
import urllib.request

import numpy as np

GATEWAY = "http://192.0.2.3/rngd/llama8b-pd/completions"
TRACE = "/home/gspark/LLMServingSim/dataset/sharegpt_req300_rate10_llama.jsonl"
SLO_MS = 150.0

# config name -> gateway model route, and the decode widths it serves
CASES = [
    ("rngd_alone", "llama8b-rngd-alone", [1, 4]),
    ("pd_forward", "llama8b-pd-forward", [1, 4]),
    ("pd_reverse", "llama8b-pd-reverse", [1, 4]),
    ("atom_alone", "llama8b-atom-alone", [1, 4]),
    ("solo", "llama8b-solo", [4]),
]

# How many cards each route holds, for the per-card column. solo is the only
# one-card configuration; everything else is a prefill pod plus a decode pod.
CARDS = {"solo": 1, "rngd_alone": 2, "pd_forward": 2,
         "pd_reverse": 2, "atom_alone": 2}


def fixed_items(P, D, n=32):
    """A synthetic workload of exactly one shape.

    The trace answers "does the SLO hold on real traffic". This answers a
    different question -- at which prompt-to-generation ratio does a dedicated
    decode card start paying for itself -- and for that the shape has to be the
    controlled variable, not a distribution. Prompts still carry a random head
    so nothing can be served out of a prefix cache.
    """
    from transformers import AutoTokenizer
    import random
    tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    body = ("Paris is the capital and most populous city of France, "
            "situated on the river Seine in the north of the country. ")
    ids = tk.encode(body * 60, add_special_tokens=False)
    rng = random.Random(0)
    out = []
    for _ in range(n):
        head = " ".join(str(rng.randint(10**6, 10**7)) for _ in range(6))
        hlen = len(tk.encode(head, add_special_tokens=False))
        text = head + " " + tk.decode(ids[:max(P - hlen, 1)])
        text = tk.decode(tk.encode(text, add_special_tokens=False)[:P])
        out.append({"prompt": text, "max_new": D, "P": P, "D": D})
    return out


def load_trace(path, S, margin=8):
    """Trace entries that fit the compiled context, plus what that cost."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    keep = []
    for r in rows:
        if r["input_toks"] + r["output_toks"] + margin <= S:
            keep.append({"prompt": tk.decode(r["input_tok_ids"]),
                         "max_new": r["output_toks"],
                         "P": r["input_toks"], "D": r["output_toks"]})
    return keep, len(rows)


def post(model, item, batch, timeout=1800):
    body = json.dumps({"model": model, "prompt": item["prompt"],
                       "max_tokens": item["max_new"], "batch": batch}).encode()
    req = urllib.request.Request(GATEWAY, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    pd = d.get("pd", {})
    return {"latency_ms": (time.perf_counter() - t0) * 1e3,
            "ttft_ms": pd.get("ttft_ms"), "tpot_ms": pd.get("tpot_ms"),
            "prefill_ms": pd.get("prefill_ms"),
            "out_tokens": d["usage"]["completion_tokens"],
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "occupancy": pd.get("occupancy"),
            "starved_lanes": pd.get("starved_lanes"),
            "sched": pd.get("sched")}


def _med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.median(vals)), 2) if vals else None


def _pct(rows, key, q):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.percentile(vals, q)), 2) if vals else None


def run_case(model, items, batch, N, seconds):
    """Hold N requests in flight for `seconds`; return every completion."""
    done, errs, lock = [], [], threading.Lock()
    stop = threading.Event()

    def client(cid):
        i = cid
        while not stop.is_set():
            try:
                r = post(model, items[i % len(items)], batch)
                with lock:
                    done.append(r)
            except Exception as e:
                with lock:
                    errs.append(repr(e)[:120])
            i += N

    t0 = time.perf_counter()
    ts = [threading.Thread(target=client, args=(k,), daemon=True) for k in range(N)]
    for t in ts:
        t.start()
    while time.perf_counter() - t0 < seconds and not stop.is_set():
        time.sleep(0.5)
    stop.set()
    for t in ts:
        t.join(timeout=1800)
    return done, errs, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--N", default="1,2,4,8")
    ap.add_argument("--B", default="1,4")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--only", default="")
    ap.add_argument("--fixed", default="",
                    help="P,D — one synthetic shape instead of the trace")
    ap.add_argument("--out", default="bench/kpi_slo.jsonl")
    a = ap.parse_args()

    if a.fixed:
        P, D = (int(v) for v in a.fixed.split(","))
        items, drop = fixed_items(P, D), 0.0
        print(f"[fixed] P={P} D={D}  P/D={P/D:.2f}  S={a.S}", flush=True)
    else:
        items, total = load_trace(a.trace, a.S)
        drop = 1 - len(items) / total
        print(f"[trace] {len(items)}/{total} fit S={a.S} "
              f"(제외 {drop*100:.1f}%)  P p50={int(np.median([i['P'] for i in items]))} "
              f"D p50={int(np.median([i['D'] for i in items]))}", flush=True)

    keep = set(a.only.split(",")) if a.only else None
    Ns = [int(v) for v in a.N.split(",")]
    Bs = [int(v) for v in a.B.split(",")]

    fh = open(a.out, "w")
    met = tot = 0
    for name, model, widths in CASES:
        if keep and name not in keep:
            continue
        for B in [b for b in Bs if b in widths]:
            for N in Ns:
                try:
                    # Warm the path, not the workload: the first trace entry
                    # can ask for 400 tokens, and on ATOM that is half a minute
                    # of decode before the measured window even opens.
                    post(model, dict(items[0], max_new=8), B)
                except Exception as e:
                    print(f"[{name} B={B}] warmup failed: {e}", flush=True)
                    continue
                done, errs, wall = run_case(model, items, B, N, a.seconds)
                if not done:
                    print(f"[{name} B={B} N={N}] 완료 요청 없음 ({len(errs)} 실패)",
                          flush=True)
                    continue
                tp = np.array([r["tpot_ms"] for r in done if r["tpot_ms"]])
                tt = np.array([r["ttft_ms"] for r in done if r["ttft_ms"]])
                lat = np.array([r["latency_ms"] for r in done])
                p99 = float(np.percentile(tp, 99)) if len(tp) else None
                ok = p99 is not None and p99 <= SLO_MS
                met += bool(ok); tot += 1
                cards = CARDS[name]
                row = {"config": name, "B": B, "N": N, "S": a.S,
                       "cards": cards,
                       "P": items[0]["P"] if a.fixed else None,
                       "D": items[0]["D"] if a.fixed else None,
                       "requests": len(done), "errors": len(errs),
                       "wall_s": round(wall, 1),
                       "tokens_s": round(sum(r["out_tokens"] for r in done) / wall, 1),
                       "tokens_s_per_card": round(
                           sum(r["out_tokens"] for r in done) / wall / cards, 1),
                       "tpot_p50": round(float(np.percentile(tp, 50)), 1) if len(tp) else None,
                       "tpot_p95": round(float(np.percentile(tp, 95)), 1) if len(tp) else None,
                       "tpot_p99": round(p99, 1) if p99 else None,
                       "ttft_p50": round(float(np.percentile(tt, 50)), 1) if len(tt) else None,
                       "ttft_p99": round(float(np.percentile(tt, 99)), 1) if len(tt) else None,
                       "lat_p99": round(float(np.percentile(lat, 99)), 1),
                       "sched": done[-1].get("sched"),
                       # Each request reports occupancy over its own window, so
                       # the median across requests is the representative one.
                       # The last request is not: it finishes into the drain,
                       # when the generator has stopped and the lanes are
                       # emptying, and reports a number from that tail.
                       "occupancy": _med(done, "occupancy"),
                       # A median of 4.0 can still hide a tail of starved
                       # requests, and the tail is what a lane-occupancy fix
                       # would target, so carry the low end too.
                       "occupancy_p10": _pct(done, "occupancy", 10),
                       "occupancy_min": _pct(done, "occupancy", 0),
                       "starved_lanes": _med(done, "starved_lanes"),
                       "starved_p90": _pct(done, "starved_lanes", 90),
                       "trace_kept_pct": round((1 - drop) * 100, 1),
                       "slo_met": bool(ok)}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                print(f"[{name:11s} B={B} N={N}] {row['requests']:3d} req  "
                      f"{row['tokens_s']:6.1f} tok/s ({row['tokens_s_per_card']:5.1f}/card)  "
                      f"TTFT p50 {row['ttft_p50']:7.1f} p99 {row['ttft_p99']:7.1f}  "
                      f"TPOT p50 {row['tpot_p50']:6.1f} p99 {row['tpot_p99']:6.1f}  "
                      f"occ {row['occupancy']} (p10 {row['occupancy_p10']}, "
                      f"min {row['occupancy_min']})  starved p90 {row['starved_p90']}  "
                      f"err {row['errors']}", flush=True)
    fh.close()
    if tot:
        print(f"\nSLO 달성률 = {met}/{tot} = {met/tot*100:.1f}%  (TPOT p99 <= {SLO_MS:.0f} ms)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
