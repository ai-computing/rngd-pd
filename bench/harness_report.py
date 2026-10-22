#!/usr/bin/env python3
"""Gap table from two bench/vendor_sweep.py runs: reference vs candidate.

    python3 bench/harness_report.py bench/harness_furiosa_llm.jsonl bench/harness_rngd_pd.jsonl

Ratios are candidate / reference. The issue #14 target is tok/s >= 0.90 and
TTFT, TPOT <= 1.10 for P <= 1024 and concurrency <= 16; rows that meet all
three are marked "ok".
"""
import json
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "concurrency" in r:
                rows[(r["P"], r["concurrency"])] = r
    return rows


def main():
    ref, cand = load(sys.argv[1]), load(sys.argv[2])
    rname = next(iter(ref.values()))["engine"]
    cname = next(iter(cand.values()))["engine"]
    print(f"| P | conc | TTFT ms {rname} / {cname} | x | TPOT ms {rname} / {cname} | x "
          f"| tok/s {rname} / {cname} | x | W {rname} / {cname} | target |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|")
    for key in sorted(set(ref) | set(cand)):
        r, c = ref.get(key, {}), cand.get(key, {})

        def pair(k, n=1):
            f = lambda v: f"{v:.{n}f}" if isinstance(v, (int, float)) else "-"
            return f"{f(r.get(k))} / {f(c.get(k))}"

        def ratio(k):
            a, b = r.get(k), c.get(k)
            return b / a if a and b else None
        rt, rp, rs = ratio("ttft_ms"), ratio("tpot_ms"), ratio("tokens_s")
        fx = lambda v: f"{v:.2f}" if v is not None else "-"
        ok = (rt is not None and rp is not None and rs is not None
              and rt <= 1.10 and rp <= 1.10 and rs >= 0.90)
        tgt = "ok" if ok else ("n/a" if key[0] > 1024 else "miss")
        if c.get("errors"):
            tgt += f" ({c['errors']} err)"
        print(f"| {key[0]} | {key[1]} | {pair('ttft_ms')} | {fx(rt)} | {pair('tpot_ms', 2)} | "
              f"{fx(rp)} | {pair('tokens_s')} | {fx(rs)} | {pair('power_w', 0)} | {tgt} |")


if __name__ == "__main__":
    main()
