#!/usr/bin/env python3
"""Free-running token gate for decode-graph changes (#14).

Every continuation of the candidate run must equal one that the reference
engine itself produces (any of the --allowed runs, e.g. furiosa-llm one at a
time and all at once) or rngd-pd's own 16-wide baseline (--baseline); and the
candidate's agreement with --primary (furiosa-llm, sequential) must not fall
below the baseline's.

    python3 bench/token_gate.py --candidate bench/greedy_rngd_pd_tiers_S1024.jsonl \
        --baseline bench/greedy_rngd_pd_base_S1024.jsonl \
        --primary bench/greedy_furiosa_llm.jsonl \
        --allowed bench/greedy_furiosa_llm_together.jsonl

Texts are compared (furiosa-llm streams no token IDs). Exit 1 on failure.
"""
import argparse
import json
import sys


def load(path):
    return {tuple(r["prompt_ids"]): r for r in map(json.loads, open(path)) if not r.get("err")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--allowed", nargs="*", default=[])
    a = ap.parse_args()
    cand, base, prim = load(a.candidate), load(a.baseline), load(a.primary)
    allowed = [prim, base] + [load(p) for p in a.allowed]
    keys = [k for k in prim if k in cand and k in base]
    ok_all, agree_c, agree_b = True, 0, 0
    for k in keys:
        txt = cand[k]["text"]
        agree_c += txt == prim[k]["text"]
        agree_b += base[k]["text"] == prim[k]["text"]
        found = any(k in run and run[k]["text"] == txt for run in allowed)
        ok_all &= found
        if not found:
            print(f"#{cand[k]['idx']:>2} P={cand[k]['P']:>5} not produced by any allowed run: "
                  f"{txt[:60]!r}")
    ok = ok_all and agree_c >= agree_b
    print(f"agreement with primary: candidate {agree_c}/{len(keys)}, baseline {agree_b}/{len(keys)}")
    print(f"every continuation allowed: {ok_all}")
    print("GATE " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
