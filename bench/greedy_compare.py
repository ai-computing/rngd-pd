#!/usr/bin/env python3
"""Compare two `vendor_sweep.py --greedy` runs token by token.

    python3 bench/greedy_compare.py bench/greedy_rngd_pd_base_S1024.jsonl bench/greedy_rngd_pd_new.jsonl

Prompts are matched by their token IDs. For each pair it prints whether the
generated IDs are identical and, if not, the first position where they differ.
Exit status 1 if any shared prompt differs (use it as the gate for a change).
"""
import json
import sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[tuple(r["prompt_ids"])] = r
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    shared = [k for k in a if k in b]
    same = 0
    for k in shared:
        x, y = a[k], b[k]
        if x.get("err") or y.get("err"):
            print(f"#{x['idx']:>2} P={x['P']:>5}  error: {x.get('err') or y.get('err')}")
            continue
        ox, oy = x["out_ids"], y["out_ids"]
        if not ox or not oy:
            # furiosa-llm's stream carries text only: compare the text then
            ox, oy = x["text"], y["text"]
        if ox == oy:
            same += 1
            unit = "tokens" if isinstance(ox, list) else "chars, text only"
            print(f"#{x['idx']:>2} P={x['P']:>5}  identical ({len(ox)} {unit})")
        else:
            d = next((i for i, (p, q) in enumerate(zip(ox, oy)) if p != q), min(len(ox), len(oy)))
            unit = "token" if isinstance(ox, list) else "char"
            print(f"#{x['idx']:>2} P={x['P']:>5}  differs at {unit} {d}: "
                  f"{x['text'][:40]!r} vs {y['text'][:40]!r}")
    print(f"{same}/{len(shared)} identical")
    sys.exit(0 if same == len(shared) else 1)


if __name__ == "__main__":
    main()
