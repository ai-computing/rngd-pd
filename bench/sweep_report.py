#!/usr/bin/env python3
"""Turn the llm-d sweep into the tables worth reading.

  python3 bench/sweep_report.py bench/sweep_llmd.jsonl
"""
import json
import sys
from collections import defaultdict

ORDER = ["rngd_alone", "pd_forward", "pd_reverse", "atom_alone"]
LABEL = {"rngd_alone": "RNGD → RNGD", "pd_forward": "RNGD → ATOM",
         "pd_reverse": "ATOM → RNGD", "atom_alone": "ATOM → ATOM"}


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return sorted(rows, key=lambda r: (ORDER.index(r["config"]), r["B"], r["P"]))


def table(rows, title, cols, fmt):
    print(f"\n{title}")
    head = "  {:<12s} {:>2s} {:>4s} " + " ".join(f"{{:>{w}s}}" for _, w in cols)
    print(head.format("config", "B", "P", *[c for c, _ in cols]))
    print("  " + "-" * (12 + 1 + 2 + 1 + 4 + 1 + sum(w + 1 for _, w in cols)))
    for r in rows:
        vals = [fmt(r, c) for c, _ in cols]
        line = "  {:<12s} {:>2d} {:>4d} " + " ".join(f"{{:>{w}s}}" for _, w in cols)
        print(line.format(LABEL[r["config"]], r["B"], r["P"], *vals))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/sweep_llmd.jsonl"
    rows = load(path)

    def f(r, c):
        v = r.get(c)
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
        return str(v)

    table(rows, "지연 · 처리량 (게이트웨이 경유)",
          [("ttft_ms", 8), ("tpot_ms", 8), ("tokens_s", 9),
           ("prefill_ms", 10), ("handoff_ms", 10)], f)
    table(rows, "전력 · 활용도 · 대역폭",
          [("rngd_power_w", 12), ("rngd_util_pct", 13), ("atom_power_w", 12),
           ("atom_util_pct", 13), ("decode_bw_gbs", 13), ("prefill_bw_gbs", 14)], f)

    print("\n배치 확장 (ATOM 디코드)")
    by = defaultdict(dict)
    for r in rows:
        by[(r["config"], r["P"])][r["B"]] = r
    print("  {:<12s} {:>4s} {:>9s} {:>9s} {:>9s} {:>8s}".format(
        "config", "P", "B=1", "B=2", "B=4", "4/1"))
    for (cfg, P), d in sorted(by.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        if len(d) < 3:
            continue
        t = [d[b]["tokens_s"] for b in (1, 2, 4)]
        print("  {:<12s} {:>4d} {:>9.2f} {:>9.2f} {:>9.2f} {:>7.2f}x".format(
            LABEL[cfg], P, *t, t[2] / t[0]))

    print("\n프리필 토큰당 비용 (ms/token)")
    print("  {:<12s} " .format("config") + " ".join(f"{'P='+str(p):>9s}"
          for p in sorted({r['P'] for r in rows})))
    for cfg in ORDER:
        sel = {r["P"]: r for r in rows if r["config"] == cfg and r["B"] == 1}
        if not sel:
            continue
        print("  {:<12s} ".format(LABEL[cfg]) + " ".join(
            f"{sel[p]['prefill_ms_per_token']:>9.3f}" if p in sel else f"{'-':>9s}"
            for p in sorted({r['P'] for r in rows})))


if __name__ == "__main__":
    main()
