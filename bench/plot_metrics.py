#!/usr/bin/env python3
"""Metric-centric figures for the wide DUP vs PD sweep (measured data only).

  python3 bench/plot_metrics.py --in bench/sweep_wide512.jsonl [--in2 <s1024.jsonl>]

Figures -> bench/plots_metrics/:
  fig_tokens.png  : tok/s vs batch(n) | vs prompt P | vs decode D
  fig_ttft.png    : TTFT vs P | vs n
  fig_tpot.png    : TPOT vs P | vs D | vs n
  fig_util.png    : avg power + PE occupancy per card role, vs P and vs n
  fig_handoff.png : PD handoff gap vs P (KV size annotated)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"dup": "#2a78d6", "pd": "#eb6834", "pd2": "#f0a37e", "ink": "#0b0b0b",
     "sec": "#52514e", "mut": "#898781", "grid": "#e1e0d9", "base": "#c3c2b7",
     "surf": "#fcfcfb"}


def style(ax):
    ax.set_facecolor(C["surf"])
    for s in ax.spines.values():
        s.set_color(C["base"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=C["mut"], labelsize=8)
    ax.grid(True, color=C["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load(paths):
    rows = []
    for p in paths:
        if p and os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    out = []
    for r in rows:
        if r["point"] == "chat":
            continue
        P, D, n = (int(x) for x in r["point"].split(":"))
        r.update(P=P, D=D, n=n)
        out.append(r)
    return out


def card(r, which, idx, key):
    cards = r.get(f"{which}_cards") or {}
    ks = sorted(cards)
    return cards.get(ks[idx], {}).get(key) if len(ks) > idx else None


def series(rows, xkey, fixed):
    sel = [r for r in rows if all(r[k] == v for k, v in fixed.items())]
    sel = {r[xkey]: r for r in sel}          # dedupe (last wins)
    return [sel[x] for x in sorted(sel)]


def duo(ax, sel, xkey, dkey, pkey, scale=1.0, labels=("DUP", "PD")):
    xs = [r[xkey] for r in sel]
    ax.plot(xs, [r[dkey] * scale for r in sel], "-o", color=C["dup"], lw=2, ms=5,
            label=labels[0])
    ax.plot(xs, [r[pkey] * scale for r in sel], "-o", color=C["pd"], lw=2, ms=5,
            label=labels[1])
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench/sweep_wide512.jsonl")
    ap.add_argument("--in2", default="")
    ap.add_argument("--outdir", default="bench/plots_metrics")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = load([args.inp, args.in2])

    P0, D0, N0 = 128, 64, 8      # anchors
    by_n = series(rows, "n", {"P": P0, "D": D0})
    by_P = series(rows, "P", {"D": D0, "n": N0})
    by_D = series(rows, "D", {"P": P0, "n": N0})

    def save(fig, name):
        fig.tight_layout()
        p = os.path.join(args.outdir, name)
        fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=C["surf"])
        plt.close(fig)
        print("[saved]", p)

    def panel3(specs, name, suptitle):
        fig, axes = plt.subplots(1, len(specs), figsize=(4.5 * len(specs), 3.7),
                                 facecolor=C["surf"])
        if len(specs) == 1:
            axes = [axes]
        for ax, (sel, xkey, xlabel, dkey, pkey, scale, ylabel, xlog) in zip(axes, specs):
            style(ax)
            if not sel:
                continue
            xs = duo(ax, sel, xkey, dkey, pkey, scale)
            if xlog:
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs)
                ax.set_xticklabels(xs)
            ax.set_xlabel(xlabel, color=C["sec"], fontsize=9)
            ax.set_ylabel(ylabel, color=C["sec"], fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(suptitle, color=C["ink"], fontsize=11, weight="bold", y=1.03)
        save(fig, name)

    # ---- token/s ----
    panel3([
        (by_n, "n", f"batch (in-flight streams), P={P0} D={D0}",
         "dup_tok_s", "pd_tok_s", 1, "output tok/s (2 NPU)", True),
        (by_P, "P", f"prompt length P, D={D0} n={N0}",
         "dup_tok_s", "pd_tok_s", 1, "output tok/s (2 NPU)", False),
        (by_D, "D", f"decode length D, P={P0} n={N0}",
         "dup_tok_s", "pd_tok_s", 1, "output tok/s (2 NPU)", True),
    ], "fig_tokens.png",
       "Throughput — DUP vs PD(KV streamed) · Llama-3.1-8B 32L · 2×RNGD · S=512")

    # ---- TTFT ----
    for r in rows:
        r["pd_ttfp_s"] = r["pd_prefill_p50_s"] + r["pd_gap_ms_p50"] / 1e3
    panel3([
        (by_P, "P", f"prompt length P, D={D0} n={N0}",
         "dup_ttft_p50_s", "pd_ttfp_s", 1e3, "TTFT p50 (ms)", False),
        (by_n, "n", f"batch, P={P0} D={D0}",
         "dup_ttft_p50_s", "pd_ttfp_s", 1e3, "TTFT p50 (ms)", True),
        (by_D, "D", f"decode length D, P={P0} n={N0}",
         "dup_ttft_p50_s", "pd_ttfp_s", 1e3, "TTFT p50 (ms)", True),
    ], "fig_ttft.png",
       "TTFT (per-request service: prefill compute; PD adds streamed-KV tail)")

    # ---- TPOT ----
    panel3([
        (by_P, "P", f"prompt length P, D={D0} n={N0}",
         "dup_tpot_p50_ms", "pd_tpot_p50_ms", 1, "TPOT p50 (ms/token)", False),
        (by_D, "D", f"decode length D, P={P0} n={N0}",
         "dup_tpot_p50_ms", "pd_tpot_p50_ms", 1, "TPOT p50 (ms/token)", True),
        (by_n, "n", f"batch, P={P0} D={D0}",
         "dup_tpot_p50_ms", "pd_tpot_p50_ms", 1, "TPOT p50 (ms/token)", True),
    ], "fig_tpot.png",
       "TPOT — decode-chip token latency (PD decodes on streamed KV)")

    # ---- utilization ----
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), facecolor=C["surf"])
    for row_i, (sel, xkey, xlabel, xlog) in enumerate((
            (by_P, "P", f"prompt length P (D={D0}, n={N0})", False),
            (by_n, "n", f"batch (P={P0}, D={D0})", True))):
        for col_i, (key, ylabel) in enumerate((("avg_W", "avg power (W)"),
                                               ("pe_util_pct", "PE occupancy (%)"))):
            ax = axes[row_i][col_i]
            style(ax)
            xs = [r[xkey] for r in sel]
            ax.plot(xs, [card(r, "dup", 0, key) for r in sel], "-o",
                    color=C["dup"], lw=2, ms=5, label="DUP (chip0)")
            ax.plot(xs, [card(r, "pd", 0, key) for r in sel], "-o",
                    color=C["pd"], lw=2, ms=5, label="PD prefill chip")
            ax.plot(xs, [card(r, "pd", 1, key) for r in sel], "--o",
                    color=C["pd2"], lw=2, ms=5, label="PD decode chip")
            if xlog:
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs)
                ax.set_xticklabels(xs)
            ax.set_xlabel(xlabel, color=C["sec"], fontsize=9)
            ax.set_ylabel(ylabel, color=C["sec"], fontsize=9)
            ax.legend(fontsize=7.5)
    fig.suptitle("Utilization per card — power (left) and PE occupancy (right)",
                 color=C["ink"], fontsize=11, weight="bold", y=1.01)
    save(fig, "fig_util.png")

    # ---- handoff ----
    fig, ax = plt.subplots(figsize=(5.8, 3.8), facecolor=C["surf"])
    style(ax)
    xs = [r["P"] for r in by_P]
    gaps = [r["pd_gap_ms_p50"] for r in by_P]
    ax.plot(xs, gaps, "-o", color=C["pd"], lw=2, ms=5)
    for r in by_P:
        mb = 2 * 32 * 8 * 128 * r["P"] * 2 / 1e6
        ax.annotate(f"{mb:.0f} MB", (r["P"], r["pd_gap_ms_p50"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=C["mut"])
    ax.set_xlabel("prompt length P (KV bf16 size annotated)", color=C["sec"],
                  fontsize=9)
    ax.set_ylabel("handoff gap p50 (ms)", color=C["sec"], fontsize=9)
    ax.set_ylim(0, max(gaps) * 1.6)
    ax.set_title("PD KV handoff tail — flat in context (streaming hides growth)",
                 color=C["ink"], fontsize=10, weight="bold")
    save(fig, "fig_handoff.png")


if __name__ == "__main__":
    main()
