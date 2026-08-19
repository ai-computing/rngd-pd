#!/usr/bin/env python3
"""Paper figures for the engine-class DUP vs PD sweep (real KV streaming).

  python3 bench/plot_sweep.py --in bench/sweep256_full.jsonl [--in2 sweep512.jsonl]
Figures -> bench/plots/:
  fig_ratio_theory.png : measured PD/DUP vs prefill fraction f, queueing-model overlay
  fig_context.png      : throughput + handoff gap vs prompt length (per S bucket)
  fig_load.png         : throughput vs in-flight n
  fig_chat.png         : chat-mix workload — DUP vs PD bars (tok/s, TTFT p50)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"dup": "#2a78d6", "pd": "#eb6834", "ideal": "#1baf7a", "ink": "#0b0b0b",
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
    for r in rows:
        if r["point"] != "chat":
            P, D, n = (int(x) for x in r["point"].split(":"))
            r.update(P=P, D=D, n=n, f=P / (P + D))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench/sweep256_full.jsonl")
    ap.add_argument("--in2", default="")
    ap.add_argument("--outdir", default="bench/plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = load([args.inp, args.in2])
    num = [r for r in rows if r["point"] != "chat"]
    chat = [r for r in rows if r["point"] == "chat"]

    def save(fig, name):
        fig.tight_layout()
        p = os.path.join(args.outdir, name)
        fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=C["surf"])
        plt.close(fig)
        print("[saved]", p)

    # ---- Fig 1: PD/DUP vs f with queueing-model overlay ----
    pts = [r for r in num if r["n"] == 8]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=C["surf"])
    style(ax)
    f = np.linspace(0.02, 0.98, 300)
    ax.plot(f, 1.0 / (2 * np.maximum(f, 1 - f)), color=C["ink"], lw=1.6, ls="--",
            label="queueing model  (Tp+Td)/(2·max(Tp,Td))")
    for S, mk in ((256, "o"), (512, "s")):
        sel = [r for r in pts if r["S"] == S]
        if not sel:
            continue
        ax.scatter([r["f"] for r in sel],
                   [r["pd_tok_s"] / r["dup_tok_s"] for r in sel],
                   s=64, color=C["pd"], marker=mk, zorder=5,
                   edgecolor=C["surf"], lw=1.2,
                   label=f"measured (S={S}, real KV streaming)")
        for r in sel:
            ax.annotate(f"P{r['P']}:D{r['D']}", (r["f"], r["pd_tok_s"] / r["dup_tok_s"]),
                        xytext=(4, 5), textcoords="offset points", fontsize=7,
                        color=C["sec"])
    ax.axhline(1.0, color=C["base"], lw=1)
    ax.set_xlabel("prefill fraction  f = Tp/(Tp+Td) ≈ P/(P+D)", color=C["sec"], fontsize=9)
    ax.set_ylabel("PD / DUP  aggregate throughput", color=C["sec"], fontsize=9)
    ax.set_title("Where PD stands without decode batching — measurement vs theory\n"
                 "(Llama-3.1-8B 32L, 2×RNGD, per-request KV streamed chip0→chip1)",
                 color=C["ink"], fontsize=10, weight="bold")
    ax.legend(fontsize=8, loc="lower center", framealpha=0.95)
    ax.set_ylim(0.4, 1.1)
    save(fig, "fig_ratio_theory.png")

    # ---- Fig 2: context sweep (D=64, n=8) ----
    sel = sorted((r for r in num if r["D"] == 64 and r["n"] == 8), key=lambda r: r["P"])
    if sel:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), facecolor=C["surf"])
        for ax in (a1, a2):
            style(ax)
        Ps = [r["P"] for r in sel]
        a1.plot(Ps, [r["dup_tok_s"] for r in sel], "-o", color=C["dup"], lw=2, ms=5,
                label="DUP")
        a1.plot(Ps, [r["pd_tok_s"] for r in sel], "-o", color=C["pd"], lw=2, ms=5,
                label="PD (KV streamed)")
        a1.set_xlabel("prompt length P (tokens)", color=C["sec"], fontsize=9)
        a1.set_ylabel("aggregate tok/s (2 NPU)", color=C["sec"], fontsize=9)
        a1.set_title("Throughput vs context", color=C["ink"], fontsize=10, weight="bold")
        a1.legend(fontsize=8)
        gaps = [r["pd_gap_ms_p50"] for r in sel]
        kv_mb = [2 * 32 * 8 * 128 * r["P"] * 2 / 1e6 for r in sel]
        a2.plot(Ps, gaps, "-o", color=C["pd"], lw=2, ms=5, label="handoff gap (streamed)")
        for x, g, m in zip(Ps, gaps, kv_mb):
            a2.annotate(f"{m:.0f} MB KV", (x, g), xytext=(0, 8),
                        textcoords="offset points", fontsize=7, ha="center",
                        color=C["mut"])
        a2.set_xlabel("prompt length P (tokens)", color=C["sec"], fontsize=9)
        a2.set_ylabel("prefill-done → decode-ready (ms)", color=C["sec"], fontsize=9)
        a2.set_title("KV handoff tail vs context", color=C["ink"], fontsize=10,
                     weight="bold")
        a2.set_ylim(0, max(gaps) * 1.5)
        a2.legend(fontsize=8)
        save(fig, "fig_context.png")

    # ---- Fig 3: load sweep (P=128, D=64) ----
    sel = sorted((r for r in num if r["P"] == 128 and r["D"] == 64),
                 key=lambda r: r["n"])
    if len(sel) > 1:
        fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor=C["surf"])
        style(ax)
        ns = [r["n"] for r in sel]
        ax.plot(ns, [r["dup_tok_s"] for r in sel], "-o", color=C["dup"], lw=2, ms=5,
                label="DUP")
        ax.plot(ns, [r["pd_tok_s"] for r in sel], "-o", color=C["pd"], lw=2, ms=5,
                label="PD (KV streamed)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels(ns)
        ax.set_xlabel("requests in flight n", color=C["sec"], fontsize=9)
        ax.set_ylabel("aggregate tok/s (2 NPU)", color=C["sec"], fontsize=9)
        ax.set_title("Pipeline saturation (P=128, D=64)\nno decode batching: "
                     "sequential streams saturate both configs early",
                     color=C["ink"], fontsize=10, weight="bold")
        ax.legend(fontsize=8)
        save(fig, "fig_load.png")

    # ---- Fig 3b: TPOT + utilization vs context (D=64, n=8) ----
    sel = sorted((r for r in num if r["D"] == 64 and r["n"] == 8
                  and "dup_tpot_p50_ms" in r), key=lambda r: r["P"])
    if sel:
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.2, 3.8),
                                         facecolor=C["surf"])
        for ax in (a1, a2, a3):
            style(ax)
        Ps = [r["P"] for r in sel]
        a1.plot(Ps, [r["dup_tpot_p50_ms"] for r in sel], "-o", color=C["dup"],
                lw=2, ms=5, label="DUP")
        a1.plot(Ps, [r["pd_tpot_p50_ms"] for r in sel], "-o", color=C["pd"],
                lw=2, ms=5, label="PD decode chip")
        a1.set_xlabel("prompt length P", color=C["sec"], fontsize=9)
        a1.set_ylabel("TPOT p50 (ms/token)", color=C["sec"], fontsize=9)
        a1.set_title("TPOT vs context", color=C["ink"], fontsize=10, weight="bold")
        a1.legend(fontsize=8)

        def card(r, which, chip_idx, key):
            cards = r.get(f"{which}_cards") or {}
            ks = sorted(cards)
            return cards.get(ks[chip_idx], {}).get(key) if len(ks) > chip_idx else None
        a2.plot(Ps, [card(r, "dup", 0, "avg_W") for r in sel], "-o",
                color=C["dup"], lw=2, ms=5, label="DUP chip0")
        a2.plot(Ps, [card(r, "pd", 0, "avg_W") for r in sel], "-o",
                color=C["pd"], lw=2, ms=5, label="PD prefill chip")
        a2.plot(Ps, [card(r, "pd", 1, "avg_W") for r in sel], "--o",
                color=C["pd"], lw=2, ms=5, alpha=0.6, label="PD decode chip")
        a2.set_xlabel("prompt length P", color=C["sec"], fontsize=9)
        a2.set_ylabel("avg power (W)", color=C["sec"], fontsize=9)
        a2.set_title("Power per card", color=C["ink"], fontsize=10, weight="bold")
        a2.legend(fontsize=7.5)
        a3.plot(Ps, [card(r, "dup", 0, "pe_util_pct") for r in sel], "-o",
                color=C["dup"], lw=2, ms=5, label="DUP chip0")
        a3.plot(Ps, [card(r, "pd", 0, "pe_util_pct") for r in sel], "-o",
                color=C["pd"], lw=2, ms=5, label="PD prefill chip")
        a3.plot(Ps, [card(r, "pd", 1, "pe_util_pct") for r in sel], "--o",
                color=C["pd"], lw=2, ms=5, alpha=0.6, label="PD decode chip")
        a3.set_xlabel("prompt length P", color=C["sec"], fontsize=9)
        a3.set_ylabel("PE occupancy (%)", color=C["sec"], fontsize=9)
        a3.set_title("PE utilization per card", color=C["ink"], fontsize=10,
                     weight="bold")
        a3.legend(fontsize=7.5)
        save(fig, "fig_tpot_util.png")

    # ---- Fig 4: chat mix ----
    if chat:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.8), facecolor=C["surf"])
        for ax in (a1, a2):
            style(ax)
        labels = [f"S={r['S']}" for r in chat]
        x = np.arange(len(chat))
        w = 0.34
        a1.bar(x - w / 2, [r["dup_tok_s"] for r in chat], w, color=C["dup"], label="DUP")
        a1.bar(x + w / 2, [r["pd_tok_s"] for r in chat], w, color=C["pd"],
               label="PD (KV streamed)")
        for xi, r in zip(x, chat):
            a1.annotate(f"{r['pd_tok_s']/r['dup_tok_s']:.2f}×",
                        (xi + w / 2, r["pd_tok_s"]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=C["sec"])
        a1.set_xticks(x)
        a1.set_xticklabels(labels)
        a1.set_ylabel("aggregate tok/s", color=C["sec"], fontsize=9)
        a1.set_title("Chat mix — throughput", color=C["ink"], fontsize=10, weight="bold")
        a1.legend(fontsize=8)
        for xi, r in zip(x, chat):
            tt = sorted(r["dup_ttfts"])
            a2.bar(xi - w / 2, tt[len(tt) // 2] * 1e3, w, color=C["dup"])
            a2.bar(xi + w / 2, r["pd_prefill_p50_s"] * 1e3 + r["pd_gap_ms_p50"], w,
                   color=C["pd"])
        a2.set_xticks(x)
        a2.set_xticklabels(labels)
        a2.set_ylabel("time to decode-start p50 (ms)", color=C["sec"], fontsize=9)
        a2.set_title("Chat mix — first-token path", color=C["ink"], fontsize=10,
                     weight="bold")
        save(fig, "fig_chat.png")


if __name__ == "__main__":
    main()
