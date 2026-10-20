#!/usr/bin/env python3
"""Render the figures for the KPI report.

Five charts, each answering one question the tables make you work for:
what the four PD pairings cost in latency, what they return in throughput,
where power efficiency actually comes from, how far each stage sits from its
own ceiling, and why batching stops paying at four.

Colour is assigned by entity, in a fixed order, and never cycled -- a chart
that drops a series must not repaint the survivors. The palette is the
validated five-slot categorical set; three of its light-mode slots sit under
3:1 on the surface, so every bar carries a visible direct label rather than
relying on the fill to be readable. No chart here has two y-scales.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from kpi import kpi_row

OUT = Path("~/workspace/01_pd_disag/report/figs").expanduser()
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"

# fixed slot order -- the entity owns the hue for the life of the report
C = {"rngd_alone": "#2a78d6", "pd_forward": "#eb6834",
     "pd_reverse": "#1baf7a", "atom_alone": "#eda100", "solo": "#e87ba4"}
LABEL = {"rngd_alone": "RNGD→RNGD", "pd_forward": "RNGD→ATOM",
         "pd_reverse": "ATOM→RNGD", "atom_alone": "ATOM→ATOM", "solo": "solo (1 card)"}
ORDER = ["rngd_alone", "pd_forward", "pd_reverse", "atom_alone"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "svg.fonttype": "none",
})


def load(f):
    p = Path("~/workspace/01_pd_disag/rngd_llmd").expanduser() / f
    return [kpi_row(json.loads(l)) for l in open(p)]


def pick(rows, **kw):
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return out[0] if len(out) == 1 else out


def rbar(ax, y, w, h, color, horiz=True):
    """One bar, anchored to the baseline.

    Rounded data-ends went out: FancyBboxPatch rounds in data units, so on an
    axis where x spans 1,500 and y spans 4 the corner radius came out as a
    bulge wider than the bar. A square end that is the right length beats a
    rounded one that lies about the value.
    """
    if horiz:
        ax.barh(y, w, height=h, color=color, zorder=3, linewidth=0)
    else:
        ax.bar(y, w, width=h, color=color, zorder=3, linewidth=0)


def tidy(ax, xlab=None, ylab=None, xgrid=True):
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(axis="y" if xgrid else "x", visible=False)
    ax.set_axisbelow(True)
    if xlab: ax.set_xlabel(xlab, fontsize=9, labelpad=7)
    if ylab: ax.set_ylabel(ylab, fontsize=9, labelpad=7)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def header(fig, t, sub, top=0.80):
    """Figure-level title block with its own reserved band.

    suptitle at y>1 plus bbox_inches="tight" was the collision: tight crops to
    the artists it finds and then the two lines land on each other. Reserving
    the band inside the figure and leaving the crop alone keeps them apart at
    any figure size.
    """
    fig.text(0.01, 0.985, t, ha="left", va="top", fontsize=13,
             fontweight="700", color=INK)
    fig.text(0.01, 0.905, sub, ha="left", va="top", fontsize=9, color=INK3)
    fig.tight_layout(rect=(0, 0, 1, top))


def title(ax, t, sub=None):
    # The subtitle sits just above the axes and the title clears it by pad.
    # Both were floating near the same y before, which read as one struck-out
    # line rather than two.
    ax.set_title(t, loc="left", fontsize=11.5, fontweight="600", color=INK,
                 pad=26 if sub else 8)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=8.8,
                color=INK3, va="bottom")


# ---------------------------------------------------------------- fig 1
def fig_latency(sw):
    """Which vendor prefills sets TTFT; which vendor decodes sets TPOT."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key, lab, note in (
            (axes[0], "ttft_ms", "TTFT (ms)  — time to first token", "set by the PREFILL card"),
            (axes[1], "tpot_ms", "TPOT (ms)  — per output token", "set by the DECODE card")):
        ys = range(len(ORDER))
        for i, cfg in enumerate(ORDER):
            r = pick(sw, config=cfg, B=4, P=448)
            v = r[key]
            rbar(ax, i, v, 0.52, C[cfg])
            ax.text(v * 1.02, i, f"{v:,.0f}", va="center", ha="left",
                    fontsize=9, color=INK2, fontweight="600")
        ax.set_yticks(list(ys)); ax.set_yticklabels([LABEL[c] for c in ORDER])
        ax.invert_yaxis()
        hi = max(pick(sw, config=c, B=4, P=448)[key] for c in ORDER)
        ax.set_xlim(0, (170 if key == "tpot_ms" else hi * 1.22))
        tidy(ax, xlab=lab)
        title(ax, lab.split("—")[0].strip(), note)
    axes[1].axvline(150, color="#8E2029", lw=1.4, ls=(0, (4, 3)), zorder=2)
    axes[1].text(148, 3.45, "SLO 150 ms", fontsize=8.5, color="#8E2029",
                 ha="right", va="bottom")
    header(fig, "Prefill and decode are set by different cards",
           "Llama-3.1-8B · S=512 · prompt 448 tok · batch 4 · through the llm-d gateway",
           top=0.84)
    fig.savefig(OUT / "fig1_latency.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_throughput(sw, solo):
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    items = [(c, pick(sw, config=c, B=1, P=448), pick(sw, config=c, B=4, P=448)) for c in ORDER]
    items.append(("solo", None, pick(solo, config="solo", B=4, P=448, D=16)))
    for i, (cfg, r1, r4) in enumerate(items):
        if r1:
            rbar(ax, i - 0.17, r1["tokens_s"], 0.3, C[cfg] + "66")
            ax.text(r1["tokens_s"] + 1.5, i - 0.17, f"{r1['tokens_s']:.1f}", va="center",
                    fontsize=8.5, color=INK3)
        rbar(ax, i + 0.17, r4["tokens_s"], 0.3, C[cfg])
        ax.text(r4["tokens_s"] + 1.5, i + 0.17, f"{r4['tokens_s']:.1f}", va="center",
                fontsize=9, color=INK2, fontweight="600")
    # The card count belongs in the row's name, not floating left of the axis
    # where it landed on the tick labels.
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{LABEL[c].split(' (')[0]}\n{r4['cards']} card{'s' if r4['cards'] > 1 else ''}"
                        for c, _, r4 in items], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    tidy(ax, xlab="output tokens / s     (light bar = batch 1, solid = batch 4)")
    header(fig, "Throughput — and what it cost in cards",
           "S=512 · prompt 448 tok · 16 output tokens. solo holds one card; every other row holds two.",
           top=0.85)
    fig.savefig(OUT / "fig2_throughput.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_power(shape, solo):
    """The KPI chart: efficiency is a function of the workload's shape."""
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    series = [("solo", solo), ("rngd_alone", shape), ("atom_alone", shape)]
    Ds = [16, 8, 4]
    for cfg, src in series:
        pts = [pick(src, config=cfg, B=4, P=448, D=d) for d in Ds]
        xs = [448 / d for d in Ds]
        ys = [p["gflops_per_w"] for p in pts]
        ax.plot(xs, ys, "-o", color=C[cfg], lw=2, ms=8, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2)
        # nudge the labels that would otherwise sit on a target rule and read
        # as struck through: RNGD lands on 341, solo lands on 1029
        dy = {"rngd_alone": -62, "solo": -78}.get(cfg, 0)
        ax.text(xs[-1] * 1.09, ys[-1] + dy, f"{LABEL[cfg]}  {ys[-1]:,.0f}",
                fontsize=9, color=C[cfg] if cfg != "atom_alone" else "#9A5415",
                fontweight="600", va="center")
    for y, lab, col in ((341, "2026 target  341", "#0F6E5C"),
                        (512, "2027 target  512", "#8a8983"),
                        (1029, "2030 target  1029", "#8a8983")):
        ax.axhline(y, color=col, lw=1.3, ls=(0, (4, 3)), zorder=2)
        ax.text(25.2, y + 22, lab, fontsize=8.5, color=col, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([28, 56, 112])
    ax.set_xticklabels(["28\n(D=16)", "56\n(D=8)", "112\n(D=4)"])
    # A log axis decorates itself with 3x10^1, 4x10^1 ... which landed on top
    # of the three ticks that actually mean something here.
    ax.minorticks_off()
    ax.set_xlim(24, 300); ax.set_ylim(0, 1200)
    tidy(ax, xlab="prompt : generated  token ratio  (P/D)",
         ylab="GFLOPS / Watt  (measured power)", xgrid=False)
    header(fig, "Power efficiency is a property of the workload, not the stack",
           "S=512 · prompt 448 tok · batch 4. A chat-shaped trace sits near P/D ≈ 0.09 — far left of this axis.",
           top=0.86)
    fig.savefig(OUT / "fig3_power.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_mfu(sw, solo):
    """Each stage against its own ceiling, so 'low MFU' stops being one number."""
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    rows = [(c, pick(sw, config=c, B=4, P=448)) for c in ORDER]
    rows.append(("solo", pick(solo, config="solo", B=4, P=448, D=16)))
    x = range(len(rows))
    w = 0.26
    for j, (key, lab, alpha) in enumerate((("prefill_mfu_pct", "prefill stage", 1.0),
                                           ("decode_mfu_pct", "decode stage", 0.45),
                                           ("system_mfu_pct", "whole system", 0.72))):
        for i, (cfg, r) in enumerate(rows):
            v = r[key]
            col = C[cfg] + ("" if alpha == 1.0 else f"{int(alpha*255):02x}")
            rbar(ax, i + (j - 1) * w, v, w * 0.86, col, horiz=False)
            ax.text(i + (j - 1) * w, v + 1.1, f"{v:.1f}", ha="center", fontsize=8,
                    color=INK2, fontweight="600" if j == 2 else "400")
    ax.axhline(20, color="#0F6E5C", lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.text(len(rows) - 0.42, 21, "2026 target 20%", fontsize=8.5, color="#0F6E5C", ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels([LABEL[c] for c, _ in rows], fontsize=9)
    ax.set_ylim(0, 50)
    tidy(ax, ylab="Model FLOPs Utilisation  (%)", xgrid=False)
    ax.text(0.0, -0.22, "left → right per group:  prefill stage · decode stage · whole system",
            transform=ax.transAxes, fontsize=8.5, color=INK3)
    header(fig, "Where the efficiency goes",
           "Decode is bandwidth-bound: its ceiling at batch 4 is 2.19%, so a two-card system is dragged down by the card that idles.",
           top=0.85)
    fig.savefig(OUT / "fig4_mfu.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_batch():
    """Emphasis form: one series, and the point is where it breaks."""
    Bs = [1, 4, 8]
    bw = [1049.7, 823.6, 220.9]
    tok = [15.30, 4.88, 9.09]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axes[0]
    for i, (b, v) in enumerate(zip(Bs, bw)):
        col = C["rngd_alone"] if b <= 4 else "#8E2029"
        rbar(ax, i, v, 0.5, col, horiz=False)
        ax.text(i, v + 32, f"{v:,.0f}", ha="center", fontsize=9, color=INK2, fontweight="600")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"B={b}" for b in Bs])
    tidy(ax, ylab="decode bandwidth (GB/s)", xgrid=False)
    ax.axhline(1500, color=INK3, lw=1.2, ls=(0, (4, 3)))
    ax.text(2.45, 1520, "HBM peak 1.5 TB/s", fontsize=8.2, color=INK3, ha="right", va="bottom")
    ax.set_ylim(0, 1750)
    title(ax, "Batch 8 collapses the weight stream")

    ax = axes[1]
    for i, (b, v) in enumerate(zip(Bs, tok)):
        col = C["rngd_alone"] if b <= 4 else "#8E2029"
        rbar(ax, i, v, 0.5, col, horiz=False)
        ax.text(i, v + 0.32, f"{v:.2f}", ha="center", fontsize=9, color=INK2, fontweight="600")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"B={b}" for b in Bs])
    ax.set_ylim(0, 18); tidy(ax, ylab="ms per output token  (lower is better)", xgrid=False)
    title(ax, "so per-token cost goes back up")

    header(fig, "Batching stops paying at four",
           "RNGD · S=256 · prompt 224 tok. B=8 and B=16 graphs both compile; B=16 never completes a request.",
           top=0.84)
    fig.savefig(OUT / "fig5_batch.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig_pd_vs_solo():
    """Does a dedicated prefill card ever pay? Across the reachable envelope, no.

    The two limits explain the shape. As generation grows, PD converges on
    solo rather than passing it: in the decode-dominated limit both are one
    card running the same kernel, and the prefill card contributes nothing to
    steady-state throughput. As the prompt grows, PD should win -- two cards
    both busy -- but the KV handoff is a fixed 237 ms a batch and there are
    too few decode steps to amortise it, because S=512 forces D small whenever
    P is large. The window needs P and D both large, and the compile ceiling
    forbids it.
    """
    import json
    rows = []
    for f in ("pd_vs_solo_balanced", "pd_vs_solo_mid",
              "pd_vs_solo_longgen", "pd_vs_solo_decodeheavy"):
        p = Path("~/workspace/01_pd_disag/rngd_llmd/bench").expanduser() / f"{f}.jsonl"
        rows += [json.loads(l) for l in open(p)]

    pts = {}
    for r in rows:
        if r["N"] not in (4, 8) or r["errors"]:
            continue
        pts.setdefault((r["P"], r["D"], r["N"]), {})[r["config"]] = r
    shapes = sorted({(P, D) for P, D, _ in pts}, key=lambda t: t[0] / t[1])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax = axes[0]
    for N, mk, al in ((4, "o", 0.55), (8, "s", 1.0)):
        xs, ys = [], []
        for P, D in shapes:
            d = pts.get((P, D, N), {})
            if "solo" in d and "rngd_alone" in d:
                xs.append(P / D)
                ys.append(d["solo"]["tokens_s"] / d["rngd_alone"]["tokens_s"])
        ax.plot(xs, ys, mk + "-", color=C["solo"], lw=2, ms=9, alpha=al,
                markeredgecolor=SURFACE, markeredgewidth=2, label=f"N={N} in flight")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points",
                        xytext=(0, 13 if N == 8 else -21), ha="center",
                        fontsize=8.5, color=INK2, fontweight="600")
    ax.axhline(1.0, color="#0F6E5C", lw=1.5, ls=(0, (4, 3)), zorder=2)
    # keep the rule's caption clear of the value labels clustered near 1.0
    ax.text(105, 1.02, "parity — below this line PD wins", fontsize=8.6,
            color="#0F6E5C", va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([0.033, 0.167, 1.07, 112])
    ax.set_xticklabels(["0.03\n16/480", "0.17\n64/384", "1.07\n256/240", "112\n448/4"])
    ax.minorticks_off()
    ax.set_ylim(0.88, 2.0)
    tidy(ax, xlab="prompt : generated  token ratio   (P / D)",
         ylab="solo throughput ÷ PD throughput", xgrid=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    title(ax, "PD never crosses parity", "one card beats or ties two, everywhere we can compile")

    ax = axes[1]
    for cfg, mk in (("solo", "o"), ("rngd_alone", "s")):
        xs, ys = [], []
        for P, D in shapes:
            d = pts.get((P, D, 8), {})
            if cfg in d:
                xs.append(P / D); ys.append(d[cfg]["ttft_p50"])
        ax.plot(xs, ys, mk + "-", color=C[cfg], lw=2, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label="solo (1 card)" if cfg == "solo" else "PD (2 cards)")
    ax.set_xscale("log")
    ax.set_xticks([0.033, 0.167, 1.07, 112])
    ax.set_xticklabels(["0.03", "0.17", "1.07", "112"])
    ax.minorticks_off()
    ax.set_ylim(0, 600)
    tidy(ax, xlab="prompt : generated  token ratio   (P / D)",
         ylab="TTFT p50  (ms)", xgrid=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    title(ax, "and it pays a KV handoff for the privilege",
          "237 ms a batch across /dev/shm, on every request")

    header(fig, "Does a dedicated prefill card ever pay?",
           "RNGD · S=512 · batch 4 · sustained closed loop, both configurations on the same cards and the same decode kernel.",
           top=0.83)
    fig.savefig(OUT / "fig6_pd_vs_solo.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- fig 7
def fig_kv_overlap():
    """What the KV handoff actually costs once you separate overlap from total.

    The sweep's handoff_ms adds the save and the load and reports ~237 ms,
    but almost all of the load happens inside the prefetch while the other
    card is still prefilling. Timing the blocking collection separately shows
    the real critical path, and then shows two fixes moving it.
    """
    stages = [
        ("as measured\nas-is", 16.1, 0.0, 157.2, 679.7),
        ("host work out\nof device lock", 9.9, 62.1, 36.2, 626.4),
        ("cast moved into\nthe prefetch", 25.6, 10.5, 42.9, 583.8),
    ]
    parts = [("kv_wait — shm→host residual", C["rngd_alone"]),
             ("prep — stack + bf16 cast", C["pd_forward"]),
             ("upload — host→card, inside lock", C["pd_reverse"])]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axes[0]
    for i, (lab, w, p, u, _wall) in enumerate(stages):
        left = 0.0
        for v, (_pl, col) in zip((w, p, u), parts):
            if v <= 0:
                continue
            ax.barh(i, v, left=left, height=0.5, color=col, zorder=3,
                    linewidth=1.6, edgecolor=SURFACE)   # 2px surface gap
            if v > 12:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8.8, color="#fcfcfb", fontweight="700")
            left += v
        ax.text(left + 4, i, f"{left:.0f} ms", va="center", fontsize=9.5,
                color=INK2, fontweight="700")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in stages], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 340)
    tidy(ax, xlab="critical-path KV cost per batch (ms)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    # bottom-right: the only quadrant no bar or total label reaches
    ax.legend(handles, [p for p, _ in parts], frameon=False, fontsize=8.4,
              loc="lower right", bbox_to_anchor=(1.0, -0.02))
    title(ax, "2.2× off the critical path",
          "prep was inside the upload before it was split out")

    ax = axes[1]
    walls = [s[4] for s in stages]
    for i, (lab, *_rest) in enumerate(stages):
        ax.barh(i, walls[i], height=0.5, color=C["solo"], zorder=3)
        ax.text(walls[i] + 8, i, f"{walls[i]:.0f}", va="center",
                fontsize=9.5, color=INK2, fontweight="700")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in stages], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 790)
    tidy(ax, xlab="request wall (ms) — P=448, D=4, B=4")
    title(ax, "…which is 14% of the request",
          "the handoff was never 237 ms; it was 173, and is now 79")

    header(fig, "Was the KV transfer overlapped? Half of it.",
           "RNGD→RNGD · S=512 · batch 4. shm→host already hid behind prefill; host→card did not, and sat inside the device lock.",
           top=0.83)
    fig.savefig(OUT / "fig7_kv_overlap.png", dpi=200)
    plt.close(fig)



# ---------------------------------------------------------------- fig 8
def fig_continuous():
    """Iteration-level scheduling, once admission stopped costing a round trip.

    The scheduler was written months ago and shelved because it lost to the
    static batcher. The reason was admission: inserting one stream downloaded
    the whole batch, scattered it on the host and uploaded all of it back.
    Writing the lane in place instead took that from 112 ms to 13.9 ms, which
    changes what the schedule is allowed to do -- and what it buys is time to
    first token, not throughput.
    """
    cases = [("P=448 D=4\nN=4", 25.5, 31.6, 290.0, 365.0),
             ("P=448 D=4\nN=8", 35.3, 31.9, 518.6, 860.6),
             ("P=64 D=384\nN=4", 171.4, 174.9, 283.4, 189.8),
             ("P=64 D=384\nN=8", 174.6, 160.8, 398.4, 122.5)]
    OLD, NEW = "#9fc3ee", C["rngd_alone"]        # one hue, two shades

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, idx, lab, hi, good_down in (
            (axes[0], (3, 4), "TTFT p50 (ms)", 980, True),
            (axes[1], (1, 2), "output tokens / s", 205, False)):
        for i, c in enumerate(cases):
            a, b = c[idx[0]], c[idx[1]]
            ax.plot([a, b], [i, i], "-", color=INK3, lw=1.6, zorder=2)
            ax.plot([a], [i], "o", ms=9, color=OLD, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.plot([b], [i], "o", ms=10, color=NEW, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            win = (b < a) if good_down else (b > a)
            pct = (b / a - 1) * 100
            ax.annotate(f"{pct:+.0f}%", ((a + b) / 2, i), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="700",
                        color="#0F6E5C" if win else "#8E2029")
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels([c[0] for c in cases], fontsize=8.8)
        # headroom above the first row: the +/-% labels sit 12pt up and were
        # landing on the axes subtitle
        ax.set_ylim(-0.75, len(cases) - 0.35)
        ax.invert_yaxis(); ax.set_xlim(0, hi)
        tidy(ax, xlab=lab)
    h = [plt.Line2D([], [], marker="o", ls="", ms=9, color=OLD,
                    markeredgecolor=SURFACE, markeredgewidth=2),
         plt.Line2D([], [], marker="o", ls="", ms=10, color=NEW,
                    markeredgecolor=SURFACE, markeredgewidth=2)]
    axes[0].legend(h, ["static batcher", "continuous (per-lane admit)"],
                   frameon=False, fontsize=8.8, loc="lower right")
    title(axes[0], "What it buys: time to first token",
          "a request joins an in-flight batch instead of waiting for it to drain")
    title(axes[1], "What it costs: a little throughput",
          "steps run 79% full — a step costs the same carrying one stream or four")

    header(fig, "Continuous batching, reconsidered",
           "RNGD→RNGD · S=512 · batch 4. Per-lane admission took the composition change from 112 ms to 13.9 ms.",
           top=0.83)
    fig.savefig(OUT / "fig8_continuous.png", dpi=200)
    plt.close(fig)



# ---------------------------------------------------------------- fig 9
def fig_fair_continuous():
    """The comparison with both sides on the same scheduler.

    The earlier continuous run put PD on the new scheduler and left the
    one-card control on the static batcher, which is the same shape of unfair
    comparison as the mismatched decode kernel. Solo runs it now too -- its
    prefill is on the card the lane lives on, so admission is a device-to-
    device slice write and costs 3.6 ms against PD's 13.9.
    """
    ratios = [("P=64 D=384\nN=8", 178.3 / 174.6, 180.1 / 177.9),
              ("P=448 D=4\nN=8", 47.9 / 35.3, 47.8 / 27.7)]
    OLD, NEW = "#9fc3ee", C["rngd_alone"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axes[0]
    w = 0.34
    for i, (lab, st, co) in enumerate(ratios):
        for j, (v, col, nm) in enumerate(((st, OLD, "static ↔ static"),
                                          (co, NEW, "continuous ↔ continuous"))):
            ax.bar(i + (j - 0.5) * w, v, width=w * 0.88, color=col, zorder=3)
            ax.text(i + (j - 0.5) * w, v + 0.02, f"{v:.2f}×", ha="center",
                    fontsize=9.5, color=INK2, fontweight="700")
    ax.axhline(1.0, color="#0F6E5C", lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.46, 1.03, "parity", fontsize=9, color="#0F6E5C", ha="left")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([r[0] for r in ratios], fontsize=9)
    ax.set_ylim(0, 2.0)
    tidy(ax, ylab="solo throughput ÷ PD throughput", xgrid=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=OLD), plt.Rectangle((0, 0), 1, 1, color=NEW)]
    ax.legend(h, ["static ↔ static", "continuous ↔ continuous"],
              frameon=False, fontsize=8.8, loc="upper left")
    title(ax, "Matched schedulers close it — in one regime",
          "chat-shaped traffic reaches parity; prompt-heavy gets worse")

    ax = axes[1]
    # The occupancy panel that stood here was wrong: it read the scheduler's
    # cumulative counters, which include every idle stretch since the pod
    # booted, and reported 81% for a batch that actually runs full. Measured
    # over each request's own window, both configurations sit at 4.00 of 4
    # lanes with zero starvation. What is left is the tail of time-to-first-
    # token, so that is what this shows.
    tt = [("PD  (2 cards)", 80.6, 688.1, C["rngd_alone"]),
          ("solo (1 card)", 34.9, 78.6, C["solo"])]
    for i, (lab, p50, p99, col) in enumerate(tt):
        ax.plot([p50, p99], [i, i], "-", color=INK3, lw=1.6, zorder=2)
        ax.plot([p50], [i], "o", ms=9, color=col, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.plot([p99], [i], "D", ms=9, color=col, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2)
        # above the marker, not left of it: at 65 ms the label ran into the
        # axis and the row label
        ax.annotate(f"p50 {p50:.0f}", (p50, i), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9,
                    color=INK2, fontweight="600")
        ax.text(p99 + 20, i, f"p99 {p99:.0f}", va="center",
                fontsize=9.5, color=INK2, fontweight="700")
    ax.set_yticks(range(len(tt)))
    ax.set_yticklabels([t[0] for t in tt], fontsize=9.5)
    ax.set_ylim(-0.6, len(tt) - 0.4)          # headroom for the p50 labels
    ax.invert_yaxis(); ax.set_xlim(0, 1080)
    tidy(ax, xlab="time to first token (ms) — ShareGPT trace, N=8")
    title(ax, "What is actually left: the TTFT tail",
          "lane occupancy is 4.00 of 4 for both — the 81% figure was a counter artifact")

    header(fig, "The fair comparison: both on continuous",
           "RNGD · S=512 · batch 4 · sustained closed loop, same cards, same decode kernel, same scheduler.",
           top=0.83)
    fig.savefig(OUT / "fig9_fair_continuous.png", dpi=200)
    plt.close(fig)



# ---------------------------------------------------------------- fig 10
def fig_prefill_width():
    """Prefill costs the same for a 16-token prompt as for a 448-token one.

    The graph is compiled for one width and every request pays all of it. The
    trace's median prompt is 21 tokens against a 512-wide graph, so the median
    request throws away 96% of the pass -- and that waste is what fills the
    prefill queue and puts the tail on TTFT. The width was hardwired to the
    decode context in two places; separating them makes it a knob.
    """
    P = [16, 64, 128, 256, 448]
    c512 = [90.2, 97.1, 96.2, 90.2, 95.2]
    c128_P = [16, 32, 64, 128]
    c128 = [48.3, 53.3, 56.9, 46.5]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    ax = axes[0]
    ax.plot(P, c512, "o-", color=C["rngd_alone"], lw=2, ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="C = 512 (as deployed)")
    ax.plot(c128_P, c128, "s-", color=C["pd_forward"], lw=2, ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="C = 128")
    ax.axvline(21, color="#0F6E5C", lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(24, 112, "trace median\nprompt = 21", fontsize=8.6, color="#0F6E5C",
            va="top")
    ax.text(300, 99, "flat: the prompt\nbuys nothing", fontsize=9,
            color=INK2, fontweight="600", ha="center")
    ax.set_xlim(0, 480); ax.set_ylim(0, 125)
    tidy(ax, xlab="prompt length (tokens)", ylab="prefill (ms)", xgrid=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    title(ax, "One compiled width, paid in full",
          "16 tokens or 448, the pass costs the same")

    ax = axes[1]
    parts = [("compute  (the pass)", C["rngd_alone"]),
             ("download  (KV off the card)", C["pd_reverse"])]
    bars = [("C = 512", 65.2, 19.7), ("C = 128", 27.3, 11.2)]
    for i, (lab, comp, dl) in enumerate(bars):
        left = 0.0
        for v, (_pl, col) in zip((comp, dl), parts):
            ax.barh(i, v, left=left, height=0.46, color=col, zorder=3,
                    linewidth=1.6, edgecolor=SURFACE)
            ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=9.5, color="#fcfcfb", fontweight="700")
            left += v
        ax.text(left + 2.5, i, f"{left:.0f} ms", va="center", fontsize=10,
                color=INK2, fontweight="700")
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=10)
    ax.invert_yaxis(); ax.set_xlim(0, 118)
    tidy(ax, xlab="prefill (ms), median prompt")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in parts]
    ax.legend(h, [p for p, _ in parts], frameon=False, fontsize=8.8,
              loc="lower right")
    title(ax, "Narrowing the graph pays on both halves",
          "1.9x on the critical path, and it is the prefill card that queues")

    header(fig, "The prefill card was doing 512 tokens of work for 21",
           "RNGD kv-free prefill · S=512 · one request at a time, so nothing queues and the number is the pass itself.",
           top=0.83)
    fig.savefig(OUT / "fig10_prefill_width.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    # the re-measurement, taken after the KV-path, scheduler and prefill-width
    # work; the originals stay on disk for the before/after tables
    sw = load("bench/rpt_sweep_S512.jsonl")
    shape = load("bench/rpt_shape.jsonl")
    solo = load("bench/rpt_solo.jsonl")
    fig_latency(sw); fig_throughput(sw, solo); fig_power(shape, solo)
    fig_mfu(sw, solo); fig_batch(); fig_pd_vs_solo(); fig_kv_overlap(); fig_continuous(); fig_fair_continuous(); fig_prefill_width()
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size/1024:.0f} KB")
