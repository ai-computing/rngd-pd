"""Replot the measured PD-vs-replica crossover (from pd_crossover.py run, 2026-08-07).
Two panels: (A) PD zero-interference (measured), (B) replica decode-stall vs PD handoff cost.
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, os

# --- measured on RNGD, Llama-3.1-8B, S=256, per-layer-wired ChipRuntime ---
T_step = 664.8
Tpf = {8:5648.7, 16:10395.2, 32:21072.7, 64:41231.9, 128:81964.5, 192:120789.1, 255:154815.7}
T_kv = 88.9                       # 33.6MB fp16 handoff over host-PCIe(shm)
alone_med, alone_p99 = 678.9, 774.3
conc_med,  conc_p99  = 688.8, 745.7      # chip1 decode WHILE chip0 prefills

BLU, ORG, AQ, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a97a8"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.7), dpi=130)

# Panel A — zero interference (the PD premise, measured)
x = np.arange(2); w = 0.36
axA.bar(x-w/2, [alone_med, conc_med], w, color=[GREY, BLU], label="median")
axA.bar(x+w/2, [alone_p99, conc_p99], w, color=[GREY, BLU], alpha=0.45, label="p99")
axA.set_xticks(x); axA.set_xticklabels(["decode alone\n(chip1)", "decode WHILE\nchip0 prefills"])
axA.set_ylabel("decode inter-token latency (ms)")
axA.set_title("A · PD: two chips don't interfere\nconcurrent prefill costs decode  +1%", fontsize=11)
axA.axhline(alone_med, color=GREY, ls=":", lw=1)
axA.text(1, conc_med+30, "+1%", ha="center", color=BLU, fontweight="bold", fontsize=12)
axA.grid(axis="y", alpha=0.25); axA.set_ylim(0, 900)
axA.legend(fontsize=9, loc="upper left")

# Panel B — replica decode-stall grows with prefill; PD handoff is a small flat cost
xs = np.array(sorted(Tpf)); ys = np.array([Tpf[p] for p in xs])
axB.plot(xs, ys, "o-", color=ORG, lw=2.3, label="REPLICA: decode STALLED by prefill = T_prefill(P)")
axB.axhline(T_kv, color=BLU, lw=2.3, label=f"PD: one-time KV handoff = {T_kv:.0f} ms")
axB.fill_between(xs, T_kv, ys, color=AQ, alpha=0.10)
axB.set_yscale("log")
axB.set_xlabel("prefill length P  (tokens)"); axB.set_ylabel("added decode latency (ms, log)")
axB.set_title("B · In replica, one prefill blocks decode for its\nwhole duration; PD never stalls decode", fontsize=11)
axB.annotate(f"{ys[-1]/1000:.0f} s stall\n@ P=255", (xs[-1], ys[-1]), color=ORG,
             fontsize=9, ha="right", va="bottom")
axB.annotate("89 ms", (12, T_kv), color=BLU, fontsize=9, va="bottom")
axB.grid(alpha=0.25, which="both")
axB.legend(fontsize=8.8, loc="center right")

fig.suptitle("RNGD  ·  PD disaggregation vs replication  ·  Llama-3.1-8B, 2×RNGD, S=256",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), "pd_crossover.png")
os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out)
print("plot →", out)
