"""2-NPU: duplication (2 full replicas) vs PD disaggregation (1 prefill + 1 decode) — throughput & latency.

Grounded in THIS session's measured single-chip components (Llama-3.1-8B, 32L-scaled):
  TPOT   = 13 ms/token (low-batch, latency-optimal; per-stream inter-token latency)
  Tp(P)  = prefill time for P tokens ≈ max(13ms floor, 0.156 ms/token · P)   [compute-bound slope
           from the B=128 serving-sweep point; prefill ≈ processing P tokens through weights once]
  T_kv   = 89 ms  KV handoff (33.6MB fp16, host-PCIe), PD only
  interference: concurrent prefill slows the OTHER chip's decode by +1% (measured) → PD isolates;
           in a replica the same prefill BLOCKS that chip's decode entirely.

Queueing model (chips are independent — validated by the +1% interference measurement):
  a request = Tp(P) prefill-work + Td=D·TPOT decode-work.
  DUP (2 replicas): each chip runs full requests → aggregate output = 2 · D/(Tp+Td).
  PD  (1P+1D)     : stages pipelined on separate chips → aggregate output = D / max(Tp, Td).
  ⇒ PD/DUP = (Tp+Td)/(2·max(Tp,Td)) ∈ [0.5, 1]:  EQUAL only when Tp==Td (balanced);
     DUP up to 2× PD when one stage dominates (decode-heavy = typical LLM serving).

  python3 pd_vs_dup.py
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TPOT = 13.0            # ms/token (32L)
TP_FLOOR, TP_SLOPE = 13.0, 0.156     # Tp(P) = max(floor, slope*P) ms
T_KV = 89.0            # ms handoff (PD only)

def Tp(P): return np.maximum(TP_FLOOR, TP_SLOPE*np.asarray(P, float))
def Td(D): return np.asarray(D, float)*TPOT

# times in ms → ×1000 for tok/s. Single-stream (batch=1); batching scales BOTH equally, ratio unchanged.
def dup_tps(P, D):  tp, td = Tp(P), Td(D); return 1000*2*np.asarray(D,float)/(tp+td)          # tok/s (2 chips)
def pd_tps(P, D):   tp, td = Tp(P), Td(D); return 1000*np.asarray(D,float)/np.maximum(tp, td)  # tok/s (1P+1D)

BLU, ORG, AQ, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a97a8", "#d1495b"
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=130)

# Panel A — aggregate throughput vs OUTPUT length, at a fixed prompt P=512 (RAG-ish)
P = 512
Ds = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048])
dA, pA = dup_tps(P, Ds), pd_tps(P, Ds)
axA.plot(Ds, dA, "o-", color=BLU, lw=2.3, label="DUP · 2 replicas (both chips decode)")
axA.plot(Ds, pA, "s-", color=ORG, lw=2.3, label="PD · 1 prefill + 1 decode")
# balance point: Td==Tp  → D* = Tp(P)/TPOT
Dstar = Tp(P)/TPOT
axA.axvline(Dstar, color=GREY, ls="--", lw=1.2)
axA.text(Dstar*1.1, dA.min()*1.4, f"balance\nD*≈{Dstar:.0f}", color=GREY, fontsize=9)
axA.set_xscale("log", base=2); axA.set_xticks(Ds); axA.set_xticklabels(Ds, fontsize=8)
axA.set_xlabel(f"output length D  (prompt P={P})"); axA.set_ylabel("aggregate throughput (tok/s)")
axA.set_title("A · 2-NPU throughput — DUP ≥ PD, up to 2× for decode-heavy", fontsize=11)
axA.grid(alpha=0.25); axA.legend(fontsize=9, loc="upper left")
axA.annotate("decode-heavy\n(typical chat/RAG):\nDUP ≈ 2× PD", (Ds[-1], pA[-1]),
             color=ORG, fontsize=8.5, ha="right", va="bottom")

# Panel B — normalized aggregate throughput vs prefill-fraction f = Tp/(Tp+Td)
f = np.linspace(0.001, 0.999, 400)
pd_norm = 1.0/np.maximum(f, 1-f)          # (Tp+Td)/max(Tp,Td) in terms of f
axB.axhline(2.0, color=BLU, lw=2.3, label="DUP (both chips, any mix)")
axB.plot(f, pd_norm, color=ORG, lw=2.3, label="PD (1P+1D)")
axB.fill_between(f, pd_norm, 2.0, color=GREY, alpha=0.10)
axB.axvline(0.5, color=GREY, ls="--", lw=1.2)
axB.plot([0.5], [2.0], "o", color=AQ, ms=9); axB.text(0.5, 2.04, "EQUAL only when\nprefill work = decode work",
         color=AQ, fontsize=9, ha="center")
# typical operating points
for lbl, ff, dx in [("chat P128/D256", 20/(20+256*13), 0.02), ("RAG P2048/D128", Tp(2048).item()/(Tp(2048).item()+128*13), 0.0)]:
    yy = 1/max(ff, 1-ff)
    axB.plot([ff], [yy], "v", color=RED, ms=8)
    axB.annotate(lbl, (ff, yy), color=RED, fontsize=8, xytext=(6, -2), textcoords="offset points")
axB.set_xlabel("prefill fraction  f = Tp / (Tp + Td)"); axB.set_ylabel("throughput ÷ single chip")
axB.set_title("B · they match ONLY at balance; real workloads sit far left", fontsize=11)
axB.set_ylim(0.9, 2.15); axB.grid(alpha=0.25); axB.legend(fontsize=9, loc="lower center")

fig.suptitle("2×RNGD · duplication vs PD disaggregation  ·  Llama-3.1-8B (measured components, 32L)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), "pd_vs_dup.png")
os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out)
print("plot →", out)

print("\n" + "="*70)
print("2-NPU DUP vs PD — verdict (measured-grounded)")
print(f"  THROUGHPUT: DUP ≥ PD always. Equal only at balance (Tp==Td, output D*≈{Tp(512)/TPOT:.0f} @P=512).")
print(f"    typical decode-heavy chat (P128,D256): DUP {dup_tps(128,256):.0f} vs PD {pd_tps(128,256):.0f} tok/s"
      f"  → DUP {dup_tps(128,256)/pd_tps(128,256):.2f}×")
print(f"    balanced (P512,D40):                   DUP {dup_tps(512,40):.0f} vs PD {pd_tps(512,40):.0f} tok/s"
      f"  → DUP {dup_tps(512,40)/pd_tps(512,40):.2f}×")
print(f"  LATENCY: PD isolates — concurrent prefill costs decode +1% (measured); a replica STALLS")
print(f"    decode for the whole prefill. PD wins TPOT-tail & TTFT under mixed load, not raw 2-chip tput.")
print("="*70)
