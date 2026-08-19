"""Serving sweep on RNGD — throughput / TTFT / TPOT vs batch & context, + roofline.

Drives llama_engine_batch.py (batched wired decode; weights read ONCE/step, shared across batch) as
subprocesses and parses its per-config line. Two sweeps at nl=8 (fast compile; latency ×4, tok/s ÷4
for the full 32L model — the ROOFLINE is nl-invariant so it is exact):

  • batch sweep  (S=256): B=1..128 → TPOT (ms/step), throughput (tok/s), roofline points.
  • context sweep (B=8) : S=128..2048 → TPOT(context), attention cost growth.

TTFT: a prefill of P prompt tokens reads the weights once and does ~P× the matmul work of a decode
step — i.e. it is the compute of a batch of P. So TTFT(P) ≈ ms/step at B=P (compute-bound), read
straight off the batch sweep. (Slight underestimate: ignores P×P attention, small for moderate P.)

  python3 serving_bench.py
"""
import subprocess, sys, re, os
PY = sys.executable
ENG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime", "llama_engine_batch.py")
NL = 8                      # 8-layer proxy; ×4 latency, ÷4 tok/s → full 32L
LINE = re.compile(r"B=\s*(\d+)\s+chip\d+:\s*([\d.]+)\s*ms/step\s*\|\s*(\d+)\s*tok/s\s*\|\s*wBW\s*(\d+)\s*GB/s\s*\|\s*compute\s*([\d.]+)\s*TFLOPS")

# roofline constants (RNGD, bf16)
PEAK_TFLOPS = 256.0; HBM_GBs = 1500.0; RIDGE = PEAK_TFLOPS*1e12/(HBM_GBs*1e9)   # 170.7 FLOP/byte
# per-step flop/bytes formulas (from llama_engine_batch.bench); nl cancels in intensity
def intensity(B):
    flop_per_nl = 2*(4096*4096*2 + 14336*4096*3)
    wbytes_per_nl = 4096*4096*2*2 + 14336*4096*2*3
    return B*flop_per_nl/wbytes_per_nl


def run(nl, bucket, batch, chip=0):
    cmd = [PY, ENG, "--nl", str(nl), "--bucket", str(bucket), "--tactic", "Decode",
           "--chip", str(chip), "--batch", ",".join(str(b) for b in batch)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        m = LINE.search(ln)
        if m:
            B, ms, toks, gbs, tf = int(m[1]), float(m[2]), int(m[3]), int(m[4]), float(m[5])
            rows.append(dict(B=B, ms=ms, toks=toks, gbs=gbs, tflops=tf, S=bucket))
    return rows


def main():
    print("=== batch sweep (nl=8, S=256) ===", flush=True)
    batch_rows = run(NL, 256, [1, 2, 4, 8, 16, 32, 64, 128], chip=0)
    for r in batch_rows:
        print(f"  B={r['B']:3d}  {r['ms']:.2f} ms  {r['toks']:5d} tok/s  {r['tflops']:.1f} TFLOPS", flush=True)

    print("=== context sweep (nl=8, B=8) ===", flush=True)
    ctx_rows = []
    for S in [128, 256, 512, 1024, 2048]:
        r = run(NL, S, [8], chip=0)
        if r: ctx_rows.append(r[0]); print(f"  S={S:5d}  {r[0]['ms']:.2f} ms/step (TPOT)", flush=True)

    # ---- plots ----
    import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    BLU, ORG, AQ, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#8a97a8", "#d1495b"
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.2), dpi=125)

    # A: throughput + TPOT vs batch  (32L-scaled)
    Bs = [r["B"] for r in batch_rows]
    tps32 = [r["toks"]/4 for r in batch_rows]        # ÷4 for 32L
    tpot32 = [r["ms"]*4 for r in batch_rows]          # ×4 for 32L
    a = ax[0][0]; a2 = a.twinx()
    a.plot(Bs, tps32, "o-", color=BLU, lw=2.3, label="throughput (tok/s)")
    a2.plot(Bs, tpot32, "s--", color=ORG, lw=2, label="TPOT (ms/token)")
    a.set_xscale("log", base=2); a.set_xticks(Bs); a.set_xticklabels(Bs)
    a.set_xlabel("batch size"); a.set_ylabel("throughput (tok/s)", color=BLU)
    a2.set_ylabel("TPOT (ms/token)", color=ORG)
    a.set_title("A · decode throughput ↑ & TPOT vs batch  (32L)", fontsize=11)
    a.grid(alpha=0.25); a.tick_params(axis="y", colors=BLU); a2.tick_params(axis="y", colors=ORG)

    # B: TTFT(P) ≈ ms/step(B=P), 32L
    Ps = Bs; ttft32 = [r["ms"]*4 for r in batch_rows]
    b = ax[0][1]
    b.plot(Ps, ttft32, "o-", color=AQ, lw=2.3)
    b.set_xscale("log", base=2); b.set_xticks(Ps); b.set_xticklabels(Ps)
    b.set_xlabel("prompt length P (tokens)"); b.set_ylabel("TTFT (ms)")
    b.set_title("B · TTFT ≈ prefill of P tokens  (≈ ms/step at B=P, 32L)", fontsize=11)
    b.grid(alpha=0.25)

    # C: TPOT vs context length (32L)
    Ss = [r["S"] for r in ctx_rows]; tpot_ctx = [r["ms"]*4 for r in ctx_rows]
    c = ax[1][0]
    c.plot(Ss, tpot_ctx, "o-", color=ORG, lw=2.3)
    c.set_xscale("log", base=2); c.set_xticks(Ss); c.set_xticklabels(Ss)
    c.set_xlabel("KV context length (tokens)"); c.set_ylabel("TPOT (ms/token)")
    c.set_title("C · TPOT vs context (B=8, 32L) — attention cost growth", fontsize=11)
    c.grid(alpha=0.25)

    # D: roofline
    d = ax[1][1]
    I = np.array([intensity(r["B"]) for r in batch_rows])
    ach = np.array([r["tflops"] for r in batch_rows])
    xr = np.logspace(np.log10(min(I)*0.4), np.log10(max(I)*2.2), 200)
    roof = np.minimum(PEAK_TFLOPS, HBM_GBs*1e9*xr/1e12)
    d.plot(xr, roof, "-", color=GREY, lw=2, label=f"RNGD roof (256 TFLOPS · 1.5 TB/s)")
    d.axvline(RIDGE, color=RED, ls=":", lw=1.4, label=f"ridge {RIDGE:.0f} FLOP/byte")
    d.plot(I, ach, "o-", color=BLU, lw=2, label="achieved (batch sweep)")
    for r, x, y in zip(batch_rows, I, ach):
        d.annotate(f"B={r['B']}", (x, y), fontsize=7.5, color=BLU, xytext=(3, 4), textcoords="offset points")
    d.set_xscale("log"); d.set_yscale("log")
    d.set_xlabel("arithmetic intensity (FLOP/byte)"); d.set_ylabel("achieved (TFLOPS)")
    d.set_title("D · roofline — batch moves decode toward the ridge", fontsize=11)
    d.grid(alpha=0.25, which="both"); d.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("RNGD serving sweep · Llama-3.1-8B (nl=8 measured → 32L scaled) · throughput / TTFT / TPOT / roofline",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), "serving_sweep.png")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out)
    print(f"\nplot → {out}", flush=True)

    # summary
    peak = max(batch_rows, key=lambda r: r["toks"])
    print("\n" + "="*66)
    print("SUMMARY (32L-scaled from nl=8)")
    print(f"  peak throughput  : {peak['toks']//4:,} tok/s @ B={peak['B']}  ({peak['tflops']:.0f} TFLOPS = {peak['tflops']/PEAK_TFLOPS*100:.0f}% peak)")
    print(f"  TPOT @ B=1        : {batch_rows[0]['ms']*4:.1f} ms/token  (latency-optimal)")
    print(f"  TTFT(P=128)       : {[r['ms']*4 for r in batch_rows if r['B']==128][0]:.0f} ms  (prefill 128 tok)")
    if ctx_rows: print(f"  TPOT ctx 128→2048 : {ctx_rows[0]['ms']*4:.1f} → {ctx_rows[-1]['ms']*4:.1f} ms/token")
    print("="*66)


if __name__ == "__main__":
    main()
