"""Where does PD disaggregation beat replication on RNGD? — empirical crossover.

Two measured facts decide it:
  • T_prefill(P): a prefill of P tokens OCCUPIES a chip this long. In REPLICA (one chip does both),
    that occupancy STALLS every in-flight decode on the chip by ~T_prefill(P).
  • T_kv       : the fixed cost PD pays to hand a request's KV from the prefill chip to the decode
    chip via the connector (shm). PD's decode chip is NEVER stalled by prefill.

So PD wins (lower decode tail latency under concurrency) once  T_prefill(P) > T_kv.
Plus a direct check: run a decode stream on chip1 WHILE chip0 prefills — if TPOT is unchanged, the
two chips don't interfere (independent HBM), which is the whole premise of PD.

  python3 pd_crossover.py --model meta-llama/Llama-3.1-8B-Instruct --S 256
"""
import argparse, sys, time, threading, statistics as st
import numpy as np, torch
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
from llama_pd_2chip import ChipRuntime
from generic_runtime import ModelConfig
from rngd_kv_connector import RngdKVConnector, KVConnectorRole
from kv_transport import ShmTransport

med = st.median
def p99(xs): return sorted(xs)[max(0, int(len(xs)*0.99)-1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--S", type=int, default=256)
    a = ap.parse_args()
    S = a.S
    from transformers import AutoModelForCausalLM
    print(f"loading {a.model}...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    NL = cfg.n_layers
    toks = [(i * 131 + 7) % 32000 for i in range(S)]

    print(f"[xover] compiling chip0 + chip1 (S={S})...", flush=True)
    rt0 = ChipRuntime(cfg, sd, S, 0); rt1 = ChipRuntime(cfg, sd, S, 1)
    print(f"  compiled: chip0 {rt0.compile_s:.0f}s, chip1 {rt1.compile_s:.0f}s", flush=True)

    # --- T_step: one decode step (median) ---
    ck, cv = rt1.new_kv()
    for p in range(4): rt1.step(toks[p], p, ck, cv)          # warm
    ts = []
    for p in range(4, 34):
        t0 = time.time(); rt1.step(toks[p], p, ck, cv); ts.append((time.time()-t0)*1000)
    T_step = med(ts)
    print(f"  T_step (1 decode token) = {T_step:.1f} ms", flush=True)

    # --- T_prefill(P): occupancy of a P-token prefill on one chip ---
    Ps = [p for p in [8, 16, 32, 64, 128, 192, S-1] if p < S]
    Tpf = {}
    filled = None
    for P in Ps:
        pck, pcv = rt0.new_kv()
        t0 = time.time()
        for p in range(P): rt0.step(toks[p], p, pck, pcv)
        Tpf[P] = (time.time()-t0)*1000
        if P == Ps[-1]: filled = (pck, pcv)
        print(f"  T_prefill(P={P:>3}) = {Tpf[P]:7.1f} ms", flush=True)

    # --- T_kv: connector handoff of the (full-bucket) KV, chip0 -> shm -> chip1 ---
    tr = ShmTransport(root="/dev/shm/rngd_xover")
    pf = RngdKVConnector(KVConnectorRole.WORKER, req_id="x", transport=tr, n_layers=NL)
    dk, dv = rt1.new_kv()
    dc = RngdKVConnector(KVConnectorRole.WORKER, req_id="x", transport=tr, n_layers=NL)
    dc.register_kv_caches((dk, dv)); dc.bind_connector_metadata(dc.build_connector_meta(None))
    pck, pcv = filled
    tkv = []
    for _ in range(8):
        t0 = time.time()
        for i in range(NL): pf.save_kv_layer(f"layer.{i}", (pck[i], pcv[i]))
        pf.wait_for_save(); dc.start_load_kv()
        for i in range(NL): dc.wait_for_layer_load(f"layer.{i}")
        tkv.append((time.time()-t0)*1000)
    T_kv = med(tkv)
    kv_mb = sum(len(__import__("kv_transport").ser(pck[i])) + len(__import__("kv_transport").ser(pcv[i])) for i in range(NL))/1e6
    print(f"  T_kv (32-layer handoff, {kv_mb:.1f}MB fp16, full bucket S={S}) = {T_kv:.1f} ms", flush=True)

    # --- crossover P*: T_prefill(P*) == T_kv (linear interp on measured curve) ---
    xs = sorted(Tpf); ys = [Tpf[p] for p in xs]
    Pstar = None
    for i in range(1, len(xs)):
        if ys[i-1] <= T_kv <= ys[i] or (ys[i-1] < T_kv and i == len(xs)-1 and ys[i] < T_kv):
            if ys[i-1] <= T_kv <= ys[i]:
                Pstar = xs[i-1] + (xs[i]-xs[i-1])*(T_kv-ys[i-1])/(ys[i]-ys[i-1]); break
    if Pstar is None:
        Pstar = T_kv / T_step if T_step > 0 else float('nan')     # fallback: P ~ T_kv/T_step
    print(f"\n  CROSSOVER P* ≈ {Pstar:.0f} tokens  (T_prefill(P*) == T_kv={T_kv:.1f}ms)", flush=True)

    # --- PD zero-interference: decode stream on chip1 alone vs while chip0 prefills ---
    def stream(rt, n):
        c, v = rt.new_kv(); lat = []
        for p in range(n):
            t0 = time.time(); rt.step(toks[p % S], p % S, c, v); lat.append((time.time()-t0)*1000)
        return lat
    base = stream(rt1, 40)                                    # chip1 alone
    stop = [False]
    def prefill_loop():
        while not stop[0]:
            c, v = rt0.new_kv()
            for p in range(64):
                if stop[0]: break
                rt0.step(toks[p], p, c, v)
    th = threading.Thread(target=prefill_loop); th.start()
    conc = stream(rt1, 40)                                     # chip1 while chip0 prefills
    stop[0] = True; th.join()
    print(f"\n  PD interference check (chip1 decode TPOT):")
    print(f"    alone           : median {med(base):.1f} ms  p99 {p99(base):.1f} ms")
    print(f"    while chip0 pref: median {med(conc):.1f} ms  p99 {p99(conc):.1f} ms")
    slow = (med(conc)/med(base)-1)*100
    print(f"    → decode slowdown from concurrent prefill on the OTHER chip: {slow:+.0f}%", flush=True)

    # --- plot ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        BLU, ORG, AQ = "#2a78d6", "#eb6834", "#1baf7a"
        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=130)
        xx = np.array(sorted(Tpf)); yy = np.array([Tpf[p] for p in xx])
        ax.plot(xx, yy, "o-", color=ORG, lw=2.2, label="REPLICA: decode stall = T_prefill(P)")
        ax.axhline(T_kv, color=BLU, lw=2.2, ls="-", label=f"PD: KV handoff = {T_kv:.0f} ms (decode never stalls)")
        ax.axvline(Pstar, color="#888", lw=1.2, ls="--")
        ax.axvspan(Pstar, xx.max(), color=AQ, alpha=0.10)
        ax.text(Pstar*1.02, yy.max()*0.9, f"PD wins →\nP > {Pstar:.0f}", color=AQ, fontsize=11, fontweight="bold")
        ax.text(Pstar, -yy.max()*0.06, f"P*≈{Pstar:.0f}", color="#666", ha="center", fontsize=9)
        ax.set_xlabel("prefill length P  (tokens)"); ax.set_ylabel("added decode latency (ms)")
        ax.set_title("RNGD: PD disaggregation vs replication — crossover\n"
                     f"Llama-3.1-8B, S={S}  ·  T_step={T_step:.0f}ms  ·  KV {kv_mb:.1f}MB over host-PCIe(shm)", fontsize=11)
        ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.25); ax.set_ylim(bottom=0)
        out = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), "pd_crossover.png")
        import os; os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout(); fig.savefig(out); print(f"\n  plot → {out}", flush=True)
    except Exception as e:
        print(f"  (plot skipped: {e})", flush=True)

    print("\n" + "=" * 66)
    print("VERDICT")
    print(f"  PD beats replication once prefill length P > ~{Pstar:.0f} tokens (T_prefill > T_kv),")
    print(f"  AND when there is concurrent decode to protect (interference check: {slow:+.0f}%).")
    print(f"  RNGD host-PCIe KV ({T_kv:.0f}ms) sets the bar; shorter/isolated requests favor replica.")
    print("=" * 66)


if __name__ == "__main__":
    main()
