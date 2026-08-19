#!/usr/bin/env python3
"""Paper-grade DUP vs PD sweep on the engine-class runtime (real KV streaming).

Workers load the engine ONCE and iterate many points, so a 12-point sweep costs
one model load instead of twelve. Points:
  "P:D:n"  — n requests, prompt P, decode D  (P+D <= S)
  "chat"   — deterministic mixed chat workload (per-request P_i, D_i)

Per point x mode: wall, out tok/s, per-request TTFT, PD handoff gap (real
streamed KV), output cross-check (PD decode == DUP decode for same request).

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_sweep_engine.py --S 256 --chips 0,1 \
    --kv-stream 64 --points "128:16:8,128:32:8,128:64:8,128:128:8,chat" \
    --out /tmp/sweep256.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import _paths  # noqa: F401
from pd_engine import (EngineSession, _bf16_bytes, build_chunk_kernels,
                       build_engine, build_lmhead)

WORK = "/dev/shm/rngd_sweep"

CHAT_256 = [(64, 32), (96, 64), (128, 48), (160, 96), (64, 128), (192, 64),
            (96, 32), (128, 80), (160, 48), (192, 32), (64, 64), (128, 96)]
CHAT_512 = [(128, 64), (256, 96), (384, 128), (320, 64), (448, 64), (192, 128),
            (256, 192), (384, 96), (128, 256), (320, 128), (448, 48), (256, 64)]


def parse_points(spec, S):
    pts = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "chat":
            reqs = CHAT_512 if S > 256 else CHAT_256
            pts.append({"name": "chat", "reqs": reqs})
        else:
            P, D, n = (int(x) for x in tok.split(":"))
            assert P + D <= S, f"{tok}: P+D > S={S}"
            pts.append({"name": tok, "reqs": [(P, D)] * n})
    return pts


def make_prompts_pt(pt, seed_base):
    rng = np.random.default_rng(seed_base)
    return [list(map(int, rng.integers(100, 120000, P))) for P, _ in pt["reqs"]]


def wait_for(path, dead=None, poll=0.005):
    while not os.path.exists(path):
        if dead and dead():
            raise RuntimeError(f"peer died waiting for {path}")
        time.sleep(poll)


# ------------------------------- workers -----------------------------------

def worker(a):
    import llama_engine_resident as LER
    from kv_transport import make_transport
    cfg, rt = build_engine(a.model, a.S, a.nl, chip=a.chip)
    lmh = build_lmhead(rt, LER, a.chip)
    s = EngineSession(rt, LER, lmhead=lmh)
    C = a.kv_stream
    if C and a.role in ("prefill", "decode"):
        mg, ms_sc, sels = build_chunk_kernels(rt, LER, a.chip, C)
    pts = parse_points(a.points, a.S)
    tag = f"{a.role}{a.chip}" if a.role == "dup" else a.role

    open(f"{WORK}/ready_{tag}", "w").write("1")
    for j, pt in enumerate(pts):
        prompts = make_prompts_pt(pt, 1000 + j)
        wait_for(f"{WORK}/go_p{j}")
        res = []
        if a.role == "dup":
            mine = [(i, prompts[i], pt["reqs"][i][1]) for i in range(len(prompts))
                    if i % 2 == (a.chip_rank)]
            for i, toks, D in mine:
                Kp, Vp = s.fresh_kv()
                t0 = time.perf_counter()
                nxt = None
                for pos in range(len(toks)):
                    nxt, Kp, Vp = s.step(toks[pos], pos, Kp, Vp)
                ttft = time.perf_counter() - t0
                out = [nxt]
                p = len(toks)
                for _ in range(D - 1):
                    nxt, Kp, Vp = s.step(nxt, p, Kp, Vp)
                    out.append(nxt)
                    p += 1
                res.append({"i": i, "ttft": ttft,
                            "e2e": time.perf_counter() - t0, "out": out})
        elif a.role == "prefill":
            for i, toks in enumerate(prompts):
                tr = make_transport("shm", root=f"{WORK}/kv_p{j}_{i}")
                Kp, Vp = s.fresh_kv()
                t0 = time.perf_counter()
                nxt = None
                n_chunks = 0
                for pos in range(len(toks)):
                    nxt, Kp, Vp = s.step(toks[pos], pos, Kp, Vp)
                    if C and (pos + 1) % C == 0 and pos + 1 < len(toks):
                        k = (pos + 1) // C - 1
                        Kc, Vc = mg(Kp, Vp, sels[k][0], sels[k][1])
                        tr.push(f"k{k}", _bf16_bytes(Kc))
                        tr.push(f"v{k}", _bf16_bytes(Vc))
                        open(f"{WORK}/kc_p{j}_{i}_{k}", "w").write("1")
                        n_chunks = k + 1
                t_pre = time.perf_counter() - t0
                t_pre_wall = time.time()
                # final (possibly partial-position) chunk: covers positions up
                # to the last one prefilled; chunk grid is fixed C so the last
                # chunk index is ceil(P/C)-1 — gather that window
                k_last = (len(toks) + C - 1) // C - 1
                Kc, Vc = mg(Kp, Vp, sels[k_last][0], sels[k_last][1])
                tr.push(f"k{k_last}", _bf16_bytes(Kc))
                tr.push(f"v{k_last}", _bf16_bytes(Vc))
                open(f"{WORK}/kc_p{j}_{i}_{k_last}", "w").write("1")
                json.dump({"next_tok": int(nxt), "t_pre_wall": t_pre_wall,
                           "n_chunks": k_last + 1},
                          open(f"{WORK}/meta_p{j}_{i}.json", "w"))
                open(f"{WORK}/kvready_p{j}_{i}", "w").write("1")
                res.append({"i": i, "t_pre": t_pre})
        elif a.role == "decode":
            import torch
            for i, toks in enumerate(prompts):
                D = pt["reqs"][i][1]
                tr = make_transport("shm", root=f"{WORK}/kv_p{j}_{i}")

                def pull_bf16(tag2, shape):
                    u = np.frombuffer(tr.pull(tag2), np.uint16).copy()
                    return torch.from_numpy(u).view(torch.bfloat16).float() \
                                .numpy().reshape(shape)
                wait_for(f"{WORK}/meta_p{j}_{i}.json".replace(".json", ".json")
                         if False else f"{WORK}/kc_p{j}_{i}_0")
                Kp, Vp = s.fresh_kv()
                k = 0
                gap0 = time.perf_counter()
                while True:
                    if os.path.exists(f"{WORK}/kc_p{j}_{i}_{k}"):
                        Kc = LER._d(pull_bf16(f"k{k}", (rt.nl, LER.NKV, LER.HD, C)))
                        Vc = LER._d(pull_bf16(f"v{k}", (rt.nl, LER.NKV, C, LER.HD)))
                        Kp, Vp = ms_sc(Kp, Vp, Kc, Vc, sels[k][0], sels[k][1])
                        k += 1
                        if os.path.exists(f"{WORK}/kvready_p{j}_{i}"):
                            meta = json.load(open(f"{WORK}/meta_p{j}_{i}.json"))
                            if k >= meta["n_chunks"]:
                                break
                    else:
                        gap0 = time.perf_counter()
                        time.sleep(0.002)
                meta = json.load(open(f"{WORK}/meta_p{j}_{i}.json"))
                gap_wall = (time.time() - meta["t_pre_wall"]) * 1e3
                nxt = meta["next_tok"]
                out = [nxt]
                p = len(toks)
                t_dec0 = time.perf_counter()
                for _ in range(D - 1):
                    nxt, Kp, Vp = s.step(nxt, p, Kp, Vp)
                    out.append(nxt)
                    p += 1
                tpot = ((time.perf_counter() - t_dec0) / (D - 1)) if D > 1 else None
                res.append({"i": i, "handoff_gap_ms": round(gap_wall, 1),
                            "tpot_s": tpot, "out": out})
        json.dump(res, open(f"{WORK}/{tag}_p{j}.json", "w"))
        open(f"{WORK}/done_{tag}_p{j}", "w").write("1")


# ----------------------------- orchestrator --------------------------------

class UtilSampler:
    """Thread-based power + PE-counter sampler (orchestrator side)."""

    def __init__(self, chips, period=0.25):
        import threading
        self.chips = chips
        self.samples = {c: [] for c in chips}   # (t, W, task_cyc, cyc)
        self.period = period
        self._stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        # furiosa_smi_py Device is thread-pinned (unsendable): create the
        # devices ON this thread, never share them across threads
        import time as _t
        import furiosa_smi_py as smi
        smi.init()
        devs = smi.list_devices()
        my = {c: devs[c] for c in self.chips}
        while not self._stop:
            t = _t.perf_counter()
            for c, d in my.items():
                try:
                    w = d.power_consumption()
                    pcs = d.device_performance_counter().pe_performance_counters()
                    task = sum(p.task_execution_cycle() for p in pcs)
                    cyc = sum(p.cycle_count() for p in pcs)
                    self.samples[c].append((t, w, task, cyc))
                except Exception:
                    pass
            _t.sleep(self.period)

    def window(self, t0, t1):
        out = {}
        for c, ss in self.samples.items():
            w = [x[1] for x in ss if t0 <= x[0] <= t1]
            pe = [(x[2], x[3]) for x in ss if t0 <= x[0] <= t1]
            util = None
            if len(pe) >= 2 and pe[-1][1] > pe[0][1]:
                util = 100.0 * (pe[-1][0] - pe[0][0]) / (pe[-1][1] - pe[0][1])
            out[str(c)] = {"avg_W": round(sum(w) / len(w), 1) if w else None,
                           "pe_util_pct": round(util, 1) if util is not None else None}
        return out

    def stop(self):
        self._stop = True


def spawn(role, chip, chip_rank, a):
    cmd = [sys.executable, os.path.abspath(__file__), "--role", role,
           "--chip", str(chip), "--chip-rank", str(chip_rank),
           "--S", str(a.S), "--nl", str(a.nl), "--model", a.model,
           "--kv-stream", str(a.kv_stream), "--points", a.points]
    return subprocess.Popen(cmd, stdout=open(f"{WORK}/{role}{chip}.log", "w"),
                            stderr=subprocess.STDOUT)


def run_mode(mode, a, chips):
    os.system(f"rm -rf {WORK} && mkdir -p {WORK}")
    if mode == "dup":
        procs = [spawn("dup", chips[0], 0, a), spawn("dup", chips[1], 1, a)]
        tags = [f"dup{chips[0]}", f"dup{chips[1]}"]
    else:
        procs = [spawn("prefill", chips[0], 0, a), spawn("decode", chips[1], 1, a)]
        tags = ["prefill", "decode"]
    # a peer that exited CLEANLY already wrote all its done markers — only a
    # nonzero exit is a real death
    dead = lambda: any(p.poll() is not None and p.returncode != 0 for p in procs)
    for t in tags:
        wait_for(f"{WORK}/ready_{t}", dead, poll=0.5)
    pts = parse_points(a.points, a.S)
    sampler = UtilSampler(chips)
    rows = []
    for j, pt in enumerate(pts):
        t0 = time.perf_counter()
        open(f"{WORK}/go_p{j}", "w").write("1")
        for t in tags:
            wait_for(f"{WORK}/done_{t}_p{j}", dead, poll=0.05)
        wall = time.perf_counter() - t0
        cards = sampler.window(t0, time.perf_counter())
        data = {t: json.load(open(f"{WORK}/{t}_p{j}.json")) for t in tags}
        rows.append({"point": pt["name"], "wall": wall, "data": data,
                     "cards": cards})
        print(f"  [{mode}] {pt['name']}: wall {wall:.1f}s", flush=True)
    sampler.stop()
    for p in procs:
        p.wait(timeout=600)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--chips", default="0,1")
    ap.add_argument("--kv-stream", type=int, default=64)
    ap.add_argument("--points", required=False,
                    default="128:16:8,128:32:8,128:64:8,128:128:8,chat")
    ap.add_argument("--out", default="")
    ap.add_argument("--role", default="orchestrator")
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--chip-rank", type=int, default=0)
    a = ap.parse_args()

    if a.role != "orchestrator":
        return worker(a)

    chips = [int(x) for x in a.chips.split(",")]
    print(f"[sweep] S={a.S} nl={a.nl} chips={chips} C={a.kv_stream} "
          f"points={a.points}", flush=True)
    dup = run_mode("dup", a, chips)
    pd = run_mode("pd", a, chips)

    def p50(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    out_rows = []
    print("\n" + "=" * 78)
    print(f"{'point':>12} | {'DUP tok/s':>9} {'PD tok/s':>8} {'PD/DUP':>6} | "
          f"{'DUP ttft':>8} | {'gap ms':>6} | match")
    for dj, pj in zip(dup, pd):
        pt = dj["point"]
        tot_out = sum(len(r["out"]) for t in dj["data"].values() for r in t)
        tps_d = tot_out / dj["wall"]
        tps_p = tot_out / pj["wall"]
        dup_out = {r["i"]: r["out"] for t in dj["data"].values() for r in t}
        pd_out = {r["i"]: r["out"] for r in pj["data"]["decode"]}
        match = all(dup_out.get(i) == pd_out.get(i) for i in pd_out)
        ttft = p50([r["ttft"] for t in dj["data"].values() for r in t])
        gap = p50([r["handoff_gap_ms"] for r in pj["data"]["decode"]])
        dup_tpots = [(r["e2e"] - r["ttft"]) / (len(r["out"]) - 1)
                     for t in dj["data"].values() for r in t if len(r["out"]) > 1]
        pd_tpots = [r["tpot_s"] for r in pj["data"]["decode"] if r.get("tpot_s")]
        tpot_d, tpot_p = p50(dup_tpots), p50(pd_tpots)
        print(f"{pt:>12} | {tps_d:9.2f} {tps_p:8.2f} {tps_p/tps_d:6.2f} | "
              f"{ttft*1e3:8.0f} | tpot {tpot_d*1e3:5.1f}/{tpot_p*1e3:5.1f} | "
              f"{gap:6.0f} | {'PASS' if match else 'FAIL'}")
        out_rows.append({"point": pt, "S": a.S, "kv_stream": a.kv_stream,
                         "dup_tok_s": round(tps_d, 2), "pd_tok_s": round(tps_p, 2),
                         "dup_wall": round(dj["wall"], 2), "pd_wall": round(pj["wall"], 2),
                         "dup_ttft_p50_s": round(ttft, 3),
                         "dup_tpot_p50_ms": round(tpot_d * 1e3, 2),
                         "pd_tpot_p50_ms": round(tpot_p * 1e3, 2),
                         "pd_gap_ms_p50": gap, "match": match,
                         "dup_cards": dj["cards"], "pd_cards": pj["cards"],
                         "dup_ttfts": [round(r["ttft"], 3) for t in dj["data"].values() for r in t],
                         "pd_gaps": [r["handoff_gap_ms"] for r in pj["data"]["decode"]],
                         "pd_prefill_p50_s": round(p50([r["t_pre"] for r in pj["data"]["prefill"]]), 3)})
    print("=" * 78)
    if a.out:
        with open(a.out, "a") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
