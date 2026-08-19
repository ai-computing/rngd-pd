#!/usr/bin/env python3
"""ROADMAP C4 — REAL 2-chip DUP vs PD throughput, measured (no queueing model).

Both chips run CONCURRENTLY (threads); PD moves the prompt KV chip0→chip1 through
the vLLM-shaped connector (shm) — the genuine handoff, not an emulation.

  DUP : chip0 and chip1 each process whole requests (prefill P + decode D).
  PD  : chip0 = prefill stage (P steps + save_kv_layer×NL), queue,
        chip1 = decode stage (wait_for_layer_load×NL + D steps).

Caveat (honest): ChipRuntime is the per-token wired path (~160 ms/tok incl. host
round-trips), so ABSOLUTE numbers are far below serving engines; the comparison
(DUP vs PD, overlap, handoff share) is the result. The queueing-model prediction
PD/DUP = (Tp+Td)/(2·max(Tp,Td)) is printed next to the measured ratio.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_vs_dup_real.py --P 48 --D 16 --n 6
"""
import argparse
import queue
import threading
import time

import numpy as np
import torch

import _paths  # noqa: F401
torch.set_grad_enabled(False)

from generic_runtime import ModelConfig                     # noqa: E402
from generic_runtime_wired import _bucket                   # noqa: E402
from llama_pd_2chip import ChipRuntime                      # noqa: E402
from rngd_kv_connector import RngdKVConnector, KVConnectorRole  # noqa: E402
from kv_transport import make_transport                     # noqa: E402


def run_dup(rts, prompts, D):
    """Each chip pulls whole requests off a shared queue."""
    q = queue.Queue()
    for i, p in enumerate(prompts):
        q.put((i, p))
    out = {}
    lock = threading.Lock()

    def worker(rt):
        while True:
            try:
                i, toks = q.get_nowait()
            except queue.Empty:
                return
            ck, cv = rt.new_kv()
            t0 = time.perf_counter()
            for pos in range(len(toks)):
                nxt = rt.step(toks[pos], pos, ck, cv)
            ttft = time.perf_counter() - t0
            n = len(toks)
            for _ in range(D - 1):
                nxt = rt.step(nxt, n, ck, cv)
                n += 1
            with lock:
                out[i] = {"ttft": ttft, "e2e": time.perf_counter() - t0}

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(rt,)) for rt in rts]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    return out, wall


def run_pd(rt0, rt1, prompts, D, NL):
    """chip0 prefill stage -> connector (shm) -> chip1 decode stage."""
    handoff = queue.Queue(maxsize=2)
    out = {}
    kv_ms = []

    def prefiller():
        for i, toks in enumerate(prompts):
            tr = make_transport("shm", root=f"/dev/shm/rngd_c4_{i}")
            pf = RngdKVConnector(KVConnectorRole.WORKER, req_id=f"r{i}",
                                 transport=tr, n_layers=NL)
            ck, cv = rt0.new_kv()
            t0 = time.perf_counter()
            for pos in range(len(toks)):
                nxt = rt0.step(toks[pos], pos, ck, cv)
            t_pre = time.perf_counter() - t0
            tk0 = time.perf_counter()
            for L in range(NL):
                pf.save_kv_layer(f"layer.{L}", (ck[L], cv[L]))
            pf.wait_for_save()
            kv_ms.append((time.perf_counter() - tk0) * 1e3)
            handoff.put((i, toks, nxt, tr, t0, t_pre))
        handoff.put(None)

    def decoder():
        while True:
            item = handoff.get()
            if item is None:
                return
            i, toks, nxt, tr, t0, t_pre = item
            dc = RngdKVConnector(KVConnectorRole.WORKER, req_id=f"r{i}",
                                 transport=tr, n_layers=NL)
            ck, cv = rt1.new_kv()
            dc.register_kv_caches((ck, cv))
            dc.bind_connector_metadata(dc.build_connector_meta(None))
            dc.start_load_kv()
            tk0 = time.perf_counter()
            for L in range(NL):
                dc.wait_for_layer_load(f"layer.{L}")
            t_load = (time.perf_counter() - tk0) * 1e3
            n = len(toks)
            for _ in range(D - 1):
                nxt = rt1.step(nxt, n, ck, cv)
                n += 1
            out[i] = {"ttft": t_pre, "e2e": time.perf_counter() - t0,
                      "kv_load_ms": t_load}

    t0 = time.perf_counter()
    tp = threading.Thread(target=prefiller)
    td = threading.Thread(target=decoder)
    tp.start(); td.start(); tp.join(); td.join()
    wall = time.perf_counter() - t0
    return out, wall, kv_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--P", type=int, default=48, help="prompt tokens")
    ap.add_argument("--D", type=int, default=16, help="decode tokens")
    ap.add_argument("--n", type=int, default=6, help="requests")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM
    print(f"loading {a.model}...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf
    NL = cfg.n_layers
    S = _bucket(a.P + a.D)
    rng = np.random.default_rng(0)
    prompts = [list(rng.integers(100, 120000, a.P)) for _ in range(a.n)]

    print(f"[c4] compiling chip0/chip1 (S={S})...", flush=True)
    rt0 = ChipRuntime(cfg, sd, S, chip=0)
    rt1 = ChipRuntime(cfg, sd, S, chip=1)
    print(f"  compiled: {rt0.compile_s:.0f}s / {rt1.compile_s:.0f}s", flush=True)

    # warm each chip once (first-call overheads)
    ck, cv = rt0.new_kv(); rt0.step(prompts[0][0], 0, ck, cv)
    ck, cv = rt1.new_kv(); rt1.step(prompts[0][0], 0, ck, cv)

    # per-step calibration for the queueing-model prediction
    ck, cv = rt0.new_kv()
    t0 = time.perf_counter()
    for pos in range(8):
        rt0.step(prompts[0][pos], pos, ck, cv)
    ms_step = (time.perf_counter() - t0) / 8 * 1e3

    dup, wall_dup = run_dup([rt0, rt1], prompts, a.D)
    pd, wall_pd, kv_ms = run_pd(rt0, rt1, prompts, a.D, NL)

    tot_out = a.n * a.D
    tps_dup = tot_out / wall_dup
    tps_pd = tot_out / wall_pd
    Tp, Td = a.P * ms_step, a.D * ms_step
    model_ratio = (Tp + Td) / (2 * max(Tp, Td))

    def p50(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    print("\n" + "=" * 72)
    print(f"C4 REAL 2-chip DUP vs PD — Llama-8B wired, P={a.P} D={a.D} n={a.n} "
          f"(step≈{ms_step:.0f} ms)")
    print(f"  DUP : {tps_dup:6.2f} tok/s | TTFT p50 {p50([r['ttft'] for r in dup.values()])*1e3:7.0f} ms | wall {wall_dup:.1f}s")
    print(f"  PD  : {tps_pd:6.2f} tok/s | TTFT p50 {p50([r['ttft'] for r in pd.values()])*1e3:7.0f} ms | wall {wall_pd:.1f}s")
    print(f"  KV handoff (real, shm): save p50 {p50(kv_ms):.0f} ms + load p50 "
          f"{p50([r['kv_load_ms'] for r in pd.values()]):.0f} ms")
    print(f"  measured PD/DUP = {tps_pd/tps_dup:.2f} | queueing-model prediction = {model_ratio:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
