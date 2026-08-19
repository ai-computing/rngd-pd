#!/usr/bin/env python3
"""ROADMAP C4 v2 — REAL 2-chip DUP vs PD, PROCESS-based (thread version hangs:
two ChipRuntimes in one process deadlock in the native runtime).

Each chip is owned by its own OS process; the PD handoff crosses processes
through the same shm connector path verified by llama_pd_llmd.py.

  orchestrator (default): spawns workers, synchronizes start, aggregates.
  --role dup    --chip N : whole requests (prefill P + decode D) on chip N.
  --role prefill --chip 0: prefill P steps + save_kv_layer×NL -> shm + marker.
  --role decode  --chip 1: wait marker, load KV, decode D steps.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_vs_dup_real2.py --P 48 --D 16 --n 6
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import _paths  # noqa: F401

WORK = "/dev/shm/rngd_c4v2"


def make_prompts(P, n):
    rng = np.random.default_rng(0)
    return [list(map(int, rng.integers(100, 120000, P))) for _ in range(n)]


# --------------------------- worker roles ---------------------------------

def load_runtime(model, S, chip):
    import torch
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM
    from generic_runtime import ModelConfig
    from llama_pd_2chip import ChipRuntime
    hf = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf
    rt = ChipRuntime(cfg, sd, S, chip=chip)
    return cfg, rt


def sync_start(tag):
    open(f"{WORK}/ready_{tag}", "w").write("1")
    while not os.path.exists(f"{WORK}/go"):
        time.sleep(0.05)


def role_dup(a):
    cfg, rt = load_runtime(a.model, a.S, a.chip)
    prompts = make_prompts(a.P, a.n)
    mine = [(i, p) for i, p in enumerate(prompts) if i % 2 == a.chip]
    sync_start(f"dup{a.chip}")
    res = []
    for i, toks in mine:
        ck, cv = rt.new_kv()
        t0 = time.perf_counter()
        for pos in range(len(toks)):
            nxt = rt.step(toks[pos], pos, ck, cv)
        ttft = time.perf_counter() - t0
        npos = len(toks)
        for _ in range(a.D - 1):
            nxt = rt.step(nxt, npos, ck, cv)
            npos += 1
        res.append({"i": i, "ttft": ttft, "e2e": time.perf_counter() - t0,
                    "t_end": time.perf_counter()})
    json.dump(res, open(f"{WORK}/dup{a.chip}.json", "w"))


def role_prefill(a):
    from rngd_kv_connector import RngdKVConnector, KVConnectorRole
    from kv_transport import make_transport
    cfg, rt = load_runtime(a.model, a.S, 0)
    NL = cfg.n_layers
    prompts = make_prompts(a.P, a.n)
    sync_start("prefill")
    res = []
    for i, toks in enumerate(prompts):
        tr = make_transport("shm", root=f"{WORK}/kv_{i}")
        pf = RngdKVConnector(KVConnectorRole.WORKER, req_id=f"r{i}",
                             transport=tr, n_layers=NL)
        ck, cv = rt.new_kv()
        t0 = time.perf_counter()
        for pos in range(len(toks)):
            nxt = rt.step(toks[pos], pos, ck, cv)
        t_pre = time.perf_counter() - t0
        tk = time.perf_counter()
        for L in range(NL):
            pf.save_kv_layer(f"layer.{L}", (ck[L], cv[L]))
        pf.wait_for_save()
        json.dump({"next_tok": int(nxt)}, open(f"{WORK}/meta_{i}.json", "w"))
        open(f"{WORK}/kvready_{i}", "w").write("1")
        res.append({"i": i, "t_pre": t_pre,
                    "kv_save_ms": (time.perf_counter() - tk) * 1e3,
                    "t_handoff": time.perf_counter()})
    json.dump(res, open(f"{WORK}/prefill.json", "w"))


def role_decode(a):
    from rngd_kv_connector import RngdKVConnector, KVConnectorRole
    from kv_transport import make_transport
    cfg, rt = load_runtime(a.model, a.S, 1)
    NL = cfg.n_layers
    prompts = make_prompts(a.P, a.n)
    sync_start("decode")
    res = []
    for i, toks in enumerate(prompts):
        while not os.path.exists(f"{WORK}/kvready_{i}"):
            time.sleep(0.02)
        t0 = time.perf_counter()
        tr = make_transport("shm", root=f"{WORK}/kv_{i}")
        dc = RngdKVConnector(KVConnectorRole.WORKER, req_id=f"r{i}",
                             transport=tr, n_layers=NL)
        ck, cv = rt.new_kv()
        dc.register_kv_caches((ck, cv))
        dc.bind_connector_metadata(dc.build_connector_meta(None))
        dc.start_load_kv()
        for L in range(NL):
            dc.wait_for_layer_load(f"layer.{L}")
        t_load = (time.perf_counter() - t0) * 1e3
        nxt = json.load(open(f"{WORK}/meta_{i}.json"))["next_tok"]
        npos = len(toks)
        for _ in range(a.D - 1):
            nxt = rt.step(nxt, npos, ck, cv)
            npos += 1
        res.append({"i": i, "kv_load_ms": t_load,
                    "t_end": time.perf_counter()})
    json.dump(res, open(f"{WORK}/decode.json", "w"))


# --------------------------- orchestrator ---------------------------------

def spawn(role, chip, a):
    cmd = [sys.executable, os.path.abspath(__file__), "--role", role,
           "--chip", str(chip), "--P", str(a.P), "--D", str(a.D),
           "--n", str(a.n), "--S", str(a.S), "--model", a.model]
    return subprocess.Popen(cmd, stdout=open(f"{WORK}/{role}{chip}.log", "w"),
                            stderr=subprocess.STDOUT)


def wait_ready_and_go(tags, procs, timeout=1200):
    t0 = time.time()
    while any(not os.path.exists(f"{WORK}/ready_{t}") for t in tags):
        if any(p.poll() is not None for p in procs):
            raise RuntimeError("worker died during startup — see logs in " + WORK)
        if time.time() - t0 > timeout:
            raise TimeoutError("workers not ready")
        time.sleep(0.5)
    t_go = time.perf_counter()
    open(f"{WORK}/go", "w").write("1")
    return t_go


def run_phase(mode, a):
    os.system(f"rm -rf {WORK} && mkdir -p {WORK}")
    if mode == "dup":
        procs = [spawn("dup", c, a) for c in (0, 1)]
        tags = ["dup0", "dup1"]
    else:
        procs = [spawn("prefill", 0, a), spawn("decode", 1, a)]
        tags = ["prefill", "decode"]
    t_go = wait_ready_and_go(tags, procs)
    for p in procs:
        p.wait(timeout=3600)
    wall = time.perf_counter() - t_go
    if mode == "dup":
        rows = (json.load(open(f"{WORK}/dup0.json")) +
                json.load(open(f"{WORK}/dup1.json")))
        return {"wall": wall, "rows": rows}
    return {"wall": wall,
            "pre": json.load(open(f"{WORK}/prefill.json")),
            "dec": json.load(open(f"{WORK}/decode.json"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--P", type=int, default=48)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--S", type=int, default=0)
    ap.add_argument("--role", default="orchestrator")
    ap.add_argument("--chip", type=int, default=0)
    a = ap.parse_args()
    if a.S == 0:
        from generic_runtime_wired import _bucket
        a.S = _bucket(a.P + a.D)

    if a.role == "dup":
        return role_dup(a)
    if a.role == "prefill":
        return role_prefill(a)
    if a.role == "decode":
        return role_decode(a)

    print(f"[c4v2] P={a.P} D={a.D} n={a.n} S={a.S} — spawning workers "
          f"(each loads 8B fp32 + compiles, ~3 min)...", flush=True)
    dup = run_phase("dup", a)
    print(f"  DUP done in {dup['wall']:.1f}s", flush=True)
    pd = run_phase("pd", a)
    print(f"  PD  done in {pd['wall']:.1f}s", flush=True)

    tot = a.n * a.D
    def p50(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]
    tps_dup, tps_pd = tot / dup["wall"], tot / pd["wall"]
    print("\n" + "=" * 70)
    print(f"C4v2 REAL 2-chip (process-parallel, real shm KV handoff)")
    print(f"  DUP: {tps_dup:6.2f} tok/s | TTFT p50 "
          f"{p50([r['ttft'] for r in dup['rows']])*1e3:6.0f} ms | wall {dup['wall']:.1f}s")
    print(f"  PD : {tps_pd:6.2f} tok/s | prefill p50 "
          f"{p50([r['t_pre'] for r in pd['pre']])*1e3:6.0f} ms | "
          f"kv save p50 {p50([r['kv_save_ms'] for r in pd['pre']]):4.0f} ms + "
          f"load p50 {p50([r['kv_load_ms'] for r in pd['dec']]):4.0f} ms | wall {pd['wall']:.1f}s")
    print(f"  measured PD/DUP = {tps_pd/tps_dup:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
