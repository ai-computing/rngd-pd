#!/usr/bin/env python3
"""BATCHED resident-KV decode — the missing piece that lets the PD decode chip
serve B streams per step (weights read once), reclaiming its idle PE.

llama_engine_batch.py proved the batched DSL kernels compile (B up to 128) but
it is a perf harness: no RoPE, no KV injection, random weights. This file is
the CORRECT batched engine: WireLayerB = resident WireLayer + batch dim,
lockstep positions (all B streams at the same pos — exact for uniform-P
benches; ragged admission is orchestration on top).

Modes:
  --validate : B lockstep decode == B independent sequential decodes
  bench      : DUP-batched (each chip batches B/2 streams) vs PD-batched
               (prefill chip streams KV per request; decode chip batches B)

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_engine_batch.py --validate --nl 2 --B 4
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

import _paths  # noqa: F401

WORK = "/dev/shm/rngd_pdb"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts_cache")

H, M, NH, NKV, HD = 4096, 14336, 32, 8, 128
G = NH // NKV
EPS = 1e-5
SCALE = 1.0 / math.sqrt(HD)


def build_batch_engine(model, S, nl, B, chip):
    import torch
    torch.set_grad_enabled(False)
    import furiosa.torch                                        # noqa: F401
    from artifact_cache import cache_key, get_or_compile
    from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig
    from furiosa.torch.custom_ops import CompileModule, TacticKernelModule
    from llama_engine_batch import (rms_dsl, softmax_dsl, swiglu_dsl,
                                    zipadd_dsl)
    from transformers import AutoModelForCausalLM
    from generic_runtime import ModelConfig

    BF = torch.bfloat16
    DEV = f"rngd:{chip}"

    class WireLayerB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rms1 = TacticKernelModule(rms_dsl(B))
            self.rms2 = TacticKernelModule(rms_dsl(B))
            self.sm = TacticKernelModule(softmax_dsl(B, S))
            self.r1 = TacticKernelModule(zipadd_dsl(B))
            self.r2 = TacticKernelModule(zipadd_dsl(B))
            self.sw = TacticKernelModule(swiglu_dsl(B))

        def _rope(self, x, nh_, rotT):
            # SLICE-FREE RoPE: last-dim slices on 3D activations miscompile
            # (batch K came out 72% wrong at pos0 where rope is identity), so
            # rotate via a constant [HD,HD] block-rotation matrix instead:
            # y = x @ R(pos)^T  — pure batched matmul, verified lowering.
            xr = x.view(B, nh_, HD).to(torch.float32)
            return torch.matmul(xr, rotT).reshape(B, nh_ * HD).to(BF)

        def forward(self, x, eps, nh, n1, wq, wk, wv, wo, n2, wg, wu, wd,
                    Kp, Vp, rotT, mask, oh):
            h = self.rms1(x, n1, eps, nh)[0]                     # [B,H]
            q = self._rope(torch.matmul(h, wq), NH, rotT)
            k = self._rope(torch.matmul(h, wk), NKV, rotT)
            v = torch.matmul(h, wv)                              # [B,Hkv]
            Kf = Kp + k.view(B, NKV, HD, 1) * oh.view(1, 1, 1, S)   # oh bf16
            Vf = Vp + v.view(B, NKV, 1, HD) * oh.view(1, 1, S, 1)
            sc = torch.matmul(q.reshape(B, NKV, G, HD), Kf) + mask.view(1, 1, 1, S)
            p = self.sm(sc.to(BF))[0]
            ao = torch.matmul(p, Vf).reshape(B, NH * HD)
            x2 = self.r1(torch.matmul(ao, wo), x)[0]
            h2 = self.rms2(x2, n2, eps, nh)[0]
            gg = torch.matmul(h2, wg)                            # [B,M]
            uu = torch.matmul(h2, wu)                            # [B,M]
            gated = self.sw(gg, uu)[0]
            return self.r2(torch.matmul(gated, wd), x2)[0], Kf, Vf

    class WireModelB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([WireLayerB() for _ in range(nl)])

        def forward(self, x, eps, nh, an1, aq, ak, av, ao, an2, awg, awu, awd,
                    Kp, Vp, rotT, mask, oh):
            ks, vs = [], []
            for i in range(nl):
                x, Kf, Vf = self.layers[i](x, eps, nh, an1[i], aq[i], ak[i],
                                           av[i], ao[i], an2[i], awg[i],
                                           awu[i], awd[i], Kp[i], Vp[i],
                                           rotT, mask, oh)
                ks.append(Kf)
                vs.append(Vf)
            return x, torch.stack(ks), torch.stack(vs)

    hf = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf

    g = lambda k: sd[k].astype(np.float32)
    sqrtH = math.sqrt(H)
    d = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)
    df = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32).to(DEV)

    # weights: matmul(h[B,H], W[H,out]) -> store TRANSPOSED once
    an1 = torch.stack([d(np.tile(g(f"model.layers.{i}.input_layernorm.weight") * sqrtH, (B, 1))) for i in range(nl)])
    an2 = torch.stack([d(np.tile(g(f"model.layers.{i}.post_attention_layernorm.weight") * sqrtH, (B, 1))) for i in range(nl)])
    aq = torch.stack([d((g(f"model.layers.{i}.self_attn.q_proj.weight") * SCALE).T.copy()) for i in range(nl)])
    ak = torch.stack([d(g(f"model.layers.{i}.self_attn.k_proj.weight").T.copy()) for i in range(nl)])
    av = torch.stack([d(g(f"model.layers.{i}.self_attn.v_proj.weight").T.copy()) for i in range(nl)])
    ao = torch.stack([d(g(f"model.layers.{i}.self_attn.o_proj.weight").T.copy()) for i in range(nl)])
    awg = torch.stack([d(g(f"model.layers.{i}.mlp.gate_proj.weight").T.copy()) for i in range(nl)])
    awu = torch.stack([d(g(f"model.layers.{i}.mlp.up_proj.weight").T.copy()) for i in range(nl)])
    awd = torch.stack([d(g(f"model.layers.{i}.mlp.down_proj.weight").T.copy()) for i in range(nl)])

    z = lambda *s: torch.zeros(*s, dtype=BF)
    half = HD // 2
    sample = (z(B, H), torch.zeros(B, H, dtype=torch.float32),
              torch.zeros(B, H, dtype=torch.float32),
              z(nl, B, H), z(nl, H, NH * HD), z(nl, H, NKV * HD),
              z(nl, H, NKV * HD), z(nl, NH * HD, H), z(nl, B, H),
              z(nl, H, M), z(nl, H, M), z(nl, M, H),
              z(nl, B, NKV, HD, S), z(nl, B, NKV, S, HD),
              torch.zeros(HD, HD, dtype=torch.float32),   # rotT (RoPE matrix)
              torch.zeros(S, dtype=torch.float32),   # mask f32 (adds pre-cast)
              z(S))                                  # onehot bf16 (exact 0/1)
    cfg_c = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                     tactic_hint=TacticHintConfig.ForLlmModelDecode)
    key = cache_key(kind="batch_resident", model=model, S=S, nl=nl, B=B, rev=3)
    m, mode, secs = get_or_compile(
        CACHE, key,
        lambda: CompileModule.from_module(WireModelB(), sample,
                                          compiler_config=cfg_c), device=None)
    m.to(DEV)
    print(f"[batch-engine chip{chip}] {mode} in {secs:.0f}s (B={B} S={S} nl={nl})",
          flush=True)

    class Runtime:
        pass
    rt = Runtime()
    rt.m = m
    rt.W = (an1, aq, ak, av, ao, an2, awg, awu, awd)
    rt.cfg = cfg
    rt.S, rt.nl, rt.B = S, nl, B
    rt.inv = cfg.inv_freq()
    rt.embed = g("model.embed_tokens.weight")
    rt.norm = g("model.norm.weight")
    rt.lm = rt.embed if cfg.tie else g("lm_head.weight")
    rt.eps_d = df(np.full((B, H), EPS * H, np.float32))
    rt.nh_d = df(np.full((B, H), -0.5, np.float32))
    rt.d, rt.df, rt.DEV = d, df, DEV
    return rt


class BatchSession:
    """Lockstep decode of B streams on a batched resident engine."""

    def __init__(self, rt, lmhead=None):
        self.rt = rt
        self.lmh = lmhead

    def fresh_kv(self):
        r = self.rt
        return (r.d(np.zeros((r.nl, r.B, NKV, HD, r.S), np.float32)),
                r.d(np.zeros((r.nl, r.B, NKV, r.S, HD), np.float32)))

    def step(self, toks, pos, Kp, Vp):
        """toks: list[B] token ids, all at the same pos. Returns next[B]."""
        r = self.rt
        an1, aq, ak, av, ao, an2, awg, awu, awd = r.W
        ang = pos * r.inv
        cosv, sinv = np.cos(ang), np.sin(ang)
        half = HD // 2
        Mrot = np.zeros((HD, HD), np.float32)
        i = np.arange(half)
        Mrot[i, i] = cosv
        Mrot[i, i + half] = -sinv
        Mrot[i + half, i + half] = cosv
        Mrot[i + half, i] = sinv
        rotT = r.df(Mrot.T.copy())
        mask = np.full(r.S, -1e4, np.float32)
        mask[:pos + 1] = 0.0
        oh = np.zeros(r.S, np.float32)
        oh[pos] = 1.0
        x = r.d(np.stack([r.embed[t] for t in toks]).astype(np.float32))
        hid, Kp, Vp = r.m(x, r.eps_d, r.nh_d, an1, aq, ak, av, ao, an2,
                          awg, awu, awd, Kp, Vp, rotT, r.df(mask), r.d(oh))
        from generic_runtime import rmsnorm
        hh = hid.detach().float().cpu().numpy()
        nxt = [int(np.argmax(r.lm @ rmsnorm(hh[b], r.norm, r.cfg.eps)))
               for b in range(r.B)]
        return nxt, Kp, Vp


def validate(a):
    """B lockstep streams == B sequential single-stream decodes."""
    from pd_engine import EngineSession, build_engine
    import llama_engine_resident as LER

    rng = np.random.default_rng(7)
    prompts = [list(map(int, rng.integers(100, 120000, a.P))) for _ in range(a.B)]

    # reference: sequential single-stream engine (cached)
    cfg, rt1 = build_engine(a.model, a.S, a.nl, chip=a.chips[0])
    s1 = EngineSession(rt1, LER)
    ref = []
    for toks in prompts:
        Kp, Vp = s1.fresh_kv()
        nxt = None
        for pos in range(len(toks)):
            nxt, Kp, Vp = s1.step(toks[pos], pos, Kp, Vp)
        out = [nxt]
        p = len(toks)
        for _ in range(a.D - 1):
            nxt, Kp, Vp = s1.step(nxt, p, Kp, Vp)
            out.append(nxt)
            p += 1
        ref.append(out)

    # batched lockstep
    rt = build_batch_engine(a.model, a.S, a.nl, a.B, chip=a.chips[1])
    sb = BatchSession(rt)
    Kp, Vp = sb.fresh_kv()
    cur = [prompts[b][0] for b in range(a.B)]
    nxt = None
    for pos in range(a.P):
        cur = [prompts[b][pos] for b in range(a.B)]
        nxt, Kp, Vp = sb.step(cur, pos, Kp, Vp)
    outs = [[nxt[b]] for b in range(a.B)]
    ts = []
    p = a.P
    for _ in range(a.D - 1):
        t0 = time.perf_counter()
        nxt, Kp, Vp = sb.step(nxt, p, Kp, Vp)
        ts.append(time.perf_counter() - t0)
        for b in range(a.B):
            outs[b].append(nxt[b])
        p += 1
    ok = outs == ref
    step_ms = sorted(ts)[len(ts) // 2] * 1e3
    print(f"batched step p50: {step_ms:.1f} ms for B={a.B} "
          f"-> {a.B / step_ms * 1e3:.1f} tok/s/chip "
          f"({step_ms / a.B:.2f} ms/token effective)")
    print(f"VALIDATE {'PASS ✅ lockstep batch == sequential' if ok else 'FAIL ❌'}")
    if not ok:
        for b in range(a.B):
            if outs[b] != ref[b]:
                print(f"  stream {b}: batch={outs[b][:8]} ref={ref[b][:8]}")


def _prompts(P, n, seed=0):
    rng = np.random.default_rng(seed)
    return [list(map(int, rng.integers(100, 120000, P))) for _ in range(n)]


def _bf16_bytes(t):
    import torch
    return t.detach().cpu().view(torch.uint16).numpy().tobytes()


def prefill_batch(sess, prompts, P):
    """Lockstep-prefill len(prompts)==B streams; returns (next[B], Kp, Vp)."""
    Kp, Vp = sess.fresh_kv()
    nxt = None
    for pos in range(P):
        nxt, Kp, Vp = sess.step([pr[pos] for pr in prompts], pos, Kp, Vp)
    return nxt, Kp, Vp


def decode_batch(sess, nxt, P, D, Kp, Vp):
    outs = [[t] for t in nxt]
    ts = []
    p = P
    for _ in range(D - 1):
        t0 = time.perf_counter()
        nxt, Kp, Vp = sess.step(nxt, p, Kp, Vp)
        ts.append(time.perf_counter() - t0)
        for b, t in enumerate(nxt):
            outs[b].append(t)
        p += 1
    tpot = sorted(ts)[len(ts) // 2] if ts else None
    return outs, tpot


def role_dup(a):
    rt = build_batch_engine(a.model, a.S, a.nl, a.B // 2, chip=a.chip)
    s = BatchSession(rt)
    half = a.B // 2
    lo = a.chip_rank * half
    res = []
    open(f"{WORK}/ready_dup{a.chip}", "w").write("1")
    while not os.path.exists(f"{WORK}/go"):
        time.sleep(0.05)
    for k in range(a.batches):
        prompts = _prompts(a.P, a.B, seed=100 + k)[lo:lo + half]
        t0 = time.perf_counter()
        nxt, Kp, Vp = prefill_batch(s, prompts, a.P)
        ttft = time.perf_counter() - t0
        outs, tpot = decode_batch(s, nxt, a.P, a.D, Kp, Vp)
        res.append({"batch": k, "ttft": ttft, "tpot": tpot, "outs": outs,
                    "t_end": time.perf_counter()})
    json.dump(res, open(f"{WORK}/dup{a.chip}.json", "w"))


def role_pf(a):
    import torch
    rt = build_batch_engine(a.model, a.S, a.nl, a.B, chip=a.chip)
    s = BatchSession(rt)
    from kv_transport import make_transport
    open(f"{WORK}/ready_pf", "w").write("1")
    while not os.path.exists(f"{WORK}/go"):
        time.sleep(0.05)
    res = []
    for k in range(a.batches):
        prompts = _prompts(a.P, a.B, seed=100 + k)
        tr = make_transport("shm", root=f"{WORK}/kv_{k}")
        t0 = time.perf_counter()
        nxt, Kp, Vp = prefill_batch(s, prompts, a.P)
        t_pre = time.perf_counter() - t0
        tk = time.perf_counter()
        tr.push("K", _bf16_bytes(Kp))
        tr.push("V", _bf16_bytes(Vp))
        json.dump({"next": nxt, "t_pre_wall": time.time()},
                  open(f"{WORK}/meta_{k}.json", "w"))
        open(f"{WORK}/kvready_{k}", "w").write("1")
        res.append({"batch": k, "t_pre": t_pre,
                    "kv_save_ms": (time.perf_counter() - tk) * 1e3})
    json.dump(res, open(f"{WORK}/pf.json", "w"))


def role_dec(a):
    import torch
    rt = build_batch_engine(a.model, a.S, a.nl, a.B, chip=a.chip)
    s = BatchSession(rt)
    from kv_transport import make_transport
    open(f"{WORK}/ready_dec", "w").write("1")
    while not os.path.exists(f"{WORK}/go"):
        time.sleep(0.05)
    res = []
    for k in range(a.batches):
        while not os.path.exists(f"{WORK}/kvready_{k}"):
            time.sleep(0.005)
        t0 = time.perf_counter()
        tr = make_transport("shm", root=f"{WORK}/kv_{k}")

        def pull(tag, shape):
            u = np.frombuffer(tr.pull(tag), np.uint16).copy()
            return torch.from_numpy(u).view(torch.bfloat16).float() \
                        .numpy().reshape(shape)
        Kh = pull("K", (a.nl, a.B, NKV, HD, a.S))
        Vh = pull("V", (a.nl, a.B, NKV, a.S, HD))
        Kp = rt.d(Kh); Vp = rt.d(Vh)
        t_load = (time.perf_counter() - t0) * 1e3
        meta = json.load(open(f"{WORK}/meta_{k}.json"))
        gap = (time.time() - meta["t_pre_wall"]) * 1e3
        outs, tpot = decode_batch(s, meta["next"], a.P, a.D, Kp, Vp)
        res.append({"batch": k, "kv_load_ms": t_load, "gap_ms": gap,
                    "tpot": tpot, "outs": outs, "t_end": time.perf_counter()})
    json.dump(res, open(f"{WORK}/dec.json", "w"))


def orchestrate(a):
    def spawn(role, chip, rank):
        cmd = [sys.executable, os.path.abspath(__file__), "--role", role,
               "--chip", str(chip), "--chip-rank", str(rank),
               "--S", str(a.S), "--nl", str(a.nl), "--B", str(a.B),
               "--P", str(a.P), "--D", str(a.D), "--batches", str(a.batches),
               "--model", a.model]
        return subprocess.Popen(cmd, stdout=open(f"{WORK}/{role}{chip}.log", "w"),
                                stderr=subprocess.STDOUT)

    def run(mode):
        os.system(f"rm -rf {WORK} && mkdir -p {WORK}")
        if mode == "dup":
            procs = [spawn("dup", a.chips[0], 0), spawn("dup", a.chips[1], 1)]
            tags = [f"dup{a.chips[0]}", f"dup{a.chips[1]}"]
        else:
            procs = [spawn("pf", a.chips[0], 0), spawn("dec", a.chips[1], 1)]
            tags = ["pf", "dec"]
        while any(not os.path.exists(f"{WORK}/ready_{t}") for t in tags):
            if any(p.poll() is not None and p.returncode != 0 for p in procs):
                raise RuntimeError(f"worker died — {WORK}/*.log")
            time.sleep(0.5)
        t0 = time.perf_counter()
        open(f"{WORK}/go", "w").write("1")
        for p in procs:
            p.wait(timeout=3600)
        wall = time.perf_counter() - t0
        if mode == "dup":
            data = {t: json.load(open(f"{WORK}/{t}.json")) for t in tags}
        else:
            data = {"pf": json.load(open(f"{WORK}/pf.json")),
                    "dec": json.load(open(f"{WORK}/dec.json"))}
        return wall, data

    n = a.B * a.batches
    print(f"[batched-bench] P={a.P} D={a.D} B={a.B} batches={a.batches} "
          f"(n={n}) S={a.S} nl={a.nl}", flush=True)
    wall_d, dd = run("dup")
    print(f"  DUP  done {wall_d:.1f}s", flush=True)
    wall_p, dp = run("pd")
    print(f"  PD   done {wall_p:.1f}s", flush=True)

    tot = n * a.D
    dup_outs = {}
    for t, rows in dd.items():
        rank = 0 if t.endswith(str(a.chips[0])) else 1
        for r in rows:
            for j, o in enumerate(r["outs"]):
                dup_outs[(r["batch"], rank * (a.B // 2) + j)] = o
    pd_outs = {}
    for r in dp["dec"]:
        for j, o in enumerate(r["outs"]):
            pd_outs[(r["batch"], j)] = o
    matches = sum(dup_outs.get(key) == pd_outs.get(key) for key in pd_outs)
    dup_tpot = np.median([r["tpot"] for t in dd.values() for r in t])
    pd_tpot = np.median([r["tpot"] for r in dp["dec"]])
    print("=" * 72)
    print(f"BATCHED DUP vs PD (lockstep, B per chip: DUP {a.B//2} / PD {a.B})")
    print(f"  DUP: {tot/wall_d:7.2f} tok/s | prefill(batch) p50 "
          f"{np.median([r['ttft'] for t in dd.values() for r in t])*1e3:6.0f} ms | "
          f"tpot {dup_tpot*1e3:5.1f} ms")
    print(f"  PD : {tot/wall_p:7.2f} tok/s | prefill(batch) p50 "
          f"{np.median([r['t_pre'] for r in dp['pf']])*1e3:6.0f} ms | "
          f"tpot {pd_tpot*1e3:5.1f} ms | kv gap p50 "
          f"{np.median([r['gap_ms'] for r in dp['dec']]):5.0f} ms")
    print(f"  PD/DUP = {wall_d/wall_p:.2f} | stream output match: "
          f"{matches}/{len(pd_outs)}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--chips", default="0,1")
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--chip-rank", type=int, default=0)
    ap.add_argument("--role", default="orchestrator")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    a.chips = [int(x) for x in a.chips.split(",")] if isinstance(a.chips, str) else a.chips
    if a.role == "dup":
        return role_dup(a)
    if a.role == "pf":
        return role_pf(a)
    if a.role == "dec":
        return role_dec(a)
    if a.validate:
        return validate(a)
    return orchestrate(a)


if __name__ == "__main__":
    main()
