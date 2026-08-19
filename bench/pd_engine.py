#!/usr/bin/env python3
"""ENGINE-CLASS 2-chip PD vs DUP — resident-KV whole-model runtime (~16 ms/tok),
one-shot KV handoff (single d2h -> shm -> single h2d per request).

The C4 wired path pays a host round-trip per LAYER (~3k/request). The resident
engine (llama_engine_resident.WireModel) keeps KV on device across steps, so a
PD request costs exactly ONE KV export after prefill and ONE import before
decode — the same shape a real KV-transfer API would have.

Modes:
  --validate          : single process, small nl — PD flow output == monolithic
  (default orchestr.) : DUP (2 engine replicas) vs PD (prefill chip0 + decode
                        chip1, shm KV blob handoff), process-per-chip,
                        artifact-cached compiles.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_engine.py --P 512 --D 64 --n 4 --nl 32
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import _paths  # noqa: F401

WORK = "/dev/shm/rngd_pd_engine"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts_cache")


def make_prompts(P, n):
    rng = np.random.default_rng(0)
    return [list(map(int, rng.integers(100, 120000, P))) for _ in range(n)]


# ------------------------- engine construction -----------------------------

def build_engine(model, S, nl, chip):
    """ResidentRuntime on a given chip, with EDF artifact caching."""
    import torch
    torch.set_grad_enabled(False)
    import llama_engine_resident as LER
    from artifact_cache import get_or_compile, cache_key
    from furiosa.torch.custom_ops import CompileModule
    from transformers import AutoModelForCausalLM
    from generic_runtime import ModelConfig

    LER.DEV = f"rngd:{chip}"                    # module-global device patch
    hf = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf

    key = cache_key(kind="resident", model=model, S=S, nl=nl)
    orig_from_module = CompileModule.from_module
    holder = {}

    def caching_from_module(m, args, **kw):
        def compile_fn():
            return orig_from_module(m, args, **kw)
        mod, mode, secs = get_or_compile(CACHE, key, compile_fn, device=None)
        holder["mode"], holder["secs"] = mode, secs
        return mod

    CompileModule.from_module = staticmethod(caching_from_module)
    try:
        t0 = time.time()
        rt = LER.ResidentRuntime(cfg, sd, S, nl)
        print(f"[engine chip{chip}] {holder.get('mode','?')} in "
              f"{holder.get('secs', time.time()-t0):.0f}s (S={S}, nl={nl})", flush=True)
    finally:
        CompileModule.from_module = orig_from_module
    return cfg, rt


def build_lmhead(rt, LER, chip):
    """Device lm_head: RMS-norm + [V,H] matmul compiled on-chip; the host then
    argmaxes 128k logits (256 KB download) instead of running a 1 GFLOP matmul
    per token on CPU (~5-7 ms of the 11.2 ms step)."""
    import math
    import torch
    import llama_engine as LE
    from artifact_cache import get_or_compile, cache_key
    from furiosa.torch.custom_ops import CompileModule, TacticKernelModule
    from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig

    H = LER.H
    V = rt.lm.shape[0]
    G = 32                      # chunk V into G groups: the monolithic [V,H]
    Vp = (V + G - 1) // G * G   # matvec tactic streams at ~23 GB/s (46 ms);
    #                             [G, V/G, H] batched form hits ~565 GB/s (1.9 ms)

    class LmHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = TacticKernelModule(LE.RMS_NORM)

        def forward(self, x, nw, eps, nh, wlm):
            h = self.rms(x, nw, eps, nh)[0]
            return torch.matmul(wlm, h)         # [G, Vp/G, H] @ [H] -> [G, Vp/G]

    BF = torch.bfloat16
    sample = (torch.zeros(H, dtype=BF), torch.zeros(H, dtype=BF),
              torch.zeros(H, dtype=torch.float32), torch.zeros(H, dtype=torch.float32),
              torch.zeros(G, Vp // G, H, dtype=BF))
    cfg_c = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                     tactic_hint=TacticHintConfig.ForLlmModelDecode)
    key = cache_key(kind="lmhead", V=V, H=H, tactic="decode-chunked", G=G)
    m, mode, secs = get_or_compile(
        CACHE, key,
        lambda: CompileModule.from_module(LmHead(), sample, compiler_config=cfg_c),
        device=None)
    m.to(f"rngd:{chip}")
    print(f"[lmhead chip{chip}] {mode} in {secs:.0f}s (V={V}, G={G})", flush=True)
    sqrtH = math.sqrt(H)
    norm_d = LER._d((rt.norm * sqrtH).astype("float32"))
    import numpy as np
    wpad = np.zeros((Vp, H), np.float32)
    wpad[:V] = rt.lm.astype(np.float32)
    wlm_d = LER._d(wpad.reshape(G, Vp // G, H))
    return m, norm_d, wlm_d, V


def build_chunk_kernels(rt, LER, chip, C):
    """Device-side KV chunk gather/scatter for STREAMING handoff.

    Naive strided slices (K[..., a:b]) host-bridge at ~68 ms/4 MB — slower than
    the one-shot. Instead a selection-matrix matmul repacks the chunk
    CONTIGUOUSLY on device (~0.3 ms), so the d2h/h2d legs shrink to ~2 ms:

      gather : Kc = K @ Sel        [nl,NKV,HD,S]@[S,C] -> [nl,NKV,HD,C]
               Vc = SelT @ V       [C,S]@[nl,NKV,S,HD] -> [nl,NKV,C,HD]
      scatter: K' = K + Kc @ SelT  (chunks land in zeroed positions)
               V' = V + Sel @ Vc
    """
    import torch
    from artifact_cache import get_or_compile, cache_key
    from furiosa.torch.custom_ops import CompileModule
    from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig

    nl, S = rt.nl, rt.S
    NKV, HD = LER.NKV, LER.HD
    BF = torch.bfloat16

    class Gather(torch.nn.Module):
        def forward(self, K, V, sel, selT):
            return torch.matmul(K, sel), torch.matmul(selT, V)

    class Scatter(torch.nn.Module):
        def forward(self, K, V, Kc, Vc, sel, selT):
            return K + torch.matmul(Kc, selT), V + torch.matmul(sel, Vc)

    cfg_c = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                     tactic_hint=TacticHintConfig.ForLlmModelDecode)
    z = lambda *s: torch.zeros(*s, dtype=BF)
    g_sample = (z(nl, NKV, HD, S), z(nl, NKV, S, HD), z(S, C), z(C, S))
    s_sample = (z(nl, NKV, HD, S), z(nl, NKV, S, HD),
                z(nl, NKV, HD, C), z(nl, NKV, C, HD), z(S, C), z(C, S))
    kg = cache_key(kind="kvgather", nl=nl, S=S, C=C)
    ks = cache_key(kind="kvscatter", nl=nl, S=S, C=C)
    mg, mode_g, sg = get_or_compile(
        CACHE, kg, lambda: CompileModule.from_module(Gather(), g_sample,
                                                     compiler_config=cfg_c))
    ms, mode_s, ss = get_or_compile(
        CACHE, ks, lambda: CompileModule.from_module(Scatter(), s_sample,
                                                     compiler_config=cfg_c))
    mg.to(f"rngd:{chip}"); ms.to(f"rngd:{chip}")
    print(f"[kvstream chip{chip}] gather {mode_g} {sg:.0f}s, "
          f"scatter {mode_s} {ss:.0f}s (C={C})", flush=True)
    # selection matrices for every chunk start (uploaded once)
    import numpy as np
    sels = []
    for a in range(0, S, C):
        m = np.zeros((S, C), np.float32)
        for j in range(C):
            if a + j < S:
                m[a + j, j] = 1.0
        sels.append((LER._d(m), LER._d(m.T.copy())))
    return mg, ms, sels


class EngineSession:
    """Streamed decode on a ResidentRuntime with exportable/importable KV."""

    def __init__(self, rt, LER, lmhead=None):
        self.rt = rt
        self.LER = LER
        self.lmh = lmhead              # (module, norm_d, wlm_d) or None

    def fresh_kv(self):
        import numpy as np
        r = self.rt
        Kp = self.LER._d(np.zeros((r.nl, self.LER.NKV, self.LER.HD, r.S), np.float32))
        Vp = self.LER._d(np.zeros((r.nl, self.LER.NKV, r.S, self.LER.HD), np.float32))
        return Kp, Vp

    def step(self, tok, pos, Kp, Vp):
        import numpy as np
        r, L = self.rt, self.LER
        an1, aq, ak, av, ao, an2, awgu, awd = r.W
        ang = pos * r.inv
        cos = L._d(np.cos(ang)); sin = L._d(np.sin(ang))
        mask = np.full(r.S, -1e4, np.float32); mask[:pos + 1] = 0.0
        oh = np.zeros(r.S, np.float32); oh[pos] = 1.0
        x = L._d(r.embed[tok].astype(np.float32))
        hid, Kp, Vp = r.m(x, r.eps_d, r.nh_d, an1, aq, ak, av, ao, an2, awgu, awd,
                          Kp, Vp, cos, sin, L._d(mask), L._d(oh))
        if self.lmh is not None:
            m, norm_d, wlm_d, V = self.lmh
            logits = m(hid, norm_d, r.eps_d, r.nh_d, wlm_d)
            nxt = int(np.argmax(
                logits.detach().float().cpu().numpy().reshape(-1)[:V]))
        else:
            from generic_runtime import rmsnorm
            nxt = int(np.argmax(r.lm @ rmsnorm(hid.detach().float().cpu().numpy(),
                                               r.norm, r.c.eps)))
        return nxt, Kp, Vp

    def export_kv(self, Kp, Vp):
        """ONE device->host download of the full stacked KV."""
        return (Kp.detach().float().cpu().numpy(), Vp.detach().float().cpu().numpy())

    def import_kv(self, Kh, Vh):
        """ONE host->device upload."""
        return self.LER._d(Kh), self.LER._d(Vh)


# ------------------------------ validate -----------------------------------

def validate(a):
    import llama_engine_resident as LER
    cfg, rt0 = build_engine(a.model, a.S, a.nl, chip=2)
    s0 = EngineSession(rt0, LER)
    prompts = make_prompts(a.P, 1)
    toks = prompts[0]

    # monolithic on chip2
    Kp, Vp = s0.fresh_kv()
    mono = []
    nxt = None
    for pos in range(len(toks)):
        nxt, Kp, Vp = s0.step(toks[pos], pos, Kp, Vp)
    mono.append(nxt)
    p = len(toks)
    for _ in range(a.D - 1):
        nxt, Kp, Vp = s0.step(nxt, p, Kp, Vp)
        mono.append(nxt)
        p += 1

    # PD: prefill chip2 -> export -> import chip3 -> decode chip3
    LER.DEV = "rngd:2"
    Kp, Vp = s0.fresh_kv()
    for pos in range(len(toks)):
        nxt1, Kp, Vp = s0.step(toks[pos], pos, Kp, Vp)
    t0 = time.perf_counter()
    Kh, Vh = s0.export_kv(Kp, Vp)
    t_exp = (time.perf_counter() - t0) * 1e3

    cfg, rt1 = build_engine(a.model, a.S, a.nl, chip=3)
    import llama_engine_resident as LER2
    s1 = EngineSession(rt1, LER2)
    t0 = time.perf_counter()
    Kp1, Vp1 = s1.import_kv(Kh, Vh)
    t_imp = (time.perf_counter() - t0) * 1e3
    pd = [nxt1]
    nxt, p = nxt1, len(toks)
    for _ in range(a.D - 1):
        nxt, Kp1, Vp1 = s1.step(nxt, p, Kp1, Vp1)
        pd.append(nxt)
        p += 1

    kv_mb = (Kh.nbytes + Vh.nbytes) / 1e6
    print(f"\nmono   : {mono}")
    print(f"pd     : {pd}")
    print(f"KV export {t_exp:.0f} ms + import {t_imp:.0f} ms ({kv_mb:.0f} MB fp32 host-staged)")
    print("VALIDATE " + ("PASS ✅ engine-class PD == monolithic" if pd == mono else "FAIL ❌"))


# ------------------------------ workers ------------------------------------

def sync_start(tag):
    open(f"{WORK}/ready_{tag}", "w").write("1")
    while not os.path.exists(f"{WORK}/go"):
        time.sleep(0.05)


def role_dup(a):
    import llama_engine_resident as LER
    cfg, rt = build_engine(a.model, a.S, a.nl, chip=a.chip)
    lmh = build_lmhead(rt, LER, a.chip) if a.lmhead else None
    s = EngineSession(rt, LER, lmhead=lmh)
    prompts = make_prompts(a.P, a.n)
    mine = [(i, p) for i, p in enumerate(prompts) if i % 2 == (a.chip % 2)]
    sync_start(f"dup{a.chip}")
    res = []
    for i, toks in mine:
        Kp, Vp = s.fresh_kv()
        t0 = time.perf_counter()
        nxt = None
        for pos in range(len(toks)):
            nxt, Kp, Vp = s.step(toks[pos], pos, Kp, Vp)
        ttft = time.perf_counter() - t0
        out = [nxt]
        p = len(toks)
        for _ in range(a.D - 1):
            nxt, Kp, Vp = s.step(nxt, p, Kp, Vp)
            out.append(nxt)
            p += 1
        res.append({"i": i, "ttft": ttft, "e2e": time.perf_counter() - t0,
                    "out": out})
    json.dump(res, open(f"{WORK}/dup{a.chip}.json", "w"))


def _bf16_bytes(t):
    """Device bf16 tensor -> raw bf16 bytes on host (bit-exact, half of fp32)."""
    import torch
    return t.detach().cpu().view(torch.uint16).numpy().tobytes()


def role_prefill(a):
    import llama_engine_resident as LER
    from kv_transport import make_transport
    cfg, rt = build_engine(a.model, a.S, a.nl, chip=a.chip)
    lmh = build_lmhead(rt, LER, a.chip) if a.lmhead else None
    s = EngineSession(rt, LER, lmhead=lmh)
    C = a.kv_stream
    if C:
        assert a.P % C == 0, "P must be a multiple of --kv-stream"
        mg, _, sels = build_chunk_kernels(rt, LER, a.chip, C)
    prompts = make_prompts(a.P, a.n)
    sync_start("prefill")
    res = []
    for i, toks in enumerate(prompts):
        tr = make_transport("shm", root=f"{WORK}/kv_{i}")
        Kp, Vp = s.fresh_kv()
        t0 = time.perf_counter()
        nxt = None
        pushed_ms = 0.0
        for pos in range(len(toks)):
            nxt, Kp, Vp = s.step(toks[pos], pos, Kp, Vp)
            if C and (pos + 1) % C == 0 and pos + 1 < len(toks):
                # mid-prefill chunk: pushed NOW — hidden from the decode side
                # behind the remaining prefill compute
                k = (pos + 1) // C - 1
                tc = time.perf_counter()
                Kc, Vc = mg(Kp, Vp, sels[k][0], sels[k][1])
                tr.push(f"r{i}/k{k}", _bf16_bytes(Kc))
                tr.push(f"r{i}/v{k}", _bf16_bytes(Vc))
                open(f"{WORK}/kvchunk_{i}_{k}", "w").write("1")
                pushed_ms += (time.perf_counter() - tc) * 1e3
        t_pre = time.perf_counter() - t0
        t_pre_wall = time.time()            # prefill COMPUTE done (pre-push)
        tk = time.perf_counter()
        if C:
            k = len(toks) // C - 1          # only the LAST chunk rides the
            Kc, Vc = mg(Kp, Vp, sels[k][0], sels[k][1])    # critical path
            tr.push(f"r{i}/k{k}", _bf16_bytes(Kc))
            tr.push(f"r{i}/v{k}", _bf16_bytes(Vc))
            open(f"{WORK}/kvchunk_{i}_{k}", "w").write("1")
        else:
            # one-shot bf16 transport: BIT-EXACT (device KV is bf16; fp32
            # round-trip is identity). fp16 is NOT safe — it collapses bf16
            # subnormals (<6e-5) and flipped argmaxes at P=192.
            import torch
            Kh, Vh = s.export_kv(Kp, Vp)
            for tag, arr in (("K", Kh), ("V", Vh)):
                b = torch.from_numpy(np.ascontiguousarray(arr, np.float32)) \
                         .to(torch.bfloat16).view(torch.uint16).numpy()
                tr.push(f"r{i}/{tag}", b.tobytes())
        json.dump({"next_tok": int(nxt), "t_pre_wall": t_pre_wall},
                  open(f"{WORK}/meta_{i}.json", "w"))
        open(f"{WORK}/kvready_{i}", "w").write("1")
        res.append({"i": i, "t_pre": t_pre,
                    "mid_chunk_push_ms": round(pushed_ms, 1),
                    "kv_save_ms": (time.perf_counter() - tk) * 1e3})
    json.dump(res, open(f"{WORK}/prefill.json", "w"))


def role_decode(a):
    import llama_engine_resident as LER
    from kv_transport import make_transport
    cfg, rt = build_engine(a.model, a.S, a.nl, chip=a.chip)
    lmh = build_lmhead(rt, LER, a.chip) if a.lmhead else None
    s = EngineSession(rt, LER, lmhead=lmh)
    C = a.kv_stream
    if C:
        _, ms_scatter, sels = build_chunk_kernels(rt, LER, a.chip, C)
    prompts = make_prompts(a.P, a.n)
    sync_start("decode")
    res = []
    import torch
    for i, toks in enumerate(prompts):
        tr = make_transport("shm", root=f"{WORK}/kv_{i}")

        def pull_bf16(tag, shape):
            u = np.frombuffer(tr.pull(f"r{i}/{tag}"), np.uint16).copy()
            return torch.from_numpy(u).view(torch.bfloat16).float() \
                        .numpy().reshape(shape)

        if C:
            # STREAMING: scatter each chunk as it lands — by the time the
            # prefill side finishes, only the last chunk is left to apply
            n_chunks = a.P // C
            Kp, Vp = s.fresh_kv()
            t0 = time.perf_counter()
            gap0 = None
            for k in range(n_chunks):
                while not os.path.exists(f"{WORK}/kvchunk_{i}_{k}"):
                    time.sleep(0.002)
                if k == n_chunks - 1:
                    gap0 = time.perf_counter()   # last-chunk critical path
                Kc = LER._d(pull_bf16(f"k{k}", (rt.nl, LER.NKV, LER.HD, C)))
                Vc = LER._d(pull_bf16(f"v{k}", (rt.nl, LER.NKV, C, LER.HD)))
                Kp, Vp = ms_scatter(Kp, Vp, Kc, Vc, sels[k][0], sels[k][1])
            t_load = (time.perf_counter() - gap0) * 1e3   # tail only
            while not os.path.exists(f"{WORK}/kvready_{i}"):
                time.sleep(0.002)
        else:
            while not os.path.exists(f"{WORK}/kvready_{i}"):
                time.sleep(0.02)
            t0 = time.perf_counter()
            Kh = pull_bf16("K", (rt.nl, LER.NKV, LER.HD, rt.S))
            Vh = pull_bf16("V", (rt.nl, LER.NKV, rt.S, LER.HD))
            Kp, Vp = s.import_kv(Kh, Vh)
            t_load = (time.perf_counter() - t0) * 1e3
        meta = json.load(open(f"{WORK}/meta_{i}.json"))
        nxt = meta["next_tok"]
        gap_wall = (time.time() - meta["t_pre_wall"]) * 1e3
        out = [nxt]
        p = len(toks)
        for _ in range(a.D - 1):
            nxt, Kp, Vp = s.step(nxt, p, Kp, Vp)
            out.append(nxt)
            p += 1
        res.append({"i": i, "kv_load_ms": t_load, "t_end": time.perf_counter(),
                    "handoff_gap_ms": round(gap_wall, 1), "out": out})
    json.dump(res, open(f"{WORK}/decode.json", "w"))


# ----------------------------- orchestrator --------------------------------

def spawn(role, chip, a):
    cmd = [sys.executable, os.path.abspath(__file__), "--role", role,
           "--chip", str(chip), "--P", str(a.P), "--D", str(a.D),
           "--n", str(a.n), "--S", str(a.S), "--nl", str(a.nl),
           "--model", a.model]
    if a.lmhead:
        cmd.append("--lmhead")
    if a.kv_stream:
        cmd += ["--kv-stream", str(a.kv_stream)]
    return subprocess.Popen(cmd, stdout=open(f"{WORK}/{role}{chip}.log", "w"),
                            stderr=subprocess.STDOUT)


def run_phase(mode, a, chips):
    os.system(f"rm -rf {WORK} && mkdir -p {WORK}")
    if mode == "dup":
        procs = [spawn("dup", c, a) for c in chips]
        tags = [f"dup{c}" for c in chips]
    else:
        procs = [spawn("prefill", chips[0], a), spawn("decode", chips[1], a)]
        tags = ["prefill", "decode"]
    t0 = time.time()
    while any(not os.path.exists(f"{WORK}/ready_{t}") for t in tags):
        if any(p.poll() is not None for p in procs):
            raise RuntimeError(f"worker died — see {WORK}/*.log")
        if time.time() - t0 > 5400:
            raise TimeoutError("workers not ready")
        time.sleep(1)
    t_go = time.perf_counter()
    open(f"{WORK}/go", "w").write("1")
    for p in procs:
        p.wait(timeout=3600)
    wall = time.perf_counter() - t_go
    out = {"wall": wall}
    if mode == "dup":
        out["rows"] = (json.load(open(f"{WORK}/dup{chips[0]}.json")) +
                       json.load(open(f"{WORK}/dup{chips[1]}.json")))
    else:
        out["pre"] = json.load(open(f"{WORK}/prefill.json"))
        out["dec"] = json.load(open(f"{WORK}/decode.json"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--nl", type=int, default=32)
    ap.add_argument("--S", type=int, default=0)
    ap.add_argument("--chips", default="2,3")
    ap.add_argument("--role", default="orchestrator")
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--lmhead", action="store_true",
                    help="run lm_head (RMS+matmul) on device; host only argmaxes")
    ap.add_argument("--kv-stream", type=int, default=0,
                    help="stream KV to the decode chip in C-token chunks DURING "
                         "prefill (0 = one-shot transfer after prefill)")
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
    if a.validate:
        return validate(a)

    chips = [int(x) for x in a.chips.split(",")]
    print(f"[pd-engine] P={a.P} D={a.D} n={a.n} nl={a.nl} S={a.S} chips={chips}",
          flush=True)
    dup = run_phase("dup", a, chips)
    print(f"  DUP done {dup['wall']:.1f}s", flush=True)
    pd = run_phase("pd", a, chips)
    print(f"  PD  done {pd['wall']:.1f}s", flush=True)

    tot = a.n * a.D
    def p50(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]
    print("\n" + "=" * 70)
    print(f"ENGINE-CLASS 2-chip DUP vs PD (resident-KV, one-shot KV handoff)")
    print(f"  DUP: {tot/dup['wall']:7.2f} tok/s | TTFT p50 "
          f"{p50([r['ttft'] for r in dup['rows']])*1e3:6.0f} ms | wall {dup['wall']:.1f}s")
    print(f"  PD : {tot/pd['wall']:7.2f} tok/s | prefill p50 "
          f"{p50([r['t_pre'] for r in pd['pre']])*1e3:6.0f} ms | "
          f"kv save p50 {p50([r['kv_save_ms'] for r in pd['pre']]):5.0f} ms + "
          f"load p50 {p50([r['kv_load_ms'] for r in pd['dec']]):5.0f} ms | wall {pd['wall']:.1f}s")
    print(f"  PD handoff gap p50 (prefill-done -> decode-ready): "
          f"{p50([r['handoff_gap_ms'] for r in pd['dec']]):.0f} ms"
          f"{f'  [kv-stream C={a.kv_stream}]' if a.kv_stream else '  [one-shot]'}")
    print(f"  measured PD/DUP = {(tot/pd['wall'])/(tot/dup['wall']):.2f}")
    dup_out = {r["i"]: r["out"] for r in dup["rows"]}
    pd_out = {r["i"]: r["out"] for r in pd["dec"]}
    match = all(dup_out.get(i) == pd_out.get(i) for i in pd_out)
    print(f"  output check (PD == DUP per request): "
          f"{'PASS ✅' if match else 'FAIL ❌'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
