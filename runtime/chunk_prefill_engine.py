#!/usr/bin/env python3
"""Real-weight chunked prefill engine for RNGD — the serving-side wrapper.

llama_prefill_chunk proved the kernel shapes and the numerics with random
weights; this loads Llama-3.1-8B into that graph and produces exactly the KV
layout the resident engine and the KV connector already speak, so it drops into
the PD prefill role without touching anything downstream.

One two-layer graph is compiled and invoked once per layer pair, because the
four-layer graph explodes in tactic search. Weights are graph inputs, so the
same compiled module walks all 32 layers.

Weight preparation matches the resident engine's conventions: norms carry
sqrt(H), q carries 1/sqrt(head_dim), and everything is pre-transposed for the
[C, in] x [in, out] matmul the chunked layer uses.

  python3 chunk_prefill_engine.py --chip 1 --prompt "The capital of France is"
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
from llama_prefill_chunk import (BF, H, M, NH, NKV, HD, G, ChunkModel,
                                 rot_matrices)
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HINT = os.environ.get("CHUNK_TACTIC_HINT", "ForLlmModelPrefill")


class ChunkPrefillEngine:
    def __init__(self, cfg, sd, S=256, C=16, group=2, chip=1, cache_dir=None):
        self.S, self.C, self.NG = S, C, group
        self.nl = cfg.n_layers
        self.ngroups = self.nl // group
        assert self.nl % group == 0
        self.dev = f"rngd:{chip}"
        self.inv = cfg.inv_freq()
        self.eps = cfg.eps
        self.embed = sd["model.embed_tokens.weight"]
        self.lm = sd["lm_head.weight"]
        self.norm = sd["model.norm.weight"]
        sq, sc = math.sqrt(H), 1.0 / math.sqrt(HD)
        d = self._d

        self.groups = []
        for g in range(self.ngroups):
            idx = range(g * group, (g + 1) * group)
            p = lambda i, k: sd[f"model.layers.{i}.{k}"].astype(np.float32)
            self.groups.append((
                torch.stack([d(np.tile(p(i, "input_layernorm.weight") * sq, (C, 1))) for i in idx]),
                torch.stack([d((p(i, "self_attn.q_proj.weight") * sc).T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.k_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.v_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.o_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(np.tile(p(i, "post_attention_layernorm.weight") * sq, (C, 1))) for i in idx]),
                torch.stack([d(p(i, "mlp.gate_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.up_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.down_proj.weight").T.copy()) for i in idx]),
            ))

        sample = (torch.zeros(C, H, dtype=BF),
                  torch.zeros(C, H, dtype=torch.float32), torch.zeros(C, H, dtype=torch.float32),
                  torch.zeros(group, C, H, dtype=BF),
                  torch.zeros(group, H, NH * HD, dtype=BF), torch.zeros(group, H, NKV * HD, dtype=BF),
                  torch.zeros(group, H, NKV * HD, dtype=BF), torch.zeros(group, NH * HD, H, dtype=BF),
                  torch.zeros(group, C, H, dtype=BF),
                  torch.zeros(group, H, M, dtype=BF), torch.zeros(group, H, M, dtype=BF),
                  torch.zeros(group, M, H, dtype=BF),
                  torch.zeros(group, NKV, HD, S, dtype=BF), torch.zeros(group, NKV, S, HD, dtype=BF),
                  torch.zeros(C, HD, HD, dtype=BF),
                  torch.zeros(C, S, dtype=BF), torch.zeros(S, C, dtype=BF),
                  torch.zeros(1, G * C, S, dtype=BF))
        ccfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                        tactic_hint=getattr(TacticHintConfig, HINT))
        t0 = time.time()
        build = lambda: CompileModule.from_module(ChunkModel(C, S, group), sample,
                                                  compiler_config=ccfg)
        if cache_dir:
            from artifact_cache import get_or_compile, cache_key
            key = cache_key(kind="chunk-prefill", model=MODEL, S=S, C=C, nl=group)
            self.m, mode, secs = get_or_compile(cache_dir, key, build, device=None)
            print(f"[chunk-engine] {mode} in {secs:.0f}s", flush=True)
        else:
            self.m = build()
            print(f"[chunk-engine] compiled in {time.time()-t0:.0f}s", flush=True)
        self.m.to(self.dev)
        self.eps_d = self._df(np.full((C, H), self.eps * H, np.float32))
        self.nh_d = self._df(np.full((C, H), -0.5, np.float32))
        self.compile_s = time.time() - t0

    def _d(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(BF).to(self.dev)

    def _df(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32).to(self.dev)

    def fresh_kv(self):
        z = lambda *s: self._d(np.zeros(s, np.float32))
        return ([z(self.NG, NKV, HD, self.S) for _ in range(self.ngroups)],
                [z(self.NG, NKV, self.S, HD) for _ in range(self.ngroups)])

    def prefill(self, tokens):
        """Whole prompt, C positions per pass. Returns (K, V, next_token)."""
        C, S = self.C, self.S
        P = len(tokens)
        pad = (-P) % C
        if P + pad > S:
            raise ValueError(f"prompt {P} (+{pad} pad) exceeds the compiled "
                             f"cache size S={S}")
        toks = list(tokens) + [tokens[-1]] * pad          # last chunk padded
        Kd, Vd = self.fresh_kv()
        hid = None
        for base in range(0, len(toks), C):
            piece = toks[base:base + C]
            ohK = np.zeros((C, S), np.float32)
            mk = np.full((C, S), -1e4, np.float32)
            for c in range(C):
                ohK[c, base + c] = 1.0
                mk[c, :base + c + 1] = 0.0
            x = self._d(self.embed[piece].astype(np.float32))
            rot = self._d(rot_matrices(self.inv, base, C))
            ohKd, ohVd = self._d(ohK), self._d(ohK.T.copy())
            maskd = self._d(np.tile(mk, (G, 1)).reshape(1, G * C, S))
            for g in range(self.ngroups):
                w = self.groups[g]
                x, Kf, Vf = self.m(x, self.eps_d, self.nh_d, *w,
                                   Kd[g], Vd[g], rot, ohKd, ohVd, maskd)
                Kd[g], Vd[g] = Kf, Vf
            hid = x
        h = hid.detach().float().cpu().numpy()[(P - 1) % C]
        from generic_runtime import rmsnorm
        nxt = int(np.argmax(self.lm @ rmsnorm(h, self.norm, self.eps)))
        K = np.concatenate([k.detach().float().cpu().numpy() for k in Kd], 0)
        V = np.concatenate([v.detach().float().cpu().numpy() for v in Vd], 0)
        return K, V, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--chip", type=int, default=1)
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    ap.add_argument("--cache", default="/home/gspark/workspace/01_pd_disag/rngd_llmd/artifacts_cache",
                    help="EDF cache so the 5-minute compile happens once")
    ap.add_argument("--long", action="store_true",
                    help="repeat the prompt to fill the cache, for steady-state timing")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from generic_runtime import ModelConfig
    print(f"loading {MODEL}...", flush=True)
    tk = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf

    if a.long:
        a.prompt = ("The three primary colors are red, blue and yellow. Mixing them "
                    "yields the secondary colors orange, green and purple. Painters "
                    "have relied on this for centuries. ") * 8 + "The capital of France is"
    eng = ChunkPrefillEngine(cfg, sd, S=a.S, C=a.C, group=a.group, chip=a.chip,
                             cache_dir=a.cache)
    toks = tk.encode(a.prompt, add_special_tokens=False)
    keep = ((a.S - 16) // a.C) * a.C   # leave room for decode, whole chunks only
    toks = toks[:keep]
    eng.prefill(toks[:a.C])            # warm up: first call carries lazy setup
    t0 = time.perf_counter()
    K, V, nxt = eng.prefill(toks)
    ms = (time.perf_counter() - t0) * 1e3
    print("\n" + "=" * 64)
    print(f"RNGD chunked prefill — {cfg.n_layers} layers, C={a.C}, group={a.group}")
    print(f"  {len(toks)} tokens in {ms:.0f} ms  ->  {ms/len(toks):.2f} ms/token")
    print(f"  sequential engine is 14.8 ms/token  ->  {14.8/(ms/len(toks)):.1f}x faster")
    print(f"  next token: {nxt} {tk.decode([nxt])!r}")
    print(f"  KV {K.nbytes*2/1e6:.0f} MB in the layout the connector expects")
    print("=" * 64)


if __name__ == "__main__":
    main()
