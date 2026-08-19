#!/usr/bin/env python3
"""Serving wrapper for the kv-free whole-prompt prefill on RNGD.

The graph is the one that closed the gap with the vendor engine: no KV input,
attention causal within the prompt, C positions per pass, `ForLlmModelPrefill`
tactics. 32 layers come from reusing one compiled two-layer module sixteen
times, since that is where tactic search still stops.

It hands back KV in the [nl, NKV, HD, S] / [nl, NKV, S, HD] layout the resident
engine and the connector already use, so the decode side needs no changes.
Rotation and mask are position-independent for prefill (base is always 0), so
they are built once at start-up rather than per request.

  python3 full_prefill_engine.py --chip 2 --C 256
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
from llama_prefill_full import (BF, H, M, NH, NKV, HD, G, PrefillModel, HINT)
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def rot_matrices_fast(inv, C):
    """[C, HD, HD] block rotations, built with array ops rather than loops."""
    half = HD // 2
    ang = np.arange(C)[:, None] * inv[None, :]
    cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    out = np.zeros((C, HD, HD), np.float32)
    i = np.arange(half)
    out[:, i, i] = cs
    out[:, i + half, i + half] = cs
    out[:, i, i + half] = sn
    out[:, i + half, i] = -sn
    return out


class FullPrefillEngine:
    def __init__(self, cfg, sd, C=256, group=2, chip=2, cache_dir=None):
        self.C, self.NG = C, group
        self.nl = cfg.n_layers
        self.ngroups = self.nl // group
        assert self.nl % group == 0
        self.dev = f"rngd:{chip}"
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
                  torch.zeros(C, HD, HD, dtype=BF), torch.zeros(1, G * C, C, dtype=BF))
        ccfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                        tactic_hint=getattr(TacticHintConfig, HINT))
        build = lambda: CompileModule.from_module(PrefillModel(C, group), sample,
                                                  compiler_config=ccfg)
        t0 = time.time()
        if cache_dir:
            from artifact_cache import get_or_compile, cache_key
            key = cache_key(kind="full-prefill", model=MODEL, C=C, nl=group, hint=HINT)
            self.m, mode, secs = get_or_compile(cache_dir, key, build, device=None)
            print(f"[full-engine] {mode} in {secs:.0f}s", flush=True)
        else:
            self.m = build()
            print(f"[full-engine] compiled in {time.time()-t0:.0f}s", flush=True)
        self.m.to(self.dev)

        inv = cfg.inv_freq()
        self.eps_d = self._df(np.full((C, H), self.eps * H, np.float32))
        self.nh_d = self._df(np.full((C, H), -0.5, np.float32))
        self.rot = d(rot_matrices_fast(inv, C))
        causal = np.triu(np.full((C, C), -1e4, np.float32), 1)
        self.mask = d(np.tile(causal, (G, 1)).reshape(1, G * C, C))
        self.compile_s = time.time() - t0

    def _d(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(BF).to(self.dev)

    def _df(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32).to(self.dev)

    def prefill(self, tokens):
        """Whole prompt in one pass per layer group. Returns (K, V, next_token)."""
        C = self.C
        P = len(tokens)
        if P > C:
            raise ValueError(f"prompt {P} exceeds the compiled prefill width C={C}")
        toks = list(tokens) + [tokens[-1]] * (C - P)   # padding is masked out anyway
        x = self._d(self.embed[toks].astype(np.float32))
        Ks, Vs = [], []
        for g in range(self.ngroups):
            x, k, v = self.m(x, self.eps_d, self.nh_d, *self.groups[g],
                             self.rot, self.mask)
            Ks.append(k)
            Vs.append(v)
        hid = x.detach().float().cpu().numpy()[P - 1]
        from generic_runtime import rmsnorm
        nxt = int(np.argmax(self.lm @ rmsnorm(hid, self.norm, self.eps)))
        # Only the prompt's own columns are worth moving: a 16-token prompt in
        # a 256-wide graph would otherwise ship 16x the bytes it needs. Stack on
        # the card first so this is two transfers, not sixty-four.
        kdev = torch.cat([k[:, :, :, :P] for k in Ks], 0)      # [nl,NKV,HD,P]
        vdev = torch.cat([v[:, :, :P, :] for v in Vs], 0)      # [nl,NKV,P,HD]
        kh = kdev.detach().float().cpu().numpy()
        vh = vdev.detach().float().cpu().numpy()
        K = np.zeros((self.nl, NKV, HD, C), np.float32)
        V = np.zeros((self.nl, NKV, C, HD), np.float32)
        K[:, :, :, :P] = kh
        V[:, :, :P, :] = vh
        return K, V, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--cache", default="/home/user/workspace/01_pd_disag/rngd_llmd/artifacts_cache")
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from generic_runtime import ModelConfig
    print(f"loading {MODEL}...", flush=True)
    tk = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf

    eng = FullPrefillEngine(cfg, sd, C=a.C, group=a.group, chip=a.chip, cache_dir=a.cache)
    toks = tk.encode(a.prompt, add_special_tokens=False)
    eng.prefill(toks)                                  # warm up
    t0 = time.perf_counter()
    K, V, nxt = eng.prefill(toks)
    ms = (time.perf_counter() - t0) * 1e3
    print("\n" + "=" * 66)
    print(f"RNGD kv-free prefill — {cfg.n_layers} layers, C={a.C}, {eng.ngroups} calls")
    print(f"  prompt {len(toks)} tokens · one pass · {ms:.1f} ms")
    print(f"  next token: {nxt} {tk.decode([nxt])!r}")
    print(f"  KV {K.nbytes*2/1e6:.0f} MB in the connector's layout")
    print("=" * 66)


if __name__ == "__main__":
    main()
