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

from rngd_pd.kernels.prefill_full import (BF, H, M, NH, NKV, HD, G, PrefillModel, HINT)
from furiosa.torch.custom_ops import CompileModule
from furiosa.torch.compiler import Config as FCConfig, TacticHintConfig
from rngd_pd.paths import ARTIFACT_CACHE

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
            from rngd_pd.engines.artifact_cache import get_or_compile, cache_key
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
        self._zeroK = self._zeroV = None      # solo path, built on demand
        self._build_lmhead(cache_dir)

    def _build_lmhead(self, cache_dir):
        """Pick the next token on the card, not in numpy.

        Prefill ends by choosing one token, and it was doing that with
        `lm @ rmsnorm(hidden)` on the host — a 2.1 GB read for a single
        argmax. The decode side already had this compiled (chunking V into 32
        groups reaches ~565 GB/s against 23 for the monolithic matvec), and its
        module takes a 1-D hidden state, which is exactly what the last prompt
        position is. Same cache key, so nothing recompiles.
        """
        import math
        from rngd_pd.kernels import dsl_b1 as LE
        from furiosa.torch.custom_ops import CompileModule, TacticKernelModule
        from rngd_pd.engines.artifact_cache import get_or_compile, cache_key
        V = self.lm.shape[0]
        NG = 32
        Vp = (V + NG - 1) // NG * NG
        self.V, self.Vp = V, Vp

        class LmHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rms = TacticKernelModule(LE.RMS_NORM)

            def forward(self, x, nw, eps, nh, wlm):
                h = self.rms(x, nw, eps, nh)[0]
                return torch.matmul(wlm, h)          # [NG, Vp/NG]

        sample = (torch.zeros(H, dtype=BF), torch.zeros(H, dtype=BF),
                  torch.zeros(H, dtype=torch.float32),
                  torch.zeros(H, dtype=torch.float32),
                  torch.zeros(NG, Vp // NG, H, dtype=BF))
        ccfg = FCConfig(allow_external_operators=True, allow_unlowered_operators=True,
                        tactic_hint=TacticHintConfig.ForLlmModelDecode)
        key = cache_key(kind="lmhead", V=V, H=H, tactic="decode-chunked", G=NG)
        build = lambda: CompileModule.from_module(LmHead(), sample,
                                                  compiler_config=ccfg)
        if cache_dir:
            self.lmh, mode, secs = get_or_compile(cache_dir, key, build, device=None)
            print(f"[prefill-lmhead] {mode} in {secs:.0f}s", flush=True)
        else:
            self.lmh = build()
        self.lmh.to(self.dev)
        wpad = np.zeros((Vp, H), np.float32)
        wpad[:V] = self.lm
        self.wlm_d = self._d(wpad.reshape(NG, Vp // NG, H))
        self.lm_norm_d = self._d(self.norm * math.sqrt(H))
        self.lm_eps_d = self._df(np.full(H, self.eps * H, np.float32))
        self.lm_nh_d = self._df(np.full(H, -0.5, np.float32))

    def _next_token(self, x, P):
        """Last prompt position -> next token, with the matmul on the card."""
        out = self.lmh(x[P - 1], self.lm_norm_d, self.lm_eps_d, self.lm_nh_d,
                       self.wlm_d)
        lg = (out[0] if isinstance(out, (tuple, list)) else out)
        return int(np.argmax(lg.detach().cpu().float().numpy().reshape(-1)[:self.V]))

    def _d(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(BF).to(self.dev)

    def _df(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32).to(self.dev)

    def prefill_device(self, tokens):
        """Prefill and leave the cache on the card, in the decode graph's layout.

        The disaggregated path has to move KV to another card, so it pays a
        download, a wire encoding and an upload. When prefill and decode share
        a card none of that is necessary: the graph already emits
        [nl, NKV, HD, S] and [nl, NKV, S, HD], which is exactly what the
        resident decode session steps on.

        The columns past the prompt still have to be cleared. Prefill pads with
        the last token, and a decode step injects its own k/v by adding at a
        one-hot slot — only equal to a write if that slot starts at zero.

        Clearing them by multiplying with a [1,1,1,S] mask cost 1497 ms against
        131 ms for the whole rest of the pass: an eager elementwise op on this
        runtime is not a compiled kernel and goes through the host. Slicing and
        concatenating a pre-allocated zero block does the same job with the two
        operations that are known to stay on the card.
        """
        C, P = self.C, len(tokens)
        if P > C:
            raise ValueError(f"prompt {P} exceeds the compiled prefill width C={C}")
        toks = list(tokens) + [tokens[-1]] * (C - P)
        x = self._d(self.embed[toks].astype(np.float32))
        Ks, Vs = [], []
        for g in range(self.ngroups):
            x, k, v = self.m(x, self.eps_d, self.nh_d, *self.groups[g],
                             self.rot, self.mask)
            Ks.append(k)
            Vs.append(v)
        nxt = self._next_token(x, P)
        if self._zeroK is None:                       # allocated once, reused
            self._zeroK = self._d(np.zeros((self.nl, NKV, HD, C), np.float32))
            self._zeroV = self._d(np.zeros((self.nl, NKV, C, HD), np.float32))
        kc = torch.cat([k[:, :, :, :P] for k in Ks], 0)
        vc = torch.cat([v[:, :, :P, :] for v in Vs], 0)
        Kp = torch.cat([kc, self._zeroK[:, :, :, P:]], -1)
        Vp = torch.cat([vc, self._zeroV[:, :, P:, :]], -2)
        return Kp, Vp, nxt

    def prefill(self, tokens):
        """Same pass, but handed back as host arrays for the connector.

        The cache is assembled at full width on the card and downloaded once.
        Building it host-side instead meant allocating and clearing 67 MB of
        zeros per request and then copying the prompt's columns in on top —
        work the card does for free while emitting the tensor anyway.
        """
        Kp, Vp, nxt = self.prefill_device(tokens)
        return (Kp.detach().cpu().float().numpy(),
                Vp.detach().cpu().float().numpy(), nxt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--cache", default=str(ARTIFACT_CACHE))
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rngd_pd.engines.model import ModelConfig
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
