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
from rngd_pd.engines.fsdk import CompileModule
from rngd_pd.engines.fsdk import FCConfig, NEW, dev_str, hint
from rngd_pd.paths import ARTIFACT_CACHE

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _compiler_config(tactic):
    legacy = {} if NEW else dict(allow_external_operators=True,
                                allow_unlowered_operators=True)
    return FCConfig(tactic_hint=hint(tactic), **legacy)


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


class _DevicePrefillStage:
    """One prefill, advanced a layer group at a time.

    advance() runs the next group and returns True when the pass is done; the
    caller holds the device lock for that call only, so a decode step can land
    between groups. result() assembles the cache exactly as the blocking path
    did.
    """

    def __init__(self, eng, tokens):
        self.e, self.P = eng, len(tokens)
        self.C = eng.pick(self.P)
        if self.P > self.C:
            raise ValueError(f"prompt {self.P} exceeds the widest compiled "
                             f"bucket C={self.C}")
        b = eng.buckets[self.C]
        toks = list(tokens) + [tokens[-1]] * (self.C - self.P)
        self.x = eng._d(eng.embed[toks].astype(np.float32))
        self.Ks, self.Vs, self.g = [], [], 0
        self._b = b

    def advance(self, n=1):
        """Run up to n layer groups; True when the pass is finished.

        One group per iteration bounds the stall at ~4 ms but pairs every
        group with a whole decode step, so admitting one stream costs sixteen
        steps -- at B=8 that was 480 ms a stream and three seconds to fill the
        batch. Several groups per iteration keeps the stall under one step
        while cutting the admission to a handful of iterations.
        """
        for _ in range(n):
            self.x, k, v = self.e._call(self._b, self.g, self.x)
            self.Ks.append(k); self.Vs.append(v)
            self.g += 1
            if self.g >= self.e.ngroups:
                return True
        return False

    def result(self):
        return self.e._device_prefill_finish(self.C, self.P, self.Ks, self.Vs, self.x)


class FullPrefillEngine:
    def __init__(self, cfg, sd, C=256, group=2, chip=2, cache_dir=None, S=0):
        # C is how wide a prompt this graph can take in one pass; S is how wide
        # the cache the consumer keeps. They were one number, which made the
        # prefill width impossible to lower without also lowering the context.
        self.C, self.NG = C, group
        self.S = S or C
        self.nl = cfg.n_layers
        self.ngroups = self.nl // group
        assert self.nl % group == 0
        self.dev = dev_str(chip)
        self.eps = cfg.eps
        self.embed = sd["model.embed_tokens.weight"]
        self.lm = sd["lm_head.weight"]
        self.norm = sd["model.norm.weight"]
        sq, sc = math.sqrt(H), 1.0 / math.sqrt(HD)
        d = self._d

        # Seven of the nine tensors a group needs do not depend on the width;
        # only the two layernorm weights are tiled to C. Splitting them is what
        # lets several widths live on one card: the 16 GB of projections are
        # shared and a bucket costs C x 0.5 MB per layer on top.
        self.gw = []
        p = lambda i, k: sd[f"model.layers.{i}.{k}"].astype(np.float32)
        for g in range(self.ngroups):
            idx = range(g * group, (g + 1) * group)
            self.gw.append((
                torch.stack([d((p(i, "self_attn.q_proj.weight") * sc).T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.k_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.v_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.o_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.gate_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.up_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.down_proj.weight").T.copy()) for i in idx]),
            ))
        # the untiled norms, kept on the host: 0.5 MB, and a new bucket needs them
        self._n1 = [p(i, "input_layernorm.weight") * sq for i in range(self.nl)]
        self._n2 = [p(i, "post_attention_layernorm.weight") * sq for i in range(self.nl)]
        self._inv = cfg.inv_freq()

        self.buckets = {}
        self._cache_dir = cache_dir
        t0 = time.time()
        self.add_bucket(C)
        self.compile_s = time.time() - t0
        self._zeroK = self._zeroV = None      # solo path, built on demand
        self._zeroKg = self._zeroVg = None    # per-group, streaming path
        self._build_lmhead(cache_dir)

    def add_bucket(self, C):
        """Compile (or load) one prefill width and build the tables it needs.

        The graph costs the same whatever the prompt, so a single width means
        a 21-token request pays a 512-wide pass -- 96% of it wasted, and it is
        the prefill card that queues. Several widths let each request take the
        smallest pass that fits. The vendor artifact carries eleven of these.
        """
        if C in self.buckets:
            return self.buckets[C]
        import torch
        from rngd_pd.engines.fsdk import CompileModule
        from rngd_pd.engines.fsdk import FCConfig, TacticHintConfig
        d = self._d
        sample = (torch.zeros(C, H, dtype=BF),
                  torch.zeros(C, H, dtype=torch.float32), torch.zeros(C, H, dtype=torch.float32),
                  torch.zeros(self.NG, C, H, dtype=BF),
                  torch.zeros(self.NG, H, NH * HD, dtype=BF),
                  torch.zeros(self.NG, H, NKV * HD, dtype=BF),
                  torch.zeros(self.NG, H, NKV * HD, dtype=BF),
                  torch.zeros(self.NG, NH * HD, H, dtype=BF),
                  torch.zeros(self.NG, C, H, dtype=BF),
                  torch.zeros(self.NG, H, M, dtype=BF), torch.zeros(self.NG, H, M, dtype=BF),
                  torch.zeros(self.NG, M, H, dtype=BF),
                  torch.zeros(C, HD, HD, dtype=BF), torch.zeros(1, G * C, C, dtype=BF))
        ccfg = _compiler_config(HINT)
        build = lambda: CompileModule.from_module(PrefillModel(C, self.NG), sample,
                                                  compiler_config=ccfg)
        t0 = time.time()
        if self._cache_dir:
            from rngd_pd.engines.artifact_cache import get_or_compile, cache_key
            key = cache_key(kind="full-prefill", model=MODEL, C=C, nl=self.NG, hint=HINT)
            m, mode, secs = get_or_compile(self._cache_dir, key, build, device=None)
            print(f"[full-engine] C={C} {mode} in {secs:.0f}s", flush=True)
        else:
            m = build()
            print(f"[full-engine] C={C} compiled in {time.time()-t0:.0f}s", flush=True)
        m.to(self.dev)
        causal = np.triu(np.full((C, C), -1e4, np.float32), 1)
        norms = [(torch.stack([d(np.tile(self._n1[i], (C, 1)))
                               for i in range(g * self.NG, (g + 1) * self.NG)]),
                  torch.stack([d(np.tile(self._n2[i], (C, 1)))
                               for i in range(g * self.NG, (g + 1) * self.NG)]))
                 for g in range(self.ngroups)]
        self.buckets[C] = {
            "m": m, "norms": norms,
            "eps": self._df(np.full((C, H), self.eps * H, np.float32)),
            "nh": self._df(np.full((C, H), -0.5, np.float32)),
            "rot": d(rot_matrices_fast(self._inv, C)),
            "mask": d(np.tile(causal, (G, 1)).reshape(1, G * C, C)),
            "zeroK": None, "zeroV": None, "zeroKg": None, "zeroVg": None}
        self.C = max(self.buckets)               # widest prompt we can take
        return self.buckets[C]

    def pick(self, P):
        """The narrowest compiled width that swallows this prompt."""
        fits = [c for c in sorted(self.buckets) if c >= P]
        return fits[0] if fits else max(self.buckets)

    def _call(self, b, g, x):
        n1, n2 = b["norms"][g]
        wq, wk, wv, wo, wg, wu, wd = self.gw[g]
        return b["m"](x, b["eps"], b["nh"], n1, wq, wk, wv, wo, n2, wg, wu, wd,
                      b["rot"], b["mask"])

    def _build_lmhead(self, cache_dir):
        """Pick the next token on the card, not in numpy.

        Prefill ends by choosing one token, and `lm @ rmsnorm(hidden)` on the
        host is a 2.1 GB read for a single argmax. The matmul runs on the card
        with V chunked into 32 groups ([32, V/32, H] @ [H, 1], 4.7 ms for
        Llama-3.1-8B). The RMS norm of the one row stays on the host: compiled
        together with the matmul it fails on furiosa-torch 2026.4 ("Inconsistent
        tensor group offsets", #15), and on one row it costs nothing.
        """
        from rngd_pd.engines.fsdk import CompileModule
        from rngd_pd.engines.artifact_cache import get_or_compile, cache_key
        V, H_ = self.lm.shape
        NG = 32
        Vp = (V + NG - 1) // NG * NG
        self.V, self.Vp = V, Vp

        class LmHead(torch.nn.Module):
            def forward(self, wlm, h):
                return torch.matmul(wlm, h)          # [NG, Vp/NG, 1]

        sample = (torch.zeros(NG, Vp // NG, H_, dtype=BF), torch.zeros(H_, 1, dtype=BF))
        ccfg = _compiler_config('ForLlmModelDecode')
        key = cache_key(kind="lmhead-matmul", V=V, H=H_, tactic="decode-chunked", G=NG)
        build = lambda: CompileModule.from_module(LmHead(), sample,
                                                  compiler_config=ccfg)
        if cache_dir:
            self.lmh, mode, secs = get_or_compile(cache_dir, key, build, device=None)
            print(f"[prefill-lmhead] {mode} in {secs:.0f}s", flush=True)
        else:
            self.lmh = build()
        self.lmh.to(self.dev)
        wpad = np.zeros((Vp, H_), np.float32)
        wpad[:V] = self.lm
        self.wlm_d = self._d(wpad.reshape(NG, Vp // NG, H_))
        self.last_logits = None

    def _logits(self, x, P):
        """Last prompt position -> logits [V] (host, float32); norm on the host, matmul on the card."""
        h = x[P - 1].detach().cpu().float().numpy()
        h = h / np.sqrt(np.mean(h * h) + self.eps) * self.norm
        out = self.lmh(self.wlm_d, self._d(h.reshape(-1, 1)))
        lg = out[0] if isinstance(out, (tuple, list)) else out
        return lg.detach().cpu().float().numpy().reshape(-1)[:self.V]

    def _next_token(self, x, P):
        """Last prompt position -> next token (greedy); the logits stay in last_logits for sampling."""
        self.last_logits = self._logits(x, P)
        return int(np.argmax(self.last_logits))

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
        P = len(tokens)
        C = self.pick(P)
        if P > C:
            raise ValueError(f"prompt {P} exceeds the widest compiled bucket C={C}")
        stage = self.begin_device_prefill(tokens)
        while not stage.advance():
            pass
        return stage.result()

    def begin_device_prefill(self, tokens):
        """Prefill as sixteen resumable steps instead of one blocking call.

        On the one-card path this pass runs between decode steps, holding the
        device for its whole length -- 35 ms for the narrowest bucket, 66 ms
        for the widest. That is one to three decode steps during which every
        stream in flight stops. The pass is already sixteen graph calls, so
        yielding between them bounds the stall at one call (~2-4 ms) without a
        new graph, without a second copy of the weights, and without the KV
        input a true chunked prefill would need.
        """
        return _DevicePrefillStage(self, tokens)

    def _device_prefill_finish(self, C, P, Ks, Vs, x):
        b = self.buckets[C]
        self.last_C = C
        nxt = self._next_token(x, P)
        if self._zeroK is None:                       # allocated once, reused
            # Padded to the decode context, not to the prefill width. Those
            # were the same number while one graph served every prompt, and
            # coupling them is what stops a narrower prefill bucket from
            # feeding the same decode cache: the consumer's resident tensor is
            # S wide whatever width produced it.
            self._zeroK = self._d(np.zeros((self.nl, NKV, HD, self.S), np.float32))
            self._zeroV = self._d(np.zeros((self.nl, NKV, self.S, HD), np.float32))
        kc = torch.cat([k[:, :, :, :P] for k in Ks], 0)
        vc = torch.cat([v[:, :, :P, :] for v in Vs], 0)
        Kp = torch.cat([kc, self._zeroK[:, :, :, P:]], -1)
        Vp = torch.cat([vc, self._zeroV[:, :, P:, :]], -2)
        return Kp, Vp, nxt

    def prefill_stream(self, tokens, sink):
        """Hand each layer group's cache over the moment it exists.

        Prefill runs sixteen graph calls and only then exports 33.6 MB, so the
        connector's encode-and-write sits idle for the whole pass and then
        costs ~20 ms with nothing else happening. Group g's cache is finished
        long before group g+1 is, so it can be on its way out while the rest
        computes.

        Only the host half overlaps. The download stays on this thread inside
        the caller's device lock, because the runtime is not reentrant — what
        moves off is the bf16 encoding and the shm write, which is the larger
        part anyway.
        """
        P = len(tokens)
        C = self.pick(P)
        b = self.buckets[C]
        if P > C:
            raise ValueError(f"prompt {P} exceeds the widest compiled bucket C={C}")
        toks = list(tokens) + [tokens[-1]] * (C - P)
        x = self._d(self.embed[toks].astype(np.float32))
        if b["zeroKg"] is None:
            b["zeroKg"] = self._d(np.zeros((self.NG, NKV, HD, self.S), np.float32))
            b["zeroVg"] = self._d(np.zeros((self.NG, NKV, self.S, HD), np.float32))
        self.last_C = C
        # Split the pass from the download. Narrowing C only shortens the
        # first; the second moves an S-wide cache whatever the prompt was, so
        # if it dominates then prefill buckets cannot pay on their own.
        t_c = t_d = 0.0
        for g in range(self.ngroups):
            t0 = time.perf_counter()
            x, k, v = self._call(b, g, x)
            # Padded to the decode context and shipped whole. Sending only
            # the prompt's columns was tried and is slower here: both transfer
            # legs are already hidden -- the save behind the rest of the pass
            # (1.5-2 ms on the clock) and the load behind prefill (7-16 ms) --
            # so narrowing them saves nothing visible, while placing a narrow
            # block into a wide cache turns one contiguous copy into a strided
            # one. TTFT p50 went 102 -> 129 ms. It becomes the right shape when
            # S grows far enough that the transfer stops fitting behind
            # prefill; at S=512 it does fit.
            kf = torch.cat([k[:, :, :, :P], b["zeroKg"][:, :, :, P:]], -1)
            vf = torch.cat([v[:, :, :P, :], b["zeroVg"][:, :, P:, :]], -2)
            t1 = time.perf_counter()
            kh = kf.detach().cpu().float().numpy()
            vh = vf.detach().cpu().float().numpy()
            t2 = time.perf_counter()
            sink(g, kh, vh)
            t_c += t1 - t0; t_d += t2 - t1
        self.last_compute_ms = t_c * 1e3
        self.last_download_ms = t_d * 1e3
        return self._next_token(x, P)

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

    eng = FullPrefillEngine(cfg, sd, C=a.C, group=a.group, chip=a.chip,
                            cache_dir=a.cache, S=getattr(a, "S", 0))
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
