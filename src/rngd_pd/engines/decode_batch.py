#!/usr/bin/env python3
"""Serving wrapper for batched decode on RNGD.

Same shape as full_prefill_engine: real Llama-8B weights arranged for the
compiled graph once at start-up, the artifact taken from the cache rather than
recompiled, and B streams advanced per call. The cache lives on the card
between steps — a decode that shipped [B,NKV,HD,S] in and out every step would
spend more time on the bus than in the layers.

Streams are independent: each carries its own position, so a batch can mix a
request on its third token with one on its hundredth.

  python3 batch_decode_engine.py --B 2 --chip 2
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from rngd_pd.kernels.decode_batch import (BF, H, M, NH, NKV, HD, G, DecodeModel,
                                rot_batch, HINT, MAX_SHAPES)
from rngd_pd.engines.fsdk import CompileModule
from rngd_pd.engines.fsdk import FCConfig, TacticHintConfig
from rngd_pd.engines.fsdk import hint as _hint
from rngd_pd.engines.fsdk import dev_str
from rngd_pd.paths import ARTIFACT_CACHE

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# 2026.3.0 dropped these two options (always on); 2026.2.0 needs them.
from rngd_pd.engines.fsdk import NEW as _NEW_SDK  # noqa: E402
_LEGACY = {} if _NEW_SDK else dict(allow_external_operators=True,
                                   allow_unlowered_operators=True)


def decode_graph_spec(B, S, group, hint=None):
    """The cache key and the build for one decode graph, in one place.

    The pod does not have to be the thing that compiles. Its 96Gi cap with the
    whole state dict already resident is where RNGD_MAX_SHAPES falls over,
    while the same config compiles fine on the bare host. artifacts_cache is a
    hostPath both share, so the host can bake the graph and the pod can just
    load it — but only if both sides derive the key identically, which is why
    this is a function and not two copies.

    The shape cap belongs in the key. Without it a capped build silently loads
    the old uncapped artifact and the change does nothing at all.
    """
    from rngd_pd.engines.artifact_cache import cache_key
    hint = hint or HINT
    key = cache_key(kind="batch-decode", model=MODEL, B=B, S=S, nl=group,
                    hint=hint, **MAX_SHAPES)
    def build():
        sample = (torch.zeros(B, H, dtype=BF),
                  torch.zeros(B, H, dtype=torch.float32), torch.zeros(B, H, dtype=torch.float32),
                  torch.zeros(group, B, H, dtype=BF),
                  torch.zeros(group, H, NH * HD, dtype=BF), torch.zeros(group, H, NKV * HD, dtype=BF),
                  torch.zeros(group, H, NKV * HD, dtype=BF), torch.zeros(group, NH * HD, H, dtype=BF),
                  torch.zeros(group, B, H, dtype=BF),
                  torch.zeros(group, H, M, dtype=BF), torch.zeros(group, H, M, dtype=BF),
                  torch.zeros(group, M, H, dtype=BF),
                  torch.zeros(group, B, NKV, HD, S, dtype=BF),
                  torch.zeros(group, B, NKV, S, HD, dtype=BF),
                  torch.zeros(B, HD, HD, dtype=BF),
                  torch.zeros(B, S, dtype=BF), torch.zeros(B, S, dtype=BF))
        ccfg = FCConfig(tactic_hint=_hint(hint), **_LEGACY, **MAX_SHAPES)
        return CompileModule.from_module(DecodeModel(B, S, group), sample,
                                         compiler_config=ccfg)
    return key, build


def resolve_group(B, S, group, cache_dir, fallback=2):
    """Only ask for a group the pod can actually get.

    Groups above 2 need RNGD_MAX_SHAPES, and compiling one inside the pod
    fails: it holds the whole state dict before it starts and lives under a
    96Gi cap, so the same graph that builds on the host raises
    UnsupportedOpError there. bench/prebake.py exists to bake it on the host
    into the shared artifacts_cache instead.

    If that bake has not happened, attempting it here costs eight minutes and
    then crash-loops the worker. Serving at the smaller group is strictly
    better than not serving, so drop to it and say so.
    """
    if not cache_dir or group == fallback:
        return group
    from rngd_pd.engines.artifact_cache import exists, path_for
    key, _ = decode_graph_spec(B, S, group)
    if exists(path_for(cache_dir, key)):
        return group
    print(f"[batch-decode] group={group} 아티팩트가 캐시에 없다 (키 {key}). "
          f"파드 안에서 굽는 것은 실패하므로 group={fallback}로 내린다. "
          f"되살리려면 호스트에서 bench/prebake.py --group {group}", flush=True)
    return fallback


# Above this width the lm_head matmul stops being matrix-vector.
LMHEAD_VECTOR_B = int(os.environ.get("RNGD_LMHEAD_VECTOR_B", "4"))


def lmhead_graph_spec(B, V, NG=32):
    """The cache key and the build for the on-card lm_head, in one place.

    The decode graph was already baked on the host for exactly this reason,
    but the lm_head was left for the pod to compile, and the pod cannot: it
    raises UnsupportedOpError at B=8 where the same config compiles on the bare
    host. That made every large-batch deployment a crash loop whose traceback
    pointed at the lm_head rather than at the batch width, so the two are baked
    together now and by the same key derivation as the decode graph.
    """
    import torch
    from rngd_pd.engines.artifact_cache import cache_key
    from rngd_pd.kernels import dsl_batch as LEB
    from rngd_pd.engines.fsdk import CompileModule, TacticKernelModule
    from rngd_pd.engines.fsdk import FCConfig, TacticHintConfig
    from rngd_pd.engines.fsdk import NEW as _NEW

    Vp = (V + NG - 1) // NG * NG

    class LmHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rms = TacticKernelModule(LEB.rms_dsl(B))

        def forward(self, x, nw, eps, nh, wlm):
            h = self.rms(x, nw, eps, nh)[0]              # [B,H]
            return torch.matmul(wlm, h.t())              # [NG,Vp/NG,B]

    # On 2026.2.0 the hint flips sign with the batch, and there is no rule to
    # it: measured over all seven, the winner is ForLlmModelDecode at B=4
    # (1.24 ms), ForMiscModel at B=8 (1.21) and Default at B=16 (1.34), while
    # the losers sit at ~45 ms -- a 37x cliff, and the whole of "batched decode
    # collapses past four", since the decode graph itself only goes 4.87 ->
    # 5.96 ms a call. LMHEAD_VECTOR_B picks its way across that by hand.
    #
    # 2026.3.0 flattens it: every one of its seven hints lands in 1.08-1.37 ms
    # at B=4, 8 and 16, so there is nothing to pick around and the width no
    # longer enters. ComputeIOBalanced is the best of them at B=4 and B=16 and
    # within 0.01 ms of the best at B=8.
    if _NEW:
        hint = TacticHintConfig.ForLlmModelComputeIOBalanced
    else:
        hint = (TacticHintConfig.ForLlmModelDecode if B <= LMHEAD_VECTOR_B
                else TacticHintConfig.NoConstraint)
    # the hint belongs in the key: two widths that share one would otherwise
    # load each other's artifact and the fix would silently do nothing
    key = cache_key(kind="batch-lmhead", model=MODEL, B=B, V=V, groups=NG,
                    hint=hint.__class__.__name__ + str(hint))
    def build():
        sample = (torch.zeros(B, H, dtype=BF), torch.zeros(B, H, dtype=BF),
                  torch.zeros(B, H, dtype=torch.float32),
                  torch.zeros(B, H, dtype=torch.float32),
                  torch.zeros(NG, Vp // NG, H, dtype=BF))
        ccfg = FCConfig(tactic_hint=hint, **_LEGACY)
        return CompileModule.from_module(LmHead(), sample, compiler_config=ccfg)
    return key, build



def greedy_tokens(logits, Vp, V, B):
    """Argmax per stream of the on-card lm_head output [NG, Vp/NG, B].

    It used to widen all Vp x B logits to float32 numpy and argmax column by
    column: 3.4-4.1 ms of every B=16 step on the host (#14 phase 2). torch's
    argmax on the bf16 tensor gives the same index -- the widening is exact
    and both return the first maximum -- in a fraction of that.
    """
    lg = logits.detach().cpu().reshape(Vp, B)[:V]
    return torch.argmax(lg, dim=0).tolist()


class BatchDecodeEngine:
    def __init__(self, cfg, sd, B=2, S=256, group=2, chip=2, cache_dir=None,
                 share=None, hint=None):
        """`share`: another BatchDecodeEngine (same S and group) whose 16 GB of
        projection weights and lm_head this one reuses. The graph takes the
        weights as inputs, so another width costs only its own graph, its
        B-tiled norms and its cache (see engines/tiers.py). `hint`: the decode
        graph's tactic hint, if not DECODE_TACTIC_HINT."""
        group = share.NG if share is not None else resolve_group(B, S, group, cache_dir)
        self.hint = hint or HINT
        self.B, self.S, self.NG = B, S, group
        self.nl = cfg.n_layers
        self.ngroups = self.nl // group
        assert self.nl % group == 0
        self.dev = dev_str(chip)
        self.eps = cfg.eps
        self.embed = sd["model.embed_tokens.weight"]
        self.lm = sd["lm_head.weight"]
        self.norm = sd["model.norm.weight"]
        self.inv = cfg.inv_freq()
        sq, sc = math.sqrt(H), 1.0 / math.sqrt(HD)
        d = self._d

        self.groups = []
        for g in range(self.ngroups):
            idx = range(g * group, (g + 1) * group)
            p = lambda i, k: sd[f"model.layers.{i}.{k}"].astype(np.float32)
            if share is not None:
                w = share.groups[g]
                self.groups.append((
                    torch.stack([d(np.tile(p(i, "input_layernorm.weight") * sq, (B, 1))) for i in idx]),
                    *w[1:5],
                    torch.stack([d(np.tile(p(i, "post_attention_layernorm.weight") * sq, (B, 1))) for i in idx]),
                    *w[6:]))
                continue
            self.groups.append((
                torch.stack([d(np.tile(p(i, "input_layernorm.weight") * sq, (B, 1))) for i in idx]),
                torch.stack([d((p(i, "self_attn.q_proj.weight") * sc).T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.k_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.v_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "self_attn.o_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(np.tile(p(i, "post_attention_layernorm.weight") * sq, (B, 1))) for i in idx]),
                torch.stack([d(p(i, "mlp.gate_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.up_proj.weight").T.copy()) for i in idx]),
                torch.stack([d(p(i, "mlp.down_proj.weight").T.copy()) for i in idx]),
            ))

        key, build = decode_graph_spec(B, S, group, self.hint)
        t0 = time.time()
        if cache_dir:
            from rngd_pd.engines.artifact_cache import get_or_compile
            self.m, mode, secs = get_or_compile(cache_dir, key, build, device=None)
            print(f"[batch-decode] {mode} in {secs:.0f}s", flush=True)
        else:
            self.m = build()
            print(f"[batch-decode] compiled in {time.time()-t0:.0f}s", flush=True)
        self.m.to(self.dev)
        self.eps_d = self._df(np.full((B, H), self.eps * H, np.float32))
        self.nh_d = self._df(np.full((B, H), -0.5, np.float32))
        self._build_tables()
        self._build_lmhead(cache_dir, share)

    def _build_tables(self):
        """Everything a step needs that depends only on position, built once.

        A step was rebuilding all of it per token: a [HD,HD] rotation per
        stream from scratch, a mask assembled with concatenate, and — worst —
        a full [S,S] identity allocated to take one row out of it, per stream
        per step. None of it varies with anything but the position, so it is a
        table lookup.
        """
        S = self.S
        self._eye = np.eye(S, dtype=np.float32)                  # one-hot rows
        self._mask = np.triu(np.full((S, S), -1e4, np.float32), 1)
        # float32 throughout: the loop computed `p * inv` in float32, and
        # arange's int64 would promote the table to float64 and round
        # differently — six of 512 rows land on the other side of a bf16
        # boundary. A speedup should not change what the card sees.
        ang = np.arange(S, dtype=np.float32)[:, None] * self.inv[None, :]
        cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        half = HD // 2
        rot = np.zeros((S, HD, HD), np.float32)
        i = np.arange(half)
        rot[:, i, i] = cs
        rot[:, i + half, i + half] = cs
        rot[:, i, i + half] = sn
        rot[:, i + half, i] = -sn
        self._rot = rot

    def _build_lmhead(self, cache_dir, share=None):
        """lm_head on the card, batched.

        bench/pd_engine.build_lmhead already established the shape that works:
        the monolithic [V,H] matvec streams at ~23 GB/s, while chunking V into
        32 groups reaches ~565 GB/s. That version takes one stream; this one
        contracts against [H,B] so the same weight read serves the whole batch.

        It matters more here than it did at B=1. The transformer step is
        B-invariant, so a host-side lm_head — 2.1 GB read per stream — is the
        one part of the step that still scales with B, and at B=4 it was three
        quarters of the time.
        """
        B = self.B
        V = self.lm.shape[0]
        NG = 32
        Vp = (V + NG - 1) // NG * NG
        self.V, self.NGV, self.Vp = V, NG, Vp
        key, build = lmhead_graph_spec(B, V, NG)
        t0 = time.time()
        if cache_dir:
            from rngd_pd.engines.artifact_cache import get_or_compile
            self.lmh, mode, secs = get_or_compile(cache_dir, key, build, device=None)
            print(f"[batch-lmhead] {mode} in {secs:.0f}s", flush=True)
        else:
            self.lmh = build()
            print(f"[batch-lmhead] compiled in {time.time()-t0:.0f}s", flush=True)
        self.lmh.to(self.dev)
        if share is not None:
            self.wlm_d = share.wlm_d               # 1 GB, the same for every width
        else:
            wlm = np.zeros((Vp, H), np.float32)
            wlm[:V] = self.lm
            self.wlm_d = self._d(wlm.reshape(NG, Vp // NG, H))
        self.norm_d = self._d(np.tile(self.norm * math.sqrt(H), (self.B, 1)))

    def _d(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(BF).to(self.dev)

    def _df(self, v):
        return torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32).to(self.dev)

    def prepare_kv(self, Ks, Vs):
        """Everything the upload needs that does not touch the card.

        The stack and the float32 -> bfloat16 cast are host work, and they are
        the larger half: at B=4, S=512 the stack copies 536 MB and the cast
        reads that and writes 268 MB, against a 268 MB transfer. Doing them
        inside the device lock charged all of it to the card and blocked every
        other device op for the duration. Split out, it can run while the other
        card is still prefilling.
        """
        B = self.B
        Ks, Vs = list(Ks), list(Vs)
        while len(Ks) < B:                       # pad the unused lanes
            Ks.append(Ks[-1]); Vs.append(Vs[-1])
        # Entries arrive either as float32 numpy (no prefetch staged them) or
        # as bfloat16 torch tensors the prefetch already cast during prefill.
        # Stacking the cast form moves half the bytes and skips the cast here,
        # which is the whole point of doing it early.
        def cast(a):
            return a if torch.is_tensor(a) else torch.from_numpy(
                np.ascontiguousarray(a)).to(BF)
        K = torch.stack([cast(a) for a in Ks], 1)   # [nl,B,NKV,HD,S]
        V = torch.stack([cast(a) for a in Vs], 1)   # [nl,B,NKV,S,HD]
        grp = lambda T, g: T[g * self.NG:(g + 1) * self.NG].contiguous()
        return ([grp(K, g) for g in range(self.ngroups)],
                [grp(V, g) for g in range(self.ngroups)])

    @staticmethod
    def stage_host_kv(K, V):
        """float32 host arrays -> bfloat16 host tensors, off the critical path.

        Runs in the prefetch thread while the other card is still prefilling,
        so the batch inherits tensors that are already the width the card
        wants. Touches no device, so it cannot contend with a decode in flight.
        """
        return (torch.from_numpy(np.ascontiguousarray(K)).to(BF),
                torch.from_numpy(np.ascontiguousarray(V)).to(BF))

    def upload_prepared_kv(self, prepared):
        """The part that actually needs the card: one transfer per layer group.

        Deliberately not named upload_kv — the class already has one, taking
        host arrays, and shadowing it silently broke every batch with a
        TypeError that pointed at the caller rather than the collision.
        """
        Kh, Vh = prepared
        return ([t.to(self.dev) for t in Kh], [t.to(self.dev) for t in Vh])

    def load_kv(self, Ks, Vs):
        """Stack per-stream caches onto the card, one transfer per layer group."""
        return self.upload_prepared_kv(self.prepare_kv(Ks, Vs))

    def write_slot_kv(self, Kd, Vd, idx, K, V):
        """Replace one lane of the resident cache, in place, on the card.

        The claim that a slot cannot be replaced in place was never tested; it
        can, bit-exactly, and it is what the continuous scheduler needed. The
        round trip it replaces moves the whole batch to admit one stream --
        B times the bytes that actually change -- and measured 67 ms per layer
        group against 2.4 ms for the lane write.

        K and V arrive either as float32 numpy or as the bfloat16 host tensors
        the prefetch staged during prefill; both are accepted so the caller
        does not have to care which path fed it.
        """
        cast = lambda a: a if torch.is_tensor(a) else torch.from_numpy(
            np.ascontiguousarray(a)).to(BF)
        Kh, Vh = cast(K), cast(V)
        for g in range(self.ngroups):
            sl = slice(g * self.NG, (g + 1) * self.NG)
            Kd[g][:, idx] = Kh[sl].to(self.dev)
            Vd[g][:, idx] = Vh[sl].to(self.dev)
        return Kd, Vd

    def download_kv(self, Kd, Vd):
        """Resident cache back to the host as [nl,B,NKV,HD,S] / [nl,B,NKV,S,HD].

        Kept for the blank/reset path and for diagnostics. Admission no longer
        uses it -- see write_slot_kv.
        """
        # Cast on the host, not the card: bf16 -> f32 is unsupported on RNGD, so
        # .float() before .cpu() routes the whole tensor through a CPU-bridged
        # copy. Coming back as bf16 first makes it a plain transfer.
        K = np.concatenate([k.detach().cpu().float().numpy() for k in Kd], 0)
        V = np.concatenate([v.detach().cpu().float().numpy() for v in Vd], 0)
        return K, V

    def upload_kv(self, K, V):
        """The inverse: [nl,B,...] host arrays back onto the card, per group."""
        NG = self.NG
        Kd = [self._d(K[g * NG:(g + 1) * NG]) for g in range(self.ngroups)]
        Vd = [self._d(V[g * NG:(g + 1) * NG]) for g in range(self.ngroups)]
        return Kd, Vd

    def adopt_kv_device(self, Ks, Vs):
        """Per-request caches that are already on the card, batched in place.

        The disaggregated path has to come through host arrays because the two
        stages are different processes. When prefill ran on this same card its
        output is already a device tensor in the right layout, so batching B of
        them is a stack and a split — no transfer at all.
        """
        while len(Ks) < self.B:                       # pad the unused lanes
            Ks.append(Ks[-1]); Vs.append(Vs[-1])
        K = torch.stack(Ks, 1)                        # [nl,B,NKV,HD,S]
        V = torch.stack(Vs, 1)                        # [nl,B,NKV,S,HD]
        NG = self.NG
        Kd = [K[g * NG:(g + 1) * NG] for g in range(self.ngroups)]
        Vd = [V[g * NG:(g + 1) * NG] for g in range(self.ngroups)]
        return Kd, Vd

    def blank_kv(self):
        """An empty resident cache — every slot free."""
        K = np.zeros((self.nl, self.B, NKV, HD, self.S), np.float32)
        V = np.zeros((self.nl, self.B, NKV, self.S, HD), np.float32)
        return self.upload_kv(K, V)

    def step(self, toks, positions, Kd, Vd):
        """One token for each of B streams. Returns next tokens and the cache."""
        logits, Kd, Vd = self.step_logits(toks, positions, Kd, Vd)
        return greedy_tokens(logits, self.Vp, self.V, self.B), Kd, Vd

    def step_logits(self, toks, positions, Kd, Vd):
        """The same step, returning the lm_head output itself ([NG, Vp/NG, B],
        on the card) -- for teacher-forced comparisons (bench/teacher_forced.py)."""
        B, S = self.B, self.S
        pos = np.asarray(positions)
        x = self._d(self.embed[list(toks)])          # [B,H], one gather
        rot = self._d(self._rot[pos])                # [B,HD,HD]
        maskd = self._d(self._mask[pos])             # [B,S], causal at p
        ohd = self._d(self._eye[pos])                # [B,S]
        for g in range(self.ngroups):
            x, Kf, Vf = self.m(x, self.eps_d, self.nh_d, *self.groups[g],
                               Kd[g], Vd[g], rot, maskd, ohd)
            Kd[g], Vd[g] = Kf, Vf
        out = self.lmh(x, self.norm_d, self.eps_d, self.nh_d, self.wlm_d)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits, Kd, Vd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cache",
                    default=str(ARTIFACT_CACHE))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rngd_pd.engines.model import ModelConfig
    print(f"loading {MODEL}...", flush=True)
    tk = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf

    eng = BatchDecodeEngine(cfg, sd, B=a.B, S=a.S, group=a.group, chip=a.chip,
                            cache_dir=a.cache)
    nl, NKV_, HD_ = cfg.n_layers, cfg.n_kv, cfg.head_dim
    Ks = [np.zeros((nl, NKV_, HD_, a.S), np.float32) for _ in range(a.B)]
    Vs = [np.zeros((nl, NKV_, a.S, HD_), np.float32) for _ in range(a.B)]
    Kd, Vd = eng.load_kv(Ks, Vs)
    toks = [tk.encode("The capital of France is",
                      add_special_tokens=False)[-1]] * a.B
    pos = [8 + i for i in range(a.B)]
    eng.step(toks, pos, Kd, Vd)                       # warm up
    ts = []
    for _ in range(a.steps):
        t0 = time.perf_counter()
        toks, Kd, Vd = eng.step(toks, pos, Kd, Vd)
        ts.append((time.perf_counter() - t0) * 1e3)
        pos = [p + 1 for p in pos]
    ts.sort()
    step = ts[len(ts) // 2]
    print("\n" + "=" * 66)
    print(f"RNGD batched decode — B={a.B}, {cfg.n_layers} layers, S={a.S}")
    print(f"  {step:.2f} ms/step -> {step/a.B:.2f} ms per token")
    print(f"  weight bandwidth {218e6*2*cfg.n_layers/(step/1000)/1e9:.0f} GB/s")
    print("=" * 66)


if __name__ == "__main__":
    main()
