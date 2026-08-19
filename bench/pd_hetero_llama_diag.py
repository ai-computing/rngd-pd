#!/usr/bin/env python3
"""Why does 8B hetero decode diverge? Teacher-forced RNGD-vs-ATOM comparison.

Free-running decode hides the cause: one flipped argmax changes every later
input. Here both cards consume the SAME token sequence, so each step's hidden
state is directly comparable, and each divergence can be scored against the
reference model's own top-1/top-2 logit margin — noise at a near-tie looks very
different from accumulated error.

  PYTHONPATH=$HOME/.local/lib/python3.10/site-packages VLLM_PLUGINS= \
    ~/venvs/furiosa/bin/python bench/pd_hetero_llama_diag.py --P 32 --D 8
"""
import argparse
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401
from rngd_pd.vendors.rbln_decode import MODEL, LlamaDecodeRBLN, rbln_compile, load_hf

torch.set_grad_enabled(False)


def rngd_step(rt, LER, tok, pos, Kp, Vp):
    """One engine step that also hands back the hidden state."""
    an1, aq, ak, av, ao, an2, awgu, awd = rt.W
    ang = pos * rt.inv
    cos = LER._d(np.cos(ang)); sin = LER._d(np.sin(ang))
    mask = np.full(rt.S, -1e4, np.float32); mask[:pos + 1] = 0.0
    oh = np.zeros(rt.S, np.float32); oh[pos] = 1.0
    x = LER._d(rt.embed[tok].astype(np.float32))
    hid, Kp, Vp = rt.m(x, rt.eps_d, rt.nh_d, an1, aq, ak, av, ao, an2, awgu, awd,
                       Kp, Vp, cos, sin, LER._d(mask), LER._d(oh))
    return hid.detach().float().cpu().numpy(), Kp, Vp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    a = ap.parse_args()

    from rngd_pd.kernels import decode_resident as LER
    from rngd_pd.engines.decode_resident import build_engine, EngineSession
    from rngd_pd.engines.model import rmsnorm
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(MODEL)
    prompt = tk.encode(a.prompt, add_special_tokens=False)[:a.P]
    P = len(prompt)

    cfg, rt = build_engine(MODEL, a.S, 32, chip=a.chip)
    sess = EngineSession(rt, LER)
    nl, NKV, HD = rt.nl, LER.NKV, LER.HD
    lm, normw, eps = rt.lm, rt.norm, rt.c.eps

    # prefill on RNGD, then keep the KV for BOTH decoders
    Kp, Vp = sess.fresh_kv()
    for pos, t in enumerate(prompt):
        hid, Kp, Vp = rngd_step(rt, LER, t, pos, Kp, Vp)
    Kh, Vh = sess.export_kv(Kp, Vp)
    logits = lm @ rmsnorm(hid, normw, eps)
    nxt = int(np.argmax(logits))
    print(f"[prefill] P={P} next={nxt} ({tk.decode([nxt])!r})", flush=True)

    # ATOM decoder over the handed-off KV
    _tk, cfg2, sd = load_hf()
    embed = sd["model.embed_tokens.weight"]
    mod = LlamaDecodeRBLN(cfg2, sd, a.S, nl).eval()
    del sd
    rrt = rbln_compile(mod, cfg2, a.S, nl, f"llama {nl}L S={a.S} fp16")
    del mod

    inv = cfg2.inv_freq()
    Ka, Va = Kh.copy(), Vh.copy()          # ATOM's own KV, seeded by the handoff
    Kr, Vr = Kp, Vp                        # RNGD's resident KV
    cur, p = nxt, P
    print(f"\n{'step':>4} {'token fed':>22} {'hid_rel':>9} {'rngd top1':>12} "
          f"{'atom top1':>12} {'margin':>8} {'margin%':>8}")
    print("-" * 82)
    flips = ties = 0
    for s in range(a.D):
        # --- RNGD (reference) ---
        hr, Kr, Vr = rngd_step(rt, LER, cur, p, Kr, Vr)
        lg_r = lm @ rmsnorm(hr, normw, eps)
        order = np.argsort(lg_r)[::-1]
        t1, t2 = int(order[0]), int(order[1])
        margin = float(lg_r[t1] - lg_r[t2])
        rel_margin = margin / max(abs(float(lg_r[t1])), 1e-9) * 100

        # --- ATOM (same input token, its own KV) ---
        ang = p * inv
        mask = np.concatenate([np.zeros(p + 1, np.float32),
                               np.full(a.S - p - 1, -1e4, np.float32)])
        out = rrt.run(embed[cur].astype(np.float32).reshape(1, -1), Ka, Va,
                      np.cos(ang).astype(np.float32),
                      np.sin(ang).astype(np.float32),
                      mask, np.eye(a.S, dtype=np.float32)[p])
        ha = np.asarray(out[0]).reshape(-1)
        Ka[:, :, :, p] = np.asarray(out[1])          # host-side KV scatter
        Va[:, :, p, :] = np.asarray(out[2])
        lg_a = lm @ ha                     # ATOM already returns the normed hidden
        ta = int(np.argmax(lg_a))
        hid_rel = float(np.linalg.norm(ha - rmsnorm(hr, normw, eps))
                        / np.linalg.norm(rmsnorm(hr, normw, eps)))

        flag = ""
        if ta != t1:
            flips += 1
            flag = "  <-- FLIP" + ("  (near-tie)" if rel_margin < 1.0 else "  (REAL)")
            if rel_margin < 1.0:
                ties += 1
        print(f"{s:>4} {tk.decode([cur])!r:>22} {hid_rel:>9.5f} "
              f"{tk.decode([t1])!r:>12} {tk.decode([ta])!r:>12} "
              f"{margin:>8.3f} {rel_margin:>7.2f}%{flag}", flush=True)
        cur = t1                           # teacher forcing: follow the reference
        p += 1

    print("-" * 82)
    print(f"flips {flips}/{a.D} — of which near-ties (<1% margin): {ties}")
    print(f"verdict: {'fp16/precision noise at low-margin decisions' if flips == ties else 'systematic error present'}")


if __name__ == "__main__":
    main()
