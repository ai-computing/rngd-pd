#!/usr/bin/env python3
"""Where does the ATOM-returned KV go wrong? One step, three questions.

The teacher-forced run showed hidden error jumping 1% → 33% the moment the
decoder starts consuming its OWN returned KV, so the suspect is the KV the
compiled module emits, not the attention it computes. After a single step from
identical inputs we ask:

  1. copy-through  — are the P already-filled slots still bit-identical?
  2. new slot      — is the freshly written column at position P correct?
  3. empty tail    — did anything leak into slots > P?

Compared against the RNGD engine's own KV after the same step, and against a
CPU fp32 computation of what the new slot should hold.
"""
import argparse

import numpy as np
import torch

import _paths  # noqa: F401
from pd_hetero_llama import MODEL, LlamaDecodeRBLN, rbln_compile, load_hf
from pd_hetero_llama_diag import rngd_step

torch.set_grad_enabled(False)


def rel(a, b):
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (d if d > 0 else 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--prompt", default="The three primary colors are red, "
                    "blue and yellow. The capital of France is")
    a = ap.parse_args()

    import llama_engine_resident as LER
    from pd_engine import build_engine, EngineSession
    from generic_runtime import rmsnorm
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(MODEL)
    prompt = tk.encode(a.prompt, add_special_tokens=False)[:a.P]
    P = len(prompt)

    cfg, rt = build_engine(MODEL, a.S, 32, chip=a.chip)
    sess = EngineSession(rt, LER)
    nl = rt.nl

    Kp, Vp = sess.fresh_kv()
    for pos, t in enumerate(prompt):
        hid, Kp, Vp = rngd_step(rt, LER, t, pos, Kp, Vp)
    Kh, Vh = sess.export_kv(Kp, Vp)
    nxt = int(np.argmax(rt.lm @ rmsnorm(hid, rt.norm, rt.c.eps)))
    print(f"[prefill] P={P} next={nxt} ({tk.decode([nxt])!r})", flush=True)

    # one more step on RNGD -> reference KV
    _hr, Kp2, Vp2 = rngd_step(rt, LER, nxt, P, Kp, Vp)
    Kr, Vr = sess.export_kv(Kp2, Vp2)

    # same step on ATOM
    _tk, cfg2, sd = load_hf()
    embed = sd["model.embed_tokens.weight"]
    mod = LlamaDecodeRBLN(cfg2, sd, a.S, nl).eval()
    rrt = rbln_compile(mod, cfg2, a.S, nl, f"llama {nl}L S={a.S} fp16")

    inv = cfg2.inv_freq()
    ang = P * inv
    mask = np.concatenate([np.zeros(P + 1, np.float32),
                           np.full(a.S - P - 1, -1e4, np.float32)])
    x = embed[nxt].astype(np.float32).reshape(1, -1)
    args = [x, Kh, Vh, np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32),
            mask, np.eye(a.S, dtype=np.float32)[P]]
    out = rrt.run(*args)
    Ka, Va = np.asarray(out[1]), np.asarray(out[2])
    # CPU fp32 run of the exact same module, for an ATOM-free reference
    cout = mod(*[torch.from_numpy(np.ascontiguousarray(v)) for v in args])
    Kc, Vc = cout[1].numpy(), cout[2].numpy()
    del sd, mod

    print(f"\nK layout [nl,NKV,HD,S]={Ka.shape}   V layout [nl,NKV,S,HD]={Va.shape}")
    print(f"{'region':>28} {'ATOM vs RNGD':>14} {'ATOM vs CPU':>14} {'CPU vs RNGD':>14}")
    print("-" * 74)
    regions = [
        ("K copy-through [<P]", Ka[..., :P], Kr[..., :P], Kc[..., :P]),
        ("K new slot [P]", Ka[..., P], Kr[..., P], Kc[..., P]),
        ("K empty tail [>P]", Ka[..., P + 1:], Kr[..., P + 1:], Kc[..., P + 1:]),
        ("V copy-through [<P]", Va[:, :, :P, :], Vr[:, :, :P, :], Vc[:, :, :P, :]),
        ("V new slot [P]", Va[:, :, P, :], Vr[:, :, P, :], Vc[:, :, P, :]),
        ("V empty tail [>P]", Va[:, :, P + 1:, :], Vr[:, :, P + 1:, :], Vc[:, :, P + 1:, :]),
    ]
    for name, at, rf, cp in regions:
        print(f"{name:>28} {rel(at, rf):>14.5f} {rel(at, cp):>14.5f} {rel(cp, rf):>14.5f}")

    # per-layer view of the worst region
    print("\nper-layer rel(ATOM,RNGD) for the copy-through region:")
    kl = [rel(Ka[i, ..., :P], Kr[i, ..., :P]) for i in range(nl)]
    vl = [rel(Va[i, :, :P, :], Vr[i, :, :P, :]) for i in range(nl)]
    print("  K:", " ".join(f"{v:.3f}" for v in kl))
    print("  V:", " ".join(f"{v:.3f}" for v in vl))
    print(f"\ninput KV was bit-identical to what RNGD handed over: "
          f"K={np.array_equal(Kh, Kh)} (by construction)")
    print(f"exactness of ATOM copy-through: K max|Δ|="
          f"{np.abs(Ka[..., :P] - Kh[..., :P]).max():.6f}  "
          f"V max|Δ|={np.abs(Va[:, :, :P, :] - Vh[:, :, :P, :]).max():.6f}"
          "   (0 = the compiled graph passes old slots through untouched)")


if __name__ == "__main__":
    main()
