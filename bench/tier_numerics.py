#!/usr/bin/env python3
"""Do decode widths agree bit for bit? Same streams through B=16 and narrower graphs.

Prefills a few prompts once, seats them in lanes 0..n-1 of every width, runs
steps, and compares per lane: the hidden state after the layers, the logits
(each width's own lm_head, and the 16-wide lm_head fed the narrow hidden
state, zero-padded), and the argmax.

  RNGD_MAX_SHAPES=8 python bench/tier_numerics.py --S 1024 --widths 4,8,16
"""
import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--widths", default="4,8,16")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    os.environ["RNGD_BUCKETS"] = "128"
    os.environ["RNGD_DECODE_TIERS"] = a.widths
    from rngd_pd.serving import cluster
    widths = sorted(int(v) for v in a.widths.split(","))
    w = cluster.SoloWorker(-1, a.S, C=256, group=2, B=widths[-1], decode_group=a.group,
                           sched="continuous")
    td = w.dec
    engines = {e.B: e for e in td.tiers}
    wide = engines[widths[-1]]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(cluster.MODEL)
    texts = ["Q: A train leaves at 3:15 pm and arrives at 6:40 pm. How long is the trip?\nA:",
             "Here is a haiku about autumn leaves:\n",
             "The derivative of x^3 * sin(x) with respect to x is",
             "The capital of France is"]
    n = min(len(texts), widths[0])
    prompts = [[tk.bos_token_id] + tk.encode(t, add_special_tokens=False) for t in texts[:n]]
    made = [w.pre.prefill_device(p) for p in prompts]
    state = {}
    for B, e in engines.items():
        Kd, Vd = e.blank_kv()
        for i, (K, V, _) in enumerate(made):
            Kd, Vd = e.write_slot_kv(Kd, Vd, i, K, V)
        state[B] = [Kd, Vd, [m[2] for m in made] + [0] * (B - n), [len(p) for p in prompts] + [0] * (B - n)]

    def run(e, st, lm_wide=False):
        Kd, Vd, toks, pos = st
        x = e._d(e.embed[list(toks)])
        p = np.asarray(pos)
        rot, mk, oh = e._d(e._rot[p]), e._d(e._mask[p]), e._d(e._eye[p])
        for g in range(e.ngroups):
            x, Kf, Vf = e.m(x, e.eps_d, e.nh_d, *e.groups[g], Kd[g], Vd[g], rot, mk, oh)
            Kd[g], Vd[g] = Kf, Vf
        hid = x.detach().cpu().float().numpy()
        out = e.lmh(x, e.norm_d, e.eps_d, e.nh_d, e.wlm_d)
        lg = (out[0] if isinstance(out, (tuple, list)) else out).detach().cpu().float().numpy()
        lg = lg.reshape(e.Vp, e.B)[:e.V]
        lgw = None
        if lm_wide and e.B < wide.B:
            xp = torch.cat([x, wide._d(np.zeros((wide.B - e.B, x.shape[1]), np.float32))], 0)
            o2 = wide.lmh(xp, wide.norm_d, wide.eps_d, wide.nh_d, wide.wlm_d)
            lgw = (o2[0] if isinstance(o2, (tuple, list)) else o2).detach().cpu().float().numpy()
            lgw = lgw.reshape(wide.Vp, wide.B)[:wide.V][:, :e.B]
        return hid, lg, lgw

    diff = {B: dict(hidden_bitexact=0, logits_bitexact=0, argmax_same=0, argmax_same_widelm=0,
                    max_hidden_abs=0.0) for B in widths[:-1]}
    total = 0
    for step in range(a.steps):
        res = {B: run(engines[B], state[B], lm_wide=True) for B in widths}
        hw, lw, _ = res[widths[-1]]
        ref_tok = lw[:, :n].argmax(0)
        for B in widths[:-1]:
            h, l, lgw = res[B]
            d = diff[B]
            d["hidden_bitexact"] += int((h[:n] == hw[:n]).all())
            d["logits_bitexact"] += int((l[:, :n] == lw[:, :n]).all())
            d["argmax_same"] += int((l[:, :n].argmax(0) == ref_tok).all())
            d["argmax_same_widelm"] += int((lgw[:, :n].argmax(0) == ref_tok).all())
            d["max_hidden_abs"] = max(d["max_hidden_abs"], float(np.abs(h[:n] - hw[:n]).max()))
        total += 1
        for B in widths:                         # all widths follow the wide graph's tokens
            st = state[B]
            st[2] = list(ref_tok) + [0] * (B - n)
            st[3] = [p + 1 if i < n else 0 for i, p in enumerate(st[3])]
    print(f"steps={total} lanes={n}")
    for B, d in diff.items():
        print(f"B={B} vs B={widths[-1]}: {d}")


if __name__ == "__main__":
    main()
