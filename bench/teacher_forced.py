#!/usr/bin/env python3
"""Teacher-forced quality gate for decode-graph changes (#14).

A change that moves bf16 rounding (another width, another tactic hint, another
KV path) is compared with the 16-wide reference graph on the *same* token
sequences, step by step, instead of on free-running text that diverges on the
first flipped near-tie:

- the reference continuations from a `vendor_sweep.py --greedy` file are fed
  through both paths (prompt prefilled once, its cache seated in each);
- per step and lane: top-1 agreement, |delta logit| at the teacher token,
  KL(old || new) over the old path's top-k, and the guard "new top-1 keeps at
  least half the old top-1's probability under the old model";
- optionally perplexity of a fixed text through both paths (--ppl-text).

  RNGD_MAX_SHAPES=8 python bench/teacher_forced.py --S 1024 --variants 4,8,4:ForMiscModel \
      --ref bench/greedy_rngd_pd_base_S1024.jsonl --ppl-text bench/ppl_text.txt \
      --out bench/teacher_forced_tiers_S1024.json

A variant is WIDTH or WIDTH:HINT and shares the reference engine's weights.
"""
import argparse
import json
import math
import os

import _bootstrap  # noqa: F401
import numpy as np
import torch


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class Stats:
    def __init__(self):
        self.n = self.top1 = self.guard_fail = 0
        self.dlogit, self.kl, self.nll_old, self.nll_new = [], [], [], []

    def add(self, old, new, teacher, topk):
        po, pn = softmax(old), softmax(new)
        ao, an = int(old.argmax()), int(new.argmax())
        self.n += 1
        self.top1 += int(ao == an)
        self.guard_fail += int(po[an] < 0.5 * po[ao])
        if teacher is not None:
            self.dlogit.append(abs(float(old[teacher]) - float(new[teacher])))
        k = np.argpartition(-po, topk)[:topk]
        self.kl.append(float(np.sum(po[k] * (np.log(po[k] + 1e-30) - np.log(pn[k] + 1e-30)))))

    def add_nll(self, old, new, teacher):
        self.nll_old.append(-math.log(softmax(old)[teacher] + 1e-30))
        self.nll_new.append(-math.log(softmax(new)[teacher] + 1e-30))

    def report(self):
        r = dict(steps=self.n, top1_agree=round(self.top1 / max(self.n, 1), 5),
                 guard_violations=self.guard_fail,
                 dlogit_mean=round(float(np.mean(self.dlogit)), 4) if self.dlogit else None,
                 dlogit_max=round(float(np.max(self.dlogit)), 4) if self.dlogit else None,
                 kl_topk_mean=round(float(np.mean(self.kl)), 6) if self.kl else None,
                 kl_topk_max=round(float(np.max(self.kl)), 6) if self.kl else None)
        if self.nll_old:
            po, pn = math.exp(np.mean(self.nll_old)), math.exp(np.mean(self.nll_new))
            r.update(ppl_tokens=len(self.nll_old), ppl_old=round(po, 4), ppl_new=round(pn, 4),
                     ppl_rel_pct=round((pn / po - 1) * 100, 3))
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--B", type=int, default=16, help="reference width")
    ap.add_argument("--decode-group", type=int, default=8)
    ap.add_argument("--variants", required=True, help="WIDTH[:HINT],... e.g. 4,8,4:ForMiscModel")
    ap.add_argument("--ref", required=True, help="greedy jsonl with prompt_ids and out_ids")
    ap.add_argument("--ppl-text", default="")
    ap.add_argument("--ppl-prompt", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--chip", type=int, default=-1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rngd_pd.devices import resolve_chip
    from rngd_pd.engines.model import ModelConfig
    from rngd_pd.engines.prefill_full import FullPrefillEngine
    from rngd_pd.engines.decode_batch import BatchDecodeEngine
    from rngd_pd.paths import ARTIFACT_CACHE
    model = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    chip = resolve_chip(a.chip)
    hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config)
    sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
    del hf
    cache = str(ARTIFACT_CACHE)
    pre = FullPrefillEngine(cfg, sd, C=512, group=2, chip=chip, S=a.S, cache_dir=cache)
    for c in (32, 128, 256):
        pre.add_bucket(c)
    ref = BatchDecodeEngine(cfg, sd, B=a.B, S=a.S, group=a.decode_group, chip=chip,
                            cache_dir=cache)
    variants = []
    for v in a.variants.split(","):
        w, _, h = v.partition(":")
        variants.append((v, BatchDecodeEngine(cfg, sd, B=int(w), S=a.S, chip=chip,
                                              cache_dir=cache, share=ref, hint=h or None)))
    del sd
    lanes = min(e.B for _, e in variants)
    tk = AutoTokenizer.from_pretrained(model)

    def logits_of(e, lg):
        return lg.detach().cpu().float().numpy().reshape(e.Vp, e.B)[:e.V]

    def run(seqs, prompts, stats, ppl):
        """seqs[i]: tokens fed after prompt i (teacher), predicting seqs[i][1:]."""
        made = [pre.prefill_device(p) for p in prompts]
        engs = [("ref", ref)] + variants
        state = {}
        for name, e in engs:
            Kd, Vd = e.blank_kv()
            for i, (K, V, _) in enumerate(made):
                Kd, Vd = e.write_slot_kv(Kd, Vd, i, K, V)
            state[name] = [Kd, Vd]
        T = max(len(s) for s in seqs) - 1
        for t in range(T):
            toks = [s[t] if t < len(s) else 0 for s in seqs]
            pos = [len(p) + t for p in prompts]
            outs = {}
            for name, e in engs:
                tt = toks + [0] * (e.B - len(toks))
                pp = pos + [0] * (e.B - len(pos))
                lg, state[name][0], state[name][1] = e.step_logits(tt, pp, *state[name])
                outs[name] = logits_of(e, lg)
            for name, _e in variants:
                for i, s in enumerate(seqs):
                    if t + 1 < len(s):
                        old, new = outs["ref"][:, i], outs[name][:, i]
                        stats[name].add(old, new, s[t + 1], a.topk)
                        if ppl:
                            stats[name].add_nll(old, new, s[t + 1])

    stats = {name: Stats() for name, _ in variants}
    rows = [json.loads(l) for l in open(a.ref)]
    rows = [r for r in rows if r.get("out_ids") and not r.get("err")]
    for g in range(0, len(rows), lanes):
        grp = rows[g:g + lanes]
        run([r["out_ids"] for r in grp], [r["prompt_ids"] for r in grp], stats, False)
        print(f"[teacher] {g + len(grp)}/{len(rows)} sequences", flush=True)
    result = {"S": a.S, "reference_width": a.B, "ref_file": os.path.basename(a.ref),
              "topk": a.topk, "teacher_forced": {k: v.report() for k, v in stats.items()}}

    if a.ppl_text:
        text = open(a.ppl_text, encoding="utf-8").read()
        ids = tk.encode(text, add_special_tokens=False)
        span = min(len(ids) // lanes, a.S - 1 - a.ppl_prompt)
        chunks = [ids[i * span:(i + 1) * span] for i in range(lanes)]
        pstats = {name: Stats() for name, _ in variants}
        prompts = [[tk.bos_token_id] + c[:a.ppl_prompt] for c in chunks]
        seqs = [c[a.ppl_prompt:] for c in chunks]
        # the first fed token is the last prompt token's successor: prefill
        # predicted it; feeding seqs[i][t] predicts seqs[i][t+1]
        run(seqs, prompts, pstats, True)
        result["perplexity"] = {k: v.report() for k, v in pstats.items()}
        result["ppl_text_tokens"] = span * lanes
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
