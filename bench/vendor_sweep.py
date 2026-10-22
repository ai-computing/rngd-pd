#!/usr/bin/env python3
"""One harness for any OpenAI-style completion server on one RNGD card.

It drives furiosa-llm `serve` (the reference) and rngd-pd's own one-card
worker (`cluster.py --role solo`, which serves /v1/completions with streaming)
the same way, so the two columns of a comparison come from one client:

- the same prompts: exactly P token IDs each, sent as a token-ID list so both
  servers see identical input. Every request starts with fresh random tokens,
  so no prefix cache can turn prefill into a lookup;
- the same grid: P x concurrency, `reps` bursts per point, each burst sending
  `concurrency` requests at once, after one untimed warm-up request per P;
- the same output: greedy, `ignore_eos`, exactly `max_tokens` tokens;
- the same clock: TTFT is send -> first streamed token; TPOT is
  (last token - first token) / (tokens - 1) per request; tokens/s is the
  burst's generated tokens over its wall time (median over bursts);
- card power and utilisation sampled from furiosa-smi while the point runs.

  python3 bench/vendor_sweep.py --engine furiosa-llm --url http://127.0.0.1:8000 \
      --model llama8b --out bench/harness_furiosa_llm.jsonl
  python3 bench/vendor_sweep.py --engine rngd-pd --url http://127.0.0.1:8100 \
      --model llama8b --out bench/harness_rngd_pd.jsonl

`--greedy N` instead sends N fixed prompts per P one at a time and records the
generated text and token IDs, for the token-equality check between engines.
bench/serve_sweep_job.sh runs either engine and this client in a one-card Job.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import threading
import time
import urllib.request

import numpy as np

FILLER = ("The three primary colors are red, blue and yellow. Mixing them yields "
          "the secondary colors orange, green and purple. Painters have relied on "
          "this relationship for centuries and printers still exploit it today. ")


class Sampler(threading.Thread):
    """Card power (W) and utilisation (%) from furiosa-smi, every `period` s.

    `npu` is the device name furiosa-smi prints (npu0 ...). "auto" takes the
    only card listed, which is what a pod granted one card sees.
    """

    def __init__(self, npu, period=0.5):
        super().__init__(daemon=True)
        self.npu, self.period, self.stop = npu, period, threading.Event()
        self.w, self.u = [], []

    @staticmethod
    def _rows(text):
        return [[c.strip() for c in line.split("|")[1:-1]]
                for line in text.splitlines() if line.startswith("|")]

    def run(self):
        while not self.stop.is_set():
            try:
                info = subprocess.run(["furiosa-smi", "info"], capture_output=True,
                                      text=True, timeout=4).stdout
                rows = [r for r in self._rows(info) if len(r) >= 5 and r[1].startswith("npu")]
                if self.npu == "auto" and len(rows) == 1:
                    self.npu = rows[0][1]
                for r in rows:
                    if r[1] == self.npu:
                        self.w.append(float(r[4].rstrip(" W")))
                status = subprocess.run(["furiosa-smi", "status"], capture_output=True,
                                        text=True, timeout=4).stdout
                for r in self._rows(status):
                    if len(r) >= 4 and r[1] == self.npu and r[3] not in ("-", ""):
                        self.u.append(float(r[3].rstrip("%").strip()))
            except Exception:
                pass
            self.stop.wait(self.period)

    def result(self):
        return (round(st.mean(self.w), 1) if self.w else None,
                round(st.mean(self.u), 1) if self.u else None)


def stream_once(url, model, prompt, max_tokens, out):
    """One streaming completion: per-token arrival times, text and token IDs."""
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, text, ids, usage, err = [], [], [], None, None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = obj.get("choices") or []
                if not ch:
                    continue
                tok = ch[0].get("token_ids") or obj.get("token_ids") or []
                piece = ch[0].get("text", "")
                # A chunk is a generated-token event if it carries a token ID or
                # text; the closing chunk with only finish_reason is not.
                n = len(tok) if tok else (1 if piece else 0)
                if n:
                    now = time.perf_counter()
                    stamps.extend([now] * n)
                    ids.extend(tok)
                    text.append(piece)
    except Exception as e:                       # noqa: BLE001 (recorded, not raised)
        err = repr(e)[:300]
    n = len(stamps)
    gen = (usage or {}).get("completion_tokens", n)
    out.append({
        "ttft": (stamps[0] - t0) * 1e3 if stamps else None,
        "tpot": (stamps[-1] - stamps[0]) * 1e3 / (n - 1) if n > 1 else None,
        "gaps": [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])],
        "n": gen, "events": n, "wall": (time.perf_counter() - t0) * 1e3,
        "text": "".join(text), "ids": ids, "err": err,
        "prompt_tokens": (usage or {}).get("prompt_tokens")})


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="llama8b", help="served model name")
    ap.add_argument("--engine", default="furiosa-llm", help="label written to every row")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--npu", default="auto", help="furiosa-smi device name, or auto")
    ap.add_argument("--contexts", default="128,512,1024,2048")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", type=int, default=0,
                    help="instead of the sweep: N fixed prompts per P, outputs recorded")
    ap.add_argument("--tag", default="", help="free text written to every row")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.tokenizer)
    filler = tk.encode(FILLER * 200, add_special_tokens=False)
    bos = [tk.bos_token_id] if tk.bos_token_id is not None else []
    rng = np.random.default_rng(a.seed)

    def prompt_ids(P, r=rng):
        """Exactly P token IDs: BOS, eight random IDs, then running text."""
        head = [int(v) for v in r.integers(1000, 100000, size=8)]
        return (bos + head + filler)[:P]

    ctxs = [int(x) for x in a.contexts.split(",")]
    meta = dict(engine=a.engine, max_tokens=a.max_tokens, tag=a.tag,
                started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if a.greedy:
        fixed = np.random.default_rng(1234)     # same prompts for every engine
        with open(a.out, "w") as fh:
            for P in ctxs:
                for i in range(a.greedy):
                    ids = prompt_ids(P, fixed)
                    res = []
                    stream_once(a.url, a.model, ids, a.max_tokens, res)
                    r = res[0]
                    row = dict(meta, P=P, idx=i, prompt_ids=ids, out_ids=r["ids"],
                               text=r["text"], err=r["err"])
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"P={P} #{i}: {r['err'] or repr(r['text'][:60])}")
        print(f"wrote -> {a.out}")
        return

    print(f"{'P':>6}{'conc':>6} | {'TTFT p50':>9} {'p90':>7} {'TPOT p50':>9} {'p90':>7} "
          f"{'tok/s':>8} | {'W':>5}{'util%':>7} err")
    print("-" * 78)
    with open(a.out, "w") as fh:
        for P in ctxs:
            res = []                              # untimed warm-up for this width
            stream_once(a.url, a.model, prompt_ids(P), a.max_tokens, res)
            if res[0]["err"]:
                print(f"{P:>6} warm-up failed: {res[0]['err']}")
            for conc in [int(x) for x in a.concurrency.split(",")]:
                smp = Sampler(a.npu)
                smp.start()
                done, toks_s = [], []
                for _ in range(a.reps):
                    res, threads = [], []
                    prompts = [prompt_ids(P) for _ in range(conc)]
                    t0 = time.perf_counter()
                    for pr in prompts:
                        th = threading.Thread(target=stream_once,
                                              args=(a.url, a.model, pr, a.max_tokens, res))
                        th.start(); threads.append(th)
                    for th in threads:
                        th.join()
                    wall = time.perf_counter() - t0
                    ok = [r for r in res if r["ttft"] is not None and not r["err"]]
                    toks_s.append(sum(r["n"] for r in ok) / wall)
                    done += res
                smp.stop.set(); smp.join(timeout=5)
                w, u = smp.result()
                ok = [r for r in done if r["ttft"] is not None and not r["err"]]
                errs = [r["err"] for r in done if r["err"] or r["ttft"] is None]
                ttft = [r["ttft"] for r in ok]
                tpot = [r["tpot"] for r in ok if r["tpot"] is not None]
                gaps = [g for r in ok for g in r["gaps"]]
                row = dict(meta, P=P, concurrency=conc, requests=len(done), errors=len(errs),
                           ttft_ms=pct(ttft, 50), ttft_p90_ms=pct(ttft, 90),
                           tpot_ms=pct(tpot, 50), tpot_p90_ms=pct(tpot, 90),
                           itl_p99_ms=pct(gaps, 99),
                           tokens_s=round(float(np.median(toks_s)), 2) if toks_s else None,
                           out_tokens=round(float(np.mean([r["n"] for r in ok])), 1) if ok else None,
                           stream_events=round(float(np.mean([r["events"] for r in ok])), 1) if ok else None,
                           prompt_tokens=ok[0]["prompt_tokens"] if ok else None,
                           power_w=w, util_pct=u, first_error=errs[0] if errs else None)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                f = lambda v, n=1: f"{v:.{n}f}" if v is not None else "-"
                print(f"{P:>6}{conc:>6} | {f(row['ttft_ms']):>9} {f(row['ttft_p90_ms']):>7} "
                      f"{f(row['tpot_ms'], 2):>9} {f(row['tpot_p90_ms'], 2):>7} "
                      f"{f(row['tokens_s']):>8} | {f(w, 0):>5}{f(u, 0):>7} {len(errs)}")
    print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
