#!/usr/bin/env python3
"""What RNGD actually does, using Furiosa's own serving engine.

The four-way sweep measured our hand-built eager engine on the RNGD side: one
token per graph pass, no batching, because that is what our compile path allows.
Those are limits of our prototype, not of the card — furiosa-llm's sealed engine
ships buckets for batch 1…32 and attention up to 4096, so it does chunked
prefill and continuous batching. Comparing our ATOM implementation against our
RNGD prototype flatters neither vendor honestly; this measures the production
engine so the RNGD column means something.

Sweeps prompt length against concurrency, reading TTFT and inter-token latency
off the streaming API and sampling card power/utilisation alongside.

  python3 bench/vendor_sweep.py --url http://127.0.0.1:8020 --npu 1
"""
import argparse
import json
import statistics as st
import subprocess
import threading
import time
import urllib.request

import numpy as np


class Sampler(threading.Thread):
    def __init__(self, npu, period=0.5):
        super().__init__(daemon=True)
        self.npu, self.period, self.stop = npu, period, threading.Event()
        self.w, self.u = [], []

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["furiosa-smi", "info"], capture_output=True,
                                     text=True, timeout=4).stdout
                for line in out.splitlines():
                    if f"npu{self.npu}" in line:
                        self.w.append(float(line.split("|")[5].strip().rstrip(" W")))
                out = subprocess.run(["furiosa-smi", "status"], capture_output=True,
                                     text=True, timeout=4).stdout
                blocks, cur = [], []
                for line in out.splitlines():
                    if line.startswith("+"):
                        if cur:
                            blocks.append(cur)
                        cur = []
                    else:
                        cur.append(line)
                if cur:
                    blocks.append(cur)
                for b in blocks:
                    if f"npu{self.npu}" not in "\n".join(b):
                        continue
                    vals = []
                    for line in b:
                        if "Core" in line:
                            v = line.split(":")[-1].replace("|", "").replace("%", "").strip()
                            if v not in ("-", ""):
                                vals.append(float(v))
                    if vals:
                        self.u.append(sum(vals) / len(vals))
                    break
            except Exception:
                pass
            time.sleep(self.period)

    def result(self):
        return (round(st.mean(self.w), 1) if self.w else None,
                round(max(self.u), 1) if self.u else None)


def stream_once(url, model, prompt, max_tokens, out):
    """One streaming completion; records first-token latency and gaps."""
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, prev, gaps, n = None, None, [], 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            txt = (obj.get("choices") or [{}])[0].get("text", "")
            if not txt:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - t0) * 1e3
            else:
                gaps.append((now - prev) * 1e3)
            prev = now
            n += 1
    out.append({"ttft": ttft, "tpot": st.median(gaps) if gaps else 0.0,
                "n": n, "wall": (time.perf_counter() - t0) * 1e3})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--npu", type=int, default=1)
    ap.add_argument("--contexts", default="128,512,1024,2048")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="/home/user/workspace/01_pd_disag/rngd_llmd/bench/sweep_vendor.jsonl")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(
        "/home/user/workspace/01_pd_disag/furiosa-bench/artifacts/llama8b-pfbatch")
    filler = ("The three primary colors are red, blue and yellow. Mixing them yields "
              "the secondary colors orange, green and purple. Painters have relied on "
              "this relationship for centuries and printers still exploit it today. ")
    ids = tk.encode(filler * 60, add_special_tokens=False)
    rng = np.random.default_rng(0)

    def unique_prompt(P):
        """A prompt no other request shares.

        A served engine keeps a prefix cache, so re-sending the same text turns
        prefill into a lookup and TTFT stops measuring prefill at all. Leading
        the prompt with fresh random tokens makes every request miss."""
        head = " ".join(str(int(v)) for v in rng.integers(10**6, 10**7, size=8))
        want = P - len(tk.encode(head, add_special_tokens=False))
        return head + " " + tk.decode(ids[:max(want, 1)])

    print(f"{'P':>6}{'conc':>6} | {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>8} | "
          f"{'W':>6}{'util%':>7}")
    print("-" * 60)
    with open(a.out, "w") as fh:
        for P in [int(x) for x in a.contexts.split(",")]:
            for conc in [int(x) for x in a.concurrency.split(",")]:
                smp = Sampler(a.npu)
                smp.start()
                ttfts, tpots, toks_s = [], [], []
                for _ in range(a.reps):
                    res, threads = [], []
                    t0 = time.perf_counter()
                    for _ in range(conc):
                        th = threading.Thread(target=stream_once,
                                              args=(a.url, a.model, unique_prompt(P),
                                                    a.max_tokens, res))
                        th.start(); threads.append(th)
                    for th in threads:
                        th.join()
                    wall = time.perf_counter() - t0
                    ok = [r for r in res if r["ttft"] is not None]
                    if not ok:
                        continue
                    ttfts.append(st.median([r["ttft"] for r in ok]))
                    tpots.append(st.median([r["tpot"] for r in ok]))
                    toks_s.append(sum(r["n"] for r in ok) / wall)
                smp.stop.set(); smp.join(timeout=3)
                w, u = smp.result()
                if not ttfts:
                    print(f"{P:>6}{conc:>6} | request failed")
                    continue
                row = dict(engine="furiosa-llm", P=P, concurrency=conc,
                           ttft_ms=round(float(np.median(ttfts)), 1),
                           tpot_ms=round(float(np.median(tpots)), 2),
                           tokens_s=round(float(np.median(toks_s)), 2),
                           power_w=w, util_pct=u)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"{P:>6}{conc:>6} | {row['ttft_ms']:>9.0f} {row['tpot_ms']:>9.2f} "
                      f"{row['tokens_s']:>8.2f} | {w or 0:>6.0f}{u or 0:>7.0f}")
    print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
