#!/usr/bin/env python3
"""Two RNGD cards, each running the whole request — the baseline PD must beat.

Duplication is the obvious way to spend a second card: run another copy of the
engine and split traffic. Any disaggregated split has to earn its keep against
this, and with the vendor engine on both cards it is a much higher bar than the
one our prototype set.

Traffic is round-robined across the two servers, so a card only sees half the
requests; aggregate throughput and per-request latency are reported together
because a split that raises one while wrecking the other is not an improvement.

  python3 bench/vendor_dup.py --urls http://127.0.0.1:8020,http://127.0.0.1:8021
"""
import argparse
import json
import statistics as st
import threading
import time

import numpy as np

from vendor_sweep import Sampler, stream_once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="http://127.0.0.1:8020,http://127.0.0.1:8021")
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--npus", default="1,2")
    ap.add_argument("--contexts", default="512,2048")
    ap.add_argument("--concurrency", default="8,32,64")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="/home/user/workspace/01_pd_disag/rngd_llmd/bench/sweep_vendor_dup.jsonl")
    a = ap.parse_args()

    urls = a.urls.split(",")
    npus = [int(x) for x in a.npus.split(",")]
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(
        "/home/user/workspace/01_pd_disag/furiosa-bench/artifacts/llama8b-pfbatch")
    filler = ("The three primary colors are red, blue and yellow. Mixing them yields "
              "the secondary colors orange, green and purple. Painters have relied on "
              "this relationship for centuries and printers still exploit it today. ")
    ids = tk.encode(filler * 90, add_special_tokens=False)
    rng = np.random.default_rng(1)

    def unique_prompt(P):
        head = " ".join(str(int(v)) for v in rng.integers(10**6, 10**7, size=8))
        want = P - len(tk.encode(head, add_special_tokens=False))
        return head + " " + tk.decode(ids[:max(want, 1)])

    print(f"{'cards':>6}{'P':>6}{'conc':>6} | {'TTFT ms':>9} {'TPOT ms':>9} {'tok/s':>8} | {'W total':>8}")
    print("-" * 62)
    with open(a.out, "w") as fh:
        for P in [int(x) for x in a.contexts.split(",")]:
            for conc in [int(x) for x in a.concurrency.split(",")]:
                for ncards in (1, 2):
                    use = urls[:ncards]
                    smps = [Sampler(n) for n in npus[:ncards]]
                    for s in smps:
                        s.start()
                    ttfts, tpots, toks_s = [], [], []
                    for _ in range(a.reps):
                        res, threads = [], []
                        t0 = time.perf_counter()
                        for i in range(conc):
                            th = threading.Thread(
                                target=stream_once,
                                args=(use[i % ncards], a.model, unique_prompt(P),
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
                    ws = []
                    for s in smps:
                        s.stop.set(); s.join(timeout=3)
                        w, _u = s.result()
                        ws.append(w or 0)
                    if not ttfts:
                        print(f"{ncards:>6}{P:>6}{conc:>6} | failed")
                        continue
                    row = dict(engine="furiosa-llm", cards=ncards, P=P, concurrency=conc,
                               ttft_ms=round(float(np.median(ttfts)), 1),
                               tpot_ms=round(float(np.median(tpots)), 2),
                               tokens_s=round(float(np.median(toks_s)), 2),
                               power_w=round(sum(ws), 1))
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"{ncards:>6}{P:>6}{conc:>6} | {row['ttft_ms']:>9.0f} "
                          f"{row['tpot_ms']:>9.2f} {row['tokens_s']:>8.2f} | {row['power_w']:>8.0f}")
    print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
