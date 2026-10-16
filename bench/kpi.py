#!/usr/bin/env python3
"""Turn a gateway sweep into the four project KPIs.

The sweep already records what the system did — TTFT, TPOT, prefill time, and
each vendor's own power reading attributed to the cards that cell used. What it
does not record is what those numbers are worth against a KPI, because that
needs a FLOP model and a peak, and both are easy to get quietly wrong:

  * the "2N" rule counts the embedding table, which is a gather, not a matmul.
    Llama-3.1-8B is 8.03 B parameters but only 7.505 B of them are multiplied,
    so per-token compute is 15.01 GFLOP and not 16.06. Using 2N inflates every
    efficiency number here by 7%.
  * attention is quadratic and belongs only to prefill. We count the causal
    half, which is the useful work; our kv-free prefill kernel actually issues
    the full [C,C] product, so the hardware does roughly twice this. MFU is
    defined on useful FLOPs, so the kernel's waste shows up as lower MFU
    rather than being hidden in the numerator.
  * a decode step streams the whole weight set, so its MFU is bounded by
    bandwidth, not by the tensor units. Reporting one blended number for a
    prefill/decode system hides which half is the constraint, so prefill,
    decode and the system as a whole are all reported.

Peak and TDP are the vendors' own published figures. Power efficiency is given
twice: once against the power the card actually drew during the cell, which is
what we measured, and once against TDP, which is the basis the proposal's
512 GFLOPS/W target was computed on (256 TFLOPS x 30% / 150 W).

    python3 bench/kpi.py bench/sweep_llmd.jsonl --md kpi_table.md
"""
import argparse
import json

# ---------------------------------------------------------------------------
# Llama-3.1-8B-Instruct
# ---------------------------------------------------------------------------
H, M, NH, NKV, HD, L, V = 4096, 14336, 32, 8, 128, 32, 128256

# Per token, per layer, everything that is a matmul.
_QKV = 2 * H * (NH * HD + 2 * NKV * HD)
_O = 2 * H * (NH * HD)
_MLP = 2 * 2 * H * M + 2 * M * H
FLOP_LAYER = _QKV + _O + _MLP
FLOP_LM = 2 * H * V
FLOP_TOKEN = L * FLOP_LAYER + FLOP_LM          # 15.01 GFLOP


def attn_flops(ctx_tokens, new_tokens):
    """QK^T and AV for `new_tokens` positions attending over `ctx_tokens`.

    Decode is one position against the whole cache; prefill is a triangle, so
    a full pass over P positions averages P/2 of context each.
    """
    return L * 4 * (NH * HD) * ctx_tokens * new_tokens


def prefill_flops(P):
    return P * FLOP_TOKEN + attn_flops(P / 2.0, P)


def decode_flops(n_tokens, ctx):
    return n_tokens * FLOP_TOKEN + attn_flops(ctx, n_tokens)


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------
CARD = {
    # name: peak dense FLOP/s at the dtype we run, TDP W, HBM/GDDR GB/s,
    # and the weight bytes a decode step has to stream on that card.
    "rngd": {"peak": 256e12, "tdp": 180.0, "bw": 1500e9, "dtype": "BF16",
             "wbytes": 8.03e9 * 2},
    # ATOM keeps embedding and lm_head on the host, so its stream is smaller.
    "atom": {"peak": 32e12, "tdp": 90.0, "bw": 256e9, "dtype": "FP16",
             "wbytes": 6.98e9 * 2},
}


def decode_ceiling(card, B):
    """The most MFU a decode step of width B can reach on this card.

    A decode step reads every weight once and does B tokens of work with them,
    so its arithmetic intensity is fixed by the batch: B x FLOP_TOKEN over the
    weight bytes. Below the card's ridge point that puts a hard bandwidth roof
    under MFU, and no kernel can cross it. Reporting this next to the measured
    value separates "our kernel is bad" from "this is what the machine is".
    """
    c = CARD[card]
    ai = B * FLOP_TOKEN / c["wbytes"]
    return min(c["bw"] * ai, c["peak"]) / c["peak"]

# which vendor prefills and which decodes, per sweep config
ROLES = {
    "rngd_alone": ("rngd", "rngd"),
    "pd_forward": ("rngd", "atom"),
    "pd_reverse": ("atom", "rngd"),
    "atom_alone": ("atom", "atom"),
    "solo": ("rngd", "rngd"),
}


def _power(row, vendor, fallback_tdp):
    """Per-card watts for that vendor during the cell, or TDP if unsampled.

    The sweep's sampler pools every sample from every card a cell used and
    takes the mean, so this is one card's draw, not the pair's. Multiplying by
    the number of cards of that vendor is the caller's job -- forgetting to is
    a clean doubling of any same-vendor configuration's GFLOPS/W.

    It is a mean over the sampling window, which spans all three timed reps and
    the gaps between them, while the reported timing is the fastest rep. Idle
    gaps therefore pull it down, so treat it as a floor on active power.
    """
    w = row.get(f"{vendor}_power_w")
    return (w, "measured") if w else (fallback_tdp, "TDP")


def kpi_row(r):
    cfg = r["config"]
    pv, dv = ROLES[cfg]
    P, D, B = r["P"], r["D"], r["B"]

    # ---- prefill: B prompts, each P tokens, on one card.
    # The span the card was busy, not the median request's own window: with
    # B > 1 the device lock serialises the passes, so dividing B*P tokens by a
    # median would credit the card with a rate it never held. Older sweeps have
    # only the median, and are marked as such rather than silently mixed in.
    pf_flops = B * prefill_flops(P)
    pf_span = r.get("prefill_span_ms")
    if cfg == "solo":
        # The two workers report prefill differently and the difference is a
        # factor of B. A disaggregated prefill pod times each request's own
        # pass; the solo worker prefills the whole batch inside one device
        # lock and divides by the batch size before reporting. Treating the
        # solo figure as a per-request window credits one card with B times
        # the throughput it had -- it read as 169% of peak, which is how the
        # error announced itself. Multiplying back recovers the batch's span.
        pf_s = (pf_span if pf_span else r["prefill_ms"]) * B / 1e3
    else:
        pf_s = (pf_span if pf_span else r["prefill_ms"]) / 1e3
    pf_rate = pf_flops / pf_s if pf_s > 0 else 0.0
    pf_mfu = pf_rate / CARD[pv]["peak"]

    # ---- decode: B streams x D tokens. tpot_ms is the step interval, so the
    # card spends D steps regardless of width; the batch is what makes those
    # steps carry B tokens each.
    ctx = P + D / 2.0
    dec_flops = B * decode_flops(D, ctx)
    dec_s = D * r["tpot_ms"] / 1e3
    dec_rate = dec_flops / dec_s if dec_s > 0 else 0.0
    dec_mfu = dec_rate / CARD[dv]["peak"]
    dec_ceil = decode_ceiling(dv, B)

    # ---- system: both cards are held for the whole request either way
    tot_flops = pf_flops + dec_flops
    wall_s = r["wall_ms"] / 1e3
    # Every card the configuration holds for the request, not one of each: a
    # deployment with two prefill cards behind one decode card occupies three,
    # and counting two would credit it with an efficiency it does not have.
    # This is the whole point of the ratio study -- a second prefill card only
    # pays for itself if prefill is actually the bottleneck, and if it is not,
    # it is idle silicon sitting in the denominator.
    n_pre = int(r.get("prefill_n", 1))
    cards = [pv] if cfg == "solo" else [pv] * n_pre + [dv]
    peak_sum = sum(CARD[c]["peak"] for c in cards)
    sys_mfu = (tot_flops / wall_s) / peak_sum if wall_s > 0 else 0.0

    # ---- power: sum what each card in the pair drew
    w_meas, w_tdp, how = 0.0, 0.0, []
    for c in dict.fromkeys(cards):
        w, src = _power(r, c, CARD[c]["tdp"])
        n = cards.count(c)
        w_meas += w * n                       # w is one card's mean draw
        w_tdp += CARD[c]["tdp"] * n
        how.append(src)
    src = "measured" if all(h == "measured" for h in how) else "mixed"

    gfw_meas = (tot_flops / wall_s) / w_meas / 1e9 if wall_s > 0 and w_meas else None
    gfw_tdp = (tot_flops / wall_s) / w_tdp / 1e9 if wall_s > 0 else None

    return {
        "config": cfg, "P": P, "D": D, "B": B, "cards": len(cards),
        "prefill_basis": "span" if pf_span else "median(추정 낙관)",
        "prefill_card": pv, "decode_card": dv,
        "ttft_ms": r.get("ttft_ms"), "tpot_ms": r.get("tpot_ms"),
        "tokens_s": r.get("tokens_s"),
        "prefill_tflops": round(pf_rate / 1e12, 2),
        "prefill_mfu_pct": round(pf_mfu * 100, 2),
        "decode_tflops": round(dec_rate / 1e12, 3),
        "decode_mfu_pct": round(dec_mfu * 100, 3),
        "decode_mfu_roof_pct": round(dec_ceil * 100, 3),
        "decode_of_roof_pct": round(dec_mfu / dec_ceil * 100, 1) if dec_ceil else None,
        "system_mfu_pct": round(sys_mfu * 100, 3),
        "power_w": round(w_meas, 1), "power_src": src,
        "power_w_tdp": round(w_tdp, 1),
        "gflops_per_w": round(gfw_meas, 1) if gfw_meas else None,
        "gflops_per_w_tdp": round(gfw_tdp, 1) if gfw_tdp else None,
        "errors": r.get("errors", 0),
        "host_load1": r.get("host_load1"),
    }


HDRS = [("config", "구성"), ("P", "P"), ("D", "D"), ("B", "B"),
        ("ttft_ms", "TTFT ms"), ("tpot_ms", "TPOT ms"),
        ("prefill_tflops", "프리필 TFLOPS"), ("prefill_mfu_pct", "프리필 MFU %"),
        ("cards", "카드"), ("decode_mfu_pct", "디코드 MFU %"), ("decode_mfu_roof_pct", "디코드 MFU 천장 %"),
        ("decode_of_roof_pct", "천장 대비 %"), ("system_mfu_pct", "시스템 MFU %"),
        ("power_w", "전력 W"), ("gflops_per_w", "GFLOPS/W"),
        ("gflops_per_w_tdp", "GFLOPS/W (TDP)")]


def to_md(rows):
    out = ["| " + " | ".join(h for _, h in HDRS) + " |",
           "|" + "|".join("---" for _ in HDRS) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            "" if r.get(k) is None else str(r[k]) for k, _ in HDRS) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default="", help="write KPI rows as JSONL")
    ap.add_argument("--md", default="", help="write a markdown table")
    a = ap.parse_args()

    rows = []
    with open(a.jsonl) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(kpi_row(json.loads(line)))

    print(f"FLOP/token (matmul only) = {FLOP_TOKEN/1e9:.2f} GFLOP"
          f"   [2N rule would say {2*8.03e9/1e9:.2f}]")
    print(to_md(rows))

    if a.out:
        with open(a.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n-> {a.out}")
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(to_md(rows) + "\n")
        print(f"-> {a.md}")


if __name__ == "__main__":
    main()
