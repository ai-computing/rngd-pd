"""Verify the integrated refined planner in both directions on the 8B fixture.

Requires the checkpoint_delta fixture and fresh-load reference from the recorded
bench_rngd run. All path arguments are site inputs; they are omitted from output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from rngd_pd.swap.bench_rngd import prompts
from rngd_pd.swap import delta, rngd
from rngd_pd.swap.variants import Variant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "new", "ranges", "reference", "cache", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    reference = json.loads(Path(a.reference).read_text())
    reference = reference.get("result_rngd_ranges.json", reference)
    v = Variant(4, 256, 8)
    expected = {"base": reference["fresh"][v.name]["tokens"],
                "new": reference["fresh_new"]["tokens"]}
    ranges = delta.load_onggi_ranges(a.ranges)
    out = dict(variant=v.name, prompt_tokens=8, new_tokens=8, refine=True, rows=[],
               source_sha256={Path(m.__file__).name: hashlib.sha256(
                   Path(m.__file__).read_bytes()).hexdigest() for m in (delta, rngd)})
    engine = rngd.SwappableDecode(a.base, group=8, chip=a.chip, cache_dir=a.cache)
    try:
        engine.swap(v)
        for round_index in range(a.rounds):
            for target, checkpoint in (("new", a.new), ("base", a.base)):
                start = time.perf_counter()
                plan = delta.plan_update(engine.ck.path, checkpoint, ranges, refine=True)
                planned = time.perf_counter()
                update = engine.apply_delta(checkpoint, plan)
                applied = time.perf_counter()
                tokens, ttft = engine.generate(prompts(v.B, 8), 8)
                out["rows"].append(dict(round=round_index, target=target,
                    plan_s=planned-start, apply_call_s=applied-planned,
                    plan=plan.summary(), apply=update,
                    request_to_first_token_s=applied-start+ttft, prompt_ttft_s=ttft,
                    tokens=tokens, tokens_equal_fresh=tokens == expected[target]))
    finally:
        try:
            engine.close()
        finally:
            Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not all(row["tokens_equal_fresh"] for row in out["rows"]):
        raise SystemExit("refined delta parity failed; results were saved")


if __name__ == "__main__":
    main()
