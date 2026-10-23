"""Full bundle reload vs shared-session/warm-pool ATOM swap, with parity.

python -m rngd_pd.swap.bench_atom_bundles --bundle <co-compiled directory> \
  --variants B1,S512 B4,S512 B8,S512 --out atom.json

Each fresh baseline uses the SAME bundle as its swap comparison. Load/prepare
time is included for unprepared bundles, and is recorded separately for a
background-prepared second bundle. Never call these graph-only timings TTFT.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .atom import AtomBundle, SwappableAtom
from .variants import Variant


def _prompts(batch):
    # Stable token IDs below the Llama vocabulary; identical across runs.
    return [[128000, 791, 6864, 315, 9822, 374] for _ in range(batch)]


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--warm-bundle", help="independently compiled bundle of the SAME shapes")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    vs = tuple(Variant.parse(t) for t in a.variants)
    if a.tokens < 1 or a.rounds < 1:
        ap.error("tokens and rounds must be positive")
    bundles = {"main": AtomBundle(a.bundle, vs)}
    if a.warm_bundle:
        bundles["warm"] = AtomBundle(a.warm_bundle, vs)
    if a.fresh:
        v = Variant.parse(a.fresh)
        with SwappableAtom(bundles, device=a.device) as e:
            start = time.perf_counter()
            info = e.swap(v, "main")
            tokens, ttft = e.generate(_prompts(v.B), a.tokens, timed=True)
            info.update(tokens=tokens, prompt_ttft_s=ttft,
                        request_to_first_token_s=info["swap_s"] + ttft,
                        request_seconds=time.perf_counter() - start)
            _write(a.out, info)
        return

    result = dict(fresh={}, swaps=[])
    for v in vs:
        output = str(a.out) + f".{v.name}.fresh.json"
        subprocess.run([sys.executable, "-m", "rngd_pd.swap.bench_atom_bundles",
                        "--bundle", a.bundle, "--variants", *a.variants,
                        "--device", str(a.device), "--tokens", str(a.tokens),
                        "--fresh", v.name, "--out", output], check=True)
        result["fresh"][v.name] = json.loads(Path(output).read_text())
    if a.warm_bundle:
        output = str(a.out) + ".warm.fresh.json"
        subprocess.run([sys.executable, "-m", "rngd_pd.swap.bench_atom_bundles",
                        "--bundle", a.warm_bundle, "--variants", *a.variants,
                        "--device", str(a.device), "--tokens", str(a.tokens),
                        "--fresh", vs[0].name, "--out", output], check=True)
        result["fresh_warm"] = json.loads(Path(output).read_text())
    with SwappableAtom(bundles, device=a.device) as e:
        result["initial_load"] = e.swap(vs[0], "main")
        for rnd in range(a.rounds):
            for v in vs:
                info = e.swap(v, "main", require_ready=True)
                tokens, ttft = e.generate(_prompts(v.B), a.tokens, timed=True)
                info.update(round=rnd, prompt_ttft_s=ttft, tokens=tokens,
                            request_to_first_token_s=info["swap_s"] + ttft,
                            tokens_equal_fresh=tokens == result["fresh"][v.name]["tokens"])
                result["swaps"].append(info)
        if a.warm_bundle:
            future = e.prepare_async(vs[0], "warm")
            # Prove the current bundle remains runnable while preparation runs.
            pending = not future.done()
            active = e.active
            t_serving = time.perf_counter()
            during_tokens, during_ttft = e.generate(_prompts(active.B), a.tokens, timed=True)
            serving_s = time.perf_counter() - t_serving
            prepared = future.result()
            info = e.swap(vs[0], "warm", require_ready=True)
            tokens, ttft = e.generate(_prompts(vs[0].B), a.tokens, timed=True)
            result["warm"] = dict(info, tokens=tokens, prompt_ttft_s=ttft,
                                  background_prepare_s=prepared["prepare_s"],
                                  prepare_pending_before_serving=pending,
                                  serving_during_prepare_s=serving_s,
                                  serving_during_prepare_ttft_s=during_ttft,
                                  token_before_prepare_completed=(
                                      t_serving + during_ttft < prepared["prepared_at_s"]),
                                  during_prepare_tokens_equal_fresh=(
                                      during_tokens == result["fresh"][active.name]["tokens"]),
                                  tokens_equal_fresh=tokens == result["fresh_warm"]["tokens"],
                                  request_to_first_token_s=info["swap_s"] + ttft)
        result["allocations"] = e.allocations()
    _write(a.out, result)
    checks = [r["tokens_equal_fresh"] for r in result["swaps"]]
    if "warm" in result:
        checks.extend([result["warm"]["tokens_equal_fresh"],
                       result["warm"]["during_prepare_tokens_equal_fresh"]])
    if not all(checks):
        raise SystemExit("token parity failed; see saved results")


if __name__ == "__main__":
    main()
