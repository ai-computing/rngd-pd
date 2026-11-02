"""Real onggi -> checkpoint planner -> resident NPU delta -> fresh token parity.

Run in an allocated device environment. Each fresh reference and the resident
engine use separate processes. ATOM compiles once per process, because rebel
0.11 cannot reliably reuse another process's device-input compilation cache.
No serving/native artifacts are modified. Outputs are machine-readable JSON.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def engine(a, checkpoint):
    if a.backend == "rngd":
        from rngd_pd.swap.rngd import SwappableDecode
        return SwappableDecode(checkpoint, chip=a.device, cache_dir=a.cache)
    if a.backend == "atom-constants":
        from rngd_pd.swap.atom_constants import ConstantWeightDecode
        return ConstantWeightDecode(checkpoint, a.cache, device=a.device)
    from rngd_pd.swap.atom_inputs import DeviceWeightDecode
    cache = str(Path(a.cache) / f"process-{os.getpid()}")
    return DeviceWeightDecode(checkpoint, cache, device=a.device)


def generated(e, a):
    if a.prompt:
        p = [a.prompt] * a.batch
    else:
        from rngd_pd.swap.bench_rngd import prompts
        p = prompts(a.batch, a.prompt_length)
    tokens, ttft = e.generate(p, a.tokens)
    return ttft, tokens


def graph_count():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


def worker(a):
    from rngd_pd.swap.variants import Variant
    started = time.perf_counter()
    e = engine(a, a.base if a.mode == "resident" else a.checkpoint)
    try:
        e.swap(Variant(a.batch, a.sequence, 8 if a.backend == "rngd" else 0))
        ready = time.perf_counter()
        ttft, tokens = generated(e, a)
        initial = dict(ttft_s=ttft, engine_load_s=ready - started,
                       load_through_first_token_s=ready - started + ttft,
                       tokens=tokens, unique_graphs=graph_count(),
                       median_decode_step_s=e.last_step_s)
        if a.mode == "fresh":
            return initial
        with open(a.reference) as f:
            references = json.load(f)
        rows = []
        for rnd in range(a.rounds):
            for ref in ("new", "base"):
                target = a.new if ref == "new" else a.base
                old_ref = "base" if ref == "new" else "new"
                before = graph_count()
                previous = list(e.W) if a.backend == "atom" else None
                native_before = tuple(e.runtimes) if a.backend == "atom-constants" else None
                t0 = time.perf_counter()
                result = e.load_delta(target, store=a.store, base_ref=old_ref,
                                      new_ref=ref, onggi=a.onggi)
                ttft, tokens = generated(e, a)
                # generate() returns after all tokens; add its measured first-token
                # interval to delta time, excluding subsequent generated tokens.
                result.update(round=rnd, target=ref, prompt_ttft_s=ttft, tokens=tokens,
                              request_to_first_token_s=result["total_s"] + ttft,
                              request_through_generation_s=time.perf_counter() - t0,
                              tokens_equal_fresh=tokens == references[ref]["tokens"],
                              median_decode_step_s=e.last_step_s,
                              unique_graphs_before=before, unique_graphs_after=graph_count())
                if native_before is not None:
                    result.update(constant_runtimes_retained=(len(native_before) == len(e.runtimes)
                        and all(a is b for a, b in zip(native_before, e.runtimes))),
                        compiled_programs_before=len(native_before), compiled_programs_after=len(e.runtimes))
                if previous is not None:
                    from rngd_pd.swap.atom_inputs import weight_names
                    changed = set(result["plan"]["names"])
                    result["unchanged_allocations_retained"] = all(
                        e.W[i] is t for i, (n, t) in enumerate(zip(weight_names(e.d["nl"]), previous))
                        if n not in changed)
                    del previous
                rows.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ("tokens", "plan")}),
                      flush=True)
        return dict(initial=initial, rows=rows,
                    destination_differs_base=references["new"]["tokens"] != references["base"]["tokens"])
    finally:
        e.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["rngd", "atom", "atom-constants"])
    ap.add_argument("--mode", default="all", choices=["all", "fresh", "resident"])
    for key in ("base", "new", "store", "cache", "out"):
        ap.add_argument("--" + key, required=True)
    ap.add_argument("--onggi", default="onggi")
    ap.add_argument("--checkpoint")
    ap.add_argument("--reference")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sequence", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--prompt", nargs="+", type=int)
    ap.add_argument("--prompt-length", type=int, default=8, help="seed-0 random IDs if --prompt is absent")
    a = ap.parse_args()
    if a.mode != "all":
        result = worker(a)
    else:
        count = 0
        def child(mode, extra):
            nonlocal count
            count += 1
            out = a.out + "." + mode + str(count) + ".json"
            cmd = [sys.executable, __file__, *sys.argv[1:], "--mode", mode, "--out", out, *extra]
            subprocess.run(cmd, check=True)
            return json.loads(Path(out).read_text())
        fresh = {n: child("fresh", ["--checkpoint", p]) for n, p in (("base", a.base), ("new", a.new))}
        reference = a.out + ".reference.json"
        Path(reference).write_text(json.dumps(fresh))
        resident = child("resident", ["--reference", reference])
        result = dict(backend=a.backend, fresh=fresh, resident=resident)
    Path(a.out).write_text(json.dumps(result, indent=2))
    resident = result.get("resident", result) if a.mode != "fresh" else None
    if resident is not None:
        assert all(r["tokens_equal_fresh"] for r in resident["rows"]), "delta token parity failed"
        assert all(r["unique_graphs_before"] == r["unique_graphs_after"]
                   for r in resident["rows"]), "delta triggered recompilation"
        if a.backend == "atom":
            assert resident["rows"] and all(r["unique_graphs_before"] > 0
                                               for r in resident["rows"]), \
                "no compiled ATOM graph; CPU fallback is not NPU evidence"
            assert all(r["unchanged_allocations_retained"] for r in resident["rows"]), \
                "delta replaced unchanged weight allocations"
        if a.backend == "atom-constants":
            assert resident["rows"] and all(r["constant_runtimes_retained"]
                and r["compiled_programs_before"] > 0
                and r["compiled_programs_before"] == r["compiled_programs_after"]
                for r in resident["rows"]), "delta replaced or lacked a native runtime"
        assert resident["destination_differs_base"], "fixture did not change generated tokens"


if __name__ == "__main__":
    main()
