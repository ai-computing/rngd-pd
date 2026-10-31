#!/usr/bin/env python3
r"""CCGrid E7: one raw RNGD resident-update or worker-restart sample.

PYTHONPATH=src python bench/ccgrid_e7.py sample --ckpt BASE --target TARGET \
    --ranges onggi.json --variant B4-S256-g2 --cache CACHE --chip 0 \
    --reference fresh-target.json --host-workers 2 --intra-op 4 \
    --mode inplace --out new-sample.json

Reference JSON requires {"tokens": [[int, ...], ...]} from an independently
fresh-loaded TARGET, one nonempty equal-length continuation per batch lane.
Optional "prompts" supplies B equal-length nonempty token-ID lists; otherwise
use bench_rngd.prompts(B, 8), NumPy seed 0 (the original producer's prompts).
Optional "variant" must match. Optional "base_tokens" contains fresh-base
continuations with the same shape; base warmup parity is gated when supplied
and recorded as null (unknown) otherwise. Length is inferred from tokens.
The complete prompts, tokens, reference hash and parity result are saved.

replace maps to apply_delta(mode="group"): each touched resident stacked
group tensor is replaced, including its unchanged neighboring layers.
inplace uses production row coalescing/density fallback, without alteration.
Both start after a synchronized base load, variant load and base generation;
planning occurs before loading the base and is reported separately.

restart is a coordinator with no engine/card: base worker loads and generates,
records base_ready_perf_ns, closes and exits; only after wait() returns does
the coordinator spawn a target worker. update_to_first_token_s includes base
close/exit, coordinator overhead, target startup/import/load and first token.
The clock is perf_counter_ns, shared across processes on supported Python
platforms. No allocator retry or wait subtraction is performed. Restart's
parent_launch_to_first_token_s is also reported separately. An outer Job must
provide exclusive card ownership and run samples serially. Missing caches
fail: allow_compile=False always. No compilation or cache construction here.
An additional from_module guard forbids compilation. Original compiled-call
outputs must use fsdk.DEV and the allocated chip index; recursive device checks
remain inside measured intervals. Native counters and partial phase results
survive failures, with cleanup errors recorded separately from the first error.

logical_device_written_bytes counts successful logical weight payloads, NOT
physical DMA/counters: inplace's strided staging can add device copies. For
restart it counts shared/LM weights and per-batch weights/constants; excludes
programs, position tables, KV and activations. Keep the raw apply.uploaded_bytes.
No aggregate statistics are computed; repeat >=5 times in the outer harness.
Outputs are exclusive-create; mismatch/error returns 1 and retains raw data.
Checkpoint manifests hash JSON metadata and record shard sizes/mtimes; full
checkpoint shard hashes must come from the owner's external input manifest.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
from importlib import metadata as packages
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


SCRIPT_ENTRY_PERF_NS = time.perf_counter_ns()


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(argv):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", "-c", "safe.directory=" + str(root),
                                "-C", str(root), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for name in ("numpy", "torch", "safetensors", "furiosa-torch", "furiosa-llm", "furiosa-runtime"):
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = None
    sources = [Path(__file__).resolve(), *[root / "src/rngd_pd/swap" / (name + ".py")
               for name in ("rngd", "delta", "variants", "bench_rngd")]]
    # Deliberate allowlist: never dump credentials or the full environment.
    env_keys = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "RNGD_MAX_SHAPES", "FURIOSA_DEVICES", "FURIOSA_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES", "KUBERNETES_SERVICE_HOST", "HOSTNAME", "POD_NAME")
    return dict(utc=utc(), pid=os.getpid(), ppid=os.getppid(), executable=sys.executable,
        invocation_argv=[sys.executable, str(Path(__file__).resolve()), *argv],
        actual_sys_argv=list(sys.argv), cwd=os.getcwd(), python=sys.version,
        platform=platform.platform(), hostname=platform.node(), packages=versions,
        environment={k: os.environ[k] for k in env_keys if k in os.environ},
        script_entry_perf_ns=SCRIPT_ENTRY_PERF_NS,
        source=dict(commit=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain"),
                    script_sha256=sha256(__file__),
                    sha256={str(p.relative_to(root)): sha256(p) for p in sources}))


def load_reference(path, variant):
    from rngd_pd.swap.variants import Variant
    v = Variant.parse(variant)
    doc = json.loads(Path(path).read_text())
    tokens = doc.get("tokens")
    def valid(rows):
        return (isinstance(rows, list) and len(rows) == v.B and all(
            isinstance(r, list) and r and all(type(t) is int and t >= 0 for t in r)
            for r in rows) and len({len(r) for r in rows}) == 1)
    if not valid(tokens):
        raise ValueError("reference tokens must contain B nonempty equal-length integer lists")
    if "base_tokens" in doc and (not valid(doc["base_tokens"])
            or len(doc["base_tokens"][0]) != len(tokens[0])):
        raise ValueError("reference base_tokens must have the same shape as tokens")
    if "variant" in doc and Variant.parse(doc["variant"]) != v:
        raise ValueError("reference variant differs from requested variant")
    prompts = doc.get("prompts")
    if prompts is None:
        from rngd_pd.swap.bench_rngd import prompts as original_prompts
        prompts = original_prompts(v.B, 8)
    if not valid(prompts) or len(prompts[0]) + len(tokens[0]) > v.S:
        raise ValueError("reference prompts or continuation do not fit requested variant")
    return v, prompts, tokens


def prepare_plan(a):
    from rngd_pd.swap import delta
    started = time.perf_counter_ns()
    ranges = delta.load_onggi_ranges(a.ranges)
    plan = delta.plan_update(a.ckpt, a.target, ranges, refine=True)
    return plan, (time.perf_counter_ns() - started) / 1e9


def full_weight_payload(engine):
    tensors = [t for group in engine.shared for t in group.values()]
    if engine.wlm_d is not None:
        tensors.append(engine.wlm_d)
    for st in engine.variants.values():
        pb = st["pb"]
        tensors.extend(t for key in ("n1", "n2") for t in pb[key])
        tensors.extend(pb[key] for key in ("norm", "eps", "nh"))
    return sum(t.numel() * t.element_size() for t in tensors)


def check_native_output(output, device_type, chip):
    import torch
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
    actual = list(tensors(output))
    if not actual or any(t.device.type != device_type or t.device.index != chip for t in actual):
        raise RuntimeError("original compiled outputs must be on %s:%d" % (device_type, chip))


@contextmanager
def observe_native(chip, row):
    from rngd_pd.engines.fsdk import CompileModule, DEV, SDK
    counts = row["native_counts"] = dict(compilation_attempts=0, native_calls=0, native_output_checks=0)
    row.update(sdk=SDK, expected_native_device=dict(type=DEV, index=chip),
               native_outputs_verified=False,
               native_observation="original compiled calls and device type/index checks; overhead included")
    original_compile = inspect.getattr_static(CompileModule, "from_module")
    original_call = CompileModule.__call__
    call_is_own = "__call__" in CompileModule.__dict__
    def no_compile(*args, **kwargs):
        counts["compilation_attempts"] += 1
        raise RuntimeError("compilation forbidden in E7")
    def call(module, *args, **kwargs):
        counts["native_calls"] += 1
        output = original_call(module, *args, **kwargs)
        check_native_output(output, DEV, chip)
        counts["native_output_checks"] += 1
        return output
    CompileModule.from_module = staticmethod(no_compile)
    CompileModule.__call__ = call
    try:
        yield
        if (not counts["native_calls"] or counts["native_calls"] != counts["native_output_checks"]
                or counts["compilation_attempts"]):
            raise RuntimeError("native graph execution verification failed")
        row["native_outputs_verified"] = True
    finally:
        CompileModule.from_module = original_compile
        if call_is_own:
            CompileModule.__call__ = original_call
        else:
            del CompileModule.__call__


def native_sample(a, row, phase="resident", plan=None):
    import torch
    from rngd_pd.swap.rngd import SwappableDecode
    v, prompts, expected = load_reference(a.reference, a.variant)
    base_expected = json.loads(Path(a.reference).read_text()).get("base_tokens")
    torch.set_num_threads(a.intra_op)
    timings = row.setdefault("timings", {})
    row.update(phase=phase, variant=v.name, prompts=prompts)
    engine, primary_error = None, None
    with observe_native(a.chip, row):
        try:
            load_started = time.perf_counter_ns()
            checkpoint = a.target if phase == "target" else a.ckpt
            engine = SwappableDecode(checkpoint, group=v.group, chip=a.chip,
                cache_dir=a.cache, allow_compile=False, host_workers=a.host_workers)
            weights_ready = time.perf_counter_ns()
            timings["engine_load_s"] = (weights_ready - load_started) / 1e9
            row["weights"] = dict(engine.timings)
            row["graph"] = engine.swap(v)
            engine._sync()
            timings["variant_load_s"] = (time.perf_counter_ns() - weights_ready) / 1e9
            if phase != "target":
                base_started = time.perf_counter_ns()
                base_tokens, base_ttft = engine.generate(prompts, len(expected[0]))
                row.update(base_tokens=base_tokens,
                    base_tokens_equal_reference=None if base_expected is None else base_tokens == base_expected)
                timings["base_prompt_ttft_s"] = base_ttft
                engine._sync()
                ready = time.perf_counter_ns()
                row["base_ready_perf_ns"] = ready
                timings["base_generation_s"] = (ready - base_started) / 1e9
                if phase == "base":
                    return
                applied = engine.apply_delta(a.target, plan,
                    mode="group" if a.mode == "replace" else "inplace")
                row.update(apply=applied, logical_device_written_bytes=applied["uploaded_bytes"])
                timings["apply_return_s"] = (time.perf_counter_ns() - ready) / 1e9
                engine._sync()
                timings["apply_s"] = (time.perf_counter_ns() - ready) / 1e9
            else:
                row["logical_device_written_bytes"] = full_weight_payload(engine)
            generation_started = time.perf_counter_ns()
            tokens, ttft = engine.generate(prompts, len(expected[0]))
            row.update(tokens=tokens, tokens_equal_reference=tokens == expected)
            timings["prompt_ttft_s"] = ttft
            first = generation_started + round(ttft * 1e9)
            row.update(first_token_timing_method="generate call start + engine-reported prompt TTFT",
                       first_token_perf_ns=first)
            timings["script_entry_to_first_token_s"] = (first - SCRIPT_ENTRY_PERF_NS) / 1e9
            if phase == "resident":
                timings["update_to_first_token_s"] = (first - ready) / 1e9
            engine._sync()
            generation_done = time.perf_counter_ns()
            timings["generation_s"] = (generation_done - generation_started) / 1e9
            if phase == "resident":
                timings["update_through_generation_s"] = (generation_done - ready) / 1e9
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if engine is not None:
                try:
                    engine.close()
                except BaseException as exc:
                    row["close_error"] = dict(type=type(exc).__name__, message=str(exc))
                    if primary_error is None:
                        raise


def launch_worker(command, out):
    launch_ns = time.perf_counter_ns()
    launched_utc = utc()
    completed = subprocess.run(command, check=False)
    exited_ns = time.perf_counter_ns()
    record = dict(argv=command, parent_launch_perf_ns=launch_ns, utc=launched_utc,
                parent_observed_exit_perf_ns=exited_ns, returncode=completed.returncode,
                result=None)
    if Path(out).exists():
        try:
            record["result"] = json.loads(Path(out).read_text())
        except (ValueError, OSError) as exc:
            record["result_read_error"] = str(exc)
    if completed.returncode or record["result"] is None:
        error = subprocess.CalledProcessError(completed.returncode or 1, command)
        error.worker_record = record
        raise error
    return record


def restart_sample(a):
    # Workers are strictly sequential; this process never constructs an engine.
    with tempfile.TemporaryDirectory(prefix="ccgrid-e7-") as directory:
        children = {}
        for phase in ("base", "target"):
            out = str(Path(directory) / (phase + ".json"))
            command = [sys.executable, str(Path(__file__).resolve()), "sample",
                "--ckpt", a.ckpt, "--target", a.target, "--ranges", a.ranges,
                "--variant", a.variant, "--cache", a.cache, "--chip", str(a.chip),
                "--reference", a.reference, "--mode", "restart", "--out", out,
                "--host-workers", str(a.host_workers), "--intra-op", str(a.intra_op),
                "--_phase", phase]
            try:
                children[phase] = launch_worker(command, out)
            except subprocess.CalledProcessError as exc:
                exc.completed_workers = children
                raise
        base, target = (children[k]["result"] for k in ("base", "target"))
        timings = dict(target["timings"])
        timings.update(update_to_first_token_s=(target["first_token_perf_ns"] - base["base_ready_perf_ns"]) / 1e9,
            base_ready_to_observed_exit_s=(children["base"]["parent_observed_exit_perf_ns"] - base["base_ready_perf_ns"]) / 1e9,
            parent_launch_to_first_token_s=(target["first_token_perf_ns"] - children["target"]["parent_launch_perf_ns"]) / 1e9)
        return dict(tokens=target["tokens"], base_tokens=base["base_tokens"],
            base_tokens_equal_reference=base.get("base_tokens_equal_reference"),
            prompts=target["prompts"], tokens_equal_reference=target["tokens_equal_reference"],
            logical_device_written_bytes=target["logical_device_written_bytes"],
            timings=timings, workers=children)


def checkpoint_manifest(directory):
    root = Path(directory)
    return dict(path=os.path.realpath(root), files=[dict(name=p.name,
        size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
        sha256=sha256(p) if p.suffix == ".json" else None)
        for p in sorted(root.iterdir()) if p.name.endswith((".safetensors", ".json"))])


def main(argv=None):
    invocation = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample"])
    for name in ("ckpt", "target", "ranges", "cache", "reference", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--variant", default="B4-S256-g2")
    ap.add_argument("--chip", required=True, type=int, help="allocated device ordinal (>=0)")
    ap.add_argument("--mode", required=True, choices=["inplace", "replace", "restart"])
    ap.add_argument("--host-workers", type=int, choices=[1, 2], default=2)
    ap.add_argument("--intra-op", type=int, default=4, help="Torch intra-op threads; positive integer")
    ap.add_argument("--_phase", choices=["base", "target"], help=argparse.SUPPRESS)
    a = ap.parse_args(invocation)
    if a.chip < 0:
        ap.error("--chip must be an explicit nonnegative allocated ordinal")
    if a.intra_op < 1:
        ap.error("--intra-op must be positive")
    # Reserve the artifact before any device operation; existing samples survive.
    with open(a.out, "x", encoding="utf-8") as destination:
        result = dict(schema="ccgrid-e7-sample-v1", mode=a.mode, variant=a.variant,
            chip=a.chip, cache=os.path.realpath(a.cache), metadata=metadata(invocation),
            allow_compile=False, allocator_retries=0,
            host_workers=a.host_workers, intra_op=a.intra_op,
            logical_device_written_bytes_note="logical weight payload; not physical DMA",
            replacement_unit="entire touched resident stacked group tensor" if a.mode == "replace" else None)
        try:
            load_reference(a.reference, a.variant)
            result.update(reference=dict(path=os.path.realpath(a.reference), sha256=sha256(a.reference)),
                ranges=dict(path=os.path.realpath(a.ranges), sha256=sha256(a.ranges)),
                checkpoints={k: checkpoint_manifest(getattr(a, k)) for k in ("ckpt", "target")})
            if a._phase:
                native_sample(a, result, phase=a._phase)
            else:
                plan, plan_s = prepare_plan(a)
                result.update(plan=plan.summary(), plan_tensors=plan.tensors, plan_s=plan_s,
                              plan_phase="before base engine load; excluded from update interval")
                if a.mode == "restart":
                    result.update(restart_sample(a))
                else:
                    native_sample(a, result, plan=plan)
            parity = (result.get("tokens_equal_reference", a._phase == "base")
                      and result.get("base_tokens_equal_reference") is not False)
            result["status"] = "ok" if parity else "token_mismatch"
        except Exception as exc:
            result.update(status="error", error=dict(type=type(exc).__name__, message=str(exc)))
            if hasattr(exc, "worker_record"):
                result.update(failed_worker=exc.worker_record,
                              workers=getattr(exc, "completed_workers", {}))
        result["completed_utc"] = utc()
        json.dump(result, destination, indent=2)
        destination.write("\n")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
