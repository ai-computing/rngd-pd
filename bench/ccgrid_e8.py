#!/usr/bin/env python3
r"""E8: one native B/S switch while retaining one checkpoint's device weights.

PYTHONPATH=src python bench/ccgrid_e8.py sample --ckpt IMMUTABLE_CHECKPOINT \
    --cache COMPILED_CACHE --reference fresh.json --chip 0 \
    --base-variant B4-S256-g2 --target-variant B4-S512-g2 \
    --graph-state first-use --host-workers 2 --intra-op 4 --out NEW.json

Reference JSON requires base_tokens, tokens, base_prompts, prompts. Each is a
list of lists of nonnegative integer token IDs, with the corresponding base
or target batch width. Each continuation must contain eight actual tokens
from an independently fresh-loaded matching variant and checkpoint. Prompts
are explicit, equal-length per batch and must fit prompt+continuation in S.
Optional base_variant/target_variant fields must match the requested variants.

first-use has no target graph prepared before the trigger. prepared actually
loads and runs the target, checks all warmup tokens, then switches back to and
runs the base again before the common base barrier. That preparation time is
reported separately and excluded from switch_to_first_token_s. Neither state
describes OS page-cache eviction or artifact delivery. Both use existing
compiled cache entries, allow_compile=False, and a hard from_module guard.

Every original CompileModule.__call__ is observed without replacing outputs:
output tensors must still be Furiosa tensors; these observation checks remain
inside measured intervals. All shared group tensors plus
the LM buffer retain both Python identity and data_ptr across the switch;
already loaded compiled modules retain identity. Full shared-weight upload
is counted and must happen exactly once, with no repeat after base setup.
Logical shared bytes exclude variant constants, programs, position tables,
KV and activations; they are not physical DMA measurements.

First-token timing is generate-call start plus native engine-reported prompt
TTFT, excluding the rest of the eight-token continuation. Raw continuations,
all parity checks, graph load/transfer/table timings and failure evidence are
saved even if a gate fails. Existing output files are never overwritten.
One outer exclusive RNGD Job runs >=5 samples for each directed switch/state;
this script neither allocates hardware nor retries failed native operations.
"""
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys
import time
import traceback


ENTRY_PERF_NS = time.perf_counter_ns()
# The public scripts are also runnable directly with only PYTHONPATH=src.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ccgrid_e7 as common  # noqa: E402


def validate(a):
    from rngd_pd.swap.variants import Variant
    base, target = Variant.parse(a.base_variant), Variant.parse(a.target_variant)
    if base == target or base.group != target.group or base.group <= 0:
        raise ValueError("E8 needs different B/S variants sharing a positive layer group")
    reference = json.loads(Path(a.reference).read_text())
    for prefix, variant in (("base_", base), ("", target)):
        for suffix in ("tokens", "prompts"):
            rows = reference.get(prefix + suffix)
            if (not isinstance(rows, list) or len(rows) != variant.B
                    or not all(isinstance(r, list) and r and all(type(t) is int and t >= 0
                        for t in r) for r in rows) or len({len(r) for r in rows}) != 1):
                raise ValueError("reference requires B equal-length integer lists: " + prefix + suffix)
        if len(reference[prefix + "tokens"][0]) != 8:
            raise ValueError("E8 requires eight reference continuation tokens per lane")
        if len(reference[prefix + "prompts"][0]) + 8 > variant.S:
            raise ValueError("reference prompt + continuation exceed variant context")
    for key, variant in (("base_variant", base), ("target_variant", target)):
        if key in reference and Variant.parse(reference[key]) != variant:
            raise ValueError("reference " + key + " differs from requested variant")
    return base, target, reference


def check_native_output(output):
    import torch
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
    actual = list(tensors(output))
    if not actual or any(t.device.type != "furiosa" for t in actual):
        raise RuntimeError("original compiled graph outputs must contain only furiosa tensors")


@contextmanager
def observe_native(engine_class, row):
    from rngd_pd.engines.fsdk import CompileModule, SDK
    counts = row["counts"] = dict(compilation_attempts=0, native_calls=0,
        native_output_checks=0, shared_uploads=0)
    phases = row["native_calls_by_phase"] = {}
    row["sdk"] = SDK
    phase = ["load"]
    original_compile = inspect.getattr_static(CompileModule, "from_module")
    original_call = CompileModule.__call__
    original_upload = engine_class._upload_shared
    def no_compile(*args, **kwargs):
        counts["compilation_attempts"] += 1
        raise RuntimeError("compilation forbidden in E8")
    def call(module, *args, **kwargs):
        counts["native_calls"] += 1
        phases[phase[0]] = phases.get(phase[0], 0) + 1
        output = original_call(module, *args, **kwargs)
        check_native_output(output)
        counts["native_output_checks"] += 1
        return output
    def upload(engine, *args, **kwargs):
        counts["shared_uploads"] += 1
        if counts["shared_uploads"] != 1:
            raise RuntimeError("full shared-weight upload repeated during E8")
        return original_upload(engine, *args, **kwargs)
    CompileModule.from_module = staticmethod(no_compile)
    CompileModule.__call__ = call
    engine_class._upload_shared = upload
    try:
        yield phase
    finally:
        CompileModule.from_module = original_compile
        CompileModule.__call__ = original_call
        engine_class._upload_shared = original_upload


def weight_buffers(engine):
    tensors = [("group%d/%s" % (i, name), tensor) for i, group in enumerate(engine.shared)
               for name, tensor in sorted(group.items())]
    tensors.append(("lm_head", engine.wlm_d))
    if len(tensors) != engine.ngroups * 7 + 1:
        raise RuntimeError("shared allocation count differs from runtime layout")
    return [dict(name=name, object_id=id(t), data_ptr=t.data_ptr(), shape=list(t.shape),
                 dtype=str(t.dtype), device=str(t.device), bytes=t.numel() * t.element_size())
            for name, t in tensors]


def graph_modules(engine):
    return {v.name: {key: id(st[key]) for key in ("m", "lmh")}
            for v, st in engine.variants.items()}


def generate(engine, prompts, expected, row, name):
    started = time.perf_counter_ns()
    tokens, ttft = engine.generate(prompts, 8)
    # Keep actual outputs before a later synchronization or cleanup can fail.
    row[name + "tokens"] = tokens
    row[name + "tokens_equal_reference"] = tokens == expected
    engine._sync()
    ended = time.perf_counter_ns()
    row[name + "generation_s"] = (ended - started) / 1e9
    row[name + "prompt_ttft_s"] = ttft
    return started, started + round(ttft * 1e9), ended


def sample(a, row, base, target, reference):
    import torch
    from rngd_pd.swap.rngd import SwappableDecode
    torch.set_num_threads(a.intra_op)
    engine = None
    with observe_native(SwappableDecode, row) as phase:
        try:
            began = time.perf_counter_ns()
            engine = SwappableDecode(a.ckpt, group=base.group, chip=a.chip,
                cache_dir=a.cache, host_workers=a.host_workers, allow_compile=False)
            weights_ready = time.perf_counter_ns()
            row.update(weights=dict(engine.timings), engine_load_s=(weights_ready - began) / 1e9)
            row["base_graph"] = dict(engine.swap(base))
            phase[0] = "base"
            generate(engine, reference["base_prompts"], reference["base_tokens"], row, "base_")
            if not row["base_tokens_equal_reference"] or not row["native_calls_by_phase"].get("base"):
                raise RuntimeError("base native generation/reference gate failed")
            if target in engine.variants:
                raise RuntimeError("target unexpectedly prepared during common base setup")
            if row["counts"]["shared_uploads"] != 1:
                raise RuntimeError("common setup must upload full shared weights exactly once")
            base_modules = graph_modules(engine)[base.name]
            row["actual_target_warmup"] = False
            preparation_started = time.perf_counter_ns()
            if a.graph_state == "prepared":
                phase[0] = "preparation"
                row["target_preparation_graph"] = dict(engine.swap(target))
                generate(engine, reference["prompts"], reference["tokens"], row, "target_warmup_")
                row["actual_target_warmup"] = True
                if not row["target_warmup_tokens_equal_reference"]:
                    raise RuntimeError("prepared target warmup differs from fresh reference")
                row["return_base_graph"] = dict(engine.swap(base))
                generate(engine, reference["base_prompts"], reference["base_tokens"], row, "barrier_base_")
                if not row["barrier_base_tokens_equal_reference"]:
                    raise RuntimeError("returned base differs from fresh reference")
            row["preparation_excluded_s"] = (time.perf_counter_ns() - preparation_started) / 1e9 if a.graph_state == "prepared" else 0.0
            engine._sync()
            row.update(weight_buffers_before=weight_buffers(engine),
                       graph_modules_before=graph_modules(engine),
                       target_graph_prepared_before=target in engine.variants)
            if row["target_graph_prepared_before"] != (a.graph_state == "prepared"):
                raise RuntimeError("target preparation state differs from requested condition")
            row["logical_shared_weight_upload_bytes"] = sum(t["bytes"] for t in row["weight_buffers_before"])
            row["base_barrier_perf_ns"] = time.perf_counter_ns()
            phase[0] = "target"
            trigger = time.perf_counter_ns()
            row["switch_trigger_perf_ns"] = trigger
            row["graph"] = dict(engine.swap(target))
            engine._sync()
            swapped = time.perf_counter_ns()
            started, first, ended = generate(engine, reference["prompts"], reference["tokens"], row, "")
            row.update(generation_start_perf_ns=started, first_token_perf_ns=first,
                weight_buffers_after=weight_buffers(engine), graph_modules_after=graph_modules(engine))
            row["timings"] = dict(swap_s=(swapped - trigger) / 1e9,
                graph_read_s=row["graph"].get("graph_read_s", 0.0),
                graph_to_device_s=row["graph"].get("graph_to_device_s", 0.0),
                tables_s=row["graph"].get("tables_s", 0.0), prompt_ttft_s=row["prompt_ttft_s"],
                switch_to_first_token_s=(first - trigger) / 1e9,
                switch_through_generation_s=(ended - trigger) / 1e9)
            row["weight_buffers_retained"] = row["weight_buffers_before"] == row["weight_buffers_after"]
            row["loaded_graph_modules_retained"] = all(
                row["graph_modules_after"].get(name) == modules
                for name, modules in row["graph_modules_before"].items())
            row["base_graph_modules_retained"] = row["graph_modules_after"].get(base.name) == base_modules
            if (not row["tokens_equal_reference"] or not row["weight_buffers_retained"]
                    or not row["loaded_graph_modules_retained"] or not row["base_graph_modules_retained"]
                    or row["counts"]["shared_uploads"] != 1 or row["counts"]["compilation_attempts"]
                    or not row["native_calls_by_phase"].get("target")
                    or row["counts"]["native_calls"] != row["counts"]["native_output_checks"]):
                raise RuntimeError("target parity, allocation, module or native execution gate failed")
        finally:
            if engine is not None:
                engine.close()


def main(argv=None):
    invocation = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample"])
    for key in ("ckpt", "cache", "reference", "out", "base-variant", "target-variant"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--chip", required=True, type=int)
    parser.add_argument("--graph-state", required=True, choices=["first-use", "prepared"])
    parser.add_argument("--host-workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--intra-op", type=int, default=4)
    a = parser.parse_args(invocation)
    if a.chip < 0 or a.intra_op < 1:
        parser.error("chip must be a nonnegative allocated ordinal and intra-op must be positive")
    with open(a.out, "x", encoding="utf-8") as destination:
        row = dict(schema="ccgrid-e8-sample-v1", graph_state=a.graph_state,
            base_variant=a.base_variant, target_variant=a.target_variant, chip=a.chip,
            host_workers=a.host_workers, intra_op=a.intra_op, allow_compile=False,
            allocator_retries=0, cache=str(Path(a.cache).resolve()), metadata=common.metadata(invocation),
            native_observation="original CompileModule calls and output-device checks; overhead included",
            first_token_timing_method="generate call start + native engine-reported prompt TTFT",
            logical_bytes_note="shared group + LM payload; not physical DMA")
        row["metadata"]["invocation_argv"] = [sys.executable, str(Path(__file__).resolve()), *invocation]
        row["metadata"]["script_entry_perf_ns"] = ENTRY_PERF_NS
        row["metadata"]["source"]["script_sha256"] = common.sha256(__file__)
        row["metadata"]["source"]["sha256"]["bench/ccgrid_e8.py"] = common.sha256(__file__)
        try:
            base, target, reference = validate(a)
            row.update(reference=dict(path=str(Path(a.reference).resolve()), sha256=common.sha256(a.reference)),
                checkpoint=common.checkpoint_manifest(a.ckpt),
                base_prompts=reference["base_prompts"], prompts=reference["prompts"])
            sample(a, row, base, target, reference)
            row["status"] = "ok"
        except Exception as exc:
            row.update(status="error", error=dict(type=type(exc).__name__, message=str(exc),
                                                  traceback=traceback.format_exc()))
        row["completed_utc"] = common.utc()
        json.dump(row, destination, indent=2)
        destination.write("\n")
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
