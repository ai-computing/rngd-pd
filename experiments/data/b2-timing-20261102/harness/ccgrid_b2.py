#!/usr/bin/env python3
"""Current unchanged RNGD B2 owner versus matched reload, allocated Job only.

python bench/ccgrid_b2.py --config CONFIG.json [--check-config]

CONFIG requires absolute source/cache/fixtures/references/input_manifest/run
paths, source_commit and phase (smoke: one pair; measure: five pairs). Optional
source_manifest defaults to input_manifest. The provider must contain the
native-interior-projection case. references is the retained refsactual.json.
Input manifest: {schema: ccgrid-b2-inputs-v1, files: [{path, bytes, sha256}],
source_commit, source_files: {relative: sha256}}. Full artifact hashes must be
computed by the controller OFF CARD; this harness checks inventory/sizes and
small input/source hashes, and records that distinction. Existing artifacts,
references and runnable caches are read only. Run must not already exist.

Each arm independently loads/generates the same base outside the trigger clock.
The trigger includes current owner.plan, owner.apply (or owner.load, closing
the base), synchronization and native prompt TTFT. Full continuation generation
is a separate wall clock. generate() returns its own first-token interval from
blank_kv through the prompt's greedy argmax (swap/rngd.py); first-token latency
is trigger-to-generate-call plus that returned interval. Python generate entry
validation/decorator work before its internal clock is not in that interval.
Observer checks and diagnostic snapshots inside the interval are included.

No runtime methods are overridden. The E7 observer calls original compiled
programs, checks their real device outputs, and rejects compilation. Per-arm
logical writes and whole-parent readbacks are not physical DMA. No rollback
is requested by the benchmark; apply's normal journaling/verification remains.
Incremental events are appended/fsynced; trial/result files are exclusive-create.
All failed attempts remain on disk. No CPU run is labeled native evidence.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback


RUNTIME_FILES = (
    "src/rngd_pd/swap/rngd_binding.py", "src/rngd_pd/swap/rngd.py",
    "src/rngd_pd/swap/delta.py", "src/rngd_pd/swap/variants.py",
    "src/rngd_pd/devices.py", "src/rngd_pd/engines/fsdk.py", "bench/ccgrid_e7.py")
VARIANT = "B4-S256-g2"
FRESH_SOURCE = "f5fa13377a3d967ba3d4022fb73ae59b1c87441c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def trial_order(phase):
    require(phase in ("smoke", "measure"), "phase must be smoke or measure")
    for pair in range(1, 2 if phase == "smoke" else 6):
        for arm in (("b2", "reload") if pair % 2 else ("reload", "b2")):
            yield pair, arm


def first_token_seconds(trigger_ns, generate_ns, prompt_ttft):
    require(generate_ns >= trigger_ns and math.isfinite(prompt_ttft) and prompt_ttft >= 0,
            "invalid first-token clock")
    return (generate_ns - trigger_ns) / 1e9 + prompt_ttft


class Journal:
    def __init__(self, run):
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=False)
        self.stream = (self.run / "events.jsonl").open("x", encoding="utf-8")

    def write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stream.close()


def write_new(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_config(path):
    config = read_json(path)
    for name in ("source", "cache", "fixtures", "references", "input_manifest", "run"):
        require(isinstance(config.get(name), str) and Path(config[name]).is_absolute(),
                name + " must be an absolute path")
        config[name] = str(Path(config[name]).resolve(strict=name != "run"))
    require(not Path(config["run"]).exists(), "run already exists; use a new attempt directory")
    require(config.get("variant", VARIANT) == VARIANT, "only the retained B4-S256-g2 fixture is supported")
    list(trial_order(config.get("phase")))
    source = Path(config["source"])
    source_manifest = Path(config.get("source_manifest", config["input_manifest"])).resolve(strict=True)
    source_doc, inputs = read_json(source_manifest), read_json(config["input_manifest"])
    require(inputs.get("schema") == "ccgrid-b2-inputs-v1", "invalid input manifest schema")
    require(isinstance(config.get("source_commit"), str) and config["source_commit"]
            and source_doc.get("source_commit") == config["source_commit"]
            and inputs.get("source_commit") == config["source_commit"], "source commit identity differs")
    source_hashes = {}
    for name in RUNTIME_FILES:
        source_hashes[name] = sha256(source / name)
        require(source_hashes[name] == source_doc.get("source_files", {}).get(name),
                "source hash differs: " + name)
    provider = read_json(config["fixtures"])
    require(provider.get("schema") == "onggi.binding-fixtures.v1", "invalid fixture provider")
    cases = [case for case in provider.get("cases", [])
             if case.get("name") == "native-interior-projection"]
    require(len(cases) == 1, "exactly one native-interior-projection case is required")
    case = dict(cases[0])
    require(case.get("expect_mode") == "units", "fixture must expect units")
    fixture_root = Path(config["fixtures"]).parent
    for name in ("base_dir", "target_dir"):
        case[name] = str((fixture_root / case[name]).resolve(strict=True))
        require(Path(case[name]).is_dir(), name + " is not a directory")
    require(case["base_dir"] != case["target_dir"], "interior target cannot equal base")
    plan_path = (fixture_root / case["switchplan"]).resolve(strict=True)
    document = read_json(plan_path)
    require(document.get("schema") == "onggi.switchplan.v1", "expected exported switchplan v1")
    for name in ("base_manifest", "manifest"):
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", document.get(name, "")), "invalid " + name)
    require(document["base_manifest"] != document["manifest"], "fixture manifests must differ")
    refs = read_json(config["references"])
    for role, directory in (("base", case["base_dir"]), ("sparse", case["target_dir"])):
        ref = refs[role]
        require(ref.get("status") == "completed" and ref.get("engine_closed") is True,
                role + " reference did not finish/close")
        require(ref.get("source_commit") == FRESH_SOURCE and ref.get("source_dirty") == "",
                role + " reference is not the retained independent source")
        require(ref.get("variant") == VARIANT and ref.get("continuation_length") == 8,
                role + " reference shape differs")
        require(Path(ref["checkpoint"]).resolve(strict=True) == Path(directory),
                role + " reference checkpoint differs")
        for name in ("prompts", "tokens"):
            values = ref.get(name)
            require(isinstance(values, list) and len(values) == 4 and all(
                isinstance(lane, list) and len(lane) == 8
                and all(type(token) is int and token >= 0 for token in lane) for lane in values),
                role + " reference needs four lanes of eight integer " + name)
        require(ref.get("counts") == {"native_calls": 255, "native_output_checks": 255,
                                      "compilation_attempts": 0}, "reference native counts differ")
    require(refs["base"]["prompts"] == refs["sparse"]["prompts"], "reference prompts differ")
    indexed = {}
    for entry in inputs["files"]:
        target = Path(entry["path"])
        require(target.is_absolute() and re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")),
                "input entries need absolute paths and SHA-256")
        target = target.resolve(strict=True)
        require(str(target) not in indexed, "duplicate manifest path: " + str(target))
        require(type(entry.get("bytes")) is int and target.stat().st_size == entry["bytes"],
                "input size differs: " + str(target))
        indexed[str(target)] = entry
    artifacts = []
    for name in ("base_dir", "target_dir"):
        paths = list(Path(case[name]).rglob("*"))
        require(not any(p.is_symlink() for p in paths), "artifact symlinks are unsupported")
        artifacts.extend(p for p in paths if p.is_file())
    cache_files = list(Path(config["cache"]).rglob("*.runnable"))
    require(cache_files, "no precompiled runnable cache files")
    small = [Path(config["fixtures"]), Path(config["references"]), plan_path]
    for target in artifacts + cache_files + small:
        require(str(target.resolve()) in indexed, "input manifest omits " + str(target))
    for target in small:
        require(sha256(target) == indexed[str(target.resolve())]["sha256"],
                "small input hash differs: " + str(target))
    return dict(config=config, case=case, document=document, references=refs,
                source_files=source_hashes, inputs=inputs, input_manifest_sha256=sha256(config["input_manifest"]),
                source_manifest_sha256=sha256(source_manifest), config_sha256=sha256(path),
                switchplan_path=str(plan_path), switchplan_sha256=sha256(plan_path))


def identities(engine):
    buffers = {f"shared/{group}/{slot}": tensor for group, slots in enumerate(engine.shared)
               for slot, tensor in slots.items()}
    buffers["lm_head"] = engine.wlm_d
    return {"engine_id": id(engine), "buffers": {
        name: {"python_id": id(tensor), "data_ptr": tensor.data_ptr(),
               "shape": list(tensor.shape), "device": str(tensor.device)}
        for name, tensor in sorted(buffers.items())}, "programs": {
        variant.name: {key: id(state[key]) for key in ("m", "lmh")}
        for variant, state in engine.variants.items()}}


def numeric_diff(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if type(value) in (int, float)}


def run_trial(owner, prepared, pair, arm, observer_record, journal, provenance, payload):
    require(arm in ("b2", "reload"), "unknown trial arm")
    case, document, refs = prepared["case"], prepared["document"], prepared["references"]
    row = dict(schema="ccgrid-current-b2-trial-v1", pair=pair, arm=arm, status="running",
               phase=prepared["config"]["phase"], native_evidence=False, started_utc=utc(),
               provenance=provenance, operations={}, timings={}, byte_scope="logical writes and readback payload, not DMA")
    journal.write(row)
    current = row["operations"]

    def snapshot():
        return dict(native=dict(observer_record["native_counts"]), diagnostics=dict(owner._metrics))

    @contextmanager
    def operation(name):
        before, began = snapshot(), time.perf_counter_ns()
        try:
            yield
        finally:
            ended, after = time.perf_counter_ns(), snapshot()
            current[name] = dict(wall_s=(ended - began) / 1e9,
                                 native_counts=numeric_diff(after["native"], before["native"]),
                                 diagnostics_before=before["diagnostics"], diagnostics_after=after["diagnostics"],
                                 diagnostics=numeric_diff(after["diagnostics"], before["diagnostics"]))

    try:
        with operation("prepare_base_outside_trigger"):
            owner.load(case["base_dir"], document["base_manifest"])
            with owner.request() as engine:
                base_tokens, base_ttft = engine.generate(refs["base"]["prompts"], 8)
                engine._sync()
                row["identities_before"] = identities(engine)
        row["base_tokens"], row["base_prompt_ttft_s"] = base_tokens, base_ttft
        require(base_tokens == refs["base"]["tokens"], "fresh-base token mismatch")
        require(owner.resident()["manifest"] == document["base_manifest"], "base identity differs")
        journal.write(row)
        trigger_diagnostics = dict(owner._metrics)
        trigger = time.perf_counter_ns()
        with operation("plan"):
            update = owner.plan(document)
        require(update["mode"] == "units" and update["units"], "interior plan was not admitted")
        row["update_plan"] = update
        with operation("apply" if arm == "b2" else "reload"):
            if arm == "b2":
                row["verification"] = owner.apply(update, case["target_dir"])
            else:
                row["load_result"] = owner.load(case["target_dir"], document["manifest"])
        with owner.request() as engine:
            with operation("sync"):
                engine._sync()
            with operation("generate_full_continuation"):
                generate_entry = time.perf_counter_ns()
                tokens, ttft = engine.generate(refs["sparse"]["prompts"], 8)
                generate_return = time.perf_counter_ns()
            row["timings"] = dict(trigger_perf_ns=trigger, generate_call_perf_ns=generate_entry,
                trigger_to_generate_call_s=(generate_entry - trigger) / 1e9,
                prompt_to_first_token_s=ttft, trigger_to_first_token_s=first_token_seconds(trigger, generate_entry, ttft),
                trigger_to_generate_return_s=(generate_return - trigger) / 1e9)
            engine._sync()
            row["identities_after"] = identities(engine)
            row["logical_device_written_bytes"] = (row["verification"]["bytes_written"]
                                                     if arm == "b2" else payload(engine))
        row.update(tokens=tokens, tokens_equal_historical_fresh=tokens == refs["sparse"]["tokens"],
                   resident_after=owner.resident(), finished_utc=utc())
        require(row["tokens_equal_historical_fresh"], "fresh-target token mismatch")
        require(row["resident_after"]["manifest"] == document["manifest"], "target identity differs")
        if arm == "b2":
            row["serving_allocations_programs_retained"] = row["identities_before"] == row["identities_after"]
            require(row["serving_allocations_programs_retained"], "B2 changed resident allocations/programs")
            require(row["verification"].get("ok") is True, "B2 verification did not succeed")
        for name in ("prepare_base_outside_trigger", "generate_full_continuation"):
            require(current[name]["native_counts"] == refs["sparse"]["counts"], "native generation counts differ")
        row["diagnostics_during_trigger"] = numeric_diff(row["resident_after"]["diagnostics"], trigger_diagnostics)
        # Operation deltas, unlike cumulative owner metrics, exclude base preparation.
        row["readback_payload_bytes_during_trigger"] = sum(
            current[name]["diagnostics"].get("readback_payload_bytes", 0)
            for name in current if name != "prepare_base_outside_trigger")
        row.update(status="completed", native_gate_status="pending_final_run_observer")
        return row
    except BaseException as error:
        row.update(status="failed", error=repr(error), traceback=traceback.format_exc(), finished_utc=utc())
        raise
    finally:
        journal.write(row)
        write_new(journal.run / f"pair-{pair:02d}-{arm}.json", row)


def run_native(prepared, journal):
    from importlib.metadata import version
    config = prepared["config"]
    record = dict(schema="ccgrid-current-b2-run-v1", status="running", native_evidence=False,
                  started_utc=utc(), config=config, config_sha256=prepared["config_sha256"], trials=[],
                  conditions=dict(os_page_cache="uncontrolled", delivery="existing verified target files; delivery excluded",
                      base_preparation="independent load, eight-token base parity gate and sync before each trigger",
                      first_token="trigger-to-generate-call plus returned native prompt TTFT",
                      included="plan, current apply/journaling/whole-parent readback or close-and-reload, sync, prompt",
                      diagnostics="numeric differences include gauges such as initial_load_s; do not sum all fields"))
    journal.write(record)
    owner = None
    try:
        sys.path.insert(0, str(Path(config["source"]) / "src"))
        import torch
        from rngd_pd import devices
        from rngd_pd.engines.fsdk import DEV
        import rngd_pd.swap.rngd_binding as binding
        require(Path(binding.__file__).resolve() == Path(config["source"], RUNTIME_FILES[0]).resolve(),
                "import selected a different runtime source")
        packages = {name: version(name) for name in ("torch", "numpy", "furiosa-torch", "furiosa-tcc")}
        require(packages["torch"] == "2.10.0+cpu" and packages["furiosa-torch"] == "2026.4.0",
                "native runtime differs from validated SDK/Torch")
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        cards = devices.allocated_cards()
        require(cards is not None and len(cards) == 1, "exactly one allocated RNGD is required")
        chip, physical = devices.resolve_chip(-1), cards[0]
        require(Path("/run/swap/rngd").read_text().strip() == "npu" + str(physical),
                "init/main card assignments differ")
        assignment = dict(pod_uid=os.environ["POD_UID"], host_index=physical, torch_ordinal=chip,
                          visible_cards=devices.visible_cards(),
                          allocation_uuid=devices.allocated_uuids(os.environ["POD_UID"], os.environ.get("CONTAINER_NAME", "run")))
        require(len(assignment["allocation_uuid"]) == 1, "ambiguous allocation UUID")
        provenance = dict(source_commit=config["source_commit"], source_files=prepared["source_files"],
            harness_sha256=sha256(__file__), input_manifest_sha256=prepared["input_manifest_sha256"],
            source_manifest_sha256=prepared["source_manifest_sha256"],
            fixture_sha256=sha256(config["fixtures"]), references_sha256=sha256(config["references"]),
            switchplan_sha256=prepared["switchplan_sha256"], assignment=assignment, packages=packages,
            input_files=prepared["inputs"]["files"], reference_assignments={
                role: prepared["references"][role]["assignment"] for role in ("base", "sparse")},
            hash_verification_scope="external off-card full hashes; native sizes/inventory plus small input/source hashes",
            python=sys.version, executable=sys.executable, platform=platform.platform(), hostname=platform.node(),
            environment={name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "FURIOSA_DEVICES", "FURIOSA_VISIBLE_DEVICES", "PYTHONPATH", "HOSTNAME", "POD_UID", "POD_NAME") if name in os.environ})
        record["provenance"] = provenance
        spec = importlib.util.spec_from_file_location("ccgrid_b2_native_observer", Path(config["source"]) / "bench/ccgrid_e7.py")
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)
        with observer.observe_native(chip, record):
            for pair, arm in trial_order(config["phase"]):
                record["active_trial"] = dict(pair=pair, arm=arm)
                owner = binding.RngdBinding(cache_dir=config["cache"], variant=VARIANT, chip=chip,
                                             expected_device=f"{DEV}:{chip}")
                row = run_trial(owner, prepared, pair, arm, record, journal, provenance,
                                observer.full_weight_payload)
                record["trials"].append(row)
                owner.close()
                owner = None
                paired = [trial for trial in record["trials"] if trial["pair"] == pair]
                if len(paired) == 2:
                    equal = paired[0]["tokens"] == paired[1]["tokens"]
                    journal.write(dict(pair=pair, tokens_equal_current_same_device_reload=equal,
                                       b2_trial=f"pair-{pair:02d}-b2.json", reload_trial=f"pair-{pair:02d}-reload.json"))
                    require(equal, "B2 tokens differ from current same-device fresh reload")
        require(record["native_outputs_verified"] and record["native_counts"]["compilation_attempts"] == 0,
                "native observer did not verify original outputs")
        for row in record["trials"]:
            row.update(native_evidence=True, native_gate_status="verified_by_final_run_observer")
        record.update(status="completed", native_evidence=True, finished_utc=utc(),
                      tokens_equal_current_same_device_reload=True)
        record.pop("active_trial", None)
    except BaseException as error:
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc(), finished_utc=utc())
        if owner is not None:
            try:
                owner.close()
            except BaseException as close_error:
                record["cleanup_error"] = repr(close_error)
    journal.write(record)
    write_new(journal.run / "result.json", record)
    return 0 if record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        prepared = check_config(args.config)
        if args.check_config:
            print(json.dumps(dict(status="config-valid", native_evidence=False,
                                  phase=prepared["config"]["phase"], trials=list(trial_order(prepared["config"]["phase"])),
                                  source_files=prepared["source_files"], config_sha256=prepared["config_sha256"])))
            return 0
        with Journal(prepared["config"]["run"]) as journal:
            return run_native(prepared, journal)
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
