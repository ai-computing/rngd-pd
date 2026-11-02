"""Loaded native caller, only inside an allocated bounded one-card Job.

Uses unchanged current B2 runtime and retained prepared-file fixtures. FIFO
request admission is the experiment caller, not a new runtime capability.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

import ccgrid_b2 as b2
from ccgrid_loaded_queue import mark_first_return, run_queue, verify_sample


def conditions(config):
    return hashlib.sha256(json.dumps({k: v for k, v in config.items()
        if k not in ("run", "phase", "smoke")}, sort_keys=True).encode()).hexdigest()


def memory():
    return {name: path.read_text() for name in ("status", "smaps_rollup")
            if (path := Path("/proc/self") / name).is_file()}


def trial(owner, prepared, arm, observed, journal):
    import torch
    from rngd_pd.engines.fsdk import CompileModule
    cfg, case, doc, refs = (prepared[k] for k in ("config", "case", "document", "references"))
    row = dict(arm=arm, status="running", cleanup_passed=False, native_evidence=False,
               device_peak_memory=None, memory_scope="host process snapshots; not device allocation peak")
    first_native, marker = threading.Event(), {}
    safe_to_close = True
    try:
        before = dict(observed["native_counts"])
        owner.load(case["base_dir"], doc["base_manifest"])
        with owner.request() as engine:
            row["warm_tokens"], row["warm_prompt_ttft_s"] = engine.generate(refs["base"]["prompts"], 8)
            engine._sync()
            row["identities_before"] = b2.identities(engine)
        row["warm_native_counts"] = b2.numeric_diff(observed["native_counts"], before)
        b2.require(row["warm_tokens"] == refs["base"]["tokens"], "initial base parity failed")
        row["memory_base_ready"] = memory()

        def generate(request):
            # Torch grad mode is thread-local; use the same inference setting.
            torch.set_grad_enabled(False)
            counts = dict(observed["native_counts"])
            with owner.request() as engine:
                if request == 0:
                    with mark_first_return(CompileModule, first_native, marker):
                        entry = time.perf_counter_ns()
                        tokens, ttft = engine.generate(refs["base"]["prompts"], 8)
                        engine._sync()
                else:
                    entry = time.perf_counter_ns()
                    tokens, ttft = engine.generate(refs["sparse"]["prompts"], 8)
                    engine._sync()
                return dict(tokens=tokens, prompt_ttft_s=ttft, generation_entry_perf_ns=entry,
                    native_first_perf_ns=marker.get("perf_ns") if request == 0 else None,
                    native_counts=b2.numeric_diff(observed["native_counts"], counts),
                    manifest=owner.resident()["manifest"])

        def switch():
            prior = dict(owner._metrics)
            update = owner.plan(doc)
            b2.require(update["mode"] == "units" and update["units"], "projection update was not admitted")
            if arm == "b2":
                verification = owner.apply(update, case["target_dir"])
            else:
                verification = owner.load(case["target_dir"], doc["manifest"])
            with owner.request() as engine:
                engine._sync()
                after = b2.identities(engine)
            return dict(update=update, verification=verification, identities_after=after,
                        diagnostics=b2.numeric_diff(owner._metrics, prior), memory_ready=memory())

        safe_to_close = False
        row["queue"] = run_queue(generate, switch, first_native, journal.write)
        safe_to_close = row["queue"]["worker_joined"]
        b2.require(row["queue"]["status"] == "completed", row["queue"].get("error", "queue failed"))
        row["status"] = "completed"
    except BaseException:
        row.update(status="failed", error=traceback.format_exc())
    finally:
        if safe_to_close:
            try:
                owner.close()
                row["cleanup_passed"] = True
            except BaseException:
                row.update(status="failed", cleanup_error=traceback.format_exc())
        else:
            row.update(status="failed", close_deferred_to_process_exit=True)
        row["memory_after"] = memory()
    if row["status"] == "completed":
        try:
            verify_sample(row, prepared)
        except BaseException:
            row.update(status="failed", error=traceback.format_exc())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    prepared = b2.check_config(args.config)
    cfg = prepared["config"]
    hashes = {name: b2.sha256(Path(__file__).with_name(name)) for name in
              ("ccgrid_loaded_native.py", "ccgrid_loaded_queue.py", "ccgrid_b2.py", "ccgrid_e7.py",
               "ccgrid_resident_delivery.py", "ccgrid_e6.py")}
    b2.require(cfg.get("harness_files") == hashes, "benchmark source hashes differ")
    b2.require(re.fullmatch(r"[0-9a-f]{40}", cfg.get("benchmark_commit", "")), "missing benchmark commit")
    key = conditions(cfg)
    if cfg["phase"] == "measure":
        smoke = b2.read_json(cfg["smoke"])
        b2.require(smoke.get("status") == "completed" and smoke.get("native_evidence") is True
                   and smoke.get("conditions_sha256") == key and len(smoke.get("trials", [])) == 2,
                   "main requires matching excluded native smoke")
    if args.check_config:
        print(json.dumps(dict(status="config-valid", native_evidence=False, conditions_sha256=key)))
        return 0
    record = dict(schema="ccgrid-loaded-native-v1", status="running", native_evidence=False,
        conditions_sha256=key, config=cfg, config_sha256=prepared["config_sha256"], trials=[],
        scope="finite prepared-file native FIFO; zero synthetic holds, no HTTP/throughput/p99 claim")
    with b2.Journal(cfg["run"]) as journal:
        try:
            sys.path.insert(0, str(Path(cfg["source"]) / "src"))
            import torch
            from importlib.metadata import version
            from rngd_pd import devices
            from rngd_pd.engines.fsdk import DEV
            from rngd_pd.swap import rngd_binding as binding
            import ccgrid_e7 as observer
            from ccgrid_resident_delivery import allocated_device
            b2.require(Path(binding.__file__).resolve() == Path(cfg["source"], b2.RUNTIME_FILES[0]).resolve(),
                       "different runtime imported")
            packages = {name: version(name) for name in ("torch", "numpy", "furiosa-torch", "furiosa-tcc")}
            b2.require(packages["torch"] == "2.10.0+cpu" and packages["furiosa-torch"] == "2026.4.0",
                       "native packages differ from retained fixture")
            chip = devices.resolve_chip(-1)
            assignment = allocated_device(dict(chip=chip))
            torch.set_num_threads(4)
            torch.set_grad_enabled(False)
            record["provenance"] = dict(assignment=assignment, packages=packages, harness_files=hashes,
                runtime_source_commit=cfg["source_commit"], runtime_files=prepared["source_files"],
                benchmark_commit=cfg["benchmark_commit"], inputs=prepared["inputs"],
                input_manifest_sha256=prepared["input_manifest_sha256"],
                source_manifest_sha256=prepared["source_manifest_sha256"],
                reference_sha256=b2.sha256(cfg["references"]),
                hash_scope="full artifact hashes external off-card; sizes/inventory and small source/input hashes here",
                python=sys.version, pid=os.getpid())
            journal.write(record)
            with observer.observe_native(chip, record):
                for pair, arm in b2.trial_order(cfg["phase"]):
                    owner = binding.RngdBinding(cache_dir=cfg["cache"], variant=b2.VARIANT, chip=chip,
                                               expected_device=f"{DEV}:{chip}")
                    row = trial(owner, prepared, arm, record, journal)
                    row["pair"] = pair
                    record["trials"].append(row)
                    b2.write_new(journal.run / f"pair-{pair:02d}-{arm}.json", row)
                    journal.write(row)
                    b2.require(row["status"] == "completed", row.get("error", "native trial/cleanup failed"))
            b2.require(record["native_outputs_verified"] and record["native_counts"] == dict(
                native_calls=len(record["trials"]) * 1020, native_output_checks=len(record["trials"]) * 1020,
                compilation_attempts=0), "aggregate native gate failed")
            record.update(status="completed", native_evidence=True)
        except BaseException:
            record.update(status="failed", error=traceback.format_exc())
        journal.write(record)
        b2.write_new(journal.run / "result.json", record)
    print(json.dumps(dict(status=record["status"], native_evidence=record["native_evidence"], trials=len(record["trials"]))))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
