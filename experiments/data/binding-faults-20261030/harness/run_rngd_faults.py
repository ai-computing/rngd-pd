"""Two bounded native compliance gates; run only in the root's allocated Job.

Uses the unchanged frozen owner, actual serving tensors and existing native
observer. This is n=1 drain/fail-closed evidence, without performance claims.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback


OLD_ROOT = Path("/home/gspark/workspace/codex_ccgrid_npu_20261030")
VARIANT = "B4-S256-g2"
RUNTIME_COMMIT = "a2ebc3281df4173a886bc6a035f2483eaa360bf5"
NATIVE_GENERATION = dict(native_calls=255, native_output_checks=255, compilation_attempts=0)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_digest(rows):
    import torch
    return hashlib.sha256(rows.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def first_half(unit):
    name, interval = unit["unit"].rsplit("@", 1)
    start, end = map(int, interval.split(":"))
    count = end - start
    require(count >= 2 and unit["length"] % count == 0, "partial rollback requires two or more whole rows")
    half = count // 2
    result = dict(unit, unit=f"{name}@{start}:{start + half}", length=half * (unit["length"] // count))
    return result, half


def identities(engine):
    buffers = {f"shared/{group}/{slot}": {"python_id": id(tensor), "data_ptr": tensor.data_ptr(),
               "shape": list(tensor.shape), "device": str(tensor.device)}
               for group, slots in enumerate(engine.shared) for slot, tensor in slots.items()}
    tensor = engine.wlm_d
    buffers["lm_head"] = {"python_id": id(tensor), "data_ptr": tensor.data_ptr(),
                          "shape": list(tensor.shape), "device": str(tensor.device)}
    return dict(buffers=buffers, programs={variant.name: {key: id(state[key]) for key in ("m", "lmh")}
                                          for variant, state in engine.variants.items()})


def main():
    root, run, source = (Path(os.environ[name]) for name in ("CCGRID_ROOT", "CCGRID_RUN", "CCGRID_SOURCE"))
    require(run.is_dir(), "CCGRID_RUN must be precreated")
    require(not (run / "result.json").exists(), "refusing to overwrite an old result")
    work = run / "work"
    work.mkdir(exist_ok=False)
    record = dict(schema="rngd-native-drain-rollback-failure-v1", status="running", utc=time.time(),
                  pod_uid=os.environ["POD_UID"], harness_sha256=sha(__file__), runtime_commit=RUNTIME_COMMIT,
                  generations=[], drain=dict(status="not-run"), rollback_failure=dict(status="not-run"),
                  validity="not yet established", native_evidence=False,
                  scope="n=1 compliance only; no performance or DMA claim")
    owner, worker = None, None

    def save():
        data = json.dumps(record, indent=2) + "\n"
        (run / "operational.json").write_text(data)
        if record["status"] in ("completed", "failed"):
            (run / "result.json").write_text(data)

    save()
    try:
        plan_path = Path(os.environ["BINDING_PLAN"]).resolve(strict=True)
        refs_path = Path(os.environ["BINDING_REFERENCES"]).resolve(strict=True)
        previous_run = root / "runs/rngd-b2-native-02"
        previous, previous_kit = load(previous_run / "result.json"), load(previous_run / "work/kit-report.json")
        document, refs = load(plan_path), load(refs_path)
        require(previous["status"] == "completed" and previous["native_evidence"], "original B2 run did not pass")
        require(sha(refs_path) == previous["inputs"]["references_actual"]["sha256"], "fresh reference bytes changed")
        measured = next(case for case in previous_kit["cases"] if case["name"] == "native-interior-projection")
        measured_plan = next(step["stdin"] for step in measured["results"]
                             if (step.get("stdin") or {}).get("schema") == "onggi.switchplan.v1")
        require(document == measured_plan, "authentic interior switch plan differs from native02")
        base, target = Path(measured["base_dir"]), Path(measured["target_dir"])
        for role, directory in (("base", base), ("sparse", target)):
            reference = refs[role]
            require(reference["status"] == "completed" and reference["engine_closed"]
                    and reference["variant"] == VARIANT and reference["counts"] == NATIVE_GENERATION
                    and reference["source_commit"] == "f5fa13377a3d967ba3d4022fb73ae59b1c87441c"
                    and reference["source_dirty"] == "" and reference["continuation_length"] == 8,
                    "invalid independent fresh reference: " + role)
            require(Path(reference["checkpoint"]).resolve(strict=True) == directory.resolve(strict=True),
                    "reference checkpoint path differs")
            require(reference["prompts"] == refs["base"]["prompts"]
                    and len(reference["tokens"]) == 4 and all(len(lane) == 8 for lane in reference["tokens"]),
                    "reference prompts/tokens differ in shape")
        record.update(inputs=dict(plan=dict(path=str(plan_path), sha256=sha(plan_path)),
                                  references=dict(path=str(refs_path), sha256=sha(refs_path))),
                      references={role: refs[role] for role in ("base", "sparse")}, source=str(source),
                      source_files={name: sha(source / name) for name in previous["source_files"]})
        require(record["source_files"] == previous["source_files"], "runtime bytes differ from frozen native02 source")
        import torch
        from importlib.metadata import version
        from rngd_pd import devices
        from rngd_pd.engines.fsdk import DEV
        import rngd_pd.swap.rngd_binding as binding
        require(Path(binding.__file__).resolve() == (source / "src/rngd_pd/swap/rngd_binding.py").resolve(), "wrong PYTHONPATH owner")
        require(torch.__version__ == "2.10.0+cpu" and version("furiosa-torch") == "2026.4.0", "runtime version changed")
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        cards = devices.allocated_cards()
        require(cards is not None and len(cards) == 1, "exactly one allocated RNGD required")
        physical, chip = cards[0], devices.resolve_chip(-1)
        require(Path("/run/swap/rngd").read_text().strip() == "npu" + str(physical), "init/main assignment mismatch")
        require(physical == previous["assignment"]["host_index"] and chip == previous["assignment"]["torch_ordinal"],
                "native02 physical/logical assignment changed")
        record.update(assignment=dict(pod_uid=os.environ["POD_UID"], host_index=physical, torch_ordinal=chip,
                                      visible_cards=devices.visible_cards(),
                                      allocation_uuid=devices.allocated_uuids(os.environ["POD_UID"], "run")),
                      packages={name: version(name) for name in ("torch", "numpy", "furiosa-torch", "furiosa-tcc")})
        spec = importlib.util.spec_from_file_location("rngd_fault_observer", source / "bench/ccgrid_e7.py")
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)
        cache = Path(os.environ.get("BINDING_CACHE", str(OLD_ROOT / "cache/graphs"))).resolve(strict=True)
        record["cache"] = str(cache)
        owner = binding.RngdBinding(cache_dir=str(cache), variant=VARIANT, chip=chip, expected_device=f"{DEV}:{chip}")

        def generate(label, role):
            row = dict(operation=label, reference_role=role, status="running", utc=time.time())
            record["generations"].append(row)
            counts = dict(record["native_counts"])
            with owner.request() as engine:
                row["identities_before"] = identities(engine)
                row["tokens"], row["prompt_ttft_s"] = engine.generate(refs["base"]["prompts"], 8)
                engine._sync()
                row["identities_after"] = identities(engine)
            row["tokens_equal_reference"] = row["tokens"] == refs[role]["tokens"]
            row["native_counts"] = {key: record["native_counts"][key] - counts[key] for key in counts}
            require(row["tokens_equal_reference"] and row["identities_before"] == row["identities_after"],
                    "native generation parity/identity gate failed: " + label)
            require(row["native_counts"] == NATIVE_GENERATION, "generation native call counts changed")
            row["status"] = "completed"
            save()

        def read_units(units):
            with owner.request():
                return {unit["unit"]: row_digest(owner._read(unit)) for unit in units}

        with observer.observe_native(chip, record):
            record["load"] = owner.load(base, document["base_manifest"])
            update = owner.plan(document)
            require(update["mode"] == "units" and len(update["units"]) == 1, "expected one interior projection unit")
            record["update"] = update
            generate("loaded-base", "base")
            base_hashes = record["base_actual_d2h_sha256"] = read_units(update["units"])
            with owner.request() as engine:
                baseline = identities(engine)
            record["identities_loaded"] = baseline
            attempted, finished = threading.Event(), threading.Event()
            outcome = {}
            drain = record["drain"] = dict(status="running")

            def apply_worker():
                drain["attempt_perf_ns"] = time.perf_counter_ns()
                attempted.set()  # Signal before entering owner.apply and its lock.
                try:
                    outcome["verification"] = owner.apply(update, target)
                except BaseException:
                    outcome["error"] = traceback.format_exc()
                finally:
                    drain["finished_perf_ns"] = time.perf_counter_ns()
                    finished.set()

            with owner.request():
                worker = threading.Thread(target=apply_worker, daemon=True)
                worker.start()
                require(attempted.wait(5), "apply worker did not signal its attempt")
                drain["finished_before_generation"] = finished.is_set()
                require(not finished.is_set(), "apply finished while whole request lock held")
                generate("whole-request-held-base", "base")
                drain["finished_after_generation_before_release"] = finished.is_set()
                drain["base_bytes_unchanged_before_release"] = read_units(update["units"]) == base_hashes
                require(not finished.is_set() and drain["base_bytes_unchanged_before_release"]
                        and owner.resident()["manifest"] == document["base_manifest"], "apply crossed native request boundary")
                drain["request_release_marker_perf_ns"] = time.perf_counter_ns()
            worker.join(timeout=300)
            require(not worker.is_alive() and finished.is_set(), "apply worker did not finish after request drained")
            require("error" not in outcome, "drained apply worker failed: " + outcome.get("error", ""))
            require(drain["finished_perf_ns"] > drain["request_release_marker_perf_ns"], "apply completed before release marker")
            record["drained_apply"] = outcome["verification"]
            drain.update(status="completed", worker_joined=True, completion_after_request_release=True)
            worker = None
            with owner.request() as engine:
                require(identities(engine) == baseline, "drained apply replaced serving buffers/programs")
            record["target_actual_d2h_sha256"] = read_units(update["units"])
            generate("drained-apply-target", "sparse")
            record["normal_rollback"] = owner.rollback()
            record["normal_rollback_actual_d2h_sha256"] = read_units(update["units"])
            require(record["normal_rollback_actual_d2h_sha256"] == base_hashes, "normal rollback failed exact actual-base restoration")
            generate("normal-rollback-base", "base")
            record["second_apply"] = owner.apply(update, target)
            generate("second-apply-target", "sparse")
            with owner.request() as failed_engine:
                require(identities(failed_engine) == baseline, "second apply replaced serving buffers/programs")
            fault = record["rollback_failure"] = dict(status="running", mechanism="real public half-unit restore, actual mixed-byte D2H, then injected exception")
            original_write = owner._write
            def partial_then_fail(unit, old_rows):
                partial, half = first_half(unit)
                before = owner._read(unit)
                mixed = before.clone()
                mixed[:half].copy_(old_rows[:half])
                fault.update(partial_unit=partial, target_sha256=row_digest(before), base_sha256=row_digest(old_rows),
                             expected_mixed_sha256=row_digest(mixed))
                original_write(partial, old_rows[:half].contiguous())
                failed_engine._sync()
                fault["actual_mixed_sha256"] = row_digest(owner._read(unit))
                require(fault["actual_mixed_sha256"] == fault["expected_mixed_sha256"]
                        and fault["actual_mixed_sha256"] not in (fault["target_sha256"], fault["base_sha256"]),
                        "partial rollback did not create verified actual mixed serving bytes")
                fault["actual_partial_restore_verified"] = True
                save()
                raise RuntimeError("intentional rollback failure after actual mixed-state readback")
            calls_before_failure = dict(record["native_counts"])
            owner._write = partial_then_fail
            try:
                try:
                    owner.rollback()
                except RuntimeError as error:
                    fault["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                    require("rollback restore failed" in str(error) and fault.get("actual_partial_restore_verified"),
                            "rollback failure occurred outside the verified partial-write injection")
                else:
                    raise RuntimeError("injected rollback failure unexpectedly succeeded")
            finally:
                owner._write = original_write
            fault["resident_after_failure"] = owner.resident()
            require(owner.resident()["state"] == "empty" and owner.resident()["manifest"] is None, "failed rollback remained resident")
            try:
                with owner.request():
                    raise AssertionError("request admitted after failed rollback")
            except RuntimeError:
                fault["request_refused"] = True
            fault["engine_closed"] = (failed_engine._failed is True and not failed_engine.shared
                                      and failed_engine.wlm_d is None and not failed_engine.variants
                                      and failed_engine.active is None and owner._quarantined is None)
            require(fault["engine_closed"], "failed native engine retained device tensors/programs")
            require(record["native_counts"] == calls_before_failure, "runtime generation occurred after injected failure")
            fault.update(status="completed", no_runtime_generation_after_failure=True)
            record["final_diagnostics"] = owner.resident()["diagnostics"]
            owner.close()
            owner = None
            record["engine_closed"] = True
        require(record["native_outputs_verified"] and record["native_counts"] == dict(
            native_calls=1275, native_output_checks=1275, compilation_attempts=0), "native count/compile gate failed")
        require(len(record["generations"]) == 5, "expected five native generation gates")
        record.update(status="completed", native_evidence=True, finished_utc=time.time(),
                      validity="whole native request drain and actual partial rollback failure/unload gates passed; n=1 compliance")
        save()
        return 0
    except BaseException:
        record.update(status="failed", traceback=traceback.format_exc(), finished_utc=time.time(),
                      validity="failed; preserve raw partial evidence")
        save()
        return 1
    finally:
        if worker is not None and worker.is_alive():
            worker.join(timeout=15)
            if worker.is_alive():
                record["cleanup_error"] = "worker still running; no concurrent native close attempted"
        if owner is not None and (worker is None or not worker.is_alive()):
            try:
                owner.close()
                record["engine_closed"] = True
            except BaseException:
                record["close_error"] = traceback.format_exc()
        save()
        owner_stat = run.stat()
        for directory, dirs, files in os.walk(run, followlinks=False):
            for name in dirs + files:
                os.chown(Path(directory) / name, owner_stat.st_uid, owner_stat.st_gid, follow_symlinks=False)
            os.chown(directory, owner_stat.st_uid, owner_stat.st_gid, follow_symlinks=False)


if __name__ == "__main__":
    sys.exit(main())
