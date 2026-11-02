"""Allocated-Job-only RNGD B2 conformance and fresh-reference generation gates.

Preparation of this script is CPU-only. Running main constructs real engines;
the owner must launch it only in its bounded, exclusively allocated RNGD Job.
No artifacts, graphs, references or fixtures are compiled/generated here.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


OLD_ROOT = Path("/home/gspark/workspace/codex_ccgrid_npu_20261030")
VARIANT = "B4-S256-g2"
FRESH_SOURCE = "f5fa13377a3d967ba3d4022fb73ae59b1c87441c"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def references(path):
    rows = read_json(path)
    for role in ("base", "sparse", "lm"):
        row = rows[role]
        require(row["status"] == "completed" and row["engine_closed"] is True,
                role + " fresh reference did not complete/close")
        require(row["source_commit"] == FRESH_SOURCE and row["source_dirty"] == "",
                role + " reference is not the independent frozen source")
        require(row["variant"] == VARIANT and row["continuation_length"] == 8,
                role + " reference variant or continuation differs")
        require(row["counts"] == {"native_calls": 255, "native_output_checks": 255,
                                  "compilation_attempts": 0}, role + " reference native counters differ")
        for key in ("tokens", "prompts"):
            value = row[key]
            require(isinstance(value, list) and len(value) == 4 and all(
                isinstance(lane, list) and len(lane) == 8
                and all(type(token) is int and token >= 0 for token in lane) for lane in value),
                role + " reference " + key + " must be four lanes of eight token IDs")
        require(row["prompts"] == rows["base"]["prompts"], "fresh reference prompts differ")
    return rows


def fixture_cases(path):
    provider = read_json(path)
    require(provider.get("schema") == "onggi.binding-fixtures.v1" and provider.get("cases"),
            "external fixture provider is missing cases")
    result, manifests = [], {}
    for case in provider["cases"]:
        require(case["expect_mode"] == "units", "RNGD native B2 fixtures must expect units")
        def resolve(value):
            return (Path(path).resolve().parent / value).resolve(strict=True)
        document = case["switchplan"]
        if isinstance(document, str):
            document = read_json(resolve(document))
        name = case["name"].lower()
        role = case.get("reference_role")
        if role is None:
            role = "base" if "identical" in name else "sparse" if "sparse" in name else "lm" if "lm" in name else None
        require(role in ("base", "sparse", "lm"), "fixture needs an explicit reference_role")
        for manifest, mapped_role in ((document["base_manifest"], "base"), (document["manifest"], role)):
            require(manifest not in manifests or manifests[manifest] == mapped_role,
                    "manifest maps to conflicting reference roles")
            manifests[manifest] = mapped_role
        result.append({"name": case["name"], "role": role, "base": resolve(case["base_dir"]),
                       "target": resolve(case["target_dir"]), "document": document})
    require({case["role"] for case in result} == {"base", "sparse", "lm"},
            "native fixture provider must include identical, sparse and LM cases")
    return result, manifests


def identities(engine):
    buffers = {f"shared/{group}/{slot}": tensor for group, slots in enumerate(engine.shared)
               for slot, tensor in slots.items()}
    buffers["lm_head"] = engine.wlm_d
    return {"buffers": {name: {"python_id": id(tensor), "data_ptr": tensor.data_ptr(),
                              "shape": list(tensor.shape), "device": str(tensor.device)}
                        for name, tensor in sorted(buffers.items())},
            "programs": {variant.name: {key: id(state[key]) for key in ("m", "lmh")}
                         for variant, state in engine.variants.items()}}


def main():
    root, run = Path(os.environ["CCGRID_ROOT"]), Path(os.environ["CCGRID_RUN"])
    source = Path(os.environ["CCGRID_SOURCE"])
    require(run.is_dir(), "CCGRID_RUN must be precreated by its owner")
    work = run / "work"
    work.mkdir(exist_ok=False)
    require(not (run / "result.json").exists(), "refusing to overwrite previous results")
    record = {"schema": "rngd-binding-native-b2-v1", "status": "running", "utc": time.time(),
              "pod_uid": os.environ["POD_UID"], "root": str(root), "source": str(source),
              "harness_sha256": sha(__file__), "generations": [], "generation_gates": False,
              "kit_ok": False, "silent_corruption_gate": {"status": "not-run"},
              "buffer_program_identity_scope": "public Tensor.data_ptr plus Python tensor/program object IDs",
              "native_evidence": False}
    owner, server, thread = None, None, None

    def save():
        data = json.dumps(record, indent=2) + "\n"
        (run / "operational.json").write_text(data)
        if record["status"] in ("completed", "failed"):
            (run / "result.json").write_text(data)

    save()
    try:
        fixture_path = Path(os.environ["BINDING_FIXTURES"]).resolve(strict=True)
        reference_path = Path(os.environ["BINDING_REFERENCES"]).resolve(strict=True)
        kit = Path(os.environ["BINDING_KIT"]).resolve(strict=True)
        if kit.is_dir():
            kit = kit / "conformance/binding/conformance.py"
        cache = Path(os.environ.get("BINDING_CACHE", str(OLD_ROOT / "cache/graphs"))).resolve(strict=True)
        refs, (cases, roles) = references(reference_path), fixture_cases(fixture_path)
        record.update(inputs={"fixtures": {"path": str(fixture_path), "sha256": sha(fixture_path)},
                              "references": {"path": str(reference_path), "sha256": sha(reference_path)},
                              "kit": {"path": str(kit), "sha256": sha(kit)}}, cache=str(cache),
                      references=refs)
        import torch
        from importlib.metadata import version
        from rngd_pd import devices
        from rngd_pd.engines.fsdk import DEV
        from rngd_pd.swap.binding_cli import create_server
        import rngd_pd.swap.rngd_binding as binding_module
        require(Path(binding_module.__file__).resolve() == (source / "src/rngd_pd/swap/rngd_binding.py").resolve(),
                "PYTHONPATH did not select the frozen new owner source")
        require(torch.__version__ == "2.10.0+cpu" and version("furiosa-torch") == "2026.4.0",
                "native runtime differs from the validated public-Torch probe")
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        cards = devices.allocated_cards()
        require(cards is not None and len(cards) == 1, "exactly one allocated RNGD is required")
        physical, chip = cards[0], devices.resolve_chip(-1)
        require(Path("/run/swap/rngd").read_text().strip() == "npu" + str(physical),
                "init/main physical assignments differ")
        assignment_file = Path(os.environ.get("BINDING_ASSIGNMENT", str(OLD_ROOT / "inputs/fixed-physical-card.json")))
        require(read_json(assignment_file)["host_index"] == physical, "frozen physical assignment changed")
        for role, reference in refs.items():
            require(reference["assignment"]["host_index"] == physical
                    and reference["assignment"]["torch_ordinal"] == chip,
                    role + " reference used a different physical/logical assignment")
        record.update(assignment={"pod_uid": os.environ["POD_UID"], "host_index": physical,
                                  "torch_ordinal": chip, "visible_cards": devices.visible_cards(),
                                  "allocation_uuid": devices.allocated_uuids(os.environ["POD_UID"], "run"),
                                  "frozen_assignment": str(assignment_file), "assignment_sha256": sha(assignment_file)},
                      packages={name: version(name) for name in ("torch", "numpy", "furiosa-torch", "furiosa-tcc")})
        paths = ["src/rngd_pd/swap/" + name + ".py" for name in ("rngd_binding", "binding_cli", "rngd", "delta", "variants")]
        paths += ["src/rngd_pd/devices.py", "src/rngd_pd/engines/fsdk.py", "bench/ccgrid_e7.py"]
        record["source_files"] = {name: sha(source / name) for name in paths}
        record["cache_files"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
                                 for path in sorted(cache.rglob("*.runnable"))]
        require(record["cache_files"], "no precompiled runnable cache files")
        spec = importlib.util.spec_from_file_location("binding_native_observer", source / "bench/ccgrid_e7.py")
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)

        class AuditedBinding(binding_module.RngdBinding):
            def generate_gate(self, operation, manifest, baseline=None):
                row = {"operation": operation, "manifest": manifest, "reference_role": roles[manifest],
                       "status": "running", "utc": time.time()}
                record["generations"].append(row)
                counts_before = dict(record["native_counts"])
                started = time.perf_counter()
                with self.request() as engine:
                    before = identities(engine)
                    row["identities_before"] = before
                    if baseline is not None:
                        row["serving_allocations_programs_retained"] = before == baseline
                        require(before == baseline, "weight/program identities changed during " + operation)
                    row["tokens"], row["prompt_ttft_s"] = engine.generate(refs["base"]["prompts"], 8)
                    engine._sync()
                    row["identities_after"] = identities(engine)
                    row["generation_preserved_identities"] = row["identities_after"] == before
                row["generation_s"] = time.perf_counter() - started
                row["native_counts"] = {key: record["native_counts"][key] - counts_before[key]
                                        for key in counts_before}
                row["tokens_equal_fresh_reference"] = row["tokens"] == refs[roles[manifest]]["tokens"]
                row["diagnostics"] = self.resident()["diagnostics"]
                save()
                require(row["tokens_equal_fresh_reference"], "fresh token mismatch after " + operation)
                require(row["generation_preserved_identities"], "generation changed weight/program identities")
                require(row["native_counts"] == refs[roles[manifest]]["counts"], "generation native call counts differ")
                row["status"] = "completed"
                save()

            def load(self, artifact, manifest):
                with self._lock:
                    result = super().load(artifact, manifest)
                    self.generate_gate("load", manifest)
                    return result

            def apply(self, update, target):
                with self._lock:
                    with self.request() as engine:
                        before = identities(engine)
                    result = super().apply(update, target)
                    self.generate_gate("apply", result["manifest"], before)
                    return result

            def rollback(self):
                with self._lock:
                    with self.request() as engine:
                        before = identities(engine)
                    result = super().rollback()
                    self.generate_gate("rollback", result["manifest"], before)
                    return result

        owner = AuditedBinding(cache_dir=str(cache), chip=chip, expected_device=f"{DEV}:{chip}")
        with observer.observe_native(chip, record):
            socket_path = work / "binding.sock"
            server = create_server(owner, socket_path)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            argv = [sys.executable, str(kit), "--level", "B2", "--fixtures", str(fixture_path),
                    "--report", str(work / "kit-report.json"), "--", sys.executable,
                    "-m", "rngd_pd.swap.binding_cli", "--socket", str(socket_path)]
            record["kit_argv"] = argv
            save()
            try:
                with (work / "kit.stdout.log").open("x") as stdout, (work / "kit.stderr.log").open("x") as stderr:
                    completed = subprocess.run(argv, stdout=stdout, stderr=stderr,
                                               timeout=int(os.environ.get("BINDING_KIT_TIMEOUT", "1200")))
                record["kit_returncode"] = completed.returncode
            finally:
                server.shutdown()
                thread.join(timeout=15)
                server.server_close()
                require(not thread.is_alive(), "binding server did not drain")
                thread, server = None, None
            kit_report = read_json(work / "kit-report.json")
            record["kit_report_sha256"] = sha(work / "kit-report.json")
            record["kit_ok"] = completed.returncode == 0 and kit_report.get("ok") is True
            save()
            require(record["kit_ok"], "neutral B2 conformance kit failed; preserve raw report")
            require(len(kit_report["cases"]) == len(cases), "kit omitted native fixture cases")
            require(len(record["generations"]) == 3 * len(cases)
                    and all(row["status"] == "completed" for row in record["generations"]),
                    "missing load/apply/rollback generation gates")

            # One public-copy fault: alter the first payload without changing
            # expected target bytes. Native D2H verification must reject it.
            fault = record["silent_corruption_gate"] = {"status": "running"}
            selected = next(case for case in cases if case["role"] == "sparse")
            require(owner.resident()["manifest"] == selected["document"]["base_manifest"],
                    "fault gate requires the kit-restored base")
            update = owner.plan(selected["document"])
            require(update["mode"] == "units" and update["units"], "fault gate needs nonempty native units")
            with owner.request() as engine:
                baseline = identities(engine)
            original_write, injected = owner._write, False
            def corrupt_once(unit, rows):
                nonlocal injected
                if not injected:
                    injected = True
                    rows = rows.clone()
                    octets = rows.contiguous().view(torch.uint8).reshape(-1)
                    octets[0] = int(octets[0]) ^ 1
                original_write(unit, rows)
            owner._write = corrupt_once
            try:
                try:
                    owner.apply(update, selected["target"])
                except RuntimeError as error:
                    fault["rejected_error"] = {"type": type(error).__name__, "message": str(error)}
                    require(injected and "readback" in str(error), "fault failed outside actual readback verification")
                else:
                    raise RuntimeError("silent corrupted device write was accepted")
            finally:
                owner._write = original_write
            require(owner.resident()["manifest"] == update["base_manifest"], "failed apply published target identity")
            owner.generate_gate("silent_corruption_restored", update["base_manifest"], baseline)
            fault.update(status="completed", actual_readback_rejected_corruption=True,
                         restored_native_base_tokens=True, identity_preserved=True,
                         diagnostics=owner.resident()["diagnostics"])
            record["generation_gates"] = all(row["status"] == "completed" for row in record["generations"])
            record["final_resident_before_close"] = owner.resident()
            owner.close()
            record["engine_closed"] = True
            owner = None
        require(record["native_counts"]["compilation_attempts"] == 0 and record["native_outputs_verified"],
                "native observation or compile prohibition failed")
        record.update(status="completed", validity="neutral B2 kit and independent fresh-token/readback/undo gates passed",
                      native_evidence=True, finished_utc=time.time())
        save()
        return 0
    except BaseException:
        record.update(status="failed", validity="failed; inspect preserved reports and logs",
                      traceback=traceback.format_exc(), finished_utc=time.time())
        save()
        return 1
    finally:
        if server is not None:
            try:
                server.shutdown()
                if thread is not None:
                    thread.join(timeout=15)
                server.server_close()
            except BaseException:
                record["server_cleanup_error"] = traceback.format_exc()
        if owner is not None:
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
