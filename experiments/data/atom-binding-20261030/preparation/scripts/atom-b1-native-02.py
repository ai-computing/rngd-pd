"""Native public ATOM B1 harness. Run only inside the root's allocated Job.

Required env: CCGRID_RUN, BINDING_FIXTURES, BINDING_KIT, ATOM_CHECKPOINT.
The binding factory is always the real public AtomSession factory.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_cases(path):
    path = Path(path).resolve()
    provider = json.loads(path.read_text())
    if provider.get("schema") != "onggi.binding-fixtures.v1" or not provider.get("cases"):
        raise ValueError("native fixtures require nonempty onggi.binding-fixtures.v1")
    cases = []
    for original in provider["cases"]:
        case = dict(original)
        for field in ("base_dir", "target_dir"):
            case[field] = str((path.parent / case[field]).resolve())
        if isinstance(case["switchplan"], str):
            case["switchplan"] = json.loads((path.parent / case["switchplan"]).read_text())
        cases.append(case)
    return cases


def summary(values):
    if not values:
        return None
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return dict(n=len(values), median=statistics.median(values), q1=q1, q3=q3, iqr=q3 - q1)


def allocated_device_id(expected, exposed, *, allow_broad=False):
    if not expected or (not set(expected) <= set(exposed) if allow_broad else sorted(expected) != sorted(exposed)):
        raise ValueError("ATOM exposure differs from init-container allocation")
    ids = {int(match.group(1)) for path in expected
           if (match := re.fullmatch(r"/dev/rbln(\d+)", path)) is not None}
    if len(ids) != 1:
        raise ValueError("exactly one allocated base /dev/rblnN device is required")
    return ids.pop()


def install_compile_guards(rebel, torch, record):
    record["compile_attempts"] = []
    record["guarded_compile_apis"] = []
    for module, prefix in ((rebel, "rebel"), (torch, "torch")):
        for name in dir(module):
            if (prefix == "rebel" and name.startswith("compile") or prefix == "torch" and name == "compile") and callable(getattr(module, name)):
                api = prefix + "." + name
                def forbidden(*args, _api=api, **kwargs):
                    record["compile_attempts"].append(_api)
                    raise RuntimeError("compilation forbidden during B1: " + _api)
                setattr(module, name, forbidden)
                record["guarded_compile_apis"].append(api)
    if "rebel.compile_from_torch" not in record["guarded_compile_apis"]:
        raise RuntimeError("public compiler guard was not installed")


class CountedRuntime:
    """Only observes calls; every run delegates to the actual native runtime."""
    def __init__(self, native, row):
        self.native, self.row = native, row
        row.update(native_runtime_type=type(native).__module__ + "." + type(native).__qualname__,
                   run_attempts=0, run_completed=0, run_seconds=[])

    def run(self, *args, **kwargs):
        self.row["run_attempts"] += 1
        started = time.monotonic()
        try:
            result = self.native.run(*args, **kwargs)
            self.row["run_completed"] += 1
            return result
        finally:
            self.row["run_seconds"].append(time.monotonic() - started)

    def __getattr__(self, name):
        return getattr(self.native, name)


def generated(session, row, prompts, n_new):
    session.rt = CountedRuntime(session.rt, row)
    row["compiled_type"] = type(session.compiled).__module__ + "." + type(session.compiled).__qualname__
    row["compiled_device_allocation"] = session.compiled.get_total_device_alloc()
    started = time.monotonic()
    tokens, ttft = session.generate(prompts, n_new)
    row.update(generation_s=time.monotonic() - started, tokens=tokens, ttft_s=ttft,
               median_decode_step_s=session.last_step_s, session_timings=dict(session.timings))
    assert row["run_completed"] == len(prompts[0]) + n_new - 1
    return tokens


def main():
    run = Path(os.environ["CCGRID_RUN"])
    run.mkdir(parents=True, exist_ok=True)
    record = dict(status="running", started_utc=time.time(), scope="native public ATOM ADR-0010 B1",
                  pod_uid=os.environ.get("POD_UID"), loads=[], repeats=[], milestones=[])
    def save():
        (run / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    def phase(name):
        record["phase"] = name
        record["milestones"].append(dict(phase=name, utc=time.time()))
        save()
    phase("initial_before_imports")
    owner = server = thread = None
    fresh = None
    try:
        phase("before_torch_import")
        import torch
        phase("after_torch_before_rebel_import")
        import rebel
        phase("after_rebel_import")
        from importlib.metadata import version
        from rngd_pd.swap import atom, atom_binding, binding_cli
        from rngd_pd.swap.atom import AtomSession, Tables
        from rngd_pd.swap.atom_binding import AtomBinding, _metadata

        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        assert os.environ.get("RBLN_WEIGHT_FREE", "0") in ("", "0", "off")
        expected = json.loads(Path("/run/swap/atom.json").read_text())
        if isinstance(expected, dict):
            expected = expected["devices"]
        exposed = sorted(str(p) for p in Path("/dev").glob("rbln*"))
        broad = os.environ.get("ATOM_ALLOW_BROAD_VISIBILITY") == "1"
        device = allocated_device_id(expected, exposed, allow_broad=broad)
        record.update(assignment=dict(sdk_device_id=device,
                      device_semantics="public create_runtime NPU device ID derived from the one allocated /dev/rblnN filename; no ordinal remapping assumed",
                      broad_visibility_allowed=broad, broader_devices_exposed=sorted(set(exposed) - set(expected)),
                      allocated_device_files=expected, actual_device_files=exposed,
                      rbln_environment={k: os.environ[k] for k in ("RBLN_DEVICES", "RBLN_DEVICE_IDS", "RBLN_WEIGHT_FREE") if k in os.environ}),
                      torch=torch.__version__, rebel=rebel.__version__, rebel_compiler=version("rebel-compiler"),
                      rebel_module=rebel.__file__, native_apis=["rebel.RBLNCompiledModel", "create_runtime(device=allocated_device_id,tensor_type=np)", "Runtime.run"],
                      vendor_internal_apis_used=[], experimental_constant_attachment=False)
        phase("after_assignment")
        install_compile_guards(rebel, torch, record)
        fixtures = Path(os.environ["BINDING_FIXTURES"]).resolve()
        kit = Path(os.environ["BINDING_KIT"]).resolve()
        cases = fixture_cases(fixtures)
        record["source_hashes"] = {str(p): sha(p) for p in (
            Path(__file__), Path(atom.__file__), Path(atom_binding.__file__), Path(binding_cli.__file__), kit, fixtures)}
        first = cases[0]
        graph = _metadata(first["base_dir"])[1]
        phase("before_graph_hash")
        started = time.monotonic()
        graph_sha = sha(graph)
        record["graph"] = dict(path=str(graph), sha256=graph_sha, bytes=graph.stat().st_size,
                               initial_hash_s=time.monotonic() - started)
        phase("after_graph_hash_before_tables")
        started = time.monotonic()
        tables = Tables(os.environ["ATOM_CHECKPOINT"])
        record["host_tables_s"] = time.monotonic() - started
        phase("after_tables")
        prompts, n_new = [[1, 42, 256, 1024, 7, 999, 2048, 13]], 8
        record.update(prompts=prompts, n_new=n_new)
        binary = Path("/host-bin/rbln-smi")
        if binary.is_file():
            command = [str(binary), "-i", str(device)]
            phase("before_public_rbln_smi")
            try:
                diagnostic = subprocess.run(command, capture_output=True, text=True, timeout=15)
                record["public_device_diagnostic"] = dict(command=command, returncode=diagnostic.returncode,
                    stdout=diagnostic.stdout[:16384], stderr=diagnostic.stderr[:16384],
                    output_truncated=len(diagnostic.stdout) > 16384 or len(diagnostic.stderr) > 16384)
            except (OSError, subprocess.TimeoutExpired) as error:
                record["public_device_diagnostic"] = dict(command=command, error=str(error))
            phase("after_public_rbln_smi")
        else:
            record["public_device_diagnostic"] = dict(skipped="/host-bin/rbln-smi absent")
        phase("before_fresh_session")
        row = record["fresh_reference"] = {}
        started = time.monotonic()
        fresh = AtomSession(str(graph), tables, device=device)
        row["load_s"] = time.monotonic() - started
        phase("after_fresh_session_before_generation")
        assert fresh.active.B == 1 and fresh.active.S == 64 and fresh.nl == 32
        reference = generated(fresh, row, prompts, n_new)
        fresh.close()
        fresh = None
        gc.collect()
        row["closed_before_binding_load"] = True
        phase("after_fresh_generation_and_close")

        class ObservedBinding(AtomBinding):
            """Observes the default public factory; no factory injection."""
            def _admission(self, artifact, manifest):
                start = time.monotonic()
                evidence = super()._admission(artifact, manifest)
                self.admission_s = time.monotonic() - start
                return evidence

            def load(self, artifact, manifest):
                row = dict(artifact=str(artifact), manifest=manifest, phase=getattr(self, "phase", "kit"))
                record["loads"].append(row)
                start = time.monotonic()
                try:
                    result = super().load(artifact, manifest)
                    row.update(load_including_hash_s=time.monotonic() - start,
                               admission_validation_and_hash_s=self.admission_s,
                               evidence=self.graph_evidence(), resident=result)
                    assert row["evidence"]["runtime_api"] == "public-AtomSession"
                    assert row["evidence"]["sha256"] == graph_sha, "compiled source changed"
                    with self.request() as session:
                        assert session.active.B == 1 and session.active.S == 64 and session.nl == 32
                        tokens = generated(session, row, prompts, n_new)
                    row["tokens_equal_fresh"] = tokens == reference
                    assert row["tokens_equal_fresh"], "native loaded tokens differ from fresh reference"
                    return result
                except BaseException:
                    row["error"] = traceback.format_exc()
                    self.close()
                    raise
                finally:
                    save()

        owner = ObservedBinding(device=device, tables=tables)
        socket_path = "/tmp/atom-b1-%s.sock" % os.getpid()
        server = binding_cli.create_server(owner, socket_path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        command = [sys.executable, str(kit), "--fixtures", str(fixtures), "--report", str(run / "kit-report.json"),
                   "--level", "B1", "--", sys.executable, "-m", "rngd_pd.swap.binding_cli", "--socket", socket_path]
        record["kit_command"] = command
        phase("before_kit")
        with (run / "kit.stdout").open("w") as stdout, (run / "kit.stderr").open("w") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=int(os.environ.get("BINDING_KIT_TIMEOUT", "1800")))
        record["kit_returncode"] = completed.returncode
        phase("after_kit")
        report = json.loads((run / "kit-report.json").read_text())
        assert completed.returncode == 0 and report["ok"]
        assert len(record["loads"]) == len(cases), "kit did not load every native fixture"
        deadline = time.monotonic() + float(os.environ.get("ATOM_REPEAT_DEADLINE_SECONDS", "600"))
        repeats = int(os.environ.get("ATOM_REPEAT_N", "5"))
        owner.phase = "repeat"
        phase("before_repeats")
        for index in range(repeats):
            if time.monotonic() >= deadline:
                record["repeat_deadline_reached"] = True
                break
            owner.load(first["base_dir"], first["switchplan"]["base_manifest"])
            record["repeats"].append(dict(record["loads"][-1], repeat=index))
        record["repeat_summary"] = {field: summary([r[field] for r in record["repeats"]]) for field in (
            "load_including_hash_s", "admission_validation_and_hash_s", "generation_s", "ttft_s", "median_decode_step_s")}
        phase("after_repeats_before_final_hash")
        started = time.monotonic()
        record["graph"]["final_sha256"] = sha(graph)
        record["graph"]["final_hash_s"] = time.monotonic() - started
        assert record["graph"]["final_sha256"] == graph_sha and not record["compile_attempts"]
        record.update(status="completed", compiled_source_reused=True, compile_attempt_count=0,
                      native_runtime_calls_completed=sum(r["run_completed"] for r in record["loads"]) + record["fresh_reference"]["run_completed"])
        phase("completed")
    except BaseException:
        record.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        try:
            if server is not None:
                server.shutdown()
                if thread is not None:
                    thread.join(timeout=5)
                server.server_close()
            if owner is not None:
                owner.close()
            if fresh is not None:
                fresh.close()
        except BaseException:
            record.update(status="failed", cleanup_error=traceback.format_exc())
            raise
        finally:
            record["finished_utc"] = time.time()
            try:
                save()
            finally:
                identity = run.stat()
                for directory, _, files in os.walk(run):
                    os.chown(directory, identity.st_uid, identity.st_gid)
                    for name in files:
                        os.chown(Path(directory) / name, identity.st_uid, identity.st_gid)


if __name__ == "__main__":
    main()
