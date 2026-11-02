"""Offline verifier tests; synthetic records are never native measurements."""
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


def test_historical_sources_verify_pinned_git_objects_after_runtime_evolves(verifier, tmp_path, monkeypatch):
    source, harness = b"historical runtime\n", b"historical harness\n"
    path = tmp_path / "src/runtime.py"
    path.parent.mkdir()
    path.write_bytes(b"new runtime implementation\n")
    path = tmp_path / "bench/ccgrid_b2.py"
    path.parent.mkdir()
    path.write_bytes(b"new harness implementation\n")
    monkeypatch.setattr(verifier, "HARNESS_SHA", verifier.digest(harness))
    def git(args, **kwargs):
        stdout = b""
        if "show" in args:
            key = args[-1]
            stdout = source if key == verifier.BASELINE + ":src/runtime.py" else harness
        return SimpleNamespace(returncode=0, stdout=stdout)
    monkeypatch.setattr(verifier.subprocess, "run", git)
    manifest = dict(source_commit=verifier.BASELINE, source_files={"src/runtime.py": verifier.digest(source)})
    provenance = dict(manifest, harness_sha256=verifier.HARNESS_SHA)
    result = verifier.verify_sources(manifest, provenance, tmp_path)
    assert result["git_objects_verified"] is True
    assert result["canonical_lf_sha256"]["src/runtime.py"] == verifier.digest(source)
    # Corrupt provenance must still fail against the actual pinned source bytes.
    manifest["source_files"]["src/runtime.py"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        verifier.verify_sources(manifest, provenance, tmp_path)


@pytest.fixture
def verifier():
    path = Path(__file__).parents[2] / "experiments/verify_b2_timing.py"
    assert path.is_file(), "offline B2 timing verifier is not implemented"
    spec = importlib.util.spec_from_file_location("verify_b2_timing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inclusive_quartiles_and_missing_samples_are_not_zero_observations(verifier):
    assert verifier.statistics_for([]) == dict(n=0, median=None, q1=None, q3=None, IQR=None, values=[])
    assert verifier.statistics_for([1, 2, 3, 4, 20]) == dict(
        n=5, median=3, q1=2, q3=4, IQR=2, values=[1, 2, 3, 4, 20])


def test_admitted_unit_must_cover_entire_changed_range(verifier):
    plan = {"files": [{"path": "weights", "ranges": [
        {"offset": 0, "length": 3, "kind": "same"},
        {"offset": 3, "length": 5, "kind": "new"}]}]}
    verifier.require_coverage(plan, [{"file": "weights", "offset": 2, "length": 7}])
    with pytest.raises(ValueError, match="coverage"):
        verifier.require_coverage(plan, [{"file": "weights", "offset": 3, "length": 4}])


def test_first_token_is_reconstructed_not_replaced_by_full_generation(verifier):
    row = {"timings": dict(trigger_perf_ns=1_000_000_000, generate_call_perf_ns=1_100_000_000,
        trigger_to_generate_call_s=0.1, prompt_to_first_token_s=0.2,
        trigger_to_first_token_s=0.3, trigger_to_generate_return_s=0.6)}
    assert verifier.verify_clock(row) == pytest.approx(0.3)
    row["timings"]["trigger_to_first_token_s"] = 0.6
    with pytest.raises(ValueError, match="clock"):
        verifier.verify_clock(row)


def test_pending_raw_may_only_differ_in_final_native_flags(verifier):
    raw = dict(status="completed", tokens=[[1]], native_evidence=False,
               native_gate_status="pending_final_run_observer")
    final = dict(raw, native_evidence=True, native_gate_status="verified_by_final_run_observer")
    verifier.compare_pending(raw, final)
    final["tokens"] = [[2]]
    with pytest.raises(ValueError, match="immutable"):
        verifier.compare_pending(raw, final)


def test_source_crlf_provenance_matches_lf_git_without_rewriting_raw_hashes(verifier):
    git = b"first\nsecond\n"
    measured = b"first\r\nsecond\r\n"
    expected = hashlib.sha256(measured).hexdigest()
    assert verifier.check_source_bytes(measured, expected, git) == hashlib.sha256(git).hexdigest()
    # A Linux source archive may retain LF while measured Windows source was CRLF.
    assert verifier.check_source_bytes(git, expected) == hashlib.sha256(git).hexdigest()
    with pytest.raises(ValueError, match="source"):
        verifier.check_source_bytes(b"changed\n", expected, git)


def test_source_archive_explicitly_reports_git_objects_unavailable(verifier, tmp_path):
    (tmp_path / "one.py").write_bytes(b"one\n")
    (tmp_path / "bench").mkdir()
    actual_repository = Path(verifier.__file__).parents[1]
    (tmp_path / "bench/ccgrid_b2.py").write_bytes((actual_repository / "bench/ccgrid_b2.py").read_bytes())
    files = {"one.py": hashlib.sha256(b"one\n").hexdigest()}
    result = verifier.verify_sources(dict(source_commit=verifier.BASELINE, source_files=files),
        dict(source_commit=verifier.BASELINE, source_files=files, harness_sha256=verifier.HARNESS_SHA), tmp_path)
    assert result["canonical_source_files_verified"] is True
    assert result["git_objects_verified"] is False
    assert result["git_object_verification"].startswith("unavailable")


def test_terminal_pod_and_cleanup_suffice_when_job_controller_status_lags(verifier, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    pod = dict(uid="pod-uid", phase="Succeeded", containers=[dict(name="run", restartCount=0,
                state=dict(terminated=dict(exitCode=0)))])
    data = {
        "job": dict(metadata=dict(name="cdx-b2-main-01"), spec=dict(backoffLimit=0, template=dict(spec=dict(
            restartPolicy="Never", containers=[dict(name="run", resources=dict(limits={"furiosa.ai/rngd": "1"}),
                command=["python", "script", "--config", "/inputs/main-01-config.json"])])))),
        "create": dict(name="cdx-b2-main-01", job_uid="job-uid"),
        "latest-status": dict(job_uid="job-uid", status=dict(active=1), pods=[pod]),
        "cleanup": dict(name="cdx-b2-main-01", job_uid="job-uid", job_absent=True, pods_absent=True, observations=[pod])}
    for suffix, doc in data.items():
        (control / ("cdx-b2-main-01-" + suffix + ".json")).write_text(json.dumps(doc))
    result = verifier.verify_job(tmp_path, "main-01", dict(provenance=dict(assignment=dict(pod_uid="pod-uid"))))
    assert result["terminal_pod_succeeded"] and result["cleanup_verified"]
    assert result["job_controller_completion_observed"] is False
    data["latest-status"]["pods"][0]["containers"][0]["state"]["terminated"]["exitCode"] = 1
    (control / "cdx-b2-main-01-latest-status.json").write_text(json.dumps(data["latest-status"]))
    with pytest.raises(ValueError, match="failed/restarted"):
        verifier.verify_job(tmp_path, "main-01", dict(provenance=dict(assignment=dict(pod_uid="pod-uid"))))


def test_archive_export_must_match_extracted_bytes(verifier, tmp_path):
    raw = b'{"status":"synthetic"}\n'
    sha = hashlib.sha256(raw).hexdigest()
    entry = dict(path="result.json", bytes=len(raw), sha256=sha)
    archive = gzip.compress(json.dumps({"files": [dict(entry, base64=base64.b64encode(raw).decode())]}).encode())
    (tmp_path / "small-evidence.json.gz").write_bytes(archive)
    (tmp_path / "result.json").write_bytes(raw)
    (tmp_path / "export-manifest.json").write_text(json.dumps(dict(files=[entry], archive=dict(
        bytes=len(archive), sha256=hashlib.sha256(archive).hexdigest(), files=1))))
    assert verifier.verify_export(tmp_path) == {"result.json"}
    (tmp_path / "result.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        verifier.verify_export(tmp_path)


def test_missing_main_is_explicit_incomplete_with_n0(verifier, tmp_path):
    (tmp_path / "package-manifest.json").write_text(json.dumps(dict(schema="synthetic-test", files=[])))
    result = verifier.verify(tmp_path)
    assert result["status"] == "incomplete"
    assert result["main"]["n_per_arm"] == 0
    assert result["main"]["arms"]["b2"]["trigger_to_first_token_s"]["median"] is None


def test_failed_attempt_does_not_summarize_completed_subset(verifier, tmp_path):
    root = tmp_path / "runs/main-01"
    root.mkdir(parents=True)
    path = root / "result.json"
    path.write_text(json.dumps(dict(status="failed", trials=[{"status": "completed"}] * 9)))
    (tmp_path / "package-manifest.json").write_text(json.dumps(dict(schema="synthetic-test", files=[dict(
        path="runs/main-01/result.json", bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())])))
    result = verifier.verify(tmp_path)
    assert result["status"] == "failed"
    assert result["main"]["observed_trials"] == 9
    assert result["main"]["n_per_arm"] == 0
    assert result["main"]["arms"]["reload"]["trigger_to_first_token_s"]["values"] == []


def test_real_package_separates_smoke_and_rejects_missing_or_rewritten_main(verifier, tmp_path):
    repository = Path(verifier.__file__).parents[1]
    original = repository / "experiments/data/b2-timing-20261102"
    if not original.exists():
        pytest.skip("immutable native package not present")
    root = tmp_path / "evidence"
    shutil.copytree(original, root)
    result = verifier.verify(root)
    assert result["verified"], result
    assert result["smoke"]["n_per_arm"] == 1 and result["smoke"]["excluded_from_main"]
    assert result["main"]["n_per_arm"] == 5 and not result["main"]["excluded_from_main"]
    assert result["main"]["arms"]["b2"]["trigger_to_first_token_s"]["median"] == pytest.approx(0.44844938095712855)
    path = root / "runs/main-01/result.json"
    native = json.loads(path.read_bytes())
    native["trials"].pop()
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="count/order"):
        verifier.verify_run(root, "main-01", "measure", repository)
    # Even a rewritten package index must not bypass the original archive/export.
    package_path = root / "package-manifest.json"
    package = json.loads(package_path.read_bytes())
    entry = next(entry for entry in package["files"] if entry["path"] == "runs/main-01/result.json")
    native["trials"].append(json.loads((original / "runs/main-01/result.json").read_bytes())["trials"][-1])
    native["trials"][0]["timings"]["trigger_to_first_token_s"] = 999
    path.write_text(json.dumps(native))
    entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    package_path.write_text(json.dumps(package))
    rejected = verifier.verify(root)
    assert rejected["status"] == "failed" and rejected["main"]["n_per_arm"] == 0
    assert "hash" in rejected["main"]["reason"]
