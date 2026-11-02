"""CPU-only harness validation; these tests provide no native device evidence."""
import importlib.util
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def harness():
    path = Path(__file__).parents[2] / "bench/ccgrid_b2.py"
    assert path.is_file(), "current B2 timing harness is not implemented"
    spec = importlib.util.spec_from_file_location("ccgrid_b2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_token_excludes_remaining_continuation(harness):
    # A 4-second continuation must not be substituted for the 0.2-second TTFT.
    assert harness.first_token_seconds(1_000_000_000, 1_100_000_000, 0.2) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        harness.first_token_seconds(1_000_000_000, 900_000_000, 0.2)
    with pytest.raises(ValueError):
        harness.first_token_seconds(0, 1, float("nan"))


def test_measure_order_alternates_pairs_and_smoke_is_separate(harness):
    assert list(harness.trial_order("smoke")) == [(1, "b2"), (1, "reload")]
    assert list(harness.trial_order("measure")) == [
        (1, "b2"), (1, "reload"), (2, "reload"), (2, "b2"),
        (3, "b2"), (3, "reload"), (4, "reload"), (4, "b2"),
        (5, "b2"), (5, "reload")]


def test_append_journal_preserves_partial_failure_and_refuses_reuse(harness, tmp_path):
    run = tmp_path / "attempt"
    with harness.Journal(run) as journal:
        journal.write({"status": "running", "operation": "plan"})
        journal.write({"status": "failed", "error": "readback failed"})
    records = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert [record["status"] for record in records] == ["running", "failed"]
    with pytest.raises(FileExistsError):
        harness.Journal(run)
    assert len((run / "events.jsonl").read_text().splitlines()) == 2


def test_config_check_is_stdlib_only_and_rejects_missing_paths(harness, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"source": str(tmp_path / "missing")}))
    completed = subprocess.run([sys.executable, "-S", str(Path(harness.__file__)),
                                "--config", str(config), "--check-config"],
                               text=True, capture_output=True)
    assert completed.returncode == 1
    assert "torch" not in completed.stderr
    assert not (tmp_path / "events.jsonl").exists()


@pytest.fixture
def valid_config(harness, tmp_path):
    source, base, target, cache = [tmp_path / name for name in ("source", "base", "target", "cache")]
    for directory in (source, base, target, cache):
        directory.mkdir()
    source_files = {}
    for name in harness.RUNTIME_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CPU validation placeholder, never imported\n")
        source_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for directory in (base, target):
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"CPU artifact placeholder")
    (cache / "graph.runnable").write_bytes(b"CPU cache placeholder")
    document = {"schema": "onggi.switchplan.v1", "base_manifest": "sha256:" + "0" * 64,
                "manifest": "sha256:" + "1" * 64, "files": []}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document))
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({"schema": "onggi.binding-fixtures.v1", "cases": [{
        "name": "native-interior-projection", "expect_mode": "units", "base_dir": str(base),
        "target_dir": str(target), "switchplan": str(plan)}]}))
    references = tmp_path / "references.json"
    refs = {}
    for role, directory, token in (("base", base, 0), ("sparse", target, 1)):
        refs[role] = dict(status="completed", engine_closed=True, source_commit=harness.FRESH_SOURCE,
            source_dirty="", variant="B4-S256-g2", continuation_length=8, checkpoint=str(directory),
            prompts=[[0] * 8 for _ in range(4)], tokens=[[token] * 8 for _ in range(4)],
            counts=dict(native_calls=255, native_output_checks=255, compilation_attempts=0))
    references.write_text(json.dumps(refs))
    files = [*base.iterdir(), *target.iterdir(), *cache.iterdir(), fixtures, references, plan]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema="ccgrid-b2-inputs-v1", source_commit="frozen-source",
        source_files=source_files, files=[dict(path=str(path), bytes=path.stat().st_size,
                                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files])))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(source=str(source), cache=str(cache), fixtures=str(fixtures),
        references=str(references), input_manifest=str(manifest), run=str(tmp_path / "output"),
        source_commit="frozen-source", phase="smoke")))
    return config


def test_valid_config_checks_without_site_packages_or_creating_output(harness, valid_config):
    completed = subprocess.run([sys.executable, "-S", str(Path(harness.__file__)),
                                "--config", str(valid_config), "--check-config"],
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["native_evidence"] is False
    assert not (valid_config.parent / "output").exists()


def test_config_rejects_missing_artifact_hash_and_changed_source(harness, valid_config):
    config = json.loads(valid_config.read_text())
    manifest_path = Path(config["input_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [entry for entry in manifest["files"]
                         if not entry["path"].endswith("target/model.safetensors")
                         and not entry["path"].endswith("target\\model.safetensors")]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits"):
        harness.check_config(valid_config)
    (Path(config["source"]) / harness.RUNTIME_FILES[0]).write_text("changed runtime")
    with pytest.raises(ValueError, match="source hash"):
        harness.check_config(valid_config)


class CpuOwner:
    """Orchestration double only; no production runtime or device evidence."""
    def __init__(self, prepared, counts, fail=False):
        self.prepared, self.counts, self.fail = prepared, counts, fail
        self._metrics = {"readback_calls": 0, "readback_payload_bytes": 0}
        self.engine = self
        self.shared, self.wlm_d, self.variants = [], self, {}
        self.shape, self.device = [1], "cpu-test-only"
        self.manifest = None

    def data_ptr(self):
        return 1

    def load(self, path, manifest):
        self.manifest = manifest
        return self.resident()

    @contextmanager
    def request(self):
        yield self

    def generate(self, prompts, length):
        self.counts["native_calls"] += 255
        self.counts["native_output_checks"] += 255
        role = "base" if self.manifest == self.prepared["document"]["base_manifest"] else "sparse"
        return self.prepared["references"][role]["tokens"], 0.2

    def _sync(self):
        pass

    def resident(self):
        return {"manifest": self.manifest, "diagnostics": dict(self._metrics)}

    def plan(self, document):
        return dict(mode="units", units=[{"length": 4}], bytes=4)

    def apply(self, update, target):
        self._metrics.update(readback_calls=2, readback_payload_bytes=128)
        if self.fail:
            raise RuntimeError("CPU injected readback failure")
        self.manifest = self.prepared["document"]["manifest"]
        return {"ok": True, "bytes_written": 4}


@pytest.mark.parametrize("arm", ["b2", "reload"])
def test_trial_gates_target_and_records_full_path_without_fake_native_evidence(harness, valid_config, tmp_path, arm):
    prepared = harness.check_config(valid_config)
    counts = dict(native_calls=0, native_output_checks=0, compilation_attempts=0)
    owner = CpuOwner(prepared, counts)
    with harness.Journal(tmp_path / "cpu-trial") as journal:
        row = harness.run_trial(owner, prepared, 1, arm, {"native_counts": counts}, journal,
                                {"evidence": "CPU orchestration test only"}, lambda engine: 64)
    assert row["native_evidence"] is False
    assert row["status"] == "completed"
    assert row["tokens"] == [[1] * 8 for _ in range(4)]
    assert row["logical_device_written_bytes"] == (4 if arm == "b2" else 64)
    assert row["readback_payload_bytes_during_trigger"] == (128 if arm == "b2" else 0)
    assert set(row["operations"]) == {"prepare_base_outside_trigger", "plan", arm if arm == "reload" else "apply",
                                       "sync", "generate_full_continuation"}
    assert row["timings"]["trigger_to_first_token_s"] >= 0.2


def test_trial_preserves_failed_apply_diagnostics(harness, valid_config, tmp_path):
    prepared = harness.check_config(valid_config)
    counts = dict(native_calls=0, native_output_checks=0, compilation_attempts=0)
    with harness.Journal(tmp_path / "failed-trial") as journal:
        with pytest.raises(RuntimeError, match="injected readback"):
            harness.run_trial(CpuOwner(prepared, counts, fail=True), prepared, 1, "b2",
                             {"native_counts": counts}, journal, {}, lambda engine: 64)
    row = json.loads((tmp_path / "failed-trial/pair-01-b2.json").read_text())
    assert row["status"] == "failed" and row["native_evidence"] is False
    assert row["operations"]["apply"]["diagnostics"]["readback_payload_bytes"] == 128
    assert "generate_full_continuation" not in row["operations"]
