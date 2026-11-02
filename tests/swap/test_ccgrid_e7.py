"""CPU integration of E7 planning and real resident updates; no SDK/card."""
import importlib.util
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import subprocess
import sys
from threading import RLock
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

from rngd_pd.swap.delta import read_header
from rngd_pd.swap.rngd import Checkpoint, SwappableDecode


@pytest.fixture
def harness():
    path = Path(__file__).parents[2] / "bench" / "ccgrid_e7.py"
    assert path.is_file(), "E7 single-sample harness has not been implemented"
    spec = importlib.util.spec_from_file_location("ccgrid_e7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_native_boundary(harness, monkeypatch):
    @contextmanager
    def cpu_observer(chip, row):
        row["native_observation"] = "CPU test adapter; no native SDK/device evidence"
        row["native_counts"] = dict(native_calls=0, native_output_checks=0, compilation_attempts=0)
        yield
    # CPU integration keeps production updates but explicitly replaces the SDK
    # observer boundary; zero native calls cannot be mistaken for NPU evidence.
    monkeypatch.setattr(harness, "observe_native", cpu_observer, raising=False)


@pytest.fixture
def case(tmp_path):
    name = "model.layers.0.self_attn.k_proj.weight"
    other = "model.layers.1.self_attn.k_proj.weight"
    base, target = tmp_path / "base", tmp_path / "target"
    values = {name: torch.zeros(8, 2, dtype=torch.bfloat16),
              other: torch.ones(8, 2, dtype=torch.bfloat16)}
    for directory in (base, target):
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"hidden_size": 2}))
        save_file(values, str(directory / "model.safetensors"))
    values[name][1, 0] = 10
    save_file(values, str(target / "model.safetensors"))
    header = read_header(target / "model.safetensors")
    ranges = tmp_path / "ranges.json"
    ranges.write_text(json.dumps({"files": [{"path": "model.safetensors",
        "base": "base", "size": (target / "model.safetensors").stat().st_size,
        "changed": [[header[name]["start"] + 4, 4]]}]}))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"tokens": [[1, 1, 1]], "prompts": [[0]],
                                    "base_tokens": [[0, 0, 0]], "variant": "B1-S16-g2"}))
    return base, target, ranges, reference


class CpuDecode(SwappableDecode):
    """Keep production transforms/apply_delta; replace only device/graph work."""
    def __init__(self, checkpoint, group, chip, cache_dir, allow_compile, host_workers):
        assert allow_compile is False
        assert host_workers == 2 and torch.get_num_threads() == 4
        self._lock, self._failed, self._generation = RLock(), False, 0
        self._kv_owner = object()
        self.torch, self.BF, self.dev = torch, torch.bfloat16, "cpu"
        self.group, self.nl, self.ngroups = group, 2, 1
        self.ck = Checkpoint(checkpoint)
        self.shared = [{"wk": self._group_slot_host(0, "wk")}]
        self.embed = self.norm = self.wlm_d = None
        self.n1 = self.n2 = []
        self.lm_name, self.variants, self.timings = "lm_head.weight", {}, {}

    def _sync(self):
        pass

    def swap(self, variant):
        self.active = variant
        return {"graph_source": ["cache", "cache"]}

    def generate(self, prompts, n_new):
        # A real tiny projection and greedy choice, derived from resident weights.
        logits = self.shared[0]["wk"][0].float().t() @ torch.tensor([1., 0.])
        token = int(torch.argmax(logits))
        return [[token] * n_new for _ in prompts], 0.001


@pytest.mark.parametrize("mode,written", [("inplace", 4), ("replace", 64)])
def test_sample_uses_actual_changed_weights_and_counts_logical_payload(harness, case, tmp_path, monkeypatch, cpu_native_boundary, mode, written):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "SwappableDecode", CpuDecode)
    base, target, ranges, reference = case
    out = tmp_path / "sample.json"
    argv = ["sample", "--ckpt", str(base), "--target", str(target), "--ranges", str(ranges),
            "--variant", "B1-S16-g2", "--cache", str(tmp_path), "--chip", "0",
            "--out", str(out), "--mode", mode, "--reference", str(reference)]
    assert harness.main(argv) == 0
    row = json.loads(out.read_text())
    assert row["tokens"] == [[1, 1, 1]]
    assert row["base_tokens"] == [[0, 0, 0]]
    assert row["tokens_equal_reference"] is True
    assert row["base_tokens_equal_reference"] is True
    assert row["logical_device_written_bytes"] == written
    assert row["apply"]["mode"] == ("group" if mode == "replace" else "inplace")
    assert row["plan"]["tensors"] == 1
    assert next(iter(row["plan_tensors"].values()))["rows"] == [[1, 2]]
    assert row["timings"]["update_to_first_token_s"] >= row["timings"]["apply_s"]
    assert row["metadata"]["invocation_argv"][2:] == argv
    assert Path(row["metadata"]["invocation_argv"][1]).name == "ccgrid_e7.py"
    assert row["metadata"]["source"]["script_sha256"]
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        harness.main(argv)
    assert out.read_bytes() == before


def test_mismatched_tokens_are_saved_and_fail(harness, case, tmp_path, monkeypatch, cpu_native_boundary):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "SwappableDecode", CpuDecode)
    base, target, ranges, reference = case
    reference.write_text(json.dumps({"tokens": [[2]], "prompts": [[0]]}))
    out = tmp_path / "failed.json"
    assert harness.main(["sample", "--ckpt", str(base), "--target", str(target),
        "--ranges", str(ranges), "--variant", "B1-S16-g2", "--cache", str(tmp_path),
        "--chip", "0", "--out", str(out), "--mode", "inplace", "--reference", str(reference)]) == 1
    row = json.loads(out.read_text())
    assert row["tokens"] == [[1]] and row["tokens_equal_reference"] is False
    assert row["base_tokens_equal_reference"] is None


def test_base_parity_failure_is_saved_and_gates_sample(harness, case, tmp_path, monkeypatch, cpu_native_boundary):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "SwappableDecode", CpuDecode)
    base, target, ranges, reference = case
    reference.write_text(json.dumps({"tokens": [[1]], "base_tokens": [[2]], "prompts": [[0]]}))
    out = tmp_path / "base-mismatch.json"
    assert harness.main(["sample", "--ckpt", str(base), "--target", str(target),
        "--ranges", str(ranges), "--variant", "B1-S16-g2", "--cache", str(tmp_path),
        "--chip", "0", "--out", str(out), "--mode", "inplace", "--reference", str(reference)]) == 1
    row = json.loads(out.read_text())
    assert row["tokens_equal_reference"] is True
    assert row["base_tokens"] == [[0]] and row["base_tokens_equal_reference"] is False


@pytest.mark.parametrize("failure", ["close", "apply", "base_sync", "apply_sync", "target_sync", "generate"])
def test_native_phase_failure_preserves_completed_evidence(harness, case, tmp_path, monkeypatch, cpu_native_boundary, failure):
    from rngd_pd.swap import rngd
    class FailingDecode(CpuDecode):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.generated = self.after_apply = 0
        def generate(self, *args):
            self.generated += 1
            if failure == "generate" and self.generated == 2:
                raise RuntimeError("target generation failed")
            return super().generate(*args)
        def apply_delta(self, *args, **kwargs):
            if failure == "apply":
                raise RuntimeError("apply failed")
            result = super().apply_delta(*args, **kwargs)
            self.after_apply = 1
            return result
        def _sync(self):
            if failure == "base_sync" and self.generated == 1:
                raise RuntimeError("base sync failed")
            if failure == "apply_sync" and self.after_apply:
                raise RuntimeError("apply sync failed")
            if failure == "target_sync" and self.generated == 2:
                raise RuntimeError("target sync failed")
        def close(self):
            # Secondary cleanup errors must not replace the original failure.
            if failure in ("close", "apply"):
                raise RuntimeError("close failed")
            super().close()
    monkeypatch.setattr(rngd, "SwappableDecode", FailingDecode)
    base, target, ranges, reference = case
    out = tmp_path / "phase-error.json"
    assert harness.main(["sample", "--ckpt", str(base), "--target", str(target),
        "--ranges", str(ranges), "--variant", "B1-S16-g2", "--cache", str(tmp_path),
        "--chip", "0", "--out", str(out), "--mode", "inplace", "--reference", str(reference)]) == 1
    row = json.loads(out.read_text())
    assert row["status"] == "error"
    assert row["base_tokens"] == [[0, 0, 0]] and row["base_tokens_equal_reference"]
    assert row["timings"]["base_prompt_ttft_s"] == 0.001
    if failure in ("close", "apply_sync", "target_sync", "generate"):
        assert row["apply"]["uploaded_bytes"] == row["logical_device_written_bytes"] == 4
    if failure in ("close", "target_sync"):
        assert row["tokens"] == [[1, 1, 1]] and row["tokens_equal_reference"]
        assert row["timings"]["update_to_first_token_s"] > 0
        assert row["timings"]["prompt_ttft_s"] == 0.001
    if failure == "close":
        assert row["timings"]["apply_s"] > 0
        assert row["error"]["message"] == row["close_error"]["message"] == "close failed"
    if failure == "apply":
        assert row["error"]["message"] == "apply failed"
        assert row["close_error"]["message"] == "close failed"


def test_git_metadata_uses_exact_command_scoped_safe_directory_and_real_hash(harness, monkeypatch):
    root = Path(harness.__file__).resolve().parents[1]
    original_run = subprocess.run
    expected = original_run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    commands = []
    def observe(command, *args, **kwargs):
        if command[0] == "git":
            commands.append(command)
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", observe)
    result = harness.metadata(["sample"])
    assert result["source"]["commit"] == expected
    assert len(commands) == 2
    assert all(command[:5] == ["git", "-c", "safe.directory=" + str(root), "-C", str(root)]
               for command in commands)


def test_native_observer_preserves_original_cpu_output_and_rejects_fallback(harness, monkeypatch):
    class CpuCompiled(torch.nn.Module):
        @staticmethod
        def from_module(*args):
            raise AssertionError("unguarded compiler")
        def forward(self, value):
            return {"nested": (value * 3,)}
    sdk = ModuleType("rngd_pd.engines.fsdk")
    sdk.CompileModule, sdk.DEV, sdk.SDK = CpuCompiled, "furiosa", "CPU-boundary-test"
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    row = {}
    original = inspect.getattr_static(CpuCompiled, "from_module")
    tensor = torch.tensor([2.0])
    with pytest.raises(RuntimeError, match="furiosa:3"):
        with harness.observe_native(3, row):
            CpuCompiled()(tensor)
    assert tensor.tolist() == [2.0]
    assert row["native_counts"]["native_calls"] == 1
    assert row["native_counts"]["native_output_checks"] == 0
    assert inspect.getattr_static(CpuCompiled, "from_module") is original
    with pytest.raises(RuntimeError, match="compilation"):
        with harness.observe_native(3, row):
            CpuCompiled.from_module(None)
    assert row["native_counts"]["compilation_attempts"] == 1


def test_native_output_check_rejects_different_allocated_ordinal(harness):
    # Real CPU storage with a controlled SDK-facing device descriptor; no GPU
    # allocation. Matching type alone must not accept another card's output.
    class WrongOrdinal(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:1")
    output = torch.tensor([1.0]).as_subclass(WrongOrdinal)
    with pytest.raises(RuntimeError, match="cuda:0"):
        harness.check_native_output({"nested": [output]}, "cuda", 0)


@pytest.mark.parametrize("doc", [{"tokens": []}, {"tokens": [[True]]},
    {"tokens": [[1], [1, 2]]}, {"tokens": [[1]], "prompts": [[]]},
    {"tokens": [[1]], "base_tokens": [[1, 2]]}])
def test_reference_rejects_invalid_token_shapes_before_opening_device(harness, doc, tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        harness.load_reference(path, "B1-S16-g2")


def test_worker_failure_is_attempted_once_without_hidden_retry(harness, tmp_path):
    count = tmp_path / "attempts"
    out = tmp_path / "failed-worker.json"
    command = [sys.executable, "-c",
        "import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.write_text(p.read_text()+'x' if p.exists() else 'x'); pathlib.Path(sys.argv[2]).write_text(json.dumps({'tokens': [[max(range(2), key=[0,9].__getitem__)]], 'status': 'error'})); sys.exit(3)", str(count), str(out)]
    with pytest.raises(subprocess.CalledProcessError) as failed:
        harness.launch_worker(command, out)
    assert count.read_text() == "x"
    assert failed.value.worker_record["result"]["tokens"] == [[1]]
    assert failed.value.worker_record["returncode"] == 3


def test_failed_child_raw_json_survives_parent_error(harness, case, tmp_path, monkeypatch):
    base, target, ranges, reference = case
    def fail(command, out):
        error = subprocess.CalledProcessError(7, command)
        error.worker_record = dict(argv=command, returncode=7,
                                   result=dict(tokens=[[5]], status="error"))
        raise error
    monkeypatch.setattr(harness, "launch_worker", fail)
    out = tmp_path / "failed-parent.json"
    assert harness.main(["sample", "--ckpt", str(base), "--target", str(target),
        "--ranges", str(ranges), "--variant", "B1-S16-g2", "--cache", str(tmp_path),
        "--chip", "0", "--out", str(out), "--mode", "restart", "--reference", str(reference)]) == 1
    row = json.loads(out.read_text())
    assert row["failed_worker"]["result"]["tokens"] == [[5]]
    assert row["failed_worker"]["returncode"] == 7


def test_restart_waits_for_actual_base_exit_before_fresh_target_process(harness, case, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import os
    base, target, ranges, reference = case
    args = SimpleNamespace(ckpt=str(base), target=str(target), ranges=str(ranges),
        variant="B1-S16-g2", cache=str(tmp_path), chip=0, reference=str(reference),
        host_workers=2, intra_op=4)
    launch = harness.launch_worker
    def cpu_worker(command, out):
        phase = command[command.index("--_phase") + 1]
        # Real subprocesses, actual tiny CPU projection/argmax, no native SDK.
        script = """
import json, os, sys, time
import torch
phase, out = sys.argv[1:]
w = torch.zeros(8, 2)
if phase == 'target':
    w[1, 0] = 10
tokens = [[int(torch.argmax(w @ torch.tensor([1., 0.])))]]
now = time.perf_counter_ns()
row = dict(metadata=dict(pid=os.getpid()), timings={}, prompts=[[0]])
if phase == 'base':
    row.update(base_tokens=tokens, base_ready_perf_ns=now)
else:
    row.update(tokens=tokens, tokens_equal_reference=tokens == [[1]],
               first_token_perf_ns=now, logical_device_written_bytes=32)
with open(out, 'x') as f:
    json.dump(row, f)
"""
        return launch([sys.executable, "-c", script, phase, str(out)], out)
    monkeypatch.setattr(harness, "launch_worker", cpu_worker)
    row = harness.restart_sample(args)
    base_worker, target_worker = row["workers"]["base"], row["workers"]["target"]
    assert row["tokens"] == [[1]] and row["base_tokens"] == [[0]]
    assert len({base_worker["result"]["metadata"]["pid"],
                target_worker["result"]["metadata"]["pid"], os.getpid()}) == 3
    assert target_worker["parent_launch_perf_ns"] >= base_worker["parent_observed_exit_perf_ns"]
    assert row["timings"]["update_to_first_token_s"] >= (
        row["timings"]["base_ready_to_observed_exit_s"] +
        row["timings"]["parent_launch_to_first_token_s"])
