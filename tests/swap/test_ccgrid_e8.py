"""Real tiny CPU weights/graphs test the E8 control flow, never NPU evidence."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from threading import RLock
import time
from types import ModuleType

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from rngd_pd.swap.rngd import Checkpoint, SwappableDecode, _LAYER


@pytest.fixture
def harness():
    path = Path(__file__).parents[2] / "bench" / "ccgrid_e8.py"
    assert path.is_file(), "E8 single-switch harness is not implemented"
    spec = importlib.util.spec_from_file_location("ccgrid_e8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CpuModule(torch.nn.Module):
    @staticmethod
    def from_module(*args, **kwargs):
        raise AssertionError("fixture must not compile")

    def forward(self, inputs, weight):
        return inputs @ weight


class CpuDecode(SwappableDecode):
    fault = None

    def __init__(self, checkpoint, group, chip, cache_dir, allow_compile, host_workers):
        assert allow_compile is False and host_workers == 2
        assert torch.get_num_threads() == 4
        self._lock, self._failed, self._generation = RLock(), False, 0
        self._kv_owner = object()
        self.torch, self.BF, self.dev = torch, torch.bfloat16, "cpu"
        self.host_workers = host_workers
        self.group, self.nl, self.ngroups = group, 2, 1
        self.ck, self.variants, self.active = Checkpoint(checkpoint), {}, None
        self.eps, self.inv = 1e-5, np.ones(1, dtype=np.float32)
        self.timings = {}
        self._load_host_side()
        self._upload_shared()

    def _sync(self):
        pass

    def _graphs(self, variant):
        if self.fault == "compile" and variant.B == 2:
            CpuModule.from_module(None, None)
        return CpuModule(), CpuModule(), ["cache", "cache"]

    def swap(self, variant):
        result = super().swap(variant)
        if variant.B == 2:
            if self.fault == "reupload":
                self._upload_shared()
            elif self.fault == "replace_weight":
                self.shared[0]["wk"] = self.shared[0]["wk"].clone()
            elif self.fault == "replace_base_graph":
                next(st for v, st in self.variants.items() if v.B == 1)["m"] = CpuModule()
        return result

    def generate(self, prompts, n_new):
        started = time.perf_counter()
        ids = torch.tensor([p[-1] for p in prompts])
        outputs = [[] for _ in prompts]
        st = self.variants[self.active]
        for i in range(n_new):
            x = torch.nn.functional.one_hot(ids, num_classes=2).to(self.BF)
            x = st["m"](x, self.shared[0]["wk"][0])
            logits = st["lmh"](x, self.wlm_d.reshape(2, 2))
            ids = torch.argmax(logits, dim=1)
            if i == 0:
                ttft = time.perf_counter() - started
            for row, token in zip(outputs, ids.tolist()):
                row.append(token)
        return outputs, ttft


@pytest.fixture
def cpu(harness, tmp_path, monkeypatch):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "H", 2)
    monkeypatch.setattr(rngd, "HD", 2)
    monkeypatch.setattr(rngd, "NG_LM", 2)
    monkeypatch.setattr(rngd, "SwappableDecode", CpuDecode)
    monkeypatch.setattr(CpuDecode, "fault", None)
    sdk = ModuleType("rngd_pd.engines.fsdk")
    sdk.CompileModule, sdk.SDK = CpuModule, "CPU-test-adapter"
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    def check_cpu(output):
        assert isinstance(output, torch.Tensor) and output.device.type == "cpu"
    # Only the unavailable SDK/device boundary changes; original graph results
    # and production weight loading, prepare(), swap() and transforms stay real.
    monkeypatch.setattr(harness, "check_native_output", check_cpu)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = {"model.embed_tokens.weight": torch.eye(2, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        "lm_head.weight": torch.eye(2, dtype=torch.bfloat16)}
    for layer in range(2):
        for suffix, (_, transform) in _LAYER.items():
            weights[f"model.layers.{layer}.{suffix}"] = (torch.ones(2, dtype=torch.bfloat16)
                if transform == "norm" else torch.eye(2, dtype=torch.bfloat16))
    save_file(weights, str(checkpoint / "model.safetensors"))
    (checkpoint / "config.json").write_text(json.dumps({"hidden_size": 2}))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"base_tokens": [[0] * 8], "tokens": [[1] * 8] * 2,
        "base_prompts": [[0] * 8], "prompts": [[1] * 8] * 2}))
    return checkpoint, reference


def args(cpu, tmp_path, state="first-use"):
    checkpoint, reference = cpu
    return ["sample", "--ckpt", str(checkpoint), "--cache", str(tmp_path),
        "--reference", str(reference), "--chip", "0", "--base-variant", "B1-S16-g2",
        "--target-variant", "B2-S32-g2", "--graph-state", state,
        "--host-workers", "2", "--intra-op", "4", "--out", str(tmp_path / "sample.json")]


@pytest.mark.parametrize("state", ["first-use", "prepared"])
def test_real_switch_retains_weights_modules_and_distinguishes_actual_target_warmup(harness, cpu, tmp_path, state):
    argv = args(cpu, tmp_path, state)
    original_compile = inspect.getattr_static(CpuModule, "from_module")
    assert harness.main(argv) == 0
    row = json.loads((tmp_path / "sample.json").read_text())
    assert row["tokens"] == [[1] * 8] * 2
    assert row["base_tokens"] == [[0] * 8]
    assert row["tokens_equal_reference"] and row["base_tokens_equal_reference"]
    assert row["target_graph_prepared_before"] is (state == "prepared")
    assert row["actual_target_warmup"] is (state == "prepared")
    assert row["weight_buffers_retained"] and row["loaded_graph_modules_retained"]
    assert row["counts"]["shared_uploads"] == 1
    assert row["counts"]["compilation_attempts"] == 0
    assert row["native_calls_by_phase"]["target"] == 16
    assert row["logical_shared_weight_upload_bytes"] == 120
    assert len(row["weight_buffers_before"]) == 8
    assert row["timings"]["switch_to_first_token_s"] >= row["timings"]["swap_s"]
    assert row["graph"]["prepared"] is (state == "first-use")
    if state == "prepared":
        assert row["target_warmup_tokens"] == [[1] * 8] * 2
        assert row["barrier_base_tokens"] == [[0] * 8]
        assert row["target_warmup_tokens_equal_reference"]
        assert row["barrier_base_tokens_equal_reference"]
        assert row["native_calls_by_phase"]["preparation"] == 32
    else:
        assert row["native_calls_by_phase"].get("preparation", 0) == 0
    assert inspect.getattr_static(CpuModule, "from_module") is original_compile
    before = (tmp_path / "sample.json").read_bytes()
    with pytest.raises(FileExistsError):
        harness.main(argv)
    assert (tmp_path / "sample.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["compile", "reupload", "replace_weight", "replace_base_graph"])
def test_invalid_switch_saves_evidence_and_fails(harness, cpu, tmp_path, monkeypatch, fault):
    monkeypatch.setattr(CpuDecode, "fault", fault)
    original_compile = inspect.getattr_static(CpuModule, "from_module")
    assert harness.main(args(cpu, tmp_path)) == 1
    row = json.loads((tmp_path / "sample.json").read_text())
    assert row["status"] == "error" and row["base_tokens"] == [[0] * 8]
    assert row["error"]["message"]
    if fault == "compile":
        assert row["counts"]["compilation_attempts"] == 1
    if fault == "reupload":
        assert row["counts"]["shared_uploads"] == 2
    if fault == "replace_weight":
        assert row["tokens_equal_reference"] is True
        assert row["weight_buffers_retained"] is False
    assert inspect.getattr_static(CpuModule, "from_module") is original_compile


@pytest.mark.parametrize("state", ["first-use", "prepared"])
def test_target_reference_mismatch_retains_actual_tokens_and_fails(harness, cpu, tmp_path, state):
    _, reference = cpu
    document = json.loads(reference.read_text())
    document["tokens"] = [[0] * 8] * 2
    reference.write_text(json.dumps(document))
    assert harness.main(args(cpu, tmp_path, state)) == 1
    row = json.loads((tmp_path / "sample.json").read_text())
    prefix = "target_warmup_" if state == "prepared" else ""
    assert row[prefix + "tokens"] == [[1] * 8] * 2
    assert row[prefix + "tokens_equal_reference"] is False


def test_cleanup_error_preserves_successfully_generated_raw_tokens(harness, cpu, tmp_path, monkeypatch):
    def failed_close(engine):
        SwappableDecode.close(engine)
        raise RuntimeError("cleanup failed after native generation")
    monkeypatch.setattr(CpuDecode, "close", failed_close)
    assert harness.main(args(cpu, tmp_path)) == 1
    row = json.loads((tmp_path / "sample.json").read_text())
    assert row["tokens"] == [[1] * 8] * 2 and row["tokens_equal_reference"]
    assert row["timings"]["switch_to_first_token_s"] > 0
    assert row["error"]["message"] == "cleanup failed after native generation"


def test_base_reference_mismatch_gates_before_target_is_prepared(harness, cpu, tmp_path):
    checkpoint, reference = cpu
    document = json.loads(reference.read_text())
    document["base_tokens"] = [[1] * 8]
    reference.write_text(json.dumps(document))
    assert harness.main(args(cpu, tmp_path, "prepared")) == 1
    row = json.loads((tmp_path / "sample.json").read_text())
    assert row["base_tokens_equal_reference"] is False
    assert row["native_calls_by_phase"].get("preparation", 0) == 0


def test_original_cpu_graph_output_is_rejected_by_native_guard(harness):
    with pytest.raises(RuntimeError, match="furiosa"):
        harness.check_native_output((torch.ones(2),))


@pytest.mark.parametrize("mutation", ["batch", "missing_base", "context", "same", "group"])
def test_invalid_reference_or_variant_fails_before_engine(harness, cpu, tmp_path, mutation):
    _, reference = cpu
    document = json.loads(reference.read_text())
    argv = args(cpu, tmp_path)
    if mutation == "batch":
        document["tokens"] = [[1] * 8]
    elif mutation == "missing_base":
        del document["base_tokens"]
    elif mutation == "context":
        document["base_prompts"] = [[0] * 16]
    elif mutation == "same":
        argv[argv.index("--target-variant") + 1] = "B1-S16-g2"
    else:
        argv[argv.index("--target-variant") + 1] = "B2-S32-g4"
    reference.write_text(json.dumps(document))
    assert harness.main(argv) == 1
    row = json.loads((tmp_path / "sample.json").read_text())
    assert row["error"]["type"] == "ValueError"
    assert "counts" not in row
