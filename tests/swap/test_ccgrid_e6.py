"""E6 rejects invalid evidence and preserves an independently owned trial."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).parents[2] / "bench/ccgrid_e6.py"


def cli(config, out, **kwargs):
    return subprocess.run([sys.executable, str(SCRIPT), "trial", "--config", str(config),
                           "--out", str(out)], capture_output=True, text=True, **kwargs)


def test_missing_actual_pod_timestamp_is_retained_failure_without_work(tmp_path):
    pod = tmp_path / "pod.json"
    pod.write_text(json.dumps({"metadata": {"name": "actual-pod", "uid": "actual-uid"}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(trial_id="empty-ts", arm="resident", cache_state="empty",
        pod_metadata_file=str(pod), work_dir=str(tmp_path / "owned"))))
    out = tmp_path / "sample.json"
    result = cli(config, out)
    assert result.returncode == 1, result.stderr
    data = json.loads(out.read_text())
    assert data["status"] == "failed" and data["validity"] == "invalid"
    assert "creationTimestamp" in data["error"]["message"]
    assert data["metrics"] == {}
    assert not (tmp_path / "owned").exists()


def test_existing_result_is_never_overwritten(tmp_path):
    out = tmp_path / "sample.json"
    out.write_text("preserved")
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = cli(config, out)
    assert result.returncode != 0
    assert out.read_text() == "preserved"


def test_existing_work_directory_is_never_consumed(tmp_path):
    work = tmp_path / "owned"
    work.mkdir()
    valuable = work / "existing"
    valuable.write_text("keep")
    pod = tmp_path / "pod.json"
    pod.write_text(json.dumps({"metadata": {"name": "pod", "uid": "uid",
        "creationTimestamp": "2026-01-01T00:00:00Z"}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(trial_id="occupied", arm="delta", cache_state="empty",
        pod_metadata_file=str(pod), clock_basis="same UTC host", work_dir=str(work))))
    out = tmp_path / "sample.json"
    assert cli(config, out).returncode == 1
    assert valuable.read_text() == "keep"
    assert json.loads(out.read_text())["metrics"] == {}


@pytest.fixture
def harness():
    assert SCRIPT.exists(), "E6 harness has not been implemented"
    spec = importlib.util.spec_from_file_location("ccgrid_e6_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_token_partition_uses_first_token_not_generation_tail(harness):
    phases = harness.first_token_partition(1_000_000_000, 3_000_000_000,
        {"generation_start_perf_ns": 5_000_000_000, "first_token_perf_ns": 5_250_000_000,
         "generation_done_perf_ns": 20_000_000_000})
    assert phases == {"artifact_delivery_s": 2.0, "delivery_to_generation_start_s": 2.0,
                      "target_prompt_to_first_token_s": 0.25}


def test_base_aware_pull_passes_owned_base_and_preserves_raw_process_failure(harness, tmp_path, monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 3, '{"partial": true}', 'delivery failed')
    monkeypatch.setattr(harness.subprocess, "run", run)
    record = {}
    with pytest.raises(RuntimeError, match="pull"):
        harness.pull(dict(onggi_binary="onggi", store="store", target_ref="target",
            base_ref="base"), tmp_path / "target", tmp_path / "node-cache", record,
            base_dir=tmp_path / "consumable")
    assert calls[0] == ["onggi", "pull", "--store", "store", "--cache", str(tmp_path / "node-cache"),
                        "--json", "--base", "base", "--base-dir", str(tmp_path / "consumable"),
                        "target", str(tmp_path / "target")]
    assert record["returncode"] == 3 and record["stdout"] == '{"partial": true}'
    assert record["stderr"] == "delivery failed"


SDK_BOUNDARY = '''
import json, sys, types
import torch
sdk = types.ModuleType("rngd_pd.engines.fsdk")
sdk.SDK, sdk.NEW = "2026.4.0", True
sdk.dev_str = lambda chip: "cpu"
sdk.open_device = lambda *a: None
sdk.sync = lambda *a: None
class CompileModule:
    @staticmethod
    def from_module(*a, **kw):
        raise AssertionError("test graph must not compile")
    def __init__(self, blob):
        self.kind = json.loads(blob)["kind"]
    def to(self, device):
        return self
    def __call__(self, *args):
        if self.kind == "decode":
            return args[0], args[12], args[13]
        logits = torch.zeros(32, args[0].shape[0])
        logits[3] = 1
        return logits
sdk.CompileModule = CompileModule
sdk.module_from_blob = lambda blob, state: CompileModule(blob)
sys.modules[sdk.__name__] = sdk
specs = types.ModuleType("rngd_pd.engines.decode_batch")
def forbidden():
    raise AssertionError("test must load a cache")
specs.decode_graph_spec = lambda B,S,g: ("decode-B%d-S%d-g%d" % (B,S,g), forbidden)
specs.lmhead_graph_spec = lambda B,V,ng: ("lm-B%d" % B, forbidden)
sys.modules[specs.__name__] = specs
import rngd_pd.swap.rngd as runtime
# Test fixture has the real runtime layout at tiny dimensions; only native SDK
# allocation/synchronization/graph outputs are replaced with CPU operations.
runtime.H, runtime.M, runtime.NH, runtime.NKV, runtime.HD = 4, 8, 2, 1, 2
'''


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    """Real checkpoint/cache readers, real engine lifecycle, tiny fake native SDK."""
    import io
    import shutil
    import zipfile
    import torch
    from safetensors.torch import save_file

    source = tmp_path / "source"
    checkpoint = source / "checkpoint"
    checkpoint.mkdir(parents=True)
    checkpoint.joinpath("config.json").write_text(json.dumps(dict(hidden_size=4,
        intermediate_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=2,
        num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=32)))
    tensors = {"model.embed_tokens.weight": torch.ones(32, 4, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(32, 4, dtype=torch.bfloat16)}
    for layer in range(2):
        for name, shape in {"self_attn.q_proj.weight": (4,4), "self_attn.k_proj.weight": (2,4),
                "self_attn.v_proj.weight": (2,4), "self_attn.o_proj.weight": (4,4),
                "mlp.gate_proj.weight": (8,4), "mlp.up_proj.weight": (8,4),
                "mlp.down_proj.weight": (4,8), "input_layernorm.weight": (4,),
                "post_attention_layernorm.weight": (4,)}.items():
            tensors[f"model.layers.{layer}.{name}"] = torch.ones(shape, dtype=torch.bfloat16)
    save_file(tensors, str(checkpoint / "model.safetensors"))
    cache = source / "cache/graphs"
    sdk_cache = cache / "2026.4.0" / str(torch.__version__)
    sdk_cache.mkdir(parents=True)
    for key, kind in (("decode-B1-S16-g2", "decode"), ("decode-B1-S32-g2", "decode"), ("lm-B1", "lm")):
        state = io.BytesIO()
        torch.save({"module_weight": None, "output_reconstructor": None}, state)
        with zipfile.ZipFile(sdk_cache / (key + ".runnable"), "w") as archive:
            archive.writestr("metadata.json", json.dumps(dict(format="runnable-v1", sdk="2026.4.0",
                                                              torch=str(torch.__version__))))
            archive.writestr("runnable.bin", json.dumps(dict(kind=kind)))
            archive.writestr("state.pt", state.getvalue())
    target = tmp_path / "target-source"
    shutil.copytree(source, target)
    # Only base graphs are visible at base startup.
    (sdk_cache / "decode-B1-S32-g2.runnable").unlink()
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(dict(prompts=[[1,2]], base_prompts=[[1,2]],
                                          tokens=[[3,3,3]], base_tokens=[[3,3,3]])))
    pod = tmp_path / "pod.json"
    from datetime import datetime, timezone
    pod.write_text(json.dumps(dict(metadata=dict(name="native-fixture", uid="fixture-uid",
        creationTimestamp=datetime.now(timezone.utc).isoformat()))))
    boundary = tmp_path / "sdk-boundary"
    boundary.mkdir()
    (boundary / "sitecustomize.py").write_text(SDK_BOUNDARY)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(boundary), str(SCRIPT.parents[1] / "src"))))
    # Python acts as the external storage CLI process on Windows too. Its script
    # is deliberately named 'pull' so the exact production argv is exercised.
    (tmp_path / "pull").write_text('''
import json, pathlib, shutil, sys
args = sys.argv[1:]
def option(name):
    return args[args.index(name)+1]
source, dest = pathlib.Path(option("--store")), pathlib.Path(args[-1])
cache = pathlib.Path(option("--cache"))
cached = cache.exists()
cache.mkdir(exist_ok=True)
cache.joinpath("filled").write_text("target cached")
shutil.copytree(source, dest)
stats = dict(bytes=1000, read_bytes=0 if cached else 1000, cache_bytes=1000 if cached else 0)
if "--base-dir" in args:
    base = pathlib.Path(option("--base-dir"))
    for p in base.rglob("*"):
        if p.is_file(): p.unlink()
    stats.update(base="immutable-base-digest", reused=800)
print(json.dumps(dict(manifest="immutable-target-digest", stats=stats)))
''')
    return dict(trial_id="tiny-native", arm="resident", cache_state="empty",
        onggi_binary=sys.executable, store=str(target), base_ref="base", target_ref="target",
        consumable_base_source=str(source), immutable_checkpoint=str(checkpoint),
        base_cache_dir=str(cache), base_variant="B1-S16-g2", target_variant="B1-S32-g2",
        chip=0, host_workers=2, intra_op=1, reference_file=str(reference), pod_metadata_file=str(pod),
        clock_basis="CPU test only, fixture API timestamp uses this host UTC clock",
        work_dir=str(tmp_path / "owned"), worker_timeout_s=60)


@pytest.mark.parametrize("arm", ["full", "delta", "resident"])
def test_real_cli_and_runtime_preserve_native_endpoint_and_ownership(native_fixture, tmp_path, arm):
    cfg = dict(native_fixture, arm=arm, cache_state="prepopulated")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    out = tmp_path / "sample.json"
    completed = cli(config, out, cwd=tmp_path)
    record = json.loads(out.read_text())
    assert completed.returncode == 0, (completed.stderr, record)
    assert record["status"] == "completed" and record["validity"] == "valid"
    assert record["metrics"]["pod_creation_to_first_token_s"] > 0
    assert record["metrics"]["reconfiguration_to_first_token_s"] > 0
    assert record["target"]["tokens"] == [[3,3,3]]
    assert sum(record["phase_s"]["trigger_to_first_partition"].values()) == pytest.approx(
        record["metrics"]["reconfiguration_to_first_token_s"])
    assert record["pulls"]["target"]["report"]["stats"]["cache_bytes"] == 1000
    assert (Path(cfg["consumable_base_source"]) / "checkpoint/model.safetensors").exists()
    assert record["target"]["counts"]["compilation_attempts"] == 0
    base_identity = (record["base"]["pid"], record["base"]["entry_perf_ns"])
    target_identity = (record["target"]["pid"], record["target"]["entry_perf_ns"])
    assert all(row["returncode"] == 0 and row["launcher_pid"] > 0 for row in record["workers"].values())
    if arm == "resident":
        assert record["target"]["resident_weights_preserved"]
        assert record["target"]["base_graphs_preserved"]
        assert base_identity == target_identity
        assert record["target"]["counts"]["shared_uploads"] == 1
        assert record["phase_s"]["target_engine_init_s"] == 0
        assert record["phase_s"]["target_weights"] == {"host_weights_s": 0, "upload_weights_s": 0}
        assert record["phase_s"]["base_weights"]["upload_weights_s"] > 0
    else:
        assert base_identity != target_identity
        assert record["target"]["checkpoint"].endswith("target" + os.sep + "checkpoint")
        assert record["phase_s"]["target_engine_init_s"] > 0
        assert record["phase_s"]["target_weights"]["upload_weights_s"] > 0


def test_worker_token_failure_preserves_native_evidence_and_invalidates_endpoints(native_fixture, tmp_path):
    cfg = native_fixture
    ref = Path(cfg["reference_file"])
    doc = json.loads(ref.read_text())
    doc["tokens"] = [[4,4,4]]
    ref.write_text(json.dumps(doc))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    out = tmp_path / "sample.json"
    completed = cli(config, out, cwd=tmp_path)
    record = json.loads(out.read_text())
    assert completed.returncode == 1
    assert record["status"] == "failed" and record["validity"] == "invalid"
    assert record["metrics"] == {}
    worker = record["worker_evidence"]["base-finished.json"]
    assert worker["tokens"] == [[3,3,3]] and not worker["tokens_equal_reference"]
    assert worker["counts"]["native_calls"] > 0


def test_native_output_validation_rejects_cpu_fallback_and_empty_outputs(harness):
    import torch
    with pytest.raises(RuntimeError, match="device"):
        harness.validate_native_outputs((torch.ones(1),), torch.device("meta"))
    with pytest.raises(RuntimeError, match="tensor"):
        harness.validate_native_outputs((None, {"metadata": 1}), torch.device("cpu"))
    assert harness.validate_native_outputs((torch.ones(1), {"nested": [torch.zeros(1)]}),
        torch.device("cpu")) == ["cpu"]


def test_worker_identity_gates_use_interpreter_start_and_allow_restart_pid_reuse(harness):
    base = dict(pid=11, entry_perf_ns=100)
    assert harness.validate_worker_identity(base, dict(pid=11, entry_perf_ns=100), "resident")
    assert harness.validate_worker_identity(base, dict(pid=11, entry_perf_ns=200), "delta")
    with pytest.raises(RuntimeError, match="identity"):
        harness.validate_worker_identity(base, dict(pid=11, entry_perf_ns=200), "resident")
    with pytest.raises(RuntimeError, match="identity"):
        harness.validate_worker_identity(base, dict(pid=11, entry_perf_ns=100), "full")
