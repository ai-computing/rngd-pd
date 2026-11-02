"""CPU checks for the resident delivery experiment's trust/timing boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[2] / "bench/ccgrid_resident_delivery.py"


@pytest.fixture
def harness():
    assert SCRIPT.exists(), "resident delivery harness has not been implemented"
    spec = importlib.util.spec_from_file_location("resident_delivery_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path):
    for name in ("checkpoint", "base-cache", "base", "target"):
        (tmp_path / name).mkdir()
    for name in ("base", "target"):
        graph = tmp_path / name / "cache/graphs"
        graph.mkdir(parents=True)
        (graph / "graph.runnable").write_bytes(name.encode())
        (tmp_path / name / "variant.json").write_text("{}")
    ref = tmp_path / "references.json"
    ref.write_text(json.dumps({key: [[1, 2]] * 4 for key in
        ("base_prompts", "base_tokens", "prompts", "tokens")}))
    return dict(trial_id="cpu-check", arm="graph-copy", phase="smoke",
        immutable_checkpoint=str(tmp_path / "checkpoint"), base_cache_dir=str(tmp_path / "base-cache"),
        base_bundle_source=str(tmp_path / "base"), target_bundle_source=str(tmp_path / "target"),
        base_ref="sha256:" + "a" * 64, target_ref="sha256:" + "b" * 64,
        store=str(tmp_path / "store"), onggi_binary="/onggi", csi_client="/csi-call",
        reference_file=str(ref), base_variant="b4-s256-g2", target_variant="b4-s512-g2",
        chip=0, host_workers=2, intra_op=4, work_dir=str(tmp_path / "owned"), provenance={})


def test_check_config_is_cpu_only_and_does_not_create_work(harness, config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    checked = subprocess.run([sys.executable, str(SCRIPT), "--check-config", "--config", str(path)],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(SCRIPT.parents[1] / "src")))
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["status"] == "config_valid"
    assert not Path(config["work_dir"]).exists()
    assert "torch" not in vars(harness)


@pytest.mark.parametrize("change,match", [
    ({"base_ref": "latest"}, "sha256"),
    ({"target_ref": "sha256:" + "a" * 64}, "different"),
    ({"host_workers": 1}, "host_workers"),
    ({"target_variant": "b4-s512-g4"}, "group"),
    ({"arm": "full-copy"}, "arm"),
])
def test_invalid_fixed_conditions_rejected(harness, config, change, match):
    with pytest.raises(ValueError, match=match):
        harness.validate(dict(config, **change))


def test_owned_path_overlap_and_existing_work_rejected(harness, config):
    with pytest.raises(ValueError, match="independent"):
        harness.validate(dict(config, work_dir=str(Path(config["target_bundle_source"]) / "new")))
    Path(config["work_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        harness.validate(config)


def test_reference_shape_and_context_checked_before_native_import(harness, config):
    path = Path(config["reference_file"])
    refs = json.loads(path.read_text())
    refs["base_tokens"][0] = [True]
    path.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="base_tokens"):
        harness.validate(config)


def test_graph_only_bundle_rejects_checkpoint_and_unexpected_files(harness, config):
    (Path(config["target_bundle_source"]) / "checkpoint").mkdir()
    with pytest.raises(ValueError, match="graph-only"):
        harness.validate(config)


def test_inventory_requires_exact_files_and_hashes(harness, config, tmp_path):
    source = Path(config["target_bundle_source"])
    expected = harness.inventory(source)
    import shutil
    target = tmp_path / "delivered"
    shutil.copytree(source, target)
    assert harness.verify_inventory(target, expected)["cache/graphs/graph.runnable"]["sha256"]
    (target / "cache/graphs/graph.runnable").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash"):
        harness.verify_inventory(target, expected)
    (target / "extra").write_text("not in artifact")
    with pytest.raises(RuntimeError, match="inventory"):
        harness.verify_inventory(target, expected)


def test_metadata_records_ownership_and_permission_witnesses(harness, config):
    root = Path(config["target_bundle_source"])
    row = harness.file_metadata(root)["cache/graphs/graph.runnable"]
    source = root / "cache/graphs/graph.runnable"
    assert row["realpath"] == str(source.resolve())
    assert row["nlink"] == source.stat().st_nlink
    assert row["mode"] == source.stat().st_mode


def test_timing_includes_hash_validation_and_excludes_generation_tail(harness):
    events = {name: {"perf_ns": value * 1_000_000_000} for name, value in
        (("demand", 1), ("delivery_finished", 3), ("hash_finished", 5), ("started", 0))}
    target = dict(generation_start_perf_ns=8_000_000_000, first_token_perf_ns=8_250_000_000,
                  generation_done_perf_ns=30_000_000_000)
    metrics = harness.timing(events, target)
    assert metrics["demand_to_first_token_s"] == 7.25
    assert metrics["hash_validation_s"] == 2
    assert metrics["delivery_s"] == 2
    assert metrics["allocation_entry_to_first_token_s"] == 8.25
    events["hash_finished"]["perf_ns"] = 9_000_000_000
    with pytest.raises(ValueError, match="ordered"):
        harness.timing(events, target)


def test_failed_config_preserves_result_without_native_or_work(harness, config, tmp_path):
    config["base_ref"] = "latest"
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    out = tmp_path / "result.json"
    process = subprocess.run([sys.executable, str(SCRIPT), "trial", "--config", str(cfg),
        "--out", str(out)], capture_output=True, text=True)
    assert process.returncode == 1
    result = json.loads(out.read_text())
    assert result["config"] == config and result["status"] == "failed"
    assert result["metrics"] == {} and not Path(config["work_dir"]).exists()


def test_readonly_mount_gate_requires_actual_exact_mount(harness):
    raw = "41 2 0:3 /artifact /owned/target rw,relatime - ext4 /dev/test rw\n"
    with pytest.raises(RuntimeError, match="read-only"):
        harness.readonly_mount("/owned/target", raw)
    raw = raw.replace(" rw,relatime ", " ro,relatime ")
    assert harness.readonly_mount("/owned/target", raw)["mount_id"] == "41"
    with pytest.raises(RuntimeError, match="exact"):
        harness.readonly_mount("/owned/other", raw)


def test_inventory_does_not_follow_symlink_artifacts(harness, config, tmp_path):
    link = Path(config["target_bundle_source"]) / "cache/graphs/escape"
    try:
        link.symlink_to(Path(config["reference_file"]))
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(RuntimeError, match="symlink"):
        harness.inventory(config["target_bundle_source"])


def test_pull_stats_proof_selects_exact_target_and_retains_refusal(harness, config):
    target, base = config["target_ref"], config["base_ref"]
    log = ("materialized " + base + ' in 1s: {"read_bytes":100}\n'
        + "pull of " + target + " took over " + base + ": 900 B left in place\n"
        + "materialized " + target + ' in 2s: {"base":"' + base + '","reused":900,"read_bytes":100}\n')
    proof = harness.delivery_log_proof(log, target, base)
    assert proof["pull_stats"] == {"base": base, "reused": 900, "read_bytes": 100}
    assert proof["claimed_base"] and proof["left_in_place_bytes"] == 900
    busy = "pull of " + target + ": no artifact to take over (1 in use, 0 used within 1s)\n"
    proof = harness.delivery_log_proof(busy, target, base)
    assert not proof["claimed_base"] and proof["busy_donor_refusal"]


def test_allocation_gate_requires_exclusive_card_and_matching_init(harness, config, monkeypatch):
    import types
    fake = types.SimpleNamespace(allocated_cards=lambda: [2], resolve_chip=lambda value: 1,
        allocated_uuids=lambda *args: ["UUID"], visible_cards=lambda: [0, 2])
    import rngd_pd
    monkeypatch.setattr(rngd_pd, "devices", fake, raising=False)
    monkeypatch.setenv("POD_UID", "pod")
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
        "npu2" if path == Path("/run/swap/rngd") else original(path, *a, **kw))
    monkeypatch.setattr(harness.platform, "system", lambda: "Linux")
    assert harness.allocated_device(dict(config, chip=1))["torch_ordinal"] == 1
    with pytest.raises(RuntimeError, match="ordinal"):
        harness.allocated_device(config)
    fake.allocated_cards = lambda: [0, 2]
    with pytest.raises(RuntimeError, match="one allocated"):
        harness.allocated_device(config)


def test_external_command_retains_real_child_failure(harness):
    row = {}
    with pytest.raises(RuntimeError, match="command failed"):
        harness.command([sys.executable, "-c", "import sys; print('partial'); sys.exit(7)"], row)
    assert row["returncode"] == 7 and row["stdout"].strip() == "partial"
    assert row["started"]["perf_ns"] <= row["finished"]["perf_ns"]


def test_cleanup_gate_rejects_residual_owned_mount_or_lease(harness, tmp_path):
    target = tmp_path / "target"
    data = tmp_path / "state"
    data.mkdir()
    harness.verify_cleanup(data, [target], "")
    raw = "41 2 0:3 /artifact " + str(target) + " ro - ext4 /dev/test rw\n"
    with pytest.raises(RuntimeError, match="mount"):
        harness.verify_cleanup(data, [target], raw)
    lease = data / "leases/base/base"
    lease.parent.mkdir(parents=True)
    lease.write_text(str(target))
    with pytest.raises(RuntimeError, match="state"):
        harness.verify_cleanup(data, [target], "")
