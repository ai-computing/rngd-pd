"""Real checkpoint/transform tests; the CPU substitutes only for NPU transfers."""
import json
import os
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from rngd_pd.swap.atom_inputs import DeviceWeightDecode, weight_names, _KV
from rngd_pd.swap.delta import plan_update
from rngd_pd.swap.rngd import Checkpoint
from rngd_pd.swap.variants import Variant


@pytest.fixture
def pair(tmp_path):
    tensors = {n: torch.ones(4 if "norm" in n else (4, 4), dtype=torch.bfloat16)
               for n in weight_names(1)}
    tensors["model.embed_tokens.weight"] = torch.ones((4, 4), dtype=torch.bfloat16)
    for sub in ("base", "new"):
        p = tmp_path / sub
        p.mkdir()
        (p / "config.json").write_text(json.dumps({"hidden_size": 4, "num_hidden_layers": 1}))
        save_file(tensors, p / "model.safetensors")
    return tmp_path / "base", tmp_path / "new", tensors


def change(pair, names):
    base, new, tensors = pair
    for n in names:
        tensors[n] = tensors[n] * 7
    save_file(tensors, new / "model.safetensors")
    return plan_update(base, new)


def engine(base):
    e = DeviceWeightDecode.__new__(DeviceWeightDecode)
    e._lock, e._closed, e._generation = RLock(), False, 0
    e.ck = Checkpoint(base)
    e.torch, e.dev, e.d = torch, "cpu", {"nl": 1}
    e.W = [e.ck.get(n).to(torch.float32 if "norm" in n else torch.float16)
           for n in weight_names(1)]
    e.tables = SimpleNamespace(embed=e.ck.get("model.embed_tokens.weight").float().numpy())
    e.active = Variant(1, 16, 0)
    e.g = Mock()
    e.graphs = {e.active: e.g}
    return e


def test_changed_inputs_equal_fresh_without_touching_other_allocations(pair):
    base, new, _ = pair
    names = ["model.layers.0.mlp.down_proj.weight", "model.layers.0.input_layernorm.weight",
             "model.embed_tokens.weight"]
    p = change(pair, names)
    e = engine(base)
    before = list(e.W)
    graph = e.g
    result = e.apply_delta(new, p)
    fresh = engine(new)
    for i, n in enumerate(weight_names(1)):
        torch.testing.assert_close(e.W[i], fresh.W[i], rtol=0, atol=0)
        assert (e.W[i] is before[i]) == (n not in names)
    np.testing.assert_array_equal(e.tables.embed, fresh.tables.embed)
    assert result["uploaded_bytes"] == 4 * 4 * 2 + 4 * 4
    assert e.g is graph and e._generation == 1


def test_failed_second_upload_preserves_active_checkpoint_and_all_inputs(pair):
    base, new, _ = pair
    p = change(pair, ["model.layers.0.mlp.down_proj.weight", "model.norm.weight"])
    e = engine(base)
    before, ck, embed = list(e.W), e.ck, e.tables.embed
    e._upload_weight = Mock(side_effect=[torch.ones(4), RuntimeError("out of device memory")])
    with pytest.raises(RuntimeError, match="device memory"):
        e.apply_delta(new, p)
    assert all(a is b for a, b in zip(before, e.W))
    assert e.ck is ck and e.tables.embed is embed and e._generation == 0


def test_rejects_wrong_base_and_graph_change_before_transfer(pair):
    base, new, _ = pair
    p = change(pair, ["model.norm.weight"])
    e = engine(base)
    e._upload_weight = Mock()
    p.base_dir = os.path.realpath(new)
    with pytest.raises(ValueError, match="plan"):
        e.apply_delta(new, p)
    p.base_dir = os.path.realpath(base)
    (new / "config.json").write_text(json.dumps({"hidden_size": 8}))
    with pytest.raises(ValueError, match="configuration"):
        e.apply_delta(new, p)
    e._upload_weight.assert_not_called()


def test_local_load_delta_and_stale_kv_rejection(pair):
    base, new, _ = pair
    change(pair, ["model.norm.weight"])
    e = engine(base)
    old = _KV(torch.zeros(1), e._generation, e.active)
    result = e.load_delta(new)
    assert result["source"] == "local" and result["plan"]["tensors"] == 1
    with pytest.raises(ValueError, match="fresh request"):
        e.step([1], [0], old, old)


def test_live_onggi_command_feeds_real_checkpoint_planner(pair, monkeypatch):
    base, new, _ = pair
    change(pair, ["model.norm.weight"])
    e = engine(base)
    import rngd_pd.swap.load as load
    run = Mock(return_value=SimpleNamespace(stdout=json.dumps({"files": [
        {"path": "model.safetensors", "size": (new / "model.safetensors").stat().st_size}]})))
    monkeypatch.setattr(load.subprocess, "run", run)
    result = e.load_delta(new, store="local-store", base_ref="base", new_ref="new")
    assert result["plan"]["tensors"] == 1 and result["source"] == "onggi"
    assert run.call_args.args[0] == ["onggi", "diff", "--store", "local-store", "--ranges",
                                     "--json", "base", "new"]
    assert run.call_args.kwargs["check"] is True


@pytest.mark.parametrize("doc", [{}, {"files": []}, {"files": [{"path": "x.rbln", "size": 3}]}])
def test_incomplete_or_compiled_artifact_diff_is_rejected(pair, doc):
    base, new, _ = pair
    e = engine(base)
    with pytest.raises(ValueError):
        e.load_delta(new, ranges=doc)
    assert e._generation == 0
