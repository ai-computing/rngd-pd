"""Exercise update/rollback with real torch tensors on CPU, without the SDK."""
from threading import RLock
from unittest.mock import Mock
import os
import pytest
import torch
import numpy as np

from rngd_pd.swap.delta import UpdatePlan
from rngd_pd.swap.rngd import Checkpoint, SwappableDecode
from rngd_pd.swap.variants import Variant


NAME = "model.layers.0.self_attn.k_proj.weight"
OTHER = "model.layers.1.self_attn.k_proj.weight"


def checkpoint(path, value):
    ck = Checkpoint.__new__(Checkpoint)
    ck.path = str(path)
    ck.config = dict(hidden_size=4, rms_norm_eps=1e-5)
    ck.where = {NAME: "x", OTHER: "x"}
    values = {NAME: torch.full((4, 4), value, dtype=torch.bfloat16),
              OTHER: torch.full((4, 4), 3.0, dtype=torch.bfloat16)}
    ck.get = lambda name: values[name]
    return ck


@pytest.fixture
def engine(tmp_path):
    e = SwappableDecode.__new__(SwappableDecode)
    e._lock, e._failed, e._generation = RLock(), False, 0
    e._kv_owner = object()
    e.torch, e.BF, e.dev = torch, torch.bfloat16, "cpu"
    e.group, e.nl, e.ngroups = 2, 2, 1
    e.ck = checkpoint(tmp_path / "old", 1.0)
    e._sync = Mock()
    e.shared = [{"wk": e._group_slot_host(0, "wk")}]
    e.embed = e.norm = e.wlm_d = None
    e.n1 = e.n2 = []
    e.lm_name = "lm_head.weight"
    e.variants = {}
    return e


def plan(e, ck):
    return UpdatePlan({NAME: {}}, 32, 64, set(), e.ck.path, ck.path)


@pytest.mark.parametrize("mode,bytes_uploaded", [("inplace", 32), ("group", 64)])
def test_delta_equals_fresh_weights_and_preserves_unchanged_layer(engine, tmp_path, mode, bytes_uploaded):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    before = e.shared[0]["wk"][1].clone()
    result = e.apply_delta(ck, plan(e, ck), mode=mode)
    torch.testing.assert_close(e.shared[0]["wk"], e._group_slot_host(0, "wk"))
    torch.testing.assert_close(e.shared[0]["wk"][1], before)
    assert result["uploaded_bytes"] == bytes_uploaded
    assert e.ck is ck


def test_failed_device_update_rolls_back(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    old, before = e.ck, e.shared[0]["wk"].clone()
    e._sync.side_effect = [None, RuntimeError("device write failed"), None]
    with pytest.raises(RuntimeError, match="device write failed"):
        e.apply_delta(ck, plan(e, ck))
    torch.testing.assert_close(e.shared[0]["wk"], before)
    assert e.ck is old and not e._failed


def test_failed_rollback_stops_serving(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    e._sync.side_effect = [None, RuntimeError("write"), RuntimeError("rollback")]
    with pytest.raises(RuntimeError, match="reload required"):
        e.apply_delta(ck, plan(e, ck))
    assert e._failed
    with pytest.raises(RuntimeError, match="reload"):
        e.swap(None)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupted_write_restores_weights_checkpoint_and_invalidates_kv(engine, tmp_path, interrupt):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    old, before = e.ck, e.shared[0]["wk"].clone()
    e.V, e.active = 128, Variant(1, 32, 2)
    old_keys, old_values = e.blank_kv()
    e._sync.side_effect = [None, interrupt("cancel after write"), None]
    with pytest.raises(interrupt, match="cancel after write"):
        e.apply_delta(ck, plan(e, ck))
    torch.testing.assert_close(e.shared[0]["wk"], before, rtol=0, atol=0)
    assert e.ck is old and not e._failed
    with pytest.raises(ValueError, match="fresh request"):
        e.step([7], [0], old_keys, old_values)


def test_interrupted_rollback_disables_serving(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    e._sync.side_effect = [None, RuntimeError("write"), KeyboardInterrupt("rollback")]
    with pytest.raises(RuntimeError, match="reload required"):
        e.apply_delta(ck, plan(e, ck))
    assert e._failed
    with pytest.raises(RuntimeError, match="reload"):
        e.swap(None)


def test_interrupted_group_commit_restores_original_allocations(engine, tmp_path):
    class InterruptOnce(dict):
        def __setitem__(self, key, value):
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt("commit")
            super().__setitem__(key, value)

    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    old, allocation = e.ck, e.shared[0]["wk"]
    state = InterruptOnce(pb=object())
    state.armed = True
    old_pb = state["pb"]
    e.variants = {Variant(1, 32, 2): state}
    with pytest.raises(KeyboardInterrupt, match="commit"):
        e.apply_delta(ck, plan(e, ck), mode="group")
    assert e.ck is old and e.shared[0]["wk"] is allocation and not e._failed
    assert state["pb"] is old_pb


def test_reject_wrong_plan_and_config_without_device_write(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    p = plan(e, ck)
    p.base_dir = os.path.realpath(tmp_path / "wrong")
    with pytest.raises(ValueError, match="plan"):
        e.apply_delta(ck, p)
    ck.config["rope_theta"] = 10000
    with pytest.raises(ValueError, match="configuration"):
        e.apply_delta(ck, plan(e, ck))
    assert e._sync.call_count == 0


def test_unknown_tensor_and_changed_shape_require_reload(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    p = plan(e, ck)
    p.tensors = {"unexpected": {}}
    with pytest.raises(ValueError, match="unsupported"):
        e.apply_delta(ck, p)
    ck.get = lambda name: torch.zeros(1)
    with pytest.raises(ValueError, match="layout"):
        e.apply_delta(ck, plan(e, ck))


def test_kv_from_before_weight_update_is_rejected(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    e._generation, e.V = 0, 128
    e.active = Variant(1, 32, 2)
    old_keys, old_values = e.blank_kv()
    e.apply_delta(ck, plan(e, ck))
    with pytest.raises(ValueError, match="fresh request"):
        e.step([7], [0], old_keys, old_values)


@pytest.mark.parametrize("fail", [False, True])
def test_norm_update_refreshes_every_prepared_batch_or_restores_all(engine, tmp_path, monkeypatch, fail):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "H", 4)
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    name = "model.norm.weight"
    old_get, new_get = e.ck.get, ck.get
    e.ck.where[name] = ck.where[name] = "x"
    e.ck.get = lambda n: torch.ones(4) if n == name else old_get(n)
    ck.get = lambda n: torch.full((4,), 3.0) if n == name else new_get(n)
    e.norm = np.ones(4, np.float32)
    e.n1 = e.n2 = [torch.ones(4), torch.ones(4)]
    e.eps = 1e-5
    vs = [Variant(1, 16, 2), Variant(2, 16, 2)]
    e.variants = {v: {"pb": e._per_batch(v.B)} for v in vs}
    previous = {v: e.variants[v]["pb"] for v in vs}
    p = UpdatePlan({name: {}}, 16, 80, set(), e.ck.path, ck.path)
    if fail:
        original = e._per_batch
        def partial(batch):
            if batch == 2:
                raise MemoryError("norm preparation failed")
            return original(batch)
        e._per_batch = partial
        with pytest.raises(MemoryError):
            e.apply_delta(ck, p)
        assert np.array_equal(e.norm, np.ones(4))
        assert all(e.variants[v]["pb"] is previous[v] for v in vs)
    else:
        e.apply_delta(ck, p)
        for v in vs:
            assert e.variants[v]["pb"] is not previous[v]
            torch.testing.assert_close(e.variants[v]["pb"]["norm"],
                                       torch.full((v.B, 4), 6.0, dtype=torch.bfloat16))
