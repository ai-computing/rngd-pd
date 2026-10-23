"""Exercise update/rollback with real torch tensors on CPU, without the SDK."""
from threading import RLock
from unittest.mock import Mock
import json
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


@pytest.mark.parametrize("suffix,slot", [
    ("self_attn.q_proj.weight", "wq"), ("self_attn.k_proj.weight", "wk"),
    ("self_attn.v_proj.weight", "wv"), ("self_attn.o_proj.weight", "wo"),
    ("mlp.gate_proj.weight", "wg"), ("mlp.up_proj.weight", "wu"),
    ("mlp.down_proj.weight", "wd")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sparse_projection_rows_equal_fresh_with_smaller_payload(engine, tmp_path, suffix, slot, dtype):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    name, other = [f"model.layers.{i}.{suffix}" for i in range(2)]
    old = torch.linspace(-1.03, 1.07, 35).reshape(7, 5).to(dtype)
    new = old.clone()
    new[1:3] += 0.113
    new[5:6] -= 0.227
    e.ck.where = ck.where = {name: "x", other: "x"}
    e.ck.get = lambda n: old if n == name else old + 3
    ck.get = lambda n: new if n == name else old + 3
    e.shared = [{slot: e._group_slot_host(0, slot)}]
    allocation = e.shared[0][slot]
    neighbor = allocation[1].clone()
    p = UpdatePlan({name: {"rows": [(1, 3), (5, 6)]}}, 60, 140, set(), e.ck.path, ck.path)
    result = e.apply_delta(ck, p)
    torch.testing.assert_close(e.shared[0][slot], e._group_slot_host(0, slot), rtol=0, atol=0)
    torch.testing.assert_close(e.shared[0][slot][1], neighbor, rtol=0, atol=0)
    assert e.shared[0][slot] is allocation
    assert result["uploaded_bytes"] == 3 * 5 * 2


@pytest.mark.parametrize("fail", [False, True])
def test_sparse_lm_rows_preserve_chunks_padding_and_rollback(engine, tmp_path, monkeypatch, fail):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "NG_LM", 2)
    monkeypatch.setattr(rngd, "H", 4)
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    name = "lm_head.weight"
    old = torch.arange(28, dtype=torch.bfloat16).reshape(7, 4)
    new = old.clone()
    new[3:5] += 10
    new[6:7] -= 10
    old_get, new_get = e.ck.get, ck.get
    e.ck.where[name] = ck.where[name] = "x"
    e.ck.get = lambda n: old if n == name else old_get(n)
    ck.get = lambda n: new if n == name else new_get(n)
    e.wlm_d = e._lm_host().to(e.dev)
    allocation = e.wlm_d
    before = allocation.clone()
    p = UpdatePlan({name: {"rows": [(3, 5), (6, 7)]}}, 24, 128, set(), e.ck.path, ck.path)
    if fail:
        copy = e._copy_rows
        armed = True
        def interrupted(*args):
            nonlocal armed
            copy(*args)
            if armed:
                armed = False
                raise KeyboardInterrupt("partial LM write")
        e._copy_rows = interrupted
        with pytest.raises(KeyboardInterrupt, match="partial LM write"):
            e.apply_delta(ck, p)
        assert e.wlm_d is allocation and not e._failed
        torch.testing.assert_close(e.wlm_d, before, rtol=0, atol=0)
        return
    result = e.apply_delta(ck, p)
    torch.testing.assert_close(e.wlm_d, e._lm_host(), rtol=0, atol=0)
    assert e.wlm_d is allocation
    assert result["uploaded_bytes"] == 24
    assert torch.count_nonzero(e.wlm_d.reshape(8, 4)[7]) == 0


@pytest.mark.parametrize("interrupt", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_after", [1, 2])
def test_partial_projection_copy_failure_restores_every_attempt(engine, tmp_path, interrupt, fail_after):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    e.ck.get = lambda n: torch.full((8, 4), 1.0 if n == NAME else 3.0, dtype=torch.bfloat16)
    ck.get = lambda n: torch.full((8, 4), 7.0 if n == NAME else 3.0, dtype=torch.bfloat16)
    e.shared = [{"wk": e._group_slot_host(0, "wk")}]
    old, before = e.ck, e.shared[0]["wk"].clone()
    copy, calls = e._copy_rows, 0
    def interrupted(*args):
        nonlocal calls
        copy(*args)
        calls += 1
        if calls == fail_after:
            raise interrupt("partial row write")
    e._copy_rows = interrupted
    p = plan(e, ck)
    p.tensors[NAME] = {"rows": [(0, 1), (2, 3)]}
    with pytest.raises(interrupt, match="partial row write"):
        e.apply_delta(ck, p)
    assert e.ck is old and not e._failed
    torch.testing.assert_close(e.shared[0]["wk"], before, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [[], [(-1, 2)], [(0, 5)], [(2, 2)],
                                  [(2, 1)], [(True, 2)], [(0.0, 2)], None])
def test_invalid_row_ranges_fail_before_native_writes(engine, tmp_path, rows):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    old, before = e.ck, e.shared[0]["wk"].clone()
    p = plan(e, ck)
    p.tensors[NAME] = {"rows": rows}
    with pytest.raises(ValueError, match="row"):
        e.apply_delta(ck, p)
    assert e.ck is old
    torch.testing.assert_close(e.shared[0]["wk"], before, rtol=0, atol=0)
    e._sync.assert_not_called()


def test_real_safetensors_plan_updates_only_one_changed_row(engine, tmp_path):
    from safetensors.torch import save_file
    from rngd_pd.swap.delta import plan_update
    e = engine
    base, new = tmp_path / "real-base", tmp_path / "real-new"
    values = {NAME: torch.ones(4, 4, dtype=torch.bfloat16),
              OTHER: torch.full((4, 4), 3.0, dtype=torch.bfloat16)}
    for directory in (base, new):
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(e.ck.config))
    save_file(values, str(base / "model.safetensors"))
    changed = {n: t.clone() for n, t in values.items()}
    changed[NAME][1, 2] = 7
    save_file(changed, str(new / "model.safetensors"))
    e.ck = Checkpoint(base)
    e.shared = [{"wk": e._group_slot_host(0, "wk")}]
    allocation = e.shared[0]["wk"]
    p = plan_update(base, new)
    assert p.tensors[NAME]["rows"] == [(1, 2)]
    result = e.apply_delta(new, p)
    assert result["uploaded_bytes"] == 8
    assert e.shared[0]["wk"] is allocation
    torch.testing.assert_close(e.shared[0]["wk"], e._group_slot_host(0, "wk"), rtol=0, atol=0)
    e.ck.close()


def test_overlapping_rows_are_coalesced_and_header_changes_update_whole(engine, tmp_path):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    p = plan(e, ck)
    p.tensors[NAME] = {"rows": [(2, 4), (0, 2), (1, 3)]}
    result = e.apply_delta(ck, p)
    assert result["uploaded_bytes"] == 32
    newer = checkpoint(tmp_path / "newer", 9.0)
    p = plan(e, newer)
    p.tensors[NAME] = {"rows": [(1, 2)]}
    p.header_changes.add(NAME)
    result = e.apply_delta(newer, p)
    assert result["uploaded_bytes"] == 32
    torch.testing.assert_close(e.shared[0]["wk"], e._group_slot_host(0, "wk"), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [[(a, a + 8) for a in range(0, 4096, 16)], [(0, 3000)]])
def test_fragmented_or_dense_rows_use_one_complete_transfer(engine, tmp_path, rows):
    e, ck = engine, checkpoint(tmp_path / "new", 7.0)
    old = torch.ones(4096, 5, dtype=torch.bfloat16)
    new = old.clone()
    for a, b in rows:
        new[a:b] = 7
    e.ck.get = lambda n: old if n == NAME else old + 3
    ck.get = lambda n: new if n == NAME else old + 3
    e.shared = [{"wk": e._group_slot_host(0, "wk")}]
    calls, copy = [], e._copy_rows
    def counted(*args):
        calls.append(args[:3])
        copy(*args)
    e._copy_rows = counted
    p = plan(e, ck)
    p.tensors[NAME] = {"rows": rows}
    result = e.apply_delta(ck, p)
    assert calls == [(NAME, 0, 4096)]
    assert result["uploaded_bytes"] == old.numel() * 2
    torch.testing.assert_close(e.shared[0]["wk"], e._group_slot_host(0, "wk"), rtol=0, atol=0)


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
