"""Real tensor state with only the native constant-buffer boundary substituted."""
from types import SimpleNamespace

import pytest
import torch

from rngd_pd.swap.atom_constants import ConstantWeightUpdater


class Handle:
    def __init__(self, params, groups):
        self.params, self.groups = params, groups
        self.device = {k: p.clone() for k, p in params.items()}
        self.applied, self.failures = set(), 0
        self.transferred = []

    def get_meta_load_key_names(self):
        return {k: str(k) for k in self.params}

    def get_host_op_keys(self):
        return []

    def extern_transform_key_groups(self):
        return self.groups

    def reset_applied_transform_graphs(self):
        self.applied.clear()

    def apply_extern_transforms(self, values):
        count = 0
        for index, group in enumerate(self.groups):
            if index in self.applied or not set(group) <= values.keys():
                continue
            for key in group:
                self.device[key].copy_(values[key])
            self.transferred.append(set(group))
            self.applied.add(index)
            if self.failures:
                self.failures -= 1
                raise RuntimeError('partial device write failed')
            count += 1
        return count


def setup():
    params = {n: torch.nn.Parameter(torch.full((2, 3), float(i)), requires_grad=False)
              for i, n in enumerate(('q', 'k', 'v'), 1)}
    by_ptr = {p.data_ptr(): p for p in params.values()}
    q, k, v = [p.data_ptr() for p in params.values()]
    handle = Handle(by_ptr, [[q, k], [v]])
    updater = ConstantWeightUpdater(params, [SimpleNamespace(_runtime_handle=handle)])
    return params, (q, k, v), handle, updater


def test_refresh_updates_only_complete_affected_transform_group_and_can_repeat():
    params, (q, k, v), handle, updater = setup()
    old_v, old_q = handle.device[v], params['q']
    for value in (7., 9.):
        result = updater.apply({'q': torch.full((2, 3), value)})
        assert torch.equal(handle.device[q], torch.full((2, 3), value))
        assert torch.equal(params['q'], handle.device[q])
        assert params['q'] is old_q
        assert handle.device[v] is old_v and torch.equal(old_v, torch.full((2, 3), 3.))
        assert result['transform_input_bytes'] == 48
        assert result['changed_weight_bytes'] == 24
    assert handle.transferred == [{q, k}, {q, k}]


def test_partial_write_failure_restores_device_and_host_values():
    params, (q, k, v), handle, updater = setup()
    handle.failures = 1
    with pytest.raises(RuntimeError, match='partial device write'):
        updater.apply({'q': torch.full((2, 3), 7.)})
    assert torch.equal(handle.device[q], torch.ones(2, 3))
    assert torch.equal(params['q'], torch.ones(2, 3))
    updater.apply({'q': torch.full((2, 3), 9.)})
    assert torch.equal(handle.device[q], torch.full((2, 3), 9.))


def test_failed_rollback_poisoned_updater_rejects_future_changes():
    params, _, handle, updater = setup()
    handle.failures = 2
    with pytest.raises(RuntimeError, match='reload required'):
        updater.apply({'q': torch.full((2, 3), 7.)})
    assert updater.failed
    with pytest.raises(RuntimeError, match='reload required'):
        updater.apply({'q': torch.full((2, 3), 9.)})


@pytest.mark.parametrize('replacement', [torch.ones(3, 2), torch.ones(2, 3).half()])
def test_layout_mismatch_rejected_before_any_native_write(replacement):
    params, _, handle, updater = setup()
    with pytest.raises(ValueError, match='layout'):
        updater.apply({'q': replacement})
    assert not handle.transferred


def test_missing_group_dependency_and_unbound_parameter_fail_before_write():
    params, (q, k, v), handle, updater = setup()
    handle.groups[0].append(123)
    with pytest.raises(ValueError, match='unmapped'):
        updater.apply({'q': torch.ones(2, 3)})
    assert not handle.transferred
    handle.groups = [[v]]
    with pytest.raises(ValueError, match='not bound'):
        updater.apply({'q': torch.ones(2, 3)})
    assert not handle.transferred


def test_failure_in_second_program_restores_first_program_too():
    params, (q, k, v), first, updater = setup()
    first.groups = [[q, k]]
    second = Handle(first.params, [[v]])
    second.failures = 1
    updater.runtimes.append(SimpleNamespace(_runtime_handle=second))
    with pytest.raises(RuntimeError, match='partial device write'):
        updater.apply({'q': torch.full((2, 3), 7.), 'v': torch.full((2, 3), 8.)})
    assert torch.equal(first.device[q], torch.ones(2, 3))
    assert torch.equal(second.device[v], torch.full((2, 3), 3.))
    assert torch.equal(params['q'], torch.ones(2, 3))
    assert torch.equal(params['v'], torch.full((2, 3), 3.))


def test_host_only_constant_refresh_uses_host_parameter_binding():
    params, (q, k, v), handle, updater = setup()
    handle.groups = [[v]]
    handle.get_host_op_keys = lambda: [q]
    def host(values):
        for key, value in values.items():
            handle.device[key].copy_(value)
    handle.apply_host_params = host
    result = updater.apply({'q': torch.full((2, 3), 7.)})
    assert torch.equal(handle.device[q], torch.full((2, 3), 7.))
    assert result['transform_input_bytes'] == 0 and result['host_parameters'] == 1
    assert not handle.transferred


@pytest.fixture
def checkpoint_pair(tmp_path):
    import json
    from safetensors.torch import save_file
    from rngd_pd.swap.atom_inputs import weight_names
    tensors = {n: torch.ones(4 if 'norm' in n else (4, 4), dtype=torch.bfloat16)
               for n in weight_names(1)}
    tensors['model.embed_tokens.weight'] = torch.ones(4, 4, dtype=torch.bfloat16)
    for folder in ('base', 'new'):
        path = tmp_path / folder
        path.mkdir()
        (path / 'config.json').write_text(json.dumps(dict(hidden_size=4, num_hidden_layers=1)))
        if folder == 'new':
            for name in ('model.layers.0.self_attn.o_proj.weight', 'model.layers.0.mlp.down_proj.weight'):
                tensors[name] = torch.arange(16).reshape(4, 4).to(torch.bfloat16)
        save_file(tensors, path / 'model.safetensors')
    return tmp_path / 'base', tmp_path / 'new'


def constant_engine(checkpoint):
    from threading import RLock
    from rngd_pd.swap.atom_constants import ConstantWeightDecode
    from rngd_pd.swap.atom_inputs import weight_names
    from rngd_pd.swap.rngd import Checkpoint
    from rngd_pd.swap.variants import Variant
    e = ConstantWeightDecode.__new__(ConstantWeightDecode)
    e._lock, e._closed, e._generation, e._kv_owner = RLock(), False, 0, object()
    e.ck, e.d = Checkpoint(checkpoint), dict(nl=1, NKV=1, G=2, HD=2, H=4)
    e.weights = {}
    for name in weight_names(1):
        t = e.ck.get(name).to(torch.float32 if 'norm' in name else torch.float16)
        if name.endswith('self_attn.o_proj.weight'):
            t = t.t().reshape(1, 4, 4).contiguous()
        e.weights[name] = t
    e.W = list(e.weights.values())
    handle = Handle({p.data_ptr(): p for p in e.W}, [[p.data_ptr()] for p in e.W])
    e.runtimes = [SimpleNamespace(_runtime_handle=handle)]
    e.updater = ConstantWeightUpdater(e.weights, e.runtimes)
    e.tables = SimpleNamespace(embed=e.ck.get('model.embed_tokens.weight').float().numpy())
    e.active = Variant(1, 8, 0)
    e.graphs, e.compiled = {e.active: e.runtimes[0]}, {}
    return e, handle


def test_checkpoint_update_preserves_runtime_and_matches_fresh_packed_constants(checkpoint_pair):
    from rngd_pd.swap.delta import plan_update
    from rngd_pd.swap.atom_inputs import _KV
    base, new = checkpoint_pair
    e, handle = constant_engine(base)
    before = dict(handle.device)
    runtime = e.graphs[e.active]
    old_kv = _KV(None, 0, e.active, e._kv_owner)
    result = e.apply_delta(new, plan_update(base, new))
    fresh, _ = constant_engine(new)
    for name, tensor in e.weights.items():
        torch.testing.assert_close(tensor, fresh.weights[name], rtol=0, atol=0)
        torch.testing.assert_close(handle.device[tensor.data_ptr()], tensor, rtol=0, atol=0)
        assert handle.device[tensor.data_ptr()] is before[tensor.data_ptr()]
    assert result['changed_weight_bytes'] == 64
    assert e.graphs[e.active] is runtime and e._generation == 1
    with pytest.raises(ValueError, match='fresh request'):
        e.step([1], [0], old_kv, old_kv)


def test_checkpoint_update_rolls_back_or_disables_engine(checkpoint_pair):
    from rngd_pd.swap.delta import plan_update
    base, new = checkpoint_pair
    for failures in (1, 2):
        e, handle = constant_engine(base)
        handle.failures = failures
        with pytest.raises(RuntimeError):
            e.apply_delta(new, plan_update(base, new))
        assert e.ck.path == base and e._generation == 0
        assert e._closed == (failures == 2)
        if failures == 2:
            with pytest.raises(RuntimeError, match='closed'):
                e.generate([[1]], 1)


def test_overlapping_subset_groups_count_all_native_eligible_transforms():
    params, (q, k, v), handle, updater = setup()
    handle.groups = [[q, k], [k], [v]]
    result = updater.apply({'q': torch.full((2, 3), 7.)})
    assert result['transform_groups'] == 2
    assert not updater.failed and torch.equal(handle.device[q], torch.full((2, 3), 7.))


def test_aliasing_replacements_are_staged_before_any_parameter_commit():
    params, (q, k, v), handle, updater = setup()
    updater.apply({'q': params['k'], 'k': params['q']})
    assert torch.equal(params['q'], torch.full((2, 3), 2.))
    assert torch.equal(params['k'], torch.ones(2, 3))
    assert torch.equal(handle.device[q], params['q'])
    assert torch.equal(handle.device[k], params['k'])


@pytest.mark.parametrize('interrupt_rollback', [False, True])
def test_interrupted_native_write_rolls_back_or_poisoned(interrupt_rollback):
    params, (q, k, v), handle, updater = setup()
    native = handle.apply_extern_transforms
    calls = 0
    def interrupted(values):
        nonlocal calls
        calls += 1
        result = native(values)
        if calls == 1 or interrupt_rollback:
            raise KeyboardInterrupt()
        return result
    handle.apply_extern_transforms = interrupted
    if interrupt_rollback:
        with pytest.raises(RuntimeError, match='reload required'):
            updater.apply({'q': torch.full((2, 3), 7.)})
        assert updater.failed
    else:
        with pytest.raises(KeyboardInterrupt):
            updater.apply({'q': torch.full((2, 3), 7.)})
        assert not updater.failed
        assert torch.equal(handle.device[q], torch.ones(2, 3))
        assert torch.equal(params['q'], torch.ones(2, 3))


def test_interruption_after_updater_commit_disables_engine(checkpoint_pair):
    from rngd_pd.swap.delta import plan_update
    base, new = checkpoint_pair
    e, _ = constant_engine(base)
    apply = e.updater.apply
    def interrupted(values):
        apply(values)
        raise KeyboardInterrupt()
    e.updater.apply = interrupted
    with pytest.raises(KeyboardInterrupt):
        e.apply_delta(new, plan_update(base, new))
    assert e._closed


def test_host_parameter_pointer_owner_survives_refresh_return():
    import gc
    import weakref
    params, (q, k, v), handle, updater = setup()
    handle.groups = [[v]]
    handle.get_host_op_keys = lambda: [q]
    pointers = []
    handle.apply_host_params = lambda values: pointers.append(weakref.ref(values[q]))
    updater.apply({'q': torch.full((2, 3), 7.)})
    gc.collect()
    assert pointers[-1]() is not None
    torch.testing.assert_close(pointers[-1](), torch.full((2, 3), 7.))


def test_gen_preparation_supplies_explicit_tensor_examples(checkpoint_pair, tmp_path, monkeypatch):
    import sys
    from rngd_pd.swap.variants import Variant
    e, handle = constant_engine(checkpoint_pair[0])
    e.torch, e.device, e.context, e.cache_dir = torch, 0, object(), tmp_path
    e.module = object()
    e.tables.cfg = SimpleNamespace(hidden=4, n_kv=1, head_dim=2)
    def compile_model(module, **kwargs):
        examples = kwargs['example_inputs']
        assert [tuple(t.shape) for t in examples] == [
            (2, 4), (1, 2, 1, 2, 16), (1, 2, 1, 16, 2), (2, 1), (2, 1), (2, 16), (2, 16)]
        assert all(t.device.type == 'cpu' and t.dtype == torch.float32 for t in examples)
        return SimpleNamespace(create_runtime=lambda **opts: SimpleNamespace(_runtime_handle=handle),
                               save=lambda path: None)
    e.rebel = SimpleNamespace(compile_from_torch=compile_model)
    monkeypatch.setitem(sys.modules, 'rebel.core.weight_free',
                        SimpleNamespace(apply_weight_free_weights=lambda runtime, module: None))
    e.prepare(Variant(2, 16, 0))
    assert len(e.graphs) == 2
