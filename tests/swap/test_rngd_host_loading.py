"""Parallel initial staging preserves exact weights and owns its worker lifetime."""
from concurrent.futures import ThreadPoolExecutor
import math
import threading
from types import SimpleNamespace

import pytest
import torch

from rngd_pd.swap.rngd import SwappableDecode


def bare(group=2, dtype=torch.bfloat16):
    e = SwappableDecode.__new__(SwappableDecode)
    e.torch, e.BF, e.dev = torch, torch.bfloat16, "cpu"
    e.group, e.ngroups, e.host_workers = group, 1, 2
    raw = [(torch.arange(70, dtype=torch.float32).reshape(7, 10)[:, ::2] / 17 - i).to(dtype)
           for i in range(group)]
    owner = threading.get_ident()

    def get(name):
        assert threading.get_ident() == owner, "checkpoint I/O escaped the caller thread"
        return raw[int(name.split(".")[2])]

    e.ck = SimpleNamespace(get=get)
    return e, raw


@pytest.mark.parametrize("slot", ["wq", "wk", "wd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("group", [1, 2, 8])
def test_parallel_groups_preserve_exact_bf16_layout(slot, dtype, group):
    e, raw = bare(group, dtype)
    expected = torch.stack([
        (t.float() * (1 / math.sqrt(128))).t().contiguous().to(e.BF)
        if slot == "wq" else t.to(e.BF).t().contiguous() for t in raw])
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = e._group_slot_host(0, slot, executor=pool)
    assert actual.dtype == e.BF and actual.is_contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_initial_pool_is_bounded_and_workers_disable_gradients():
    e, raw = bare()
    for t in raw:
        t.requires_grad_()
    owner = threading.get_ident()
    workers = set()
    gate = threading.Barrier(2, timeout=5)
    original = e._transform_weight

    def transform(t, kind):
        assert threading.get_ident() != owner and not torch.is_grad_enabled()
        workers.add(threading.current_thread())
        gate.wait()
        return original(t, kind)

    e._transform_weight = transform
    e._lm_host = lambda: torch.ones(1, dtype=e.BF)
    e._upload_shared()
    assert len(workers) == 2 and all(not t.is_alive() for t in workers)
    assert len(e.shared) == 1 and all(not t.requires_grad for t in e.shared[0].values())


def test_failed_transform_joins_workers_without_device_upload():
    e, _ = bare()
    gate = threading.Barrier(2, timeout=5)
    workers = set()

    def fail(t, kind):
        workers.add(threading.current_thread())
        gate.wait()
        raise RuntimeError("bad checkpoint tensor")

    e._transform_weight = fail
    e._lm_host = lambda: pytest.fail("failed projections must not upload LM")
    with pytest.raises(RuntimeError, match="bad checkpoint tensor"):
        e._upload_shared()
    assert len(workers) == 2 and all(not t.is_alive() for t in workers)
    assert e.shared == []


@pytest.mark.parametrize("value", [0, 3, -1, True, 1.5, None, "2"])
def test_invalid_worker_budget_rejected_before_sdk_import(value):
    with pytest.raises(ValueError, match="host_workers"):
        SwappableDecode("unused", host_workers=value)


def test_default_staging_stays_on_calling_thread():
    e, _ = bare()
    e.host_workers = 1
    owner = threading.get_ident()
    original = e._transform_weight

    def transform(t, kind):
        assert threading.get_ident() == owner
        return original(t, kind)

    e._transform_weight = transform
    e._lm_host = lambda: torch.ones(1, dtype=e.BF)
    e._upload_shared()
    assert len(e.shared) == 1
