"""Real request lifecycle with CPU tensors in place of device transfers/graphs."""
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rngd_pd.swap.atom_inputs import DeviceWeightDecode
from rngd_pd.swap.rngd import SwappableDecode
from rngd_pd.swap.variants import Variant


def atom_engine():
    e = DeviceWeightDecode.__new__(DeviceWeightDecode)
    e._lock, e._closed, e._generation = RLock(), False, 0
    e._kv_owner = object()
    e.torch, e.dev = torch, "cpu"
    e.d = dict(nl=1, NKV=1, HD=2)
    e.active = Variant(1, 8, 0)
    e.W = []
    e.tables = SimpleNamespace(step_inputs=lambda toks, pos, S: [
        np.zeros((1, 4), np.float32), np.zeros((1, 1), np.float32),
        np.zeros((1, 1), np.float32), np.zeros((1, S), np.float32),
        np.zeros((1, S), np.float32)])
    e.graphs = {e.active: lambda x, K, V, *args: (torch.tensor([[0., 1., 0.]]), K, V)}
    return e


def rngd_engine(monkeypatch):
    from rngd_pd.swap import rngd
    monkeypatch.setattr(rngd, "NKV", 1)
    monkeypatch.setattr(rngd, "HD", 2)
    e = SwappableDecode.__new__(SwappableDecode)
    e._lock, e._failed, e._generation = RLock(), False, 0
    e._kv_owner = object()
    e.torch, e.BF, e.dev = torch, torch.bfloat16, "cpu"
    e.group, e.ngroups, e.V, e.Vp = 1, 1, 3, 3
    e.active = Variant(1, 8, 1)
    e.embed = np.zeros((3, 4), np.float32)
    e.shared = [{n: torch.zeros(1) for n in ("wq", "wk", "wv", "wo", "wg", "wu", "wd")}]
    e.variants = {e.active: dict(
        tab={n: np.zeros((8, 4), np.float32) for n in ("rot", "mask", "eye")},
        pb=dict(eps=torch.zeros(1), nh=torch.zeros(1), n1=[torch.zeros(1)],
                n2=[torch.zeros(1)], norm=torch.zeros(1)),
        m=lambda *args: (args[0], args[-5], args[-4]),
        lmh=lambda *args: torch.tensor([[0.], [1.], [0.]]))}
    e.wlm_d = torch.zeros(1)
    return e


@pytest.fixture(params=["atom", "rngd"])
def engines(request, monkeypatch):
    make = atom_engine if request.param == "atom" else lambda: rngd_engine(monkeypatch)
    return make(), make()


@pytest.mark.parametrize("foreign", ["both", "keys", "values"])
def test_foreign_engine_kv_is_rejected_even_with_matching_generation_and_shape(engines, foreign):
    source, target = engines
    source_K, source_V = source.blank_kv()
    target_K, target_V = target.blank_kv()
    K = target_K if foreign == "values" else source_K
    V = target_V if foreign == "keys" else source_V
    with pytest.raises(ValueError, match="fresh request"):
        target.step([1], [0], K, V)


def test_same_engine_kv_remains_valid_across_steps(engines):
    e, _ = engines
    K, V = e.blank_kv()
    tokens, K, V = e.step([1], [0], K, V)
    tokens, _, _ = e.step(tokens, [1], K, V)
    assert tokens == [1]
