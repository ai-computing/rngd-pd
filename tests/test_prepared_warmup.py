"""Warmup executes real prepared adapters; only the native SDK boundary is replaced."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rngd_pd.serving import cluster, prepared
from test_prepared_serving import tiny_checkpoint


def atom_config(tmp_path, warmup):
    tiny_checkpoint(tmp_path / 'ck')
    (tmp_path / 'buckets.rbln').write_bytes(b'compiled')
    (tmp_path / 'buckets.rbln.json').write_text(json.dumps(
        dict(layers=1, variants=['B1-S8', 'B2-S16'])))
    cfg = dict(engine='atom', checkpoint='ck', checkpoint_id='immutable',
               artifact='buckets.rbln', variants=['B1-S8', 'B2-S16'])
    if warmup is not None:
        cfg['warmup'] = warmup
    path = tmp_path / 'prepared.json'
    path.write_text(json.dumps(cfg))
    return path


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_warmup_requires_exact_boolean(tmp_path, value):
    cfg = atom_config(tmp_path, False)
    values = json.loads(cfg.read_text())
    values['warmup'] = value
    cfg.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='warmup'):
        prepared.read_config(cfg, 16)


@pytest.mark.parametrize('warmup', [None, False, True])
def test_factory_warms_every_variant_before_readiness_and_discards_kv(tmp_path, monkeypatch, warmup):
    calls = []
    monkeypatch.setitem(cluster.STATE, 'ready', False)

    class Compiled:
        def __init__(self, path):
            pass

        def create_runtime(self, **kwargs):
            return self

        def run(self, x, K, V, cos, sin, mask, oh):
            assert not cluster.STATE['ready']
            assert not K.any() and not V.any()
            calls.append((len(x), mask.shape[1]))
            logits = np.zeros((len(x), 8)); logits[:, 3] = 1
            return logits, np.ones((1, len(x), 1, 4)), np.ones((1, len(x), 1, 4))

    monkeypatch.setitem(sys.modules, 'rebel', SimpleNamespace(RBLNCompiledModel=Compiled))
    adapter = prepared.build_prepared(atom_config(tmp_path, warmup), 16, 0)
    assert calls == ([(1, 8), (2, 16)] if warmup else [])
    assert adapter._K is None and adapter._V is None
    assert not cluster.STATE['ready']
    cluster.PreparedDecodeWorker(adapter, 16, wait_ms=0)
    assert cluster.STATE['ready']
    adapter.engine.close()


def test_startup_validation_failure_closes_initialized_engine(tmp_path, monkeypatch):
    from rngd_pd.swap.atom import AtomSession
    closed = []
    original_close = AtomSession.close

    def close(engine):
        original_close(engine)
        closed.append(engine)

    class Compiled:
        def __init__(self, path):
            pass

        def create_runtime(self, **kwargs):
            return self

    cfg = atom_config(tmp_path, True)
    (tmp_path / 'buckets.rbln.json').write_text(json.dumps(
        dict(layers=1, variants=['B1-S8'])))
    monkeypatch.setitem(sys.modules, 'rebel', SimpleNamespace(RBLNCompiledModel=Compiled))
    monkeypatch.setattr(AtomSession, 'close', close)
    with pytest.raises(ValueError, match='every configured variant'):
        prepared.build_prepared(cfg, 16, 0)
    assert len(closed) == 1 and closed[0].rt is None and closed[0].compiled is None


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_warmup_failure_or_cancellation_discards_kv_and_closes_engine(tmp_path, monkeypatch, failure):
    from rngd_pd.swap.atom import AtomSession
    discarded, closed = [], []
    original_discard, original_close = prepared.PreparedEngine.discard_kv, AtomSession.close
    monkeypatch.setitem(cluster.STATE, 'ready', False)

    def discard(adapter):
        original_discard(adapter)
        discarded.append(adapter)

    def close(engine):
        original_close(engine)
        closed.append(engine)

    class Compiled:
        def __init__(self, path):
            pass

        def create_runtime(self, **kwargs):
            return self

        def run(self, *args):
            raise failure('native step interrupted')

    monkeypatch.setattr(prepared.PreparedEngine, 'discard_kv', discard)
    monkeypatch.setattr(AtomSession, 'close', close)
    monkeypatch.setitem(sys.modules, 'rebel', SimpleNamespace(RBLNCompiledModel=Compiled))
    with pytest.raises(failure, match='native step interrupted'):
        prepared.build_prepared(atom_config(tmp_path, True), 16, 0)
    assert len(discarded) >= 1
    assert all(a._K is None and a._V is None for a in discarded)
    assert len(closed) == 1 and closed[0].rt is None and closed[0].compiled is None
    assert not cluster.STATE['ready']


@pytest.mark.parametrize('failure', [KeyboardInterrupt, SystemExit, RuntimeError])
def test_rngd_cleanup_sync_failure_preserves_startup_error_and_owns_buffers(tmp_path, monkeypatch, failure):
    import torch
    from rngd_pd.swap import rngd
    tiny_checkpoint(tmp_path / 'ck')
    for name, value in dict(H=4, M=8, NH=1, NKV=1, HD=4).items():
        monkeypatch.setattr(rngd, name, value)
    startup_error = failure('warmup interrupted')
    cleanup_error = RuntimeError('device synchronization unavailable')
    sync_failed = [False]
    engines, adapters, owned = [], [], {}

    def sync(chip):
        if sync_failed[0]:
            raise cleanup_error

    monkeypatch.setitem(sys.modules, 'rngd_pd.engines.fsdk', SimpleNamespace(
        dev_str=lambda chip: 'cpu', open_device=lambda *args: None, sync=sync))
    monkeypatch.setitem(cluster.STATE, 'ready', False)
    original_discard = prepared.PreparedEngine.discard_kv

    def discard(adapter):
        original_discard(adapter)
        adapters.append(adapter)

    class Decoder(torch.nn.Module):
        def forward(self, *args):
            engine = engines[0]
            owned.update(weight=engine.shared[0]['wq'], head=engine.wlm_d,
                         variant=next(iter(engine.variants.values())),
                         checkpoint_handles=dict(engine.ck._open))
            sync_failed[0] = True
            raise startup_error

    def graphs(engine, variant):
        engines.append(engine)
        return Decoder(), torch.nn.Identity(), ['cache', 'cache']

    monkeypatch.setattr(rngd.SwappableDecode, '_graphs', graphs)
    monkeypatch.setattr(prepared.PreparedEngine, 'discard_kv', discard)
    cfg = tmp_path / 'prepared.json'
    cfg.write_text(json.dumps(dict(engine='rngd', checkpoint='ck',
        checkpoint_id='immutable', variants=['B1-S8-g1'], warmup=True)))
    with pytest.raises(BaseException) as caught:
        prepared.build_prepared(cfg, 8, 0)
    assert caught.value is startup_error
    assert caught.value.__cause__ is cleanup_error
    engine = engines[0]
    assert engine._failed
    assert not cluster.STATE['ready']
    assert adapters and all(a._K is None and a._V is None for a in adapters)
    assert engine.shared[0]['wq'] is owned['weight']
    assert engine.wlm_d is owned['head']
    assert next(iter(engine.variants.values())) is owned['variant']
    assert engine.ck._open == owned['checkpoint_handles']
    with pytest.raises(RuntimeError, match='unavailable'):
        engine.swap(adapters[0].variants[0])
    sync_failed[0] = False
    engine.close()
    assert engine._failed
    assert engine.shared == [] and engine.variants == {}
    assert engine.wlm_d is None and engine.active is None and engine.ck._open == {}
