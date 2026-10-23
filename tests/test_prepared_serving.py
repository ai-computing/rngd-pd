"""Prepared graph selection and imported KV use the real deployed batch path."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rngd_pd.serving import cluster
from rngd_pd.swap.variants import Variant


class Resident:
    """Only the external graph/device boundary is replaced."""
    def __init__(self, variants):
        self.variants, self.active, self.swaps, self.imports = variants, None, [], []
        self.tables = SimpleNamespace(cfg=SimpleNamespace(n_kv=1, head_dim=1))
        self.nl = 1
        self.weight = object()

    def swap(self, variant):
        assert variant in self.variants
        self.active = variant
        self.swaps.append(variant)
        return dict(swap_s=0, in_session=True)

    def blank_kv(self):
        v = self.active
        return np.zeros((1, v.B, 1, 1, v.S)), np.zeros((1, v.B, 1, v.S, 1))

    def step(self, toks, pos, K, V):
        for i, p in enumerate(pos):
            assert 0 <= p < self.active.S
            self.imports.append(float(K[0, i, 0, 0, 0]))
        return [t + 1 for t in toks], K, V


def worker(variants=None):
    from rngd_pd.serving.prepared import PreparedEngine
    variants = variants or [Variant(1, 8, 0), Variant(2, 8, 0), Variant(2, 16, 0)]
    native = Resident(variants)
    adapter = PreparedEngine(native, variants, "atom", "sha256:model-a", 1, 1, 1)
    w = cluster.PreparedDecodeWorker(adapter, S=16, wait_ms=0)
    w._load_kv = lambda rid: (np.full((1, 1, 1, 16), 11., np.float32),
                              np.full((1, 1, 16, 1), 12., np.float32), 0., 0., None)
    return w, native


def test_prepared_worker_selects_shape_and_reuses_weights():
    w, native = worker()
    weight = native.weight
    a = w.run("a", 1, 3, 2, checkpoint_id="sha256:model-a")
    b = w.run("b", 2, 10, 3, checkpoint_id="sha256:model-a")
    assert a["tokens"] == [1, 2]
    assert b["tokens"] == [2, 3, 4]
    assert native.swaps == [Variant(1, 8, 0), Variant(2, 16, 0)]
    assert native.weight is weight
    assert 11. in native.imports


def test_prepared_batch_has_independent_budgets_and_safe_padding():
    w, native = worker()
    # Exercise the actual runner directly so timing cannot determine batch composition.
    batch = [dict(req_id="a", first=1, pos=6, max_new=1),
             dict(req_id="b", first=2, pos=2, max_new=5)]
    got = w._run_batch(batch)
    assert [r["tokens"] for r in got] == [[1], [2, 3, 4, 5, 6]]
    assert native.active == Variant(2, 8, 0)


@pytest.mark.parametrize("identity", [None, "sha256:model-b"])
def test_prepared_worker_rejects_foreign_prefill_before_kv_load(identity):
    w, native = worker()
    w._load_kv = lambda rid: pytest.fail("foreign KV must not be consumed")
    with pytest.raises(cluster.AdmissionError) as err:
        w.run("a", 1, 3, 2, checkpoint_id=identity)
    assert err.value.status == 409
    assert native.swaps == []


def test_prepared_worker_refuses_delta_during_active_request():
    w, native = worker()
    entered, release = threading.Event(), threading.Event()
    original = w._load_kv
    errors = []

    def load(rid):
        entered.set()
        assert release.wait(3)
        return original(rid)

    w._load_kv = load
    t = threading.Thread(target=lambda: w.run("a", 1, 3, 2,
                                             checkpoint_id="sha256:model-a"), daemon=True)
    t.start()
    try:
        assert entered.wait(2)
        with pytest.raises(cluster.AdmissionError) as err:
            w.load_delta("new-checkpoint", "sha256:model-b")
        assert err.value.status == 409
    finally:
        release.set(); t.join(3)


def test_delta_changes_identity_only_after_success():
    w, native = worker()
    native.load_delta = lambda ck, **kw: dict(uploaded_bytes=8)
    result = w.load_delta("new-checkpoint", "sha256:model-b")
    assert result["uploaded_bytes"] == 8
    with pytest.raises(cluster.AdmissionError):
        w.run("old", 1, 3, 1, checkpoint_id="sha256:model-a")
    assert w.run("new", 1, 3, 1, checkpoint_id="sha256:model-b")["tokens"] == [1]


def test_failed_delta_retains_checkpoint_identity():
    w, native = worker()

    def fail(ck, **kw):
        raise RuntimeError("upload failed")

    native.load_delta = fail
    with pytest.raises(RuntimeError, match="upload failed"):
        w.load_delta("new-checkpoint", "sha256:model-b")
    assert w.run("old", 1, 3, 1, checkpoint_id="sha256:model-a")["tokens"] == [1]


def test_cancelled_batch_member_does_not_start_another_kv_load():
    w, native = worker()
    cancel = threading.Event()
    calls = []
    original = w._load_kv

    def load(rid):
        calls.append(rid)
        cancel.set()
        return original(rid)

    w._load_kv = load
    batch = [dict(req_id="a", first=1, pos=3, max_new=1),
             dict(req_id="cancelled", first=2, pos=3, max_new=1, cancel=cancel)]
    for item in batch:
        cluster._init_request(item)
    assert w._run_batch(batch)[0]["tokens"] == [1]
    assert calls == ["a"]


def test_manifest_rejects_shape_set_that_cannot_serve_long_batches(tmp_path):
    from rngd_pd.serving.prepared import read_config
    p = tmp_path / "prepared.json"
    p.write_text(json.dumps(dict(engine="rngd", checkpoint="ck", checkpoint_id="immutable",
                                 variants=["B1-S16-g8", "B2-S8-g8"])))
    with pytest.raises(ValueError, match="largest"):
        read_config(p, S=16)


def test_gateway_forwards_prefill_checkpoint_identity():
    from test_serving_limits import gateway, post
    seen = []

    def endpoint(url, body, **kw):
        result = post(url, body, **kw)
        if url.endswith("prefill"):
            result["checkpoint_id"] = "sha256:model-a"
        else:
            seen.append(body)
        return result

    with patch.object(cluster, "_post", endpoint), patch.object(cluster, "PREFETCH", False):
        gateway().complete(dict(prompt="abc", max_tokens=1))
    assert seen[0]["checkpoint_id"] == "sha256:model-a"


def tiny_checkpoint(path):
    import torch
    from safetensors.torch import save_file
    path.mkdir()
    config = dict(hidden_size=4, intermediate_size=8, num_attention_heads=1,
                  num_key_value_heads=1, num_hidden_layers=1, vocab_size=8,
                  rms_norm_eps=1e-5)
    (path / "config.json").write_text(json.dumps(config))
    tensors = {"model.embed_tokens.weight": torch.ones(8, 4),
               "model.norm.weight": torch.ones(4), "lm_head.weight": torch.ones(8, 4)}
    for name, shape in [("input_layernorm.weight", (4,)),
                        ("post_attention_layernorm.weight", (4,)),
                        ("self_attn.q_proj.weight", (4, 4)),
                        ("self_attn.k_proj.weight", (4, 4)),
                        ("self_attn.v_proj.weight", (4, 4)),
                        ("self_attn.o_proj.weight", (4, 4)),
                        ("mlp.gate_proj.weight", (8, 4)),
                        ("mlp.up_proj.weight", (8, 4)),
                        ("mlp.down_proj.weight", (4, 8))]:
        tensors["model.layers.0." + name] = torch.ones(shape)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model.safetensors" for name in tensors}}))


def test_native_atom_factory_uses_one_session_for_all_requests(tmp_path, monkeypatch):
    import sys
    from rngd_pd.serving.prepared import build_prepared
    ck = tmp_path / "ck"
    tiny_checkpoint(ck)
    artifact = tmp_path / "buckets.rbln"
    artifact.write_bytes(b"compiled")
    (tmp_path / "buckets.rbln.json").write_text(json.dumps(
        dict(layers=1, variants=["B1-S8", "B2-S16"])))
    loaded, shapes = [], []

    class Compiled:
        def __init__(self, path):
            loaded.append(path)

        def create_runtime(self, **kwargs):
            return self

        def run(self, x, K, V, cos, sin, mask, oh):
            shapes.append((x.shape[0], mask.shape[1], float(K[0, 0, 0, 0, 0])))
            lg = np.zeros((len(x), 8)); lg[:, 3] = 1
            return lg, np.zeros((1, len(x), 1, 4)), np.zeros((1, len(x), 1, 4))

    monkeypatch.setitem(sys.modules, "rebel", SimpleNamespace(RBLNCompiledModel=Compiled))
    cfg = tmp_path / "prepared.json"
    cfg.write_text(json.dumps(dict(engine="atom", checkpoint="ck", checkpoint_id="immutable",
                                  artifact="buckets.rbln", variants=["B1-S8", "B2-S16"])))
    adapter = build_prepared(cfg, 16, 0)
    w = cluster.PreparedDecodeWorker(adapter, 16, wait_ms=0)
    w._load_kv = lambda rid: (np.full((1, 1, 4, 16), 11., np.float32),
                              np.zeros((1, 1, 16, 4), np.float32), 0., 0., None)
    assert w.run("a", 1, 2, 2, checkpoint_id="immutable")["tokens"] == [1, 3]
    assert w.run("b", 2, 10, 2, checkpoint_id="immutable")["tokens"] == [2, 3]
    assert len(loaded) == 1
    assert shapes == [(1, 8, 11.), (2, 16, 11.)]


def test_rngd_factory_imports_kv_into_real_engine_without_reuploading_weights(tmp_path, monkeypatch):
    import sys
    import torch
    from rngd_pd.swap import rngd
    from rngd_pd.serving.prepared import build_prepared
    ck = tmp_path / "ck"
    tiny_checkpoint(ck)
    for name, val in dict(H=4, M=8, NH=1, NKV=1, HD=4).items():
        monkeypatch.setattr(rngd, name, val)
    monkeypatch.setitem(sys.modules, "rngd_pd.engines.fsdk", SimpleNamespace(
        dev_str=lambda chip: "cpu", open_device=lambda *args: None, sync=lambda chip: None))
    observed = []

    class Decoder(torch.nn.Module):
        def forward(self, *a):
            observed.append(float(a[-5][0, 0, 0, 0, 0]))
            return a[0], a[-5], a[-4]

    class Head(torch.nn.Module):
        def forward(self, x, *args):
            logits = torch.zeros((32, 1, len(x))); logits[1] = 1
            return logits

    monkeypatch.setattr(rngd.SwappableDecode, "_graphs", lambda self, v:
                        (Decoder(), Head(), ["cache", "cache"]))
    cfg = tmp_path / "prepared.json"
    cfg.write_text(json.dumps(dict(engine="rngd", checkpoint="ck", checkpoint_id="immutable",
                                  variants=["B1-S8-g1", "B2-S16-g1"])))
    adapter = build_prepared(cfg, 16, 0)
    resident_weight = adapter.engine.shared[0]["wq"]
    w = cluster.PreparedDecodeWorker(adapter, 16, wait_ms=0)
    w._load_kv = lambda rid: (np.full((1, 1, 4, 16), 11., np.float32),
                              np.zeros((1, 1, 16, 4), np.float32), 0., 0., None)
    assert w.run("a", 2, 2, 2, checkpoint_id="immutable")["tokens"] == [2, 1]
    assert w.run("b", 3, 10, 2, checkpoint_id="immutable")["tokens"] == [3, 1]
    assert adapter.engine.shared[0]["wq"] is resident_weight
    assert observed == [11., 11.]


def test_atom_device_input_factory_and_worker_delta_use_real_cpu_tensors(tmp_path, monkeypatch):
    import shutil
    import sys
    import torch
    from safetensors.torch import load_file, save_file
    from rngd_pd.serving.prepared import build_prepared
    ck = tmp_path / "ck"
    tiny_checkpoint(ck)
    monkeypatch.setitem(sys.modules, "rebel", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_rbln", SimpleNamespace())
    monkeypatch.setattr(torch, "rbln", SimpleNamespace(synchronize=lambda dev: None), raising=False)
    original_to = torch.Tensor.to

    def cpu_device(tensor, *args, **kwargs):
        args = tuple("cpu" if isinstance(a, str) and a.startswith("rbln:") else a for a in args)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_device)
    compiled = []

    def external_compile(fn, **kwargs):
        compiled.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", external_compile)
    cfg = tmp_path / "prepared.json"
    cfg.write_text(json.dumps(dict(engine="atom-inputs", checkpoint="ck", checkpoint_id="base",
                                  cache_dir="cache", variants=["B1-S8", "B2-S16"])))
    adapter = build_prepared(cfg, 16, 0)
    w = cluster.PreparedDecodeWorker(adapter, 16, wait_ms=0)
    w._load_kv = lambda rid: (np.zeros((1, 1, 4, 16), np.float32),
                              np.zeros((1, 1, 16, 4), np.float32), 0., 0., None)
    before = list(adapter.engine.W)
    assert w.run("a", 1, 2, 2, checkpoint_id="base")["tokens"] == [1, 0]
    assert w.run("b", 2, 10, 2, checkpoint_id="base")["tokens"] == [2, 0]
    assert all(a is b for a, b in zip(before, adapter.engine.W))
    new = tmp_path / "new"
    shutil.copytree(ck, new)
    weights = {name: t.clone() for name, t in load_file(str(new / "model.safetensors")).items()}
    weights["model.layers.0.self_attn.k_proj.weight"][0, 0] = 2.
    save_file(weights, str(new / "model.safetensors"))
    result = w.load_delta(str(new), "new")
    assert result["uploaded_bytes"] == 32
    assert sum(a is not b for a, b in zip(before, adapter.engine.W)) == 1
    assert len(compiled) == 1
    assert w.run("new", 1, 2, 2, checkpoint_id="new")["tokens"] == [1, 0]
