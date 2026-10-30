"""CPU transaction tests; these do not demonstrate native RNGD residency."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import threading

import pytest
import torch

from rngd_pd.swap.delta import read_header
from rngd_pd.swap.rngd_binding import RngdBinding


BASE, TARGET, THIRD = ("sha256:" + char * 64 for char in "abc")
K = "model.layers.0.self_attn.k_proj.weight"
V = "model.layers.0.self_attn.v_proj.weight"
Q = "model.layers.0.self_attn.q_proj.weight"
LM = "lm_head.weight"


def raw(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def artifact(path, tensors=None, gap=0):
    path.mkdir()
    tensors = tensors or {K: torch.arange(12).reshape(4, 3).to(torch.bfloat16),
                          V: torch.ones(4, 3, dtype=torch.bfloat16),
                          Q: torch.ones(4, 3, dtype=torch.bfloat16),
                          LM: torch.arange(15).reshape(5, 3).to(torch.bfloat16)}
    header, data = {}, bytearray()
    for i, (name, value) in enumerate(tensors.items()):
        if i and gap:
            data.extend(b"\0" * gap)
        start = len(data)
        data.extend(raw(value))
        header[name] = {"dtype": "BF16" if value.dtype == torch.bfloat16 else "F32",
                        "shape": list(value.shape), "data_offsets": [start, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (path / "config.json").write_text(json.dumps({"hidden_size": 3, "num_hidden_layers": 2,
                                                 "vocab_size": tensors[LM].shape[0],
                                                 "tie_word_embeddings": False}))
    return path


def switch(base, target, spans=(), base_manifest=BASE, manifest=TARGET):
    files = []
    for path in sorted(target.iterdir()):
        size = path.stat().st_size
        ranges, cursor = [], 0
        for off, length in sorted(spans if path.suffix == ".safetensors" else []):
            if cursor < off:
                ranges.append({"offset": cursor, "length": off - cursor, "kind": "same"})
            ranges.append({"offset": off, "length": length, "kind": "new"})
            cursor = off + length
        if cursor < size:
            ranges.append({"offset": cursor, "length": size - cursor, "kind": "same"})
        changed = sum(r["length"] for r in ranges if r["kind"] == "new")
        files.append({"path": path.name, "base": path.name, "size": size,
                      "same": size - changed, "moved": 0, "new": changed, "ranges": ranges})
    return {"schema": "onggi.switchplan.v1", "base_manifest": base_manifest,
            "manifest": manifest, "files": files,
            **{key: sum(f[key] for f in files) for key in ("size", "same", "moved", "new")}}


class Engine:
    def __init__(self, path, **kwargs):
        self.options = kwargs
        self.torch, self.dev, self.group, self.nl = torch, "cpu", 2, 2
        self._lock, self._generation, self._failed = threading.RLock(), 0, False
        self.lm_name, self.closed = LM, False
        self.shared = [{slot: torch.zeros(2, 3, 4, dtype=torch.bfloat16)
                        for slot in ("wk", "wv", "wq")}]
        headers = read_header(Path(path) / "model.safetensors")
        values = {}
        with (Path(path) / "model.safetensors").open("rb") as stream:
            for name, info in headers.items():
                stream.seek(info["start"])
                dtype = torch.bfloat16 if info["dtype"] == "BF16" else torch.float32
                values[name] = torch.frombuffer(bytearray(stream.read(info["end"] - info["start"])),
                                               dtype=dtype).reshape(info["shape"]).clone()
        for name, slot in ((K, "wk"), (V, "wv"), (Q, "wq")):
            if name in values:
                rows, cols = values[name].shape
                self.shared[0][slot] = torch.zeros(2, cols, rows, dtype=torch.bfloat16)
                self.shared[0][slot][0].copy_(values[name].t())
        self.Vp = (values[LM].shape[0] + 31) // 32 * 32
        self.wlm_d = torch.zeros(32, self.Vp // 32, 3, dtype=torch.bfloat16)
        self.wlm_d.reshape(-1, 3)[:values[LM].shape[0]].copy_(values[LM])

    def swap(self, variant):
        self.active = variant

    def _sync(self):
        pass

    def close(self):
        self.closed, self._failed = True, True


@pytest.fixture
def setup(tmp_path):
    base = artifact(tmp_path / "base")
    target = tmp_path / "target"
    import shutil
    shutil.copytree(base, target)
    headers = read_header(base / "model.safetensors")
    off = headers[K]["start"] + 6
    with (target / "model.safetensors").open("r+b") as stream:
        stream.seek(off)
        stream.write(raw(torch.tensor([[50, 51, 52]], dtype=torch.bfloat16)))
    engines = []
    def factory(path, **kwargs):
        engine = Engine(path, **kwargs)
        engines.append(engine)
        return engine
    owner = RngdBinding(factory=factory, expected_device="cpu")
    owner.load(base, BASE)
    yield owner, engines[0], base, target, switch(base, target, [(off, 6)])
    owner.close()


def test_plan_is_pure_deterministic_and_keeps_equal_row_false_positives(setup, monkeypatch):
    owner, engine, base, target, doc = setup
    # The second row here is deliberately unchanged; the chunk still covers it.
    start = read_header(base / "model.safetensors")[K]["start"]
    doc = switch(base, target, [(start + 1, 11)])
    def forbidden(*args, **kwargs):
        raise AssertionError("planning attempted I/O or device access")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    first = owner.plan(doc)
    assert first == owner.plan(doc)
    assert first["mode"] == "units"
    assert first["bytes"] == 12
    assert first["units"] == [{"file": "model.safetensors", "offset": start,
                               "length": 12, "unit": K + "@0:2", "source": "target"}]
    assert engine._generation == 0


def test_apply_actual_readback_and_successful_rollback_remain_available(setup):
    owner, engine, base, target, doc = setup
    plan = owner.plan(doc)
    before = engine.shared[0]["wk"].clone()
    result = owner.apply(plan, target)
    assert owner.resident()["manifest"] == TARGET
    assert result["schema"] == "onggi.verification.v1"
    assert result["ok"] is True and result["op"] == "apply"
    assert result["units"][0]["sha256"] == hashlib.sha256(
        raw(torch.tensor([[50, 51, 52]], dtype=torch.bfloat16))).hexdigest()
    assert result["bytes_written"] == 6
    assert result["diagnostics"]["readback_calls"] >= 2
    assert result["diagnostics"]["readback_payload_bytes"] >= 96
    assert torch.equal(engine.shared[0]["wk"][0, :, 1], torch.tensor([50, 51, 52]))
    restored = owner.rollback()
    assert restored["op"] == "rollback" and restored["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], before)
    assert engine._generation == 2
    with pytest.raises(RuntimeError, match="journal"):
        owner.rollback()


@pytest.mark.parametrize("change", ["units", "manifest", "proof", "bytes", "base_manifest"])
def test_tampered_update_is_rejected_before_writes(setup, change):
    owner, engine, _, target, doc = setup
    update = owner.plan(doc)
    if change == "units":
        update["units"] = []
    elif change == "proof":
        update.pop(change)
    elif change == "bytes":
        update[change] = 0
    else:
        update[change] = THIRD
    before = engine.shared[0]["wk"].clone()
    with pytest.raises(ValueError):
        owner.apply(update, target)
    assert owner.resident()["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], before)


def test_wrong_base_plans_reload_without_mutation(setup):
    owner, _, _, target, doc = setup
    doc["base_manifest"] = THIRD
    update = owner.plan(doc)
    assert update["mode"] == "reload" and update["reason"]
    with pytest.raises((NotImplementedError, ValueError)):
        owner.apply(update, target)
    assert owner.resident()["manifest"] == BASE


@pytest.mark.parametrize("where", ["header", "q"])
def test_header_and_scaled_query_changes_require_reload(setup, where):
    owner, _, base, target, _ = setup
    headers = read_header(base / "model.safetensors")
    off = 0 if where == "header" else headers[Q]["start"]
    assert owner.plan(switch(base, target, [(off, 1)]))["mode"] == "reload"


def test_same_path_target_patch_uses_actual_device_journal(setup):
    owner, engine, base, target, doc = setup
    before = engine.shared[0]["wk"].clone()
    update = owner.plan(doc)
    # onggi can install target bytes in place after the base was loaded.
    (base / "model.safetensors").write_bytes((target / "model.safetensors").read_bytes())
    owner.apply(update, base)
    assert owner.resident()["manifest"] == TARGET
    assert not torch.equal(engine.shared[0]["wk"], before)
    owner.rollback()
    assert owner.resident()["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], before)
    assert (base / "model.safetensors").read_bytes() == (target / "model.safetensors").read_bytes()


@pytest.mark.parametrize("change", ["deleted", "modified"])
def test_apply_and_rollback_do_not_read_previous_checkpoint_bytes(setup, change):
    owner, engine, base, target, doc = setup
    before = engine.shared[0]["wk"].clone()
    update = owner.plan(doc)
    import shutil
    if change == "deleted":
        shutil.rmtree(base)
    else:
        (base / "model.safetensors").write_bytes(b"no longer a checkpoint")
    owner.apply(update, target)
    # Undo is also independent of either artifact remaining on the filesystem.
    shutil.rmtree(target)
    result = owner.rollback()
    assert result["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], before)


def test_partial_write_failure_restores_every_attempted_unit(setup, monkeypatch):
    owner, engine, _, target, doc = setup
    before = engine.shared[0]["wk"].clone()
    original = owner._write
    calls = 0
    def fail_after_copy(unit, rows):
        nonlocal calls
        calls += 1
        original(unit, rows)
        if calls == 1:
            raise KeyboardInterrupt("partial copy")
    monkeypatch.setattr(owner, "_write", fail_after_copy)
    with pytest.raises(KeyboardInterrupt):
        owner.apply(owner.plan(doc), target)
    assert calls == 2
    assert owner.resident()["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], before)


def test_silent_write_corruption_is_detected_and_restored(setup, monkeypatch):
    owner, engine, _, target, doc = setup
    before = engine.shared[0]["wk"].clone()
    original = owner._write
    calls = 0
    def corrupt(unit, rows):
        nonlocal calls
        calls += 1
        original(unit, rows)
        if calls == 1:
            engine.shared[0]["wk"][0, 0, 1] = 999
    monkeypatch.setattr(owner, "_write", corrupt)
    with pytest.raises(RuntimeError, match="readback"):
        owner.apply(owner.plan(doc), target)
    assert owner.resident()["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], before)


def test_failed_restoration_unloads_instead_of_serving_mixed_weights(setup, monkeypatch):
    owner, engine, _, target, doc = setup
    def failure(unit, rows):
        engine.shared[0]["wk"][0, 0, 1] = 999
        raise RuntimeError("copy failed")
    monkeypatch.setattr(owner, "_write", failure)
    with pytest.raises(RuntimeError, match="restore"):
        owner.apply(owner.plan(doc), target)
    assert engine.closed
    assert owner.resident()["state"] == "empty"
    with pytest.raises(RuntimeError):
        with owner.request():
            pass


def test_request_context_blocks_apply_for_the_whole_request(setup):
    owner, engine, _, target, doc = setup
    update, entered, finished = owner.plan(doc), threading.Event(), threading.Event()
    failures = []
    def apply():
        entered.set()
        try:
            owner.apply(update, target)
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()
    with owner.request() as request_engine:
        assert request_engine is engine
        thread = threading.Thread(target=apply)
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(0.05)
        assert owner.resident()["manifest"] == BASE
    thread.join(2)
    assert finished.is_set() and not failures
    assert owner.resident()["manifest"] == TARGET


def test_failure_while_publishing_restores_previous_identity_and_bytes(setup, monkeypatch):
    owner, engine, _, target, doc = setup
    before = engine.shared[0]["wk"].clone()
    def fail(*args):
        raise KeyboardInterrupt("interrupted while producing verification")
    monkeypatch.setattr(owner, "_verification", fail)
    with pytest.raises(KeyboardInterrupt):
        owner.apply(owner.plan(doc), target)
    assert owner.resident()["manifest"] == BASE
    assert owner.resident()["state"] == "ready"
    assert torch.equal(engine.shared[0]["wk"], before)


def test_adjacent_tensors_and_lm_padding_map_to_complete_rows(setup):
    owner, engine, base, target, _ = setup
    header = read_header(base / "model.safetensors")
    off = header[K]["end"] - 1
    update = owner.plan(switch(base, target, [(off, 2)]))
    assert update["mode"] == "units"
    assert [(u["unit"], u["length"]) for u in update["units"]] == [(K + "@3:4", 6), (V + "@0:1", 6)]
    lm = header[LM]
    update = owner.plan(switch(base, target, [(lm["start"], lm["end"] - lm["start"])]))
    assert [u["unit"] for u in update["units"]] == [LM + f"@{row}:{row + 1}" for row in range(5)]
    before_pad = engine.wlm_d.reshape(-1, 3)[5:].clone()
    result = owner.apply(update, target)
    assert result["bytes_written"] == 30
    assert torch.equal(engine.wlm_d.reshape(-1, 3)[5:], before_pad)
    owner.rollback()


def test_plan_unions_overlapping_rounded_rows(setup):
    owner, _, base, target, _ = setup
    start = read_header(base / "model.safetensors")[K]["start"]
    update = owner.plan(switch(base, target, [(start, 1), (start + 3, 1), (start + 7, 1)]))
    assert len(update["units"]) == 1
    assert update["units"][0]["unit"] == K + "@0:2"


def test_identical_plan_is_a_valid_zero_write_transaction(setup):
    owner, engine, base, _, _ = setup
    result = owner.apply(owner.plan(switch(base, base)), base)
    assert result["units"] == [] and result["bytes_written"] == 0
    assert result["manifest"] == TARGET
    assert owner.rollback()["manifest"] == BASE


def test_failed_second_apply_keeps_previous_successful_journal(setup, tmp_path, monkeypatch):
    owner, engine, base, target, doc = setup
    initial_k, initial_v = engine.shared[0]["wk"].clone(), engine.shared[0]["wv"].clone()
    owner.apply(owner.plan(doc), target)
    import shutil
    third = tmp_path / "third"
    shutil.copytree(target, third)
    header = read_header(third / "model.safetensors")
    offsets = [header[K]["start"] + 12, header[V]["start"]]
    with (third / "model.safetensors").open("r+b") as stream:
        for off in offsets:
            stream.seek(off)
            stream.write(raw(torch.tensor([[80, 81, 82]], dtype=torch.bfloat16)))
    next_plan = owner.plan(switch(target, third, [(off, 6) for off in offsets], TARGET, THIRD))
    original, calls = owner._write, 0
    def failure(unit, rows):
        nonlocal calls
        calls += 1
        original(unit, rows)
        if calls == 2:
            raise RuntimeError("second unit partially failed")
    monkeypatch.setattr(owner, "_write", failure)
    with pytest.raises(RuntimeError, match="second unit"):
        owner.apply(next_plan, third)
    assert calls == 4
    assert owner.resident()["manifest"] == TARGET
    monkeypatch.setattr(owner, "_write", original)
    owner.rollback()
    assert owner.resident()["manifest"] == BASE
    assert torch.equal(engine.shared[0]["wk"], initial_k)
    assert torch.equal(engine.shared[0]["wv"], initial_v)


def test_old_load_epoch_plan_is_rejected_even_with_same_manifest(setup):
    owner, _, base, target, doc = setup
    update = owner.plan(doc)
    owner.close()
    owner.load(base, BASE)
    with pytest.raises(ValueError, match="epoch"):
        owner.apply(update, target)


@pytest.mark.parametrize("change", ["config", "header", "index", "file"])
def test_target_compatibility_is_checked_before_any_write(setup, change, monkeypatch):
    owner, _, _, target, doc = setup
    update = owner.plan(doc)
    if change == "config":
        config = json.loads((target / "config.json").read_text())
        config["rope_theta"] = 100
        (target / "config.json").write_text(json.dumps(config))
    elif change == "header":
        path = target / "model.safetensors"
        path.write_bytes(path.read_bytes().replace(b'"BF16"', b'"F16 "', 1))
    elif change == "index":
        (target / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    else:
        (target / "extra.txt").write_text("new file")
    def forbidden(*args):
        raise AssertionError("compatibility failure attempted a write")
    monkeypatch.setattr(owner, "_write", forbidden)
    with pytest.raises(ValueError):
        owner.apply(update, target)
    assert owner.resident()["manifest"] == BASE


def test_gaps_and_float32_weights_are_not_lossless_units(tmp_path):
    tensors = {K: torch.ones(4, 3, dtype=torch.float32),
               V: torch.ones(4, 3, dtype=torch.bfloat16),
               LM: torch.ones(5, 3, dtype=torch.bfloat16)}
    path = artifact(tmp_path / "gap", tensors, gap=2)
    header = read_header(path / "model.safetensors")
    with_owner = RngdBinding(factory=Engine, expected_device="cpu")
    with_owner.load(path, BASE)
    try:
        for off in (header[K]["start"], header[K]["end"]):
            assert with_owner.plan(switch(path, path, [(off, 1)]))["mode"] == "reload"
    finally:
        with_owner.close()


def test_excessive_fragmentation_requires_reload(tmp_path):
    tensors = {K: torch.ones(130, 3, dtype=torch.bfloat16),
               LM: torch.ones(5, 3, dtype=torch.bfloat16)}
    path = artifact(tmp_path / "fragmented", tensors)
    start = read_header(path / "model.safetensors")[K]["start"]
    owner = RngdBinding(factory=Engine, expected_device="cpu")
    owner.load(path, BASE)
    try:
        result = owner.plan(switch(path, path, [(start + row * 6, 1) for row in range(0, 130, 2)]))
        assert result["mode"] == "reload" and "64" in result["reason"]
    finally:
        owner.close()


def test_successive_loads_validate_before_replacing_and_never_overlap_engines(setup):
    owner, initial, _, target, _ = setup
    original = owner._factory
    def replacement(path, **kwargs):
        assert initial.closed, "new allocation began before old owner closed"
        return original(path, **kwargs)
    owner._factory = replacement
    owner.load(target, TARGET)
    assert owner.resident()["manifest"] == TARGET
    with owner.request() as engine:
        assert engine is not initial and not engine.closed


def test_invalid_replacement_metadata_preserves_current_engine(setup, tmp_path):
    owner, engine, _, _, _ = setup
    bad = tmp_path / "invalid"
    bad.mkdir()
    (bad / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        owner.load(bad, TARGET)
    assert owner.resident()["manifest"] == BASE and not engine.closed


def test_replacement_factory_failure_leaves_no_published_identity(setup):
    owner, engine, _, target, _ = setup
    def failure(*args, **kwargs):
        assert engine.closed
        raise RuntimeError("allocation failed")
    owner._factory = failure
    with pytest.raises(RuntimeError, match="allocation failed"):
        owner.load(target, TARGET)
    assert owner.resident()["state"] == "empty"
    assert owner.resident()["manifest"] is None


def test_old_engine_close_failure_prevents_replacement_allocation(setup, monkeypatch):
    owner, engine, _, target, _ = setup
    def close_failure():
        raise RuntimeError("device did not close")
    def forbidden(*args, **kwargs):
        raise AssertionError("replacement overlapped an engine that failed to close")
    monkeypatch.setattr(engine, "close", close_failure)
    owner._factory = forbidden
    with pytest.raises(RuntimeError, match="did not close"):
        owner.load(target, TARGET)
    assert owner.resident()["state"] == "empty"
    # Permit fixture teardown to confirm release after testing quarantine.
    monkeypatch.setattr(engine, "close", lambda: setattr(engine, "closed", True))


def test_failed_close_quarantines_owner_until_explicit_successful_close(setup, monkeypatch):
    owner, engine, _, target, _ = setup
    close_calls, allocations = 0, 0
    original_close, original_factory = engine.close, owner._factory
    def close_failure():
        nonlocal close_calls
        close_calls += 1
        raise RuntimeError("native context release failed")
    def factory(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        assert engine.closed
        return original_factory(*args, **kwargs)
    monkeypatch.setattr(engine, "close", close_failure)
    owner._factory = factory
    with pytest.raises(RuntimeError, match="release failed"):
        owner.load(target, TARGET)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="close"):
            owner.load(target, TARGET)
    assert close_calls == 1 and allocations == 0
    assert owner.resident()["state"] == "empty"
    with pytest.raises(RuntimeError):
        with owner.request():
            pass
    with pytest.raises(RuntimeError, match="release failed"):
        owner.close()
    assert close_calls == 2 and allocations == 0
    with pytest.raises(RuntimeError, match="close"):
        owner.load(target, TARGET)
    monkeypatch.setattr(engine, "close", original_close)
    owner.close()
    owner.load(target, TARGET)
    assert allocations == 1 and owner.resident()["manifest"] == TARGET
