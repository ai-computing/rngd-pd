"""B1 owner behavior on CPU; these tests do not establish native conformance."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from rngd_pd.swap.atom_binding import AtomBinding


BASE = "sha256:" + "a" * 64
TARGET = "sha256:" + "b" * 64


@pytest.fixture
def artifact(tmp_path):
    profile = {"schema": "onggi.profile.v1", "device": "RBLN-CA22",
               "runtime": "rebel", "format": "rbln", "batch": 1,
               "future": {"nested": [1]}}
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    (tmp_path / "session.rbln").write_bytes(b"opaque precompiled graph")
    (tmp_path / "session.rbln.json").write_text(json.dumps(
        {"layers": 1, "variants": ["B1-S8"]}))
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "atom-binding.json").write_text(json.dumps(
        {"graph": "session.rbln", "checkpoint": "checkpoint"}))
    return tmp_path


@pytest.fixture
def switch():
    return {"schema": "onggi.switchplan.v1", "base_manifest": BASE,
            "manifest": TARGET, "size": 8, "same": 3, "moved": 3, "new": 2,
            "files": [{"path": "model.rbln", "base": "model.rbln", "size": 8,
                       "same": 3, "moved": 3, "new": 2, "ranges": [
                           {"offset": 0, "length": 3, "kind": "same"},
                           {"offset": 3, "length": 3, "kind": "elsewhere",
                            "source_file": "model.rbln", "source_offset": 19},
                           {"offset": 6, "length": 2, "kind": "new"}]}]}


def test_describe_reads_publisher_metadata_without_loading_runtime(artifact):
    def forbidden(*args):
        pytest.fail("describe must not load a runtime")
    owner = AtomBinding(factory=forbidden)
    profile = owner.describe(artifact)
    assert profile["device"] == "RBLN-CA22"
    profile["future"]["nested"].append(2)
    assert owner.describe(artifact)["future"] == {"nested": [1]}
    assert owner.resident() == {"schema": "onggi.resident.v1", "manifest": None,
                                "state": "empty"}


@pytest.mark.parametrize("profile", [[], {}, {"schema": "onggi.profile.v2"},
                                    {"schema": "onggi.profile.v1", "batch": True}])
def test_describe_rejects_invalid_profile(artifact, profile):
    (artifact / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        AtomBinding().describe(artifact)


@pytest.mark.parametrize("manifest", [None, "base", "sha256:" + "A" * 64, True])
def test_invalid_manifest_never_loads_device(artifact, manifest):
    def forbidden(*args):
        pytest.fail("invalid manifest reached device factory")
    owner = AtomBinding(factory=forbidden)
    with pytest.raises(ValueError):
        owner.load(artifact, manifest)
    assert owner.resident()["state"] == "empty"


def test_load_admits_identity_only_after_success_and_close_clears_it(artifact):
    closed = []
    engine = SimpleNamespace(close=lambda: closed.append(True))
    owner = AtomBinding(factory=lambda path: engine)
    owner.load(artifact, BASE)
    assert owner.engine is engine
    assert owner.resident() == {"schema": "onggi.resident.v1", "manifest": BASE,
                                "state": "ready"}
    owner.close()
    assert closed == [True]
    assert owner.resident()["manifest"] is None


def test_load_failure_never_admits_manifest(artifact):
    def fail(path):
        raise RuntimeError("device allocation failed")
    owner = AtomBinding(factory=fail)
    with pytest.raises(RuntimeError, match="allocation"):
        owner.load(artifact, BASE)
    assert owner.resident()["state"] == "empty"


def test_plan_always_reloads_and_is_pure_for_empty_ready_and_wrong_base(artifact, switch):
    engine = SimpleNamespace(close=lambda: None)
    owner = AtomBinding(factory=lambda path: engine)
    for manifest in (None, BASE, TARGET):
        if manifest is not None:
            owner.close()
            owner.load(artifact, manifest)
        before, original = owner.resident(), copy.deepcopy(switch)
        plan = owner.plan(switch)
        assert plan == owner.plan(switch)
        assert plan["schema"] == "onggi.updateplan.v1"
        assert (plan["base_manifest"], plan["manifest"]) == (BASE, TARGET)
        assert plan["mode"] == "reload" and plan["reason"] and plan["units"] == []
        assert plan["bytes"] == 8
        assert owner.resident() == before and switch == original
    assert "base" in owner.plan(switch)["reason"]


def test_plan_uses_strict_switchplan_parser(switch):
    switch["files"][0]["ranges"][0]["length"] = 2
    with pytest.raises(ValueError, match="partition"):
        AtomBinding().plan(switch)


def test_consecutive_loads_release_old_runtime_before_loading_replacement(artifact):
    live = []
    def factory(path):
        assert live == [], "two native owners must not overlap on one card"
        engine = SimpleNamespace(close=lambda: live.remove(engine))
        live.append(engine)
        return engine
    owner = AtomBinding(factory=factory)
    owner.load(artifact, BASE)
    original = owner.engine
    owner.load(artifact, TARGET)
    assert owner.engine is not original and live == [owner.engine]
    assert owner.resident()["manifest"] == TARGET


def test_replacement_failure_leaves_empty_after_releasing_old_runtime(artifact):
    closed = []
    def factory(path):
        if closed:
            raise RuntimeError("replacement allocation failed")
        return SimpleNamespace(close=lambda: closed.append(True))
    owner = AtomBinding(factory=factory)
    owner.load(artifact, BASE)
    with pytest.raises(RuntimeError, match="replacement"):
        owner.load(artifact, TARGET)
    assert closed == [True] and owner.engine is None
    assert owner.resident()["state"] == "empty" and owner.graph_evidence() is None


def test_invalid_replacement_is_rejected_before_closing_current_runtime(artifact):
    closed = []
    owner = AtomBinding(factory=lambda path: SimpleNamespace(close=lambda: closed.append(True)))
    owner.load(artifact, BASE)
    with pytest.raises(ValueError, match="manifest"):
        owner.load(artifact, "invalid")
    assert closed == [] and owner.resident()["manifest"] == BASE


@pytest.mark.parametrize("operation", ["load", "close"])
def test_request_serializes_runtime_replacement_and_teardown(artifact, operation):
    closed, attempted, finished = Event(), Event(), Event()
    owner = AtomBinding(factory=lambda path: SimpleNamespace(close=closed.set))
    owner.load(artifact, BASE)
    def change():
        attempted.set()
        if operation == "load":
            owner.load(artifact, TARGET)
        else:
            owner.close()
        finished.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with owner.request() as engine:
            assert engine is owner.engine
            pending = pool.submit(change)
            assert attempted.wait(1)
            assert not finished.wait(0.03) and not closed.is_set()
        pending.result(timeout=1)
    assert finished.is_set() and closed.is_set()


def test_request_requires_live_resident_and_releases_lock_on_inference_error(artifact):
    owner = AtomBinding(factory=lambda path: SimpleNamespace(close=lambda: None))
    with pytest.raises(RuntimeError, match="resident"):
        with owner.request():
            pytest.fail("an empty owner cannot serve")
    owner.load(artifact, BASE)
    with pytest.raises(ValueError, match="inference"):
        with owner.request():
            raise ValueError("inference error")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(owner.close).result(timeout=1)
    assert owner.resident()["state"] == "empty"


def test_retained_constant_graph_attachment_does_not_call_prepare(artifact):
    def forbidden(*args):
        pytest.fail("binding must not compile/prepare an attached graph")
    engine = SimpleNamespace(prepare=forbidden, close=lambda: None,
                             active="B1-S8", graphs={"B1-S8": object()},
                             compiled={"B1-S8": object()})
    owner = AtomBinding(factory=forbidden)
    owner.attach(artifact, BASE, engine)
    assert owner.engine is engine and owner.resident()["manifest"] == BASE
    assert "experimental" in owner.graph_evidence()["runtime_api"]
    with pytest.raises(NotImplementedError):
        owner.apply({}, artifact)
    with pytest.raises(NotImplementedError):
        owner.rollback()


@pytest.mark.parametrize("graph", ["../outside.rbln", "C:/outside.rbln", "/outside.rbln"])
def test_graph_paths_cannot_escape_artifact(artifact, graph):
    (artifact / "atom-binding.json").write_text(json.dumps({"graph": graph}))
    owner = AtomBinding(factory=lambda path: pytest.fail("invalid graph reached factory"))
    with pytest.raises(ValueError, match="relative"):
        owner.load(artifact, BASE)
    assert owner.resident()["state"] == "empty"


def test_unprepared_constant_engine_cannot_claim_residency(artifact):
    owner = AtomBinding()
    with pytest.raises(ValueError, match="prepared"):
        owner.attach(artifact, BASE, SimpleNamespace(active=None, close=lambda: None))
    assert owner.resident()["state"] == "empty"


def test_native_loader_reuses_precompiled_file_and_host_tables_without_compilation(artifact, monkeypatch):
    from rngd_pd.swap import atom

    def forbidden(*args, **kwargs):
        pytest.fail("binding must not compile or reload supplied host tables")

    parsed = []

    class Compiled:
        def __init__(self, path):
            parsed.append(path)

        def create_runtime(self, *, device, tensor_type):
            assert device == 2 and tensor_type == "np"
            return SimpleNamespace(run=lambda *args: None)

    monkeypatch.setitem(sys.modules, "rebel", SimpleNamespace(
        RBLNCompiledModel=Compiled, compile_from_torch=forbidden))
    monkeypatch.setattr(atom, "Tables", forbidden)
    tables = object()
    owner = AtomBinding(device=2, tables=tables)
    before = hashlib.sha256((artifact / "session.rbln").read_bytes()).hexdigest()
    for _ in range(2):
        owner.load(artifact, BASE)
        assert owner.engine.tables is tables and owner.engine.active.name == "B1-S8"
        evidence = owner.graph_evidence()
        assert evidence["sha256"] == before
        assert evidence["runtime_api"] == "public-AtomSession"
        evidence["sha256"] = "mutated"
        assert owner.graph_evidence()["sha256"] == before
        owner.close()
    assert parsed == [str(artifact / "session.rbln")] * 2
    assert hashlib.sha256((artifact / "session.rbln").read_bytes()).hexdigest() == before


def test_attached_closed_engine_cannot_remain_ready(artifact):
    engine = SimpleNamespace(active="B1-S8", graphs={"B1-S8": object()},
                             compiled={"B1-S8": object()}, close=lambda: None, _closed=False)
    owner = AtomBinding()
    owner.attach(artifact, BASE, engine)
    engine._closed = True
    assert owner.resident() == {"schema": "onggi.resident.v1", "manifest": None,
                                "state": "empty"}


@pytest.mark.parametrize("sidecar", [{}, {"layers": 1, "variants": []}])
def test_native_loader_rejects_missing_shape_metadata_before_runtime_creation(artifact, sidecar, monkeypatch):
    from rngd_pd.swap.atom_binding import load_atom_session
    def forbidden(*args, **kwargs):
        pytest.fail("missing shapes must be rejected before runtime creation")
    (artifact / "session.rbln.json").write_text(json.dumps(sidecar))
    monkeypatch.setitem(sys.modules, "rebel", SimpleNamespace(RBLNCompiledModel=forbidden))
    with pytest.raises(ValueError, match="metadata"):
        load_atom_session(artifact, tables=object())
