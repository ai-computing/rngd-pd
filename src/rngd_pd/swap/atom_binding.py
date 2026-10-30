"""Compile-free ATOM B1 owner for ADR-0010. Transport belongs to the worker.

Artifacts are verified by onggi before admission. A manifest is the caller's
verified identity, never a digest guessed from a graph or checkpoint file.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import re
from threading import RLock

from .delta import parse_onggi_switchplan


def _manifest(value):
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("manifest must be a lowercase sha256 digest")


def _inside(artifact, relative):
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("expected an artifact-relative path")
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError("expected an artifact-relative path")
    root = Path(artifact).resolve(strict=True)
    result = (root / str(path)).resolve(strict=True)
    if not result.is_relative_to(root):
        raise ValueError("path escapes the artifact")
    return result


def _metadata(artifact):
    with _inside(artifact, "atom-binding.json").open() as stream:
        meta = json.load(stream)
    if not isinstance(meta, dict):
        raise ValueError("atom-binding.json must be an object")
    graph = _inside(artifact, meta.get("graph"))
    if not graph.is_file():
        raise ValueError("graph must be a compiled file")
    return meta, graph


def load_atom_session(artifact, *, device=0, tables=None):
    """Public native hook: parse an existing .rbln and create its runtime.

    tables may retain the serving worker's already loaded host embedding tables.
    No compile or ConstantWeightDecode.prepare call is reachable here.
    """
    from .atom import AtomSession, Tables
    from .variants import Variant

    meta, graph = _metadata(artifact)
    # AtomSession reads documented publisher sidecar metadata, not vendor bytes.
    sidecar = _inside(artifact, str(Path(str(graph) + ".json").relative_to(Path(artifact).resolve())))
    with sidecar.open() as stream:
        shape = json.load(stream)
    if (not isinstance(shape, dict) or type(shape.get("layers")) is not int
            or shape["layers"] <= 0 or not isinstance(shape.get("variants"), list)
            or not shape["variants"] or any(not isinstance(v, str) for v in shape["variants"])):
        raise ValueError("compiled graph metadata needs positive layers and nonempty variants")
    variants = [Variant.parse(v, 0) for v in shape["variants"]]
    if tables is None:
        tables = Tables(str(_inside(artifact, meta.get("checkpoint"))))
    return AtomSession(str(graph), tables, device=device, variants=variants)


class AtomBinding:
    """Persistent, single-owner B1 binding with serialized load and inference.

    factory(artifact) must return a fully loaded, closeable native owner or raise
    without leaving a usable runtime behind. Tests can inject a CPU factory;
    those tests are not evidence of native residency.
    """

    def __init__(self, factory=None, *, device=0, tables=None):
        if type(device) is not int or device < 0:
            raise ValueError("an explicitly allocated device index is required")
        self._runtime_api = "public-AtomSession" if factory is None else "injected-factory"
        self._factory = factory or (lambda artifact: load_atom_session(
            artifact, device=device, tables=tables))
        self._lock = RLock()
        self._engine = None
        self._manifest = None
        self._evidence = None

    def describe(self, artifact):
        """Return the publisher's profile.json; never inspect compiled bytes."""
        with _inside(artifact, "profile.json").open() as stream:
            profile = json.load(stream)
        if not isinstance(profile, dict) or profile.get("schema") != "onggi.profile.v1":
            raise ValueError("unsupported onggi profile schema")
        for key in ("device", "runtime", "format", "layout"):
            if key in profile and not isinstance(profile[key], str):
                raise ValueError(f"profile {key} must be a string")
        if "batch" in profile and (type(profile["batch"]) is not int or profile["batch"] < 0):
            raise ValueError("profile batch must be a nonnegative integer")
        return profile

    def _admission(self, artifact, manifest):
        _manifest(manifest)
        self.describe(artifact)
        _, graph = _metadata(artifact)
        digest = hashlib.sha256()
        with graph.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        return {"graph": str(graph), "sha256": digest.hexdigest()}

    def load(self, artifact, manifest):
        """Validate, drain and replace; a failed replacement leaves empty state."""
        with self._lock:
            evidence = self._admission(artifact, manifest)
            self.close()
            engine = self._factory(artifact)
            if engine is None or not callable(getattr(engine, "close", None)):
                raise ValueError("factory must return a loaded closeable engine")
            self._engine, self._manifest = engine, manifest
            self._evidence = dict(evidence, runtime_api=self._runtime_api)
            return self.resident()

    def attach(self, artifact, manifest, engine):
        """Take ownership of an already prepared experimental constant engine.

        The worker must verify artifact/engine provenance before this call. This
        hook does not compile, restore saved weight-free pointer metadata, or
        claim that ConstantWeightDecode is a public vendor ABI.
        """
        with self._lock:
            if self._engine is not None:
                raise RuntimeError("a model is already loaded; close the owner before attaching")
            evidence = self._admission(artifact, manifest)
            active = getattr(engine, "active", None)
            if (active is None or active not in getattr(engine, "graphs", {})
                    or active not in getattr(engine, "compiled", {})
                    or getattr(engine, "_closed", False)
                    or not callable(getattr(engine, "close", None))):
                raise ValueError("attach requires an already prepared constant graph")
            self._engine, self._manifest = engine, manifest
            self._evidence = dict(evidence, runtime_api="experimental-constant-weight-free-abi")
            return self.resident()

    def resident(self):
        with self._lock:
            ready = (self._engine is not None and self._manifest is not None
                     and not getattr(self._engine, "_closed", False)
                     and getattr(self._engine, "rt", True) is not None)
            return {"schema": "onggi.resident.v1", "manifest": self._manifest if ready else None,
                    "state": "ready" if ready else "empty"}

    @property
    def engine(self):
        """Compatibility view for inspection; inference must use request()."""
        with self._lock:
            return self._engine

    @contextmanager
    def request(self):
        """Keep the owner alive and block reload/close for a whole request."""
        with self._lock:
            if self.resident()["state"] != "ready":
                raise RuntimeError("no live resident ATOM engine")
            yield self._engine

    def graph_evidence(self):
        """Source SHA-256 for reuse auditing; not a device readback digest."""
        with self._lock:
            return dict(self._evidence) if self._evidence is not None else None

    def plan(self, switch_plan):
        parsed = parse_onggi_switchplan(switch_plan)
        with self._lock:
            reason = ("base manifest is not resident" if self.resident()["manifest"] != parsed["base_manifest"]
                      else "B1 ATOM binding requires a full runtime reload")
        return {"schema": "onggi.updateplan.v1", "base_manifest": parsed["base_manifest"],
                "manifest": parsed["manifest"], "mode": "reload", "reason": reason,
                "bytes": parsed["size"], "units": []}

    def apply(self, update_plan, target):
        raise NotImplementedError("B1 ATOM binding does not implement apply")

    def rollback(self):
        raise NotImplementedError("B1 ATOM binding does not implement rollback")

    def close(self):
        with self._lock:
            try:
                if self._engine is not None:
                    self._engine.close()
                self._engine = None
            finally:
                self._manifest = self._evidence = None
