"""RNGD B2 owner: lossless BF16 rows, actual serving readback and retained undo.

The artifact/manifest is verified by the caller before admission. Public Torch
copies are used here; SwappableDecode's inherited graph/cache SDK internals are
not promoted to a public vendor ABI. CPU factories are transaction tests only.
"""
from contextlib import contextmanager
import copy
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from threading import RLock
from time import perf_counter

from .delta import parse_onggi_switchplan, read_header
from .variants import Variant


_SLOTS = {"self_attn.k_proj.weight": "wk", "self_attn.v_proj.weight": "wv",
          "self_attn.o_proj.weight": "wo", "mlp.gate_proj.weight": "wg",
          "mlp.up_proj.weight": "wu", "mlp.down_proj.weight": "wd"}


def _manifest(value):
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("manifest must be a lowercase sha256 digest")


def _config(artifact):
    root = Path(artifact).resolve(strict=True)
    if not root.is_dir() or (root / "config.json").is_symlink():
        raise ValueError("artifact must contain a public config.json")
    with (root / "config.json").open() as stream:
        config = json.load(stream)
    if (not isinstance(config, dict) or any(type(config.get(k)) is not int or config[k] <= 0
            for k in ("hidden_size", "num_hidden_layers", "vocab_size"))):
        raise ValueError("invalid public model configuration")
    return config


def _metadata(artifact):
    root = Path(artifact).resolve(strict=True)
    config, files, headers, names = _config(root), {}, {}, {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlinks are unsupported")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("artifact entries must be regular files")
        relative = path.relative_to(root).as_posix()
        files[relative] = path.stat().st_size
        if path.suffix == ".safetensors":
            headers[relative] = read_header(path)
            for name in headers[relative]:
                if name in names:
                    raise ValueError("duplicate checkpoint tensor: " + name)
                names[name] = relative
    if not headers:
        raise ValueError("artifact has no safetensors weights")
    index = None
    if "model.safetensors.index.json" in files:
        with (root / "model.safetensors.index.json").open() as stream:
            index = json.load(stream)
        if not isinstance(index, dict) or index.get("weight_map") != names:
            raise ValueError("safetensors index disagrees with tensor headers")
    return {"config": config, "files": files, "headers": headers, "index": index}


def _bytes(rows):
    import torch
    return rows.contiguous().view(torch.uint8).numpy().tobytes()


def _digest(rows):
    return hashlib.sha256(_bytes(rows)).hexdigest()


def _parts(unit):
    name, span = unit["unit"].rsplit("@", 1)
    start, end = map(int, span.split(":"))
    return name, start, end


class RngdBinding:
    def __init__(self, factory=None, cache_dir=None, variant="B4-S256-g2", chip=0,
                 expected_device="furiosa:0", profile=None):
        if type(chip) is not int or chip < 0:
            raise ValueError("an explicitly allocated chip index is required")
        if expected_device == "cpu" and factory is None:
            raise ValueError("CPU is permitted only with an injected test factory")
        self._factory, self._cache_dir, self._chip = factory, cache_dir, chip
        self._variant = Variant.parse(variant) if isinstance(variant, str) else variant
        self._expected_device = expected_device
        self._profile = copy.deepcopy(profile) if profile is not None else {
            "schema": "onggi.profile.v1", "device": "rngd", "runtime": "furiosa-torch",
            "format": "safetensors", "layout": "lossless-bf16-rows", "batch": self._variant.B}
        if (not isinstance(self._profile, dict) or self._profile.get("schema") != "onggi.profile.v1"
                or any(k in self._profile and not isinstance(self._profile[k], str)
                       for k in ("device", "runtime", "format", "layout"))
                or ("batch" in self._profile and (type(self._profile["batch"]) is not int
                                                 or self._profile["batch"] < 0))):
            raise ValueError("invalid publisher profile")
        self._lock = RLock()
        self._engine = self._manifest = self._path = self._meta = None
        self._quarantined = None
        self._journal = None
        self._layout, self._key, self._state = {}, secrets.token_bytes(32), "empty"
        self._metrics = {"initial_load_s": 0, "readback_calls": 0,
                         "readback_payload_bytes": 0, "readback_s": 0,
                         "readback_source": "actual-serving-allocation.cpu()",
                         "runtime_dependencies": "inherited SwappableDecode graph/cache SDK internals"}

    def describe(self, artifact):
        _config(artifact)
        return copy.deepcopy(self._profile)

    def _check_tensor(self, tensor):
        if (str(tensor.device) != self._expected_device
                or tensor.dtype != self._engine.torch.bfloat16):
            raise ValueError("serving tensor must be BF16 on " + self._expected_device)
        return tensor

    def _cache_layout(self):
        engine, layout = self._engine, {}
        if str(engine.dev) != self._expected_device:
            raise ValueError("engine device does not match explicitly allocated device")
        for slots in engine.shared:
            for tensor in slots.values():
                self._check_tensor(tensor)
        self._check_tensor(engine.wlm_d)
        for header in self._meta["headers"].values():
            for name, info in header.items():
                if info["dtype"] != "BF16" or len(info["shape"]) != 2:
                    continue
                rows, cols = info["shape"]
                match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
                if match and match[2] in _SLOTS and int(match[1]) < engine.nl:
                    group, layer = divmod(int(match[1]), engine.group)
                    slot = _SLOTS[match[2]]
                    tensor = engine.shared[group].get(slot)
                    if tensor is not None and tuple(tensor.shape) == (engine.group, cols, rows):
                        layout[name] = (group, slot, layer)
                elif (name == "lm_head.weight" and engine.lm_name == name
                      and not self._meta["config"].get("tie_word_embeddings", False)
                      and len(engine.wlm_d.shape) == 3 and engine.wlm_d.shape[2] == cols
                      and engine.wlm_d.shape[0] * engine.wlm_d.shape[1] >= rows):
                    layout[name] = ("lm", int(engine.wlm_d.shape[1]))
        self._layout = layout

    def load(self, artifact, manifest):
        with self._lock:
            if self._quarantined is not None:
                raise RuntimeError("engine release failed; retry close() successfully before load()")
            _manifest(manifest)
            started = perf_counter()
            self.describe(artifact)
            metadata = _metadata(artifact)
            # Invalid replacement metadata preserves the current serving owner.
            # Once native replacement starts, failure leaves an empty identity.
            self._unload(strict=True)
            factory = self._factory
            if factory is None:
                from .rngd import SwappableDecode
                factory = SwappableDecode
            engine = factory(str(artifact), group=self._variant.group, chip=self._chip,
                             cache_dir=self._cache_dir, allow_compile=False, host_workers=2)
            self._engine, self._meta = engine, metadata
            try:
                engine.swap(self._variant)
                self._cache_layout()
            except BaseException:
                self._unload()
                raise
            self._path, self._manifest = Path(artifact).resolve(), manifest
            self._key, self._journal, self._state = secrets.token_bytes(32), None, "ready"
            self._metrics["initial_load_s"] = perf_counter() - started
            return self.resident()

    def resident(self):
        with self._lock:
            available = self._engine is not None and not getattr(self._engine, "_failed", False)
            return {"schema": "onggi.resident.v1", "manifest": self._manifest if available else None,
                    "state": self._state if available else "empty", "diagnostics": dict(self._metrics)}

    def _ready(self):
        if self.resident()["state"] != "ready":
            raise RuntimeError("resident engine is unavailable")

    @contextmanager
    def request(self):
        """Hold ownership for generate/step/KV sessions only.

        Legacy engine load_delta/apply_delta and direct weight mutation are
        unsupported under this owner. The engine's initial Checkpoint handle
        stays open; binding._path records artifact identity, not an immutable
        source of old bytes. Undo always comes from actual serving readback.
        """
        with self._lock:
            self._ready()
            with self._engine._lock:
                yield self._engine

    def _proof(self, update):
        payload = {k: v for k, v in update.items() if k != "proof"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hmac.new(self._key, encoded, hashlib.sha256).hexdigest()

    def plan(self, document):
        parsed = parse_onggi_switchplan(document)
        with self._lock:
            update = {"schema": "onggi.updateplan.v1", "base_manifest": parsed["base_manifest"],
                      "manifest": parsed["manifest"], "mode": "units", "bytes": 0, "units": []}
            reason, rows = None, {}
            if self.resident()["manifest"] != parsed["base_manifest"]:
                reason = "base manifest is not resident"
            elif {f["path"] for f in parsed["files"]} != set(self._meta["files"]):
                reason = "file inventory changed"
            else:
                for fd in parsed["files"]:
                    if (fd.get("base") != fd["path"]
                            or self._meta["files"].get(fd["path"]) != fd["size"]):
                        reason = "file pairing or size changed"
                        break
                    header = self._meta["headers"].get(fd["path"], {})
                    for span in fd["ranges"]:
                        if span["kind"] == "same":
                            continue
                        cursor, end = span["offset"], span["offset"] + span["length"]
                        for name, info in sorted(header.items(), key=lambda item: item[1]["start"]):
                            if info["end"] <= cursor:
                                continue
                            if info["start"] > cursor or name not in self._layout:
                                break
                            row_bytes = info["shape"][1] * 2
                            stop = min(end, info["end"])
                            rows.setdefault((fd["path"], name), []).append(
                                ((cursor - info["start"]) // row_bytes,
                                 (stop - info["start"] + row_bytes - 1) // row_bytes))
                            cursor = stop
                            if cursor == end:
                                break
                        if cursor != end:
                            reason = "changed header, gap, host, scaled or unsupported tensor bytes"
                            break
                    if reason:
                        break
            if not reason:
                for (file, name), spans in sorted(rows.items()):
                    merged = []
                    for start, end in sorted(spans):
                        if merged and start <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], end)
                        else:
                            merged.append([start, end])
                    info = self._meta["headers"][file][name]
                    for start, end in merged:
                        while start < end:
                            layout = self._layout[name]
                            stop = min(end, (start // layout[1] + 1) * layout[1]) if layout[0] == "lm" else end
                            length = (stop - start) * info["shape"][1] * 2
                            update["units"].append({"file": file, "offset": info["start"] + start * info["shape"][1] * 2,
                                                    "length": length, "unit": f"{name}@{start}:{stop}", "source": "target"})
                            start = stop
                if len(update["units"]) > 64:
                    reason = "row fragmentation exceeds 64 units"
            if reason:
                update.update(mode="reload", reason=reason, units=[], bytes=parsed["size"])
            else:
                update["bytes"] = sum(unit["length"] for unit in update["units"])
            update["proof"] = self._proof(update)
            return update

    def _file_rows(self, root, unit):
        name, start, end = _parts(unit)
        info = self._meta["headers"][unit["file"]][name]
        with (root / unit["file"]).open("rb") as stream:
            stream.seek(unit["offset"])
            data = stream.read(unit["length"])
        if len(data) != unit["length"]:
            raise ValueError("checkpoint row bytes were truncated")
        return self._engine.torch.frombuffer(bytearray(data), dtype=self._engine.torch.bfloat16).reshape(
            end - start, info["shape"][1]).clone()

    def _read(self, unit):
        name, start, end = _parts(unit)
        layout, engine = self._layout[name], self._engine
        tensor = engine.wlm_d if layout[0] == "lm" else engine.shared[layout[0]][layout[1]]
        self._check_tensor(tensor)
        began = perf_counter()
        host = tensor.cpu()
        self._metrics["readback_s"] += perf_counter() - began
        self._metrics["readback_calls"] += 1
        self._metrics["readback_payload_bytes"] += tensor.numel() * tensor.element_size()
        # Select on CPU: a device view/contiguous operation would test a different path.
        if layout[0] == "lm":
            rows = host.reshape(-1, host.shape[-1])[start:end]
        else:
            rows = host[layout[2], :, start:end].t()
        return rows.contiguous().clone()

    def _write(self, unit, rows):
        # CPU payloads are all staged first; each NPU staging/copy can still
        # fail, and is covered by the attempted-unit restore transaction.
        name, start, end = _parts(unit)
        layout, engine = self._layout[name], self._engine
        if layout[0] == "lm":
            parent = self._check_tensor(engine.wlm_d)
            chunk, local = divmod(start, layout[1])
            destination = parent[chunk, local:local + end - start]
            prepared = rows.contiguous().to(self._expected_device)
        else:
            parent = self._check_tensor(engine.shared[layout[0]][layout[1]])
            destination = parent[layout[2], :, start:end]
            prepared = rows.t().contiguous().to(self._expected_device)
        self._check_tensor(prepared)
        self._check_tensor(destination).copy_(prepared)

    def _verify(self, units, expected):
        verified = []
        for unit, rows in zip(units, expected):
            digest = _digest(self._read(unit))
            if digest != _digest(rows):
                raise RuntimeError("serving readback did not match expected row bytes")
            verified.append({"file": unit["file"], "offset": unit["offset"],
                             "length": unit["length"], "sha256": digest})
        return verified

    def _verification(self, operation, units):
        return {"schema": "onggi.verification.v1", "ok": True, "op": operation, "mode": "units",
                "manifest": self._manifest, "units": units,
                "bytes_written": sum(unit["length"] for unit in units),
                "bytes_written_scope": "logical bytes copied into serving units; excludes temporary staging and D2H, not DMA",
                "diagnostics": dict(self._metrics)}

    def apply(self, update, target):
        with self._lock:
            self._ready()
            if (not isinstance(update, dict) or not isinstance(update.get("proof"), str)
                    or not hmac.compare_digest(update["proof"], self._proof(update))):
                raise ValueError("update plan proof is missing, tampered or from another load epoch")
            if update["base_manifest"] != self._manifest:
                raise ValueError("update base manifest is not resident")
            if update["mode"] == "reload":
                raise NotImplementedError("unsupported switch requires an explicit reload")
            if update["mode"] != "units":
                raise ValueError("unsupported update mode")
            with self._engine._lock:
                engine = self._engine
                engine._sync()
                target = Path(target).resolve(strict=True)
                if _metadata(target) != self._meta:
                    raise ValueError("target header, config, index or inventory is incompatible")
                units = update["units"]
                staged = [self._file_rows(target, unit) for unit in units]
                # The old artifact path may already contain an in-place target
                # patch or no longer exist. Device readback is the undo source.
                old = [self._read(unit) for unit in units]
                previous = self._manifest, self._path, self._journal
                journal = {"units": copy.deepcopy(units), "rows": old,
                           "manifest": self._manifest, "path": self._path}
                attempted = []
                try:
                    self._state = "staged"
                    engine._generation += 1
                    for index, (unit, rows) in enumerate(zip(units, staged)):
                        attempted.append(index)
                        self._write(unit, rows)
                    engine._sync()
                    verified = self._verify(units, staged)
                    self._journal = journal
                    self._manifest, self._path, self._state = update["manifest"], target, "ready"
                    return self._verification("apply", verified)
                except BaseException:
                    try:
                        for index in attempted:
                            self._write(units[index], old[index])
                        engine._sync()
                        self._verify([units[i] for i in attempted], [old[i] for i in attempted])
                        self._manifest, self._path, self._journal = previous
                        self._state = "ready"
                    except BaseException as restore_error:
                        self._unload()
                        raise RuntimeError("update restore failed; engine unloaded") from restore_error
                    raise

    def rollback(self):
        with self._lock:
            self._ready()
            if self._journal is None:
                raise RuntimeError("no successful update journal is available")
            with self._engine._lock:
                engine, journal = self._engine, self._journal
                engine._generation += 1
                self._state = "staged"
                try:
                    engine._sync()
                    for unit, rows in zip(journal["units"], journal["rows"]):
                        self._write(unit, rows)
                    engine._sync()
                    verified = self._verify(journal["units"], journal["rows"])
                    self._manifest, self._path = journal["manifest"], journal["path"]
                    self._journal, self._state = None, "ready"
                    return self._verification("rollback", verified)
                except BaseException as error:
                    self._unload()
                    raise RuntimeError("rollback restore failed; engine unloaded") from error

    def _unload(self, *, strict=False):
        engine = self._engine if self._engine is not None else self._quarantined
        self._engine = None
        self._manifest = self._path = self._meta = self._journal = None
        self._state, self._layout = "empty", {}
        if engine is not None:
            try:
                engine.close()
            except BaseException as error:
                # Keep the failed owner reachable. Only an explicit successful
                # close retry permits another native allocation in this owner.
                self._quarantined = engine
                engine._failed = True
                self._metrics["release_failure"] = str(error)
                if strict:
                    raise
            else:
                self._quarantined = None
                self._metrics.pop("release_failure", None)

    def close(self):
        """Retry a quarantined release; load remains blocked until this succeeds."""
        with self._lock:
            self._unload(strict=True)
