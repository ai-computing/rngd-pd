"""Compiled module cache: legacy EDF pairs or atomic SDK/Torch-checked Runnables.

2026.2 retains its .edf/.w format. New SDKs use a .runnable archive containing
opaque native serialization and trusted Torch weights/output metadata. Only
load cache files from trusted builds. Vendor imports remain lazy for CPU hosts.
"""
import os, time, hashlib
from dataclasses import replace
import io
import json
import tempfile
import zipfile
import torch


def cache_key(**kw):
    return hashlib.blake2b(repr(sorted(kw.items())).encode(), digest_size=12).hexdigest()


def path_for(cache_dir, key):
    """Artifacts live under the SDK that built them; new SDKs also pin Torch.

    A 2026.2.0 EDF loading under 2026.3.0 would succeed and only be wrong in
    the numbers, which is the failure mode that has already overturned three
    conclusions here. Separating by directory rather than by key keeps the
    existing artifacts usable -- they only have to move, not be rebuilt.
    """
    from rngd_pd.engines.fsdk import SDK, NEW
    d = os.path.join(cache_dir, SDK)
    if NEW:
        d = os.path.join(d, str(torch.__version__))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key)


def exists(path):
    return all(os.path.isfile(p) for p in _files(path))


def _files(path):
    from rngd_pd.engines.fsdk import NEW
    return (path + ".runnable",) if NEW else (path + ".edf", path + ".w")


def size(path):
    """Bytes in the active SDK's complete cache entry."""
    return sum(os.path.getsize(p) for p in _files(path))


def _metadata():
    from rngd_pd.engines.fsdk import SDK
    return dict(format="runnable-v1", sdk=SDK, torch=str(torch.__version__))


def _output_metadata(rec):
    """Replace export-time FakeTensors with shape-only, picklable metadata."""
    if rec is None:
        return None

    def meta(t):
        return torch.empty_strided(tuple(t.shape), tuple(t.stride()),
                                   dtype=t.dtype, device="meta")

    zero = rec.zero_sized_tensor_info
    if zero is not None:
        zero = replace(zero, output_indices_and_tensors=tuple(
            (index, meta(t)) for index, t in zero.output_indices_and_tensors))
    return replace(rec,
        zero_sized_tensor_info=zero,
        passthrough_pairs=tuple(replace(p, target_tensor=meta(p.target_tensor))
                                for p in rec.passthrough_pairs))


def _save_runnable(module, path):
    from rngd_pd.engines.fsdk import runnable_blob
    state = io.BytesIO()
    torch.save(dict(module_weight=module.module_weight,
                    output_reconstructor=_output_metadata(module.output_reconstructor)), state)
    blob = runnable_blob(module)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                               dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "wb") as f:
            with zipfile.ZipFile(f, "w") as archive:
                archive.writestr("metadata.json", json.dumps(_metadata()))
                archive.writestr("runnable.bin", blob)
                archive.writestr("state.pt", state.getvalue())
        os.replace(tmp, path + ".runnable")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return size(path)


def save(module, path):
    """Persist a compiled module in the active SDK's format; return bytes."""
    from rngd_pd.engines.fsdk import NEW
    if NEW:
        return _save_runnable(module, path)
    blob = module.edf.serialize()                       # native EDF serialization (binaries+scatter+shapes)
    tmp = path + ".edf.tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path + ".edf")
    torch.save(module.module_weight, path + ".w.tmp")   # ExportedProgramWeight (flatten/unflatten specs)
    os.replace(path + ".w.tmp", path + ".w")
    return os.path.getsize(path + ".edf") + os.path.getsize(path + ".w")


def load(path, device=None):
    """Rebuild a compiled CompileModule from a saved artifact. No recompile."""
    from rngd_pd.engines.fsdk import NEW
    if NEW:
        from rngd_pd.engines.fsdk import module_from_blob
        with zipfile.ZipFile(path + ".runnable") as archive:
            if json.loads(archive.read("metadata.json")) != _metadata():
                raise ValueError("incompatible artifact cache format, SDK or Torch version; rebake")
            # Cache files are trusted executable/pickle artifacts, just as legacy .w files are.
            state = torch.load(io.BytesIO(archive.read("state.pt")),
                               weights_only=False, map_location="cpu")
            m = module_from_blob(archive.read("runnable.bin"), state)
        if device is not None:
            m.to(device)
        return m
    from furiosa.native_torch import ir
    from rngd_pd.engines.fsdk import CompileModule
    from rngd_pd.engines.fsdk import EdfModule
    with open(path + ".edf", "rb") as f:
        edf = ir.Edf.deserialize(f.read())
    weight = torch.load(path + ".w", weights_only=False)
    m = CompileModule(EdfModule(edf), weight)
    if device is not None:
        m.to(device)
    return m


def get_or_compile(cache_dir, key, compile_fn, device=None):
    """Return (module, mode, seconds). mode ∈ {'load','compile'}. compile_fn() -> CompileModule."""
    os.makedirs(cache_dir, exist_ok=True)
    path = path_for(cache_dir, key)
    if exists(path):
        t0 = time.time(); m = load(path, device); return m, "load", time.time() - t0
    t0 = time.time(); m = compile_fn()
    try:
        save(m, path)
    except Exception as e:                              # never fail the request path over caching
        print(f"[artifact_cache] WARN: could not persist compiled module ({e}); cold-start cache disabled for {key}")
    return m, "compile", time.time() - t0


if __name__ == "__main__":
    k1 = cache_key(model="llama", H=4096, bucket=64, tactic="Decode")
    k2 = cache_key(bucket=64, tactic="Decode", H=4096, model="llama")
    k3 = cache_key(model="llama", H=4096, bucket=128, tactic="Decode")
    assert k1 == k2 and k1 != k3
    print(f"artifact_cache key self-test: PASS ✅ (stable={k1}, differs on bucket)")
