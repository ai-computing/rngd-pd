"""Cache publication and compatibility checks without the vendor SDK."""
from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rngd_pd.engines import artifact_cache as ac


@dataclass
class ZeroOutputs:
    output_indices_and_tensors: tuple


@dataclass
class Passthrough:
    target_tensor: torch.Tensor
    offset_src_to_dst: int


@dataclass
class Reconstructor:
    zero_sized_tensor_info: ZeroOutputs | None
    passthrough_pairs: tuple
    output_reconstruction_indices: tuple | None


@pytest.fixture
def sdk(monkeypatch):
    sdk = ModuleType("rngd_pd.engines.fsdk")
    sdk.NEW, sdk.SDK = True, "2026.4.0"
    sdk.runnable_blob = Mock(return_value=b"opaque native program")
    sdk.module_from_blob = Mock(side_effect=lambda blob, state: SimpleNamespace(
        blob=blob, **state, to=Mock()))
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    return sdk


def module():
    # Non-contiguous and zero-sized FakeTensor metadata cannot be pickled.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        target = torch.empty(3, 4).t()
        zero = torch.empty(0, 4)
    rec = Reconstructor(ZeroOutputs(((3, zero),)),
                        (Passthrough(target, 7),), (0, 0, 1, 2))
    return SimpleNamespace(module_weight={"constant": torch.arange(4)},
                           output_reconstructor=rec)


def test_new_cache_roundtrip_preserves_output_metadata_without_mutating_module(tmp_path, sdk):
    m = module()
    path = ac.path_for(str(tmp_path), "kernel")
    size = ac.save(m, path)
    assert ac.exists(path) and size == ac.size(path)
    restored = ac.load(path, device="furiosa:0")
    assert restored.blob == b"opaque native program"
    assert torch.equal(restored.module_weight["constant"], m.module_weight["constant"])
    rec = restored.output_reconstructor
    assert rec.output_reconstruction_indices == (0, 0, 1, 2)
    assert rec.passthrough_pairs[0].offset_src_to_dst == 7
    for old, new in [(m.output_reconstructor.passthrough_pairs[0].target_tensor,
                      rec.passthrough_pairs[0].target_tensor),
                     (m.output_reconstructor.zero_sized_tensor_info.output_indices_and_tensors[0][1],
                      rec.zero_sized_tensor_info.output_indices_and_tensors[0][1])]:
        assert type(old).__name__ == "FakeTensor"
        assert type(new) is torch.Tensor and new.device.type == "meta"
        assert (old.shape, old.stride(), old.dtype) == (new.shape, new.stride(), new.dtype)
    restored.to.assert_called_once_with("furiosa:0")


def test_default_output_reconstructor_has_no_zero_metadata(tmp_path, sdk):
    m = module()
    m.output_reconstructor = Reconstructor(None, (), None)
    path = ac.path_for(str(tmp_path), "kernel")
    ac.save(m, path)
    assert ac.load(path).output_reconstructor == Reconstructor(None, (), None)


@pytest.mark.parametrize("field", ["SDK", "torch"])
def test_cache_mismatch_rejected_before_unpickling(tmp_path, sdk, monkeypatch, field):
    path = ac.path_for(str(tmp_path), "kernel")
    ac.save(module(), path)
    if field == "SDK":
        sdk.SDK = "2026.5.0"
    else:
        monkeypatch.setattr(torch, "__version__", "different-torch")
    unpickle = Mock(side_effect=AssertionError("must not unpickle incompatible data"))
    monkeypatch.setattr(torch, "load", unpickle)
    with pytest.raises(ValueError, match="incompatible"):
        ac.load(path)
    unpickle.assert_not_called()
    sdk.module_from_blob.assert_not_called()


def test_interrupted_publication_retains_previous_archive(tmp_path, sdk, monkeypatch):
    path = ac.path_for(str(tmp_path), "kernel")
    ac.save(module(), path)
    before = {p.name: p.read_bytes() for p in (tmp_path / sdk.SDK / str(torch.__version__)).iterdir()}
    monkeypatch.setattr(ac.os, "replace", Mock(side_effect=KeyboardInterrupt("cancel")))
    with pytest.raises(KeyboardInterrupt):
        ac.save(module(), path)
    assert {p.name: p.read_bytes() for p in (tmp_path / sdk.SDK / str(torch.__version__)).iterdir()} == before
    ac.load(path)


def test_native_load_failure_never_silently_compiles(tmp_path, sdk):
    path = ac.path_for(str(tmp_path), "kernel")
    ac.save(module(), path)
    sdk.module_from_blob.side_effect = RuntimeError("invalid native program")
    compile_fn = Mock()
    with pytest.raises(RuntimeError, match="invalid native program"):
        ac.get_or_compile(str(tmp_path), "kernel", compile_fn)
    compile_fn.assert_not_called()


def test_sdk_torch_and_format_namespaces_do_not_mix(tmp_path, sdk, monkeypatch):
    new = ac.path_for(str(tmp_path), "kernel")
    ac.save(module(), new)
    monkeypatch.setattr(torch, "__version__", "next-torch")
    assert ac.path_for(str(tmp_path), "kernel") != new
    sdk.NEW = False
    legacy = ac.path_for(str(tmp_path), "kernel")
    assert legacy == str(tmp_path / sdk.SDK / "kernel")
    assert not ac.exists(legacy)
    sdk.NEW = True
    assert not ac.exists(legacy)


def test_legacy_edf_roundtrip_remains_available(tmp_path, sdk, monkeypatch):
    sdk.NEW, sdk.SDK = False, "2026.2.0"
    sdk.CompileModule = Mock(side_effect=lambda edf, weight: SimpleNamespace(edf=edf, weight=weight))
    sdk.EdfModule = lambda edf: edf
    native = ModuleType("furiosa.native_torch")
    native.ir = SimpleNamespace(Edf=SimpleNamespace(deserialize=lambda blob: ("restored", blob)))
    monkeypatch.setitem(sys.modules, native.__name__, native)
    path = ac.path_for(str(tmp_path), "kernel")
    m = SimpleNamespace(edf=SimpleNamespace(serialize=lambda: b"legacy edf"),
                        module_weight={"constant": torch.arange(4)})
    size = ac.save(m, path)
    assert ac.exists(path) and ac.size(path) == size
    restored = ac.load(path)
    assert restored.edf == ("restored", b"legacy edf")
    assert torch.equal(restored.weight["constant"], m.module_weight["constant"])
