"""Cache-key lookup must not build multi-GB compiler example inputs."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import torch


def test_cached_graph_specs_allocate_no_example_tensors(monkeypatch):
    kernel = ModuleType("rngd_pd.kernels.decode_batch")
    for name, value in dict(BF=torch.bfloat16, H=4, M=8, NH=2, NKV=1, HD=2,
                            G=2, DecodeModel=Mock(), rot_batch=Mock(),
                            HINT="decode", MAX_SHAPES={}).items():
        setattr(kernel, name, value)
    sdk = ModuleType("rngd_pd.engines.fsdk")
    sdk.CompileModule = SimpleNamespace(from_module=Mock())
    sdk.TacticKernelModule = Mock()
    sdk.FCConfig = Mock()
    sdk.TacticHintConfig = SimpleNamespace(ForLlmModelDecode="decode", NoConstraint="none")
    sdk.hint = lambda value: value
    sdk.NEW = False
    dsl = ModuleType("rngd_pd.kernels.dsl_batch")
    dsl.rms_dsl = Mock()
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    monkeypatch.setitem(sys.modules, dsl.__name__, dsl)
    path = Path(__file__).parents[2] / "src/rngd_pd/engines/decode_batch.py"
    spec = importlib.util.spec_from_file_location("decode_specs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    zeros = Mock(wraps=torch.zeros)
    monkeypatch.setattr(torch, "zeros", zeros)
    decode_key, decode_build = module.decode_graph_spec(2, 16, 2)
    lm_key, lm_build = module.lmhead_graph_spec(2, 32)
    assert decode_key != lm_key and zeros.call_count == 0
    decode_build()
    assert zeros.call_count > 0
    zeros.reset_mock()
    lm_build()
    assert zeros.call_count > 0
