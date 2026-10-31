"""Exercise the verification command with controlled result rows, without a card."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize("retained", [False, True])
def test_verifier_enforces_unchanged_allocation_retention(tmp_path, monkeypatch, retained):
    path = Path(__file__).parents[2] / "experiments" / "verify_npu_delta.py"
    spec = importlib.util.spec_from_file_location("verify_npu_delta", path)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    result = dict(destination_differs_base=True, rows=[dict(
        tokens_equal_fresh=True, unique_graphs_before=1, unique_graphs_after=1,
        unchanged_allocations_retained=retained)])
    monkeypatch.setattr(verify, "worker", lambda args: result)
    out = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--backend", "atom", "--mode", "resident",
        "--base", "base", "--new", "new", "--store", "store", "--cache", "cache",
        "--out", str(out)])
    if retained:
        verify.main()
        assert json.loads(out.read_text()) == result
    else:
        with pytest.raises(AssertionError, match="allocation"):
            verify.main()


def test_verifier_rejects_atom_cpu_fallback_even_when_outputs_match(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / "experiments" / "verify_npu_delta.py"
    spec = importlib.util.spec_from_file_location("verify_npu_delta", path)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    result = dict(destination_differs_base=True, rows=[dict(
        tokens_equal_fresh=True, unique_graphs_before=0, unique_graphs_after=0,
        unchanged_allocations_retained=True)])
    monkeypatch.setattr(verify, "worker", lambda args: result)
    monkeypatch.setattr(sys, "argv", [str(path), "--backend", "atom", "--mode", "resident",
        "--base", "base", "--new", "new", "--store", "store", "--cache", "cache",
        "--out", str(tmp_path / "result.json")])
    with pytest.raises(AssertionError, match="no compiled ATOM graph"):
        verify.main()


def test_constant_verifier_requires_retained_native_runtimes(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / "experiments" / "verify_npu_delta.py"
    spec = importlib.util.spec_from_file_location("verify_npu_delta", path)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    result = dict(destination_differs_base=True, rows=[dict(
        tokens_equal_fresh=True, unique_graphs_before=0, unique_graphs_after=0,
        constant_runtimes_retained=False, compiled_programs_before=1, compiled_programs_after=1)])
    monkeypatch.setattr(verify, "worker", lambda args: result)
    monkeypatch.setattr(sys, "argv", [str(path), "--backend", "atom-constants", "--mode", "resident",
        "--base", "base", "--new", "new", "--store", "store", "--cache", "cache",
        "--out", str(tmp_path / "result.json")])
    with pytest.raises(AssertionError, match="native runtime"):
        verify.main()
