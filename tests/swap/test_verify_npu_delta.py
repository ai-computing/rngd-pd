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
