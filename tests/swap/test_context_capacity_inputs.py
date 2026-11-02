import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('capacity_inputs', ROOT / 'bench/ccgrid_context_capacity.py')
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def test_capacity_workload_actually_crosses_base_context():
    rows = [list(range(8)), list(range(8, 16)), list(range(16, 24)), list(range(24, 32))]
    prompts = capacity.capacity_prompts(rows)
    assert len(prompts) == 4 and {len(p) for p in prompts} == {257}
    assert all(p[:8] == seed and p[-1] == seed[0] for p, seed in zip(prompts, rows))
    assert 256 < len(prompts[0]) + 8 <= 512


def test_empty_seed_is_rejected_before_native_allocation():
    with pytest.raises(ValueError):
        capacity.capacity_prompts([[], [], [], []])
