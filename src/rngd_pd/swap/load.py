"""Checkpoint-to-resident-engine delta load, including a live onggi diff.

The checkpoint directories must remain immutable while planning and applying.
onggi refs describe these safetensors directories, not embedded .rbln artifacts.
"""
import json
import os
import subprocess
import time

from .delta import load_onggi_ranges, plan_update


def onggi_ranges(store, base_ref, new_ref, executable="onggi", timeout=120):
    """Ask onggi for ranges; fail explicitly instead of scanning on an error."""
    if not all(isinstance(x, str) and x for x in (store, base_ref, new_ref)):
        raise ValueError("onggi requires store, base_ref and new_ref")
    if base_ref.startswith("-") or new_ref.startswith("-"):
        raise ValueError("onggi refs cannot start with '-'")
    proc = subprocess.run(
        [os.fspath(executable), "diff", "--store", store, "--ranges", "--json",
         base_ref, new_ref], capture_output=True, text=True, check=True, timeout=timeout)
    return load_onggi_ranges(json.loads(proc.stdout))


def load_delta(engine, new_checkpoint, *, ranges=None, store=None, base_ref=None,
               new_ref=None, onggi="onggi", timeout=120, refine=True, **apply_options):
    """Plan and apply one delta at a drained request boundary.

    Pass ranges as onggi JSON (a dict/string/path), or store and two refs to run
    onggi itself. With neither, compare checkpoint bytes locally. The result
    separates planning time from device update time; total_s includes both.
    """
    if ranges is not None and any(x is not None for x in (store, base_ref, new_ref)):
        raise ValueError("choose ranges JSON or an onggi store and refs")
    with engine._lock:
        t0 = time.perf_counter()
        if any(x is not None for x in (store, base_ref, new_ref)):
            changed = onggi_ranges(store, base_ref, new_ref, onggi, timeout)
            source = "onggi"
        else:
            changed = load_onggi_ranges(ranges) if ranges is not None else None
            source = "ranges" if ranges is not None else "local"
        if changed is not None:
            expected = {f for f in os.listdir(new_checkpoint) if f.endswith(".safetensors")}
            if not expected.issubset(changed):
                raise ValueError("onggi diff must describe every target safetensors file")
        plan = plan_update(engine.ck.path, new_checkpoint, changed, refine=refine)
        planned = time.perf_counter()
        result = engine.apply_delta(new_checkpoint, plan, **apply_options)
        return dict(result, plan=plan.summary(), source=source,
                    planning_s=planned - t0, total_s=time.perf_counter() - t0)
