"""Synthetic CPU queue checks; these do not provide native timing evidence."""
import importlib.util
import json
from pathlib import Path
import threading
import time
import sys

import pytest


def load():
    path = Path(__file__).parents[2] / "bench/ccgrid_loaded_queue.py"
    spec = importlib.util.spec_from_file_location("loaded_queue_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_real_queue_preserves_base_then_serves_updated_file(tmp_path):
    h = load()
    source = tmp_path / "state"
    source.write_bytes(b"base")
    native, release = threading.Event(), threading.Event()

    def generate(request):
        entered = time.perf_counter_ns()
        contents = source.read_bytes()
        first = time.perf_counter_ns()
        if request == 0:
            native.set()
            assert release.wait(2)
        return dict(generation_entry_perf_ns=entered, prompt_ttft_s=0,
                    native_first_perf_ns=first, contents=contents.decode())

    def switch():
        source.write_bytes(b"target")
        return {"verified": source.read_bytes() == b"target"}

    def emit(event):
        if event["event"] == "queued":
            release.set()

    result = h.run_queue(generate, switch, native, emit, timeout=2)
    assert result["status"] == "completed", result
    assert result["worker_joined"] and result["observed_inflight_at_trigger"]
    assert [request["contents"] for request in result["requests"]] == ["base", "target", "target"]
    base, one, two = result["requests"]
    assert result["trigger_perf_ns"] < base["returned_perf_ns"] <= result["switch"]["started_perf_ns"]
    assert result["switch"]["ready_perf_ns"] <= one["started_perf_ns"] < two["started_perf_ns"]
    assert one["arrival_perf_ns"] < result["switch"]["started_perf_ns"]
    assert result["metrics"]["drain_wait_s"] > 0


def test_late_native_trigger_is_failed_not_zero_drain(tmp_path):
    h = load()
    native = threading.Event()
    switches = []

    def generate(request):
        start = time.perf_counter_ns()
        return dict(generation_entry_perf_ns=start, prompt_ttft_s=0, native_first_perf_ns=start)

    result = h.run_queue(generate, lambda: switches.append(True), native,
        lambda event: native.set() if event["event"] == "request_return" else None, timeout=.1)
    assert result["status"] == "failed" and result["worker_joined"]
    assert "in-flight" in result["error"]
    assert not switches and len(result["requests"]) == 1


def test_failed_switch_never_runs_queued_target_requests():
    h = load()
    native, release = threading.Event(), threading.Event()
    requests = []

    def generate(request):
        requests.append(request)
        start = time.perf_counter_ns()
        native.set()
        assert release.wait(2)
        return dict(generation_entry_perf_ns=start, prompt_ttft_s=0, native_first_perf_ns=start)

    def switch():
        raise RuntimeError("actual update failed")

    result = h.run_queue(generate, switch, native,
        lambda event: release.set() if event["event"] == "queued" else None, timeout=2)
    assert result["status"] == "failed" and result["worker_joined"]
    assert "actual update failed" in result["error"]
    assert requests == [0] and len(result["requests"]) == 1
    assert "ready_perf_ns" not in result["switch"]


def test_early_worker_exception_is_retained():
    h = load()

    def generate(request):
        raise ValueError("bad actual request")

    result = h.run_queue(generate, lambda: None, threading.Event(), lambda event: None, timeout=.05)
    assert result["status"] == "failed" and result["worker_joined"]
    assert "bad actual request" in result["error"]
    assert not result["requests"]


@pytest.mark.parametrize("fail", [False, True])
def test_native_marker_follows_original_return_and_restores_call(fail, tmp_path):
    h = load()
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"actual payload")
    native, marker = threading.Event(), {}

    class Program:
        def __call__(self):
            assert not native.is_set()
            contents = artifact.read_bytes()
            if fail:
                raise RuntimeError("original call failed")
            return contents

    original = Program.__call__
    with h.mark_first_return(Program, native, marker):
        if fail:
            with pytest.raises(RuntimeError, match="original call failed"):
                Program()()
        else:
            assert Program()() == b"actual payload"
            assert native.is_set() and marker["perf_ns"] <= time.perf_counter_ns()
    assert Program.__call__ is original
    assert native.is_set() is (not fail)


@pytest.mark.parametrize("damage", ["target_tokens", "calls", "identity", "worker"])
def test_native_acceptance_rejects_wrong_output_counts_or_ownership(damage):
    h = load()
    counts = dict(native_calls=255, native_output_checks=255, compilation_attempts=0)
    row = dict(arm="b2", identities_before={"buffer": 1}, warm_tokens=[[1]],
        warm_native_counts=dict(counts), cleanup_passed=True,
        queue=dict(status="completed", worker_joined=True, observed_inflight_at_trigger=True,
            requests=[dict(request=i, tokens=[[1 if i == 0 else 2]], native_counts=dict(counts),
                manifest="base" if i == 0 else "target") for i in range(3)],
            switch=dict(outcome=dict(identities_after={"buffer": 1}, verification={"ok": True}))))
    refs = dict(base=dict(tokens=[[1]], counts=counts), sparse=dict(tokens=[[2]], counts=counts))
    prepared = dict(references=refs, document=dict(base_manifest="base", manifest="target"))
    h.verify_sample(row, prepared)
    if damage == "target_tokens":
        row["queue"]["requests"][1]["tokens"] = [[1]]
    elif damage == "calls":
        row["queue"]["requests"][2]["native_counts"]["native_calls"] = 0
    elif damage == "identity":
        row["queue"]["switch"]["outcome"]["identities_after"] = {"buffer": 9}
    else:
        row["queue"]["worker_joined"] = False
    with pytest.raises(ValueError):
        h.verify_sample(row, prepared)


def test_timeout_aborts_followup_and_preserves_snapshot_and_observer():
    h = load()
    native, release = threading.Event(), threading.Event()
    calls, events = [], []

    class Program:
        def __call__(self):
            return 7

    original = Program.__call__

    def checked(program):
        assert original(program) == 7
        return 7

    Program.__call__ = checked

    def generate(request):
        calls.append(request)
        entry, marker = time.perf_counter_ns(), {}
        with h.mark_first_return(Program, native, marker):
            assert Program()() == 7
            assert release.wait(2)
        return dict(generation_entry_perf_ns=entry, prompt_ttft_s=0,
                    native_first_perf_ns=marker["perf_ns"])

    before = set(threading.enumerate())
    result = h.run_queue(generate, lambda: calls.append("switch"), native, events.append, timeout=.02)
    assert result["status"] == "failed" and not result["worker_joined"]
    frozen, event_count = json.dumps(result, sort_keys=True), len(events)
    Program.__call__ = original  # Outer observer unwinds on failed native run.
    release.set()
    for thread in set(threading.enumerate()) - before:
        thread.join(2)
        assert not thread.is_alive()
    assert calls == [0], "a failed caller must not later switch or serve queued requests"
    assert json.dumps(result, sort_keys=True) == frozen
    assert len(events) == event_count
    assert Program.__call__ is original


def test_recorder_delay_is_not_native_drain_or_callback_work():
    h = load()
    native, release = threading.Event(), threading.Event()

    def generate(request):
        entry = time.perf_counter_ns()
        if request == 0:
            native.set()
            assert release.wait(2)
        return dict(generation_entry_perf_ns=entry, prompt_ttft_s=0, native_first_perf_ns=entry)

    def slow_recorder(event):
        if event["event"] == "queued":
            release.set()
        if event["event"] in ("request_return", "switch_start"):
            threading.Event().wait(.03)  # Deliberate CPU-test instrumentation cost.

    result = h.run_queue(generate, lambda: {}, native, slow_recorder, timeout=2)
    assert result["status"] == "completed", result
    base, sw = result["requests"][0], result["switch"]
    actual_drain = (base["returned_perf_ns"] - result["trigger_perf_ns"]) / 1e9
    assert result["metrics"]["drain_wait_s"] == actual_drain
    assert result["metrics"]["dispatch_s"] >= .02
    assert result["metrics"]["switch_work_s"] < .02
    assert sw["callback_started_perf_ns"] - sw["started_perf_ns"] >= 20_000_000
