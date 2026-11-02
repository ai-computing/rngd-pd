"""Finite FIFO caller over real generation/update callbacks; no native evidence by itself."""
from contextlib import contextmanager
import math
import queue
import threading
import time
import traceback


def verify_sample(row, prepared):
    from ccgrid_b2 import require
    refs, document, q = prepared["references"], prepared["document"], row["queue"]
    require(row["arm"] in ("b2", "reload"), "unknown arm")
    require(q["status"] == "completed" and q["worker_joined"]
            and q["observed_inflight_at_trigger"] and row["cleanup_passed"], "request/owner lifecycle failed")
    require(refs["base"]["tokens"] != refs["sparse"]["tokens"], "target fixture must affect outputs")
    require(row["warm_tokens"] == refs["base"]["tokens"]
            and row["warm_native_counts"] == refs["base"]["counts"], "initial base parity/counts differ")
    require([r["request"] for r in q["requests"]] == [0, 1, 2], "missing/duplicate requests")
    for request in q["requests"]:
        role = "base" if request["request"] == 0 else "sparse"
        manifest = document["base_manifest" if role == "base" else "manifest"]
        require(request["tokens"] == refs[role]["tokens"] and request["manifest"] == manifest,
                "request has wrong native output or resident identity")
        require(request["native_counts"] == refs[role]["counts"], "request native counts differ")
    if row["arm"] == "b2":
        outcome = q["switch"]["outcome"]
        require(outcome["verification"]["ok"] and outcome["identities_after"] == row["identities_before"],
                "B2 verification or retained allocation/program identity failed")


@contextmanager
def mark_first_return(program_class, first_native, marker):
    """Nest inside the existing native-output checker; never replace its result."""
    original, owned = program_class.__call__, "__call__" in program_class.__dict__

    def call(program, *args, **kwargs):
        value = original(program, *args, **kwargs)
        if not first_native.is_set():
            marker["perf_ns"] = time.perf_counter_ns()
            first_native.set()
        return value

    program_class.__call__ = call
    try:
        yield
    finally:
        if owned:
            program_class.__call__ = original
        else:
            del program_class.__call__


def run_queue(generate, switch, first_native, emit, timeout=300):
    """Request0 is running when switch and requests1/2 arrive; preserve failures.

    generate(id) returns generation_entry_perf_ns, prompt_ttft_s and the actual
    first verified native return marker for request0. emit receives incremental
    events. A false worker_joined forbids owner close/replacement until process
    exit; an outer bounded Job enforces that final fallback.
    """
    tasks, lock = queue.Queue(), threading.RLock()
    base_finished, abort = threading.Event(), threading.Event()
    initial = time.perf_counter_ns()
    result = dict(schema="ccgrid-loaded-queue-v1", status="running", requests=[],
                  observed_inflight_at_trigger=False, worker_joined=False,
                  arrivals={0: initial}, switch={})

    def event(name, **fields):
        emit(dict(event=name, at_perf_ns=time.perf_counter_ns(), **fields))

    def worker():
        try:
            for index in range(4):
                if abort.is_set():
                    return
                task = ("request", 0) if index == 0 else tasks.get(timeout=timeout)
                if abort.is_set() or task[0] == "stop":
                    return
                started = time.perf_counter_ns()
                if task[0] == "switch":
                    with lock:
                        result["switch"]["started_perf_ns"] = started
                        event("switch_start")
                    value = switch()
                    ready = time.perf_counter_ns()
                    with lock:
                        result["switch"].update(ready_perf_ns=ready, outcome=value)
                        event("switch_ready", outcome=value)
                else:
                    request = task[1]
                    value = generate(request)
                    returned = time.perf_counter_ns()
                    if request == 0:
                        base_finished.set()
                    entry, ttft = value["generation_entry_perf_ns"], value["prompt_ttft_s"]
                    if not (math.isfinite(ttft) and ttft >= 0):
                        raise ValueError("invalid prompt TTFT")
                    first = entry + round(ttft * 1e9)
                    if not started <= entry <= first <= returned:
                        raise ValueError("request first-token clock is outside its execution")
                    with lock:
                        row = dict(value, request=request, arrival_perf_ns=result["arrivals"][request],
                                   started_perf_ns=started, first_token_perf_ns=first,
                                   returned_perf_ns=returned)
                        result["requests"].append(row)
                        event("request_return", request=row)
        except BaseException:
            with lock:
                result["error"] = traceback.format_exc()
                abort.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        if not first_native.wait(min(timeout, 10)):
            raise RuntimeError("no verified native return while request is in-flight")
        with lock:
            if base_finished.is_set() or abort.is_set():
                raise RuntimeError("no in-flight base request at trigger")
            trigger = result["trigger_perf_ns"] = time.perf_counter_ns()
            result["observed_inflight_at_trigger"] = True
            for request in (1, 2):
                result["arrivals"][request] = time.perf_counter_ns()
            tasks.put(("switch", None))
            tasks.put(("request", 1))
            tasks.put(("request", 2))
            event("queued", trigger_perf_ns=trigger, arrivals=dict(result["arrivals"]))
    except BaseException:
        with lock:
            result.setdefault("error", traceback.format_exc())
            abort.set()
            tasks.put(("stop", None))
    thread.join(timeout)
    # A live native worker must not be raced by closing/replacing its owner.
    result["worker_joined"] = not thread.is_alive()
    if thread.is_alive():
        result.setdefault("error", "worker remains active; owner release deferred to process exit")
    if "error" not in result:
        try:
            base, one, two = result["requests"]
            switch_row, trigger = result["switch"], result["trigger_perf_ns"]
            if not (base["native_first_perf_ns"] <= trigger < base["returned_perf_ns"]
                    <= switch_row["started_perf_ns"] <= switch_row["ready_perf_ns"]
                    <= one["started_perf_ns"] <= one["returned_perf_ns"] <= two["started_perf_ns"]):
                raise RuntimeError("actual request/switch overlap is invalid")
            if any(row["arrival_perf_ns"] > switch_row["started_perf_ns"] for row in (one, two)):
                raise RuntimeError("target requests did not arrive before switch work")
            result["metrics"] = dict(
                drain_wait_s=(switch_row["started_perf_ns"] - trigger) / 1e9,
                switch_work_s=(switch_row["ready_perf_ns"] - switch_row["started_perf_ns"]) / 1e9,
                trigger_to_target_first_token_s=(one["first_token_perf_ns"] - trigger) / 1e9,
                queued_requests=[dict(request=row["request"],
                    wait_s=(row["started_perf_ns"] - row["arrival_perf_ns"]) / 1e9,
                    ttft_s=(row["first_token_perf_ns"] - row["arrival_perf_ns"]) / 1e9,
                    completion_s=(row["returned_perf_ns"] - row["arrival_perf_ns"]) / 1e9)
                    for row in (one, two)])
            result["status"] = "completed"
        except BaseException:
            result["error"] = traceback.format_exc()
    if "error" in result:
        result["status"] = "failed"
    event("queue_finished", status=result["status"], worker_joined=result["worker_joined"])
    return result
