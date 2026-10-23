"""Request budgets, ownership and finite admission at the deployed worker boundary."""
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from rngd_pd.serving import cluster


class Engine:
    B = 2

    def __init__(self):
        self.positions = []

    def blank_kv(self):
        return None, None

    def write_slot_kv(self, K, V, idx, k, v):
        return K, V

    def prepare_kv(self, Ks, Vs):
        return Ks, Vs

    def upload_prepared_kv(self, prepared):
        return prepared

    def adopt_kv_device(self, Ks, Vs):
        return Ks, Vs

    def step(self, cur, pos, K, V):
        assert all(0 <= p < 8 for p in pos), pos
        self.positions.append(list(pos))
        return [c + 1 for c in cur], K, V


def items():
    return [dict(req_id="short", first=1, pos=6, max_new=1, tokens=[1] * 6),
            dict(req_id="long", first=2, pos=2, max_new=5, tokens=[2] * 2)]


@pytest.mark.parametrize("kind", ["rngd", "atom", "solo"])
def test_static_batch_never_advances_finished_lanes(kind):
    cls = {"rngd": cluster.DecodeRNGDBatch, "atom": cluster.DecodeATOMBatch,
           "solo": cluster.SoloWorker}[kind]
    worker = cls.__new__(cls)
    worker.B, worker.S, worker.nl, worker.NKV, worker.HD = 2, 8, 1, 1, 1
    worker.eng = worker.dec = Engine()
    worker._load_kv = lambda rid: (np.zeros((1, 1, 1, 8)),
                                  np.zeros((1, 1, 8, 1)), 0., 0., None)
    worker.pre = type("Prefill", (), {
        "prefill_device": lambda self, toks: (None, None, toks[0])})()
    worker.embed = np.zeros((32, 1), np.float32)
    worker._cos = worker._sin = worker._mask = worker._eye = np.eye(8)

    class Runtime:
        def run(self, *args):
            logits = np.zeros((2, 32)); logits[:, 3] = 1
            return logits, np.zeros((1, 2, 1, 1)), np.zeros((1, 2, 1, 1))

    worker.rt = Runtime()
    out = worker._run_batch(items())
    assert [len(o["tokens"]) for o in out] == [1, 5]
    assert out[0]["tokens"] == [1]


@pytest.mark.parametrize("kind", ["rngd", "atom", "solo", "prepared"])
def test_static_cancellation_during_step_stops_further_device_work(kind):
    from test_prepared_serving import worker as prepared_worker
    cancel = threading.Event()
    entered, release = threading.Event(), threading.Event()
    steps, errors = [], []
    if kind == "prepared":
        w, native = prepared_worker()
        original = native.step
    else:
        cls = {"rngd": cluster.DecodeRNGDBatch, "atom": cluster.DecodeATOMBatch,
               "solo": cluster.SoloWorker}[kind]
        w = cls.__new__(cls)
        w.B, w.S, w.nl = 2, 8, 1
        w.eng = w.dec = Engine()
        w._load_kv = lambda rid: (np.zeros((1, 1, 1, 8)),
                                 np.zeros((1, 1, 8, 1)), 0., 0., None)
        w.pre = type("Pre", (), {"prefill_device": lambda self, toks: (None, None, 1)})()
        w.embed = np.zeros((32, 1), np.float32)
        w._cos = w._sin = w._mask = w._eye = np.eye(8)
        original = w.eng.step

    def step(*args):
        steps.append(1); entered.set()
        assert release.wait(3)
        return original(*args)

    if kind == "prepared":
        native.step = step
    elif kind == "atom":
        class Runtime:
            def run(self, *args):
                steps.append(1); entered.set()
                assert release.wait(3)
                return np.zeros((2, 32)), np.zeros((1, 2, 1, 1)), np.zeros((1, 2, 1, 1))
        w.rt = Runtime()
    else:
        w.eng.step = step
    batcher = cluster._Batcher(2, 0, w._run_batch)

    def run():
        try:
            batcher.submit(dict(req_id="x", first=1, pos=2, tokens=[1, 2],
                                max_new=5, cancel=cancel))
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=run, daemon=True); t.start()
    try:
        assert entered.wait(2)
        cancel.set()
    finally:
        release.set(); t.join(3)
    assert len(steps) == 1
    assert len(errors) == 1 and errors[0].status == 408


def test_continuous_first_token_can_finish_without_decode():
    eng = Engine()
    sched = cluster._ContinuousScheduler(2, eng, lambda it: (None, None, 7, 6, 0.))
    emitted = []
    out = sched.submit(dict(max_new=1, on_token=emitted.append))
    assert out["tokens"] == emitted == [7]
    assert eng.positions == []


def gateway():
    g = cluster.Gateway.__new__(cluster.Gateway)
    g.S, g.prefill, g.decode, g.targets = 16, "p", "d", {}
    g.seq, g.seq_lock = 0, threading.Lock()
    g.tk = type("Tokenizer", (), {"encode": lambda self, s, **kw: list(range(len(s))),
                                   "decode": lambda self, ts: "answer"})()
    return g


def post(url, body, **kw):
    if url.endswith("prefill"):
        return dict(next=1, pos=len(body["tokens"]), prefill_ms=0, kv_mb=0, save_ms=0)
    return dict(tokens=[1], load_ms=0, tpot_ms=0)


def test_gateway_ids_are_unique_across_replicas():
    with patch.object(cluster, "_post", post), patch.object(cluster, "PREFETCH", False), \
            patch.object(cluster.time, "time", return_value=123.456):
        a = gateway().complete(dict(prompt="abc", max_tokens=1))
        b = gateway().complete(dict(prompt="def", max_tokens=1))
    assert a["id"] != b["id"]


@pytest.mark.parametrize("body", [dict(prompt="x" * 20, max_tokens=2),
                                      dict(prompt="abc", max_tokens=0),
                                      dict(prompt="abc", max_tokens=-1),
                                      dict(prompt="abc", max_tokens=True),
                                      dict(prompt="abc", max_tokens=1.5)])
def test_gateway_rejects_invalid_budget_without_sending(body):
    with patch.object(cluster, "_post", side_effect=AssertionError("must reject locally")):
        with pytest.raises(ValueError):
            gateway().complete(body)


def test_continuous_bounds_load_ahead_and_rejects_overload(monkeypatch):
    monkeypatch.setenv("PD_MAX_REQUESTS", "3")
    monkeypatch.setenv("PD_KV_LOAD_AHEAD", "1")
    entered, release = threading.Event(), threading.Event()
    produced, results = [], []

    def produce(it):
        produced.append(it["req_id"])
        entered.set()
        assert release.wait(3)
        return None, None, 1, 2, 0.

    sched = cluster._ContinuousScheduler(2, Engine(), produce, produce_on_device=False)
    threads = []
    try:
        for n in range(3):
            t = threading.Thread(target=lambda n=n: results.append(
                sched.submit(dict(req_id=str(n), max_new=1))), daemon=True)
            t.start(); threads.append(t)
        assert entered.wait(2)
        until = time.monotonic() + 2
        while len(sched.pending) != 3 and time.monotonic() < until:
            time.sleep(.001)
        assert len(sched.pending) == 3
        # Admission has to precede loading, including requests waiting behind a loader.
        with pytest.raises(cluster.AdmissionError) as err:
            sched.submit(dict(req_id="excess", max_new=1))
        assert err.value.status == 429
        assert len(produced) == 1
    finally:
        release.set()
        for t in threads:
            t.join(3)
    assert len(results) == 3
    assert sched.submit(dict(req_id="after", max_new=1))["tokens"] == [1]


def test_cancelled_loading_request_keeps_admission_until_loader_returns(monkeypatch):
    monkeypatch.setenv("PD_MAX_REQUESTS", "1")
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def produce(it):
        entered.set()
        assert release.wait(3)
        return None, None, 1, 1, 0.

    sched = cluster._ContinuousScheduler(1, Engine(), produce, produce_on_device=False)

    def run():
        try:
            sched.submit(dict(max_new=1, cancel=cancel))
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=run, daemon=True); t.start()
    try:
        assert entered.wait(2)
        cancel.set()
        with pytest.raises(cluster.AdmissionError) as err:
            sched.submit(dict(max_new=1))
        assert err.value.status == 429
    finally:
        release.set(); t.join(3)
    assert len(errors) == 1 and errors[0].status == 408


def test_cancellation_during_step_does_not_emit_or_release_lane_early():
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
    emitted, errors = [], []

    class Blocking(Engine):
        def step(self, cur, pos, K, V):
            entered.set()
            assert release.wait(3)
            return [2], K, V

    sched = cluster._ContinuousScheduler(1, Blocking(), lambda it: (None, None, 1, 1, 0.))

    def run():
        try:
            sched.submit(dict(max_new=3, cancel=cancel, on_token=emitted.append))
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=run, daemon=True); t.start()
    try:
        assert entered.wait(2)
        cancel.set()
        assert sched.slots[0] is not None
    finally:
        release.set(); t.join(3)
    assert emitted == [1]
    assert len(errors) == 1 and errors[0].status == 408


@pytest.mark.parametrize("cls", [cluster.DecodeATOM, cluster.DecodeATOMBatch,
                                 cluster.DecodeRNGD, cluster.DecodeRNGDBatch])
def test_raw_decode_rejects_context_overflow_before_loading(cls):
    worker = cls.__new__(cls)
    worker.S = 8
    worker._load_kv = lambda rid: pytest.fail("must reject before loading")
    worker.batcher = type("Batcher", (), {"submit": lambda self, item:
                                         pytest.fail("must reject before admission")})()
    with pytest.raises(ValueError):
        worker.run("bad", 1, 7, 2)


def test_http_overload_is_429():
    import json
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    def full(body):
        raise cluster.AdmissionError(429, "admission queue full")

    with patch.dict(cluster.STATE, ready=True):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), cluster.make_handler({"/test": full}))
        t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/test", data=b"{}")
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req, timeout=3)
            assert err.value.code == 429
            assert json.loads(err.value.read())["error"]["type"] == "admission_error"
        finally:
            srv.shutdown(); srv.server_close(); t.join(2)


def test_cancel_during_admission_never_emits_first_token():
    entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
    emitted, errors = [], []

    def produce(item):
        entered.set()
        assert release.wait(3)
        return None, None, 7, 1, 0.

    sched = cluster._ContinuousScheduler(1, Engine(), produce)

    def run():
        try:
            sched.submit(dict(max_new=1, cancel=cancel, on_token=emitted.append))
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=run, daemon=True); t.start()
    try:
        assert entered.wait(2)
        cancel.set()
    finally:
        release.set(); t.join(3)
    assert emitted == []
    assert len(errors) == 1 and errors[0].status == 408


def test_prefetch_expiry_never_forgets_running_loader(monkeypatch):
    monkeypatch.setenv("PD_MAX_REQUESTS", "1")
    monkeypatch.setenv("PD_KV_LOAD_AHEAD", "1")
    entered, release = threading.Event(), threading.Event()
    w = cluster.DecodeWorkerBase()
    w._prefetch, w._prefetch_cap, w._prefetch_ttl_s = {}, 1, 0.

    def job(rid):
        entered.set()
        assert release.wait(3)
        return None, None, 0., None

    w._prefetch_job = job
    try:
        assert w.prefetch("old")["prefetch"] == "started"
        assert entered.wait(2)
        assert w.prefetch("new")["prefetch"] == "skipped (full)"
    finally:
        release.set()
        for pool, fut, _ in list(w._prefetch.values()):
            fut.result(3); pool.shutdown()


def test_prefetch_obeys_memory_budget_before_starting(monkeypatch):
    monkeypatch.setenv("PD_MAX_REQUESTS", "1")
    monkeypatch.setenv("PD_KV_LOAD_AHEAD", "1")
    release = threading.Event()
    w = cluster.DecodeWorkerBase()
    w._prefetch, w._prefetch_cap = {}, 8
    w._prefetch_job = lambda rid: (release.wait(3), None, 0., None)
    try:
        assert w.prefetch("one")["prefetch"] == "started"
        assert w.prefetch("two")["prefetch"] == "skipped (full)"
    finally:
        release.set()
        for pool, _f, _ in list(w._prefetch.values()):
            pool.shutdown()


def test_post_preserves_upstream_admission_status():
    import urllib.error
    from io import BytesIO
    error = urllib.error.HTTPError("http://worker", 429, "full", {},
                                   BytesIO(b'{"error":{"message":"queue full"}}'))
    with patch.object(cluster.urllib.request, "urlopen", side_effect=error):
        with pytest.raises(cluster.AdmissionError) as got:
            cluster._post("http://worker", {})
    assert got.value.status == 429


def test_stream_reports_admission_failure_before_returning_generator():
    class FullWorker:
        pre = type("Pre", (), {"buckets": {8: None}})()

        def run(self, *args, **kw):
            raise cluster.AdmissionError(429, "queue full")

    tk = type("Tokenizer", (), {"decode": lambda self, ids, **kw: "a"})()
    front = cluster.SoloOpenAI(FullWorker(), 16, "m", tokenizer=tk)
    with pytest.raises(cluster.AdmissionError) as got:
        front.complete(dict(prompt=[1], max_tokens=1, stream=True))
    assert got.value.status == 429


def test_http_bounds_synchronous_worker_requests(monkeypatch):
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    monkeypatch.setenv("PD_MAX_REQUESTS", "1")
    entered, release = threading.Event(), threading.Event()
    results = []

    def block(body):
        entered.set()
        assert release.wait(3)
        return {}

    with patch.dict(cluster.STATE, ready=True):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), cluster.make_handler({"/test": block}))
        loop = threading.Thread(target=srv.serve_forever, daemon=True); loop.start()
        url = f"http://127.0.0.1:{srv.server_port}/test"

        def first():
            with urllib.request.urlopen(urllib.request.Request(url, data=b"{}"), timeout=4) as r:
                results.append(r.status)

        t = threading.Thread(target=first, daemon=True); t.start()
        try:
            assert entered.wait(2)
            with pytest.raises(urllib.error.HTTPError) as got:
                urllib.request.urlopen(urllib.request.Request(url, data=b"{}"), timeout=2)
            assert got.value.code == 429
        finally:
            release.set(); t.join(3)
            srv.shutdown(); srv.server_close(); loop.join(2)
        assert results == [200]
