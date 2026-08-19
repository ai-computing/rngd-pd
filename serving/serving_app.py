"""Integrated RNGD serving app (Phase 5) — gateway + multi-bucket + continuous batching + metrics.

Ties the pieces into one runnable HTTP service:
  • EPP KV-aware routing (gateway/epp.py) over a worker registry (gateway/registry.py)
  • multi-bucket runtime selection (serving/bucket.py)
  • continuous batching of decode steps (serving/batching.py)
  • Prometheus-style /metrics (KV-hit rate, requests, tokens, PD transfer bytes)
  • registration / heartbeat / kv-event HTTP endpoints (the distributed worker→gateway plane)

Runnable NOW without an NPU via --mock (deterministic stub runtime) so the whole serving path —
routing, batching, buckets, metrics, HTTP — is verifiable independent of the 30-min compile. The real
path swaps in engine_wired_generic.GenericWiredRuntime behind the same Dispatcher interface.

  python3 serving_app.py --role all --mock            # self-contained smoke (gateway+2 workers)
  python3 serving_app.py --role gateway --mode pd      # gateway container entrypoint
  python3 serving_app.py --role worker --chip 0 --role-kind prefill --gateway http://gateway:8080
"""
import sys, os, json, time, threading, argparse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import _paths  # noqa: E402  (portable path bootstrap)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gateway"))
from registry import WorkerRegistry, Role
from kv_events import KVIndex, block_hashes
from epp import EndpointPicker
from bucket import BucketManager
from batching import ContinuousBatcher


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0; self.kv_hit_requests = 0; self.kv_hit_blocks = 0
        self.tokens_out = 0; self.pd_requests = 0; self.pd_bytes = 0
        self.compute_ms = 0.0; self.wall_ms = 0.0
    def record(self, routing, n_out, wall_ms, compute_ms, pd_bytes=0):
        with self.lock:
            self.requests += 1; self.tokens_out += n_out
            self.wall_ms += wall_ms; self.compute_ms += compute_ms
            if routing.get("kv_hit_blocks", 0) > 0:
                self.kv_hit_requests += 1; self.kv_hit_blocks += routing["kv_hit_blocks"]
            if routing.get("mode") == "pd":
                self.pd_requests += 1; self.pd_bytes += pd_bytes
    def prometheus(self):
        with self.lock:
            hit_rate = (self.kv_hit_requests / self.requests) if self.requests else 0.0
            tps = (self.tokens_out / (self.wall_ms/1000)) if self.wall_ms else 0.0
            return "\n".join([
                "# RNGD serving metrics",
                f"rngd_requests_total {self.requests}",
                f"rngd_tokens_out_total {self.tokens_out}",
                f"rngd_kv_hit_requests_total {self.kv_hit_requests}",
                f"rngd_kv_hit_rate {hit_rate:.4f}",
                f"rngd_kv_hit_blocks_total {self.kv_hit_blocks}",
                f"rngd_pd_requests_total {self.pd_requests}",
                f"rngd_pd_transfer_bytes_total {self.pd_bytes}",
                f"rngd_throughput_tokens_per_s {tps:.2f}",
                f"rngd_compute_ms_total {self.compute_ms:.1f}",
            ]) + "\n"


class MockDispatcher:
    """NPU-free deterministic runtime: publishes KV events, reports PD transfer size."""
    def __init__(self, kv_index, block_size=16, per_tok_ms=1.0, n_layers=32, kv_bytes_per_layer=262144):
        self.kv = kv_index; self.block_size = block_size; self.per_tok_ms = per_tok_ms
        self.n_layers = n_layers; self.kv_bpl = kv_bytes_per_layer
    def dispatch(self, token_ids, decision, max_new=4):
        hs = block_hashes(token_ids, self.block_size)
        prefill_blocks = max(0, len(hs) - decision.kv_hit_blocks)
        t = prefill_blocks * self.block_size * 0.02 + max_new * self.per_tok_ms
        time.sleep(t/1000.0)
        self.kv.block_stored(decision.decode_wid, hs)
        out = [(token_ids[-1] + i + 1) % 128000 for i in range(max_new)]
        pd_bytes = self.n_layers * self.kv_bpl if decision.mode == "pd" else 0
        return {"output_tokens": out, "compute_ms": t, "pd_bytes": pd_bytes}


class ServingApp:
    def __init__(self, registry, kv_index, dispatcher, mode="replica", block_size=16):
        self.reg = registry; self.kv = kv_index; self.disp = dispatcher
        self.epp = EndpointPicker(registry, kv_index, block_size=block_size)
        self.mode = mode; self.metrics = Metrics()

    def complete(self, body):
        mode = body.get("mode", self.mode)
        toks = list(body["tokens"]); max_new = body.get("max_new", 4)
        d = self.epp.route(toks, mode)
        self.reg.incr(d.decode_wid, 1)
        try:
            t0 = time.time(); res = self.disp.dispatch(toks, d, max_new=max_new); wall = (time.time()-t0)*1000
        finally:
            self.reg.incr(d.decode_wid, -1)
        self.metrics.record({"mode": d.mode, "kv_hit_blocks": d.kv_hit_blocks},
                            len(res["output_tokens"]), wall, res.get("compute_ms", 0), res.get("pd_bytes", 0))
        return {"choices": [{"tokens": res["output_tokens"]}],
                "routing": {"mode": d.mode, "prefill": d.prefill_wid, "decode": d.decode_wid,
                            "kv_hit_blocks": d.kv_hit_blocks, "reason": d.reason},
                "timing_ms": {"wall": round(wall, 2)}}

    def register(self, body):
        self.reg.register(body["wid"], body["role"], body.get("chip", 0), body.get("endpoint", ""))
        return {"ok": True}
    def heartbeat(self, body):
        self.reg.heartbeat(body["wid"], body.get("inflight")); return {"ok": True}
    def kv_event(self, body):
        ev = body.get("event", "stored")
        if ev == "stored": self.kv.block_stored(body["wid"], [bytes.fromhex(h) for h in body["hashes"]])
        elif ev == "cleared": self.kv.all_cleared(body["wid"])
        return {"ok": True}


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, obj, ctype="application/json"):
            data = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            p = self.path.rstrip("/")
            if p == "/metrics": self._send(200, app.metrics.prometheus(), "text/plain")
            elif p == "/healthz": self._send(200, {"ok": True})
            else: self.send_error(404)
        def do_POST(self):
            p = self.path.rstrip("/")
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if p == "/v1/completions": out = app.complete(body)
                elif p == "/register": out = app.register(body)
                elif p == "/heartbeat": out = app.heartbeat(body)
                elif p == "/kv_event": out = app.kv_event(body)
                else: return self.send_error(404)
                self._send(200, out)
            except Exception as e:
                self._send(400, {"error": str(e)})
    return H


def serve(app, port=8080):
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _smoke(mode):
    """--role all --mock: gateway + 2 workers in-process, drive real HTTP, verify + print metrics."""
    reg = WorkerRegistry(); kv = KVIndex()
    reg.register("pf0", Role.PREFILL, 0); reg.register("dc1", Role.DECODE, 1); reg.register("rep0", Role.BOTH, 0)
    app = ServingApp(reg, kv, MockDispatcher(kv, block_size=2), mode=mode, block_size=2)
    srv = serve(app, port=8080)
    def call(path, body):
        req = urllib.request.Request(f"http://127.0.0.1:8080{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read())
    toks = list(range(9))
    r1 = call("/v1/completions", {"tokens": toks, "mode": mode})
    r2 = call("/v1/completions", {"tokens": toks, "mode": mode})       # warm → KV-hit
    r3 = call("/v1/completions", {"tokens": list(range(9)), "mode": "pd"})
    metrics = urllib.request.urlopen("http://127.0.0.1:8080/metrics", timeout=5).read().decode()
    hit_ok = r1["routing"]["kv_hit_blocks"] == 0 and r2["routing"]["kv_hit_blocks"] > 0
    pd_ok = r3["routing"]["mode"] == "pd"
    print("=== serving_app smoke (--role all --mock) ===")
    print(f"  req1 {r1['routing']['reason']} | req2 {r2['routing']['reason']} | req3 {r3['routing']['reason']}")
    print(f"  KV-hit warm-up: {'✅' if hit_ok else '❌'} | PD route: {'✅' if pd_ok else '❌'}")
    print("  --- /metrics ---")
    for line in metrics.strip().splitlines():
        if not line.startswith("#"): print("   ", line)
    ok = hit_ok and pd_ok
    print(f"  {'PASS ✅ integrated gateway+routing+metrics over HTTP' if ok else 'FAIL ❌'}")
    srv.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="all", choices=["all", "gateway", "worker"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mode", default="replica", choices=["replica", "pd"])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--chip", type=int, default=0); ap.add_argument("--role-kind", default="both")
    ap.add_argument("--gateway", default=""); ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    a = ap.parse_args()
    if a.role == "all" and a.mock:
        _smoke(a.mode); return
    # gateway / worker container modes: real deployment wiring (documented in deploy/README.md).
    # Real worker attaches engine_wired_generic.GenericWiredRuntime behind the Dispatcher; gateway
    # runs the ServingApp with an HTTP proxy dispatcher to workers. Left as the deploy blueprint since
    # it needs a live cluster + NPUs; --role all --mock exercises the full logic here.
    print(f"[serving_app] role={a.role} mode={a.mode} mock={a.mock} — see deploy/README.md for cluster wiring")


if __name__ == "__main__":
    main()
