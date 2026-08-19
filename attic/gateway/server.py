"""HTTP gateway for the RNGD serving stack (Phase 2) — the llm-d Inference Gateway, minimized.

Exposes an OpenAI-ish POST /v1/completions. Per request: tokenize → EPP.route() (KV-aware) → hand to
a Dispatcher that actually runs the model (prefill→KV-handoff→decode for pd, or single-worker for
replica) → publish KV events so the next request with the same prefix routes to the warm worker.

The Dispatcher is injected so the SAME gateway serves both:
  • MockDispatcher : deterministic, NPU-free — validates routing / KV-hit / A/B logic instantly.
  • RngdDispatcher : real ChipRuntimes + rngd_kv_connector (see demo_e2e.py) — real tokens, real NPU.

Design: this is the "functionally-equivalent minimal gateway" the plan scopes — no Envoy/K8s; that
is Phase 5. Everything here is plain stdlib so it runs anywhere.
"""
import sys, os, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import WorkerRegistry, Role
from kv_events import KVIndex, block_hashes
from epp import EndpointPicker


class Dispatcher:
    """Runs a routed request. Subclasses do the real work; base is abstract."""
    def dispatch(self, token_ids, decision) -> dict:
        raise NotImplementedError


class MockDispatcher(Dispatcher):
    """NPU-free: 'decodes' deterministically and reports which worker ran + kv reuse. Publishes KV
    events for the served prefix so subsequent identical prefixes become KV-hits."""
    def __init__(self, kv_index, block_size=16, per_tok_ms=1.0):
        self.kv = kv_index; self.block_size = block_size; self.per_tok_ms = per_tok_ms
    def dispatch(self, token_ids, decision):
        hs = block_hashes(token_ids, self.block_size)
        # simulate compute: prefill cost skipped for cached blocks (kv-hit), decode is per-token
        prefill_blocks = max(0, len(hs) - decision.kv_hit_blocks)
        t = prefill_blocks * self.block_size * 0.02 + 4 * self.per_tok_ms   # 4 generated tokens
        time.sleep(t / 1000.0)
        # the worker now holds this prefix
        self.kv.block_stored(decision.decode_wid, hs)
        out = [(token_ids[-1] + i + 1) % 128000 for i in range(4)]          # deterministic stub
        return {"output_tokens": out, "compute_ms": t}


class Gateway:
    def __init__(self, registry, kv_index, dispatcher, tokenizer=None, default_mode="replica", block_size=16):
        self.reg = registry; self.kv = kv_index; self.disp = dispatcher
        self.epp = EndpointPicker(registry, kv_index, block_size=block_size)
        self.tok = tokenizer; self.default_mode = default_mode

    def _tokenize(self, body):
        if "tokens" in body: return list(body["tokens"])
        if self.tok is None: raise ValueError("no tokenizer configured; send {'tokens': [...]}")
        return self.tok.encode(body["prompt"], add_special_tokens=False)

    def complete(self, body):
        mode = body.get("mode", self.default_mode)
        toks = self._tokenize(body)
        decision = self.epp.route(toks, mode)
        self.reg.incr(decision.decode_wid, 1)
        try:
            t0 = time.time()
            res = self.disp.dispatch(toks, decision)
            wall = (time.time() - t0) * 1000
        finally:
            self.reg.incr(decision.decode_wid, -1)
        return {
            "choices": [{"tokens": res["output_tokens"]}],
            "routing": {"mode": decision.mode, "prefill": decision.prefill_wid,
                        "decode": decision.decode_wid, "kv_hit_blocks": decision.kv_hit_blocks,
                        "reason": decision.reason},
            "timing_ms": {"wall": round(wall, 2), "compute": round(res.get("compute_ms", 0), 2)},
        }


def make_handler(gw):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/completions":
                self.send_error(404); return
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                out = gw.complete(body); code = 200
            except Exception as e:
                out = {"error": str(e)}; code = 400
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
    return H


def serve(gw, port=8080):
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(gw))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------- NPU-free end-to-end test (mock dispatcher over real HTTP) ----------------
def _selftest():
    import urllib.request
    reg = WorkerRegistry(); kv = KVIndex()
    reg.register("w0", Role.BOTH, 0); reg.register("w1", Role.BOTH, 1)
    gw = Gateway(reg, kv, MockDispatcher(kv), default_mode="replica")
    srv = serve(gw, port=8091)
    def call(body):
        req = urllib.request.Request("http://127.0.0.1:8091/v1/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read())

    toks = list(range(64))
    r1 = call({"tokens": toks, "mode": "replica"})
    r2 = call({"tokens": toks, "mode": "replica"})        # same prefix → must KV-hit the same worker
    # third request, brand-new prefix → cold, load-balanced
    r3 = call({"tokens": list(range(2000, 2064)), "mode": "replica"})

    ok_hit = r1["routing"]["kv_hit_blocks"] == 0 and r2["routing"]["kv_hit_blocks"] == 4
    ok_same = r2["routing"]["decode"] == r1["routing"]["decode"]
    ok_out = r1["choices"][0]["tokens"] == [64, 65, 66, 67]
    ok_cold = r3["routing"]["kv_hit_blocks"] == 0

    # PD mode routing through the gateway
    reg.register("p0", Role.PREFILL, 2); reg.register("d0", Role.DECODE, 3)
    rp = call({"tokens": list(range(3000, 3064)), "mode": "pd"})
    ok_pd = rp["routing"]["mode"] == "pd" and rp["routing"]["prefill"] and rp["routing"]["decode"]

    print("=== gateway HTTP self-test (mock dispatcher, real HTTP) ===")
    print(f"  req1 route: {r1['routing']}")
    print(f"  req2 route: {r2['routing']}   (same prefix → KV-hit, same worker)")
    print(f"  req3 route: {r3['routing']}   (new prefix → cold)")
    print(f"  pd   route: {rp['routing']}")
    ok = ok_hit and ok_same and ok_out and ok_cold and ok_pd
    print(f"  {'PASS ✅ KV-aware routing + PD/replica dispatch over HTTP' if ok else 'FAIL ❌ '+str([ok_hit,ok_same,ok_out,ok_cold,ok_pd])}")
    srv.shutdown()


if __name__ == "__main__":
    _selftest()
