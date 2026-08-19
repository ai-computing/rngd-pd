#!/usr/bin/env python3
"""Cluster-side shim: an InferencePool member that forwards to the PD gateway.

The PD workers hold NPU cards through vendor toolchains (furiosa-torch on RNGD,
rebel-compiler on ATOM) that are installed on the node, not in a container
image, so they run as host processes. This pod carries the pool's label and
forwards to them, which is what lets the llm-d gateway → InferencePool → EPP
path reach a real KV-transfer PD backend.

Standard library only — the container borrows the node's Python.

  UPSTREAM=http://10.x.x.x:8100 python3 pd_proxy.py --port 8000
"""
import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8100")
TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "600"))


def _fetch(path, data=None):
    req = urllib.request.Request(
        UPSTREAM + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, r.read()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _relay(self, path, data=None):
        try:
            code, body = _fetch(path, data)
        except urllib.error.HTTPError as e:
            code, body = e.code, e.read()
        except Exception as e:
            code = 503
            body = json.dumps({"error": f"upstream unreachable: {e!r}",
                               "upstream": UPSTREAM}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.rstrip("/") or "/health"
        self._relay(p if p in ("/health", "/metrics", "/v1/models") else "/health")

    def _read_body(self):
        """Envoy forwards chunked, so Content-Length alone loses the body."""
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()               # trailing CRLF
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_POST(self):
        self._relay(self.path.rstrip("/"), self._read_body())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"[proxy] :{a.port} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
