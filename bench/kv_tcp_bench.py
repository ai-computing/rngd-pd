"""Cross-node KV over the `tcp` transport: one-way time and bandwidth per request, alone and concurrent.

Each request is what a prefill worker hands over at S=512: 32 layers x (K, V) blobs of 8 x 512 x 128
on the bf16 wire (1 MiB each, 64 blobs, 67 MB), pushed through the connector's Transport to a
TcpKVServer and taken back by a second client, as the decode worker does. Run it on the NPU node with
the relay on another node, and both legs cross the network:

    python3 bench/kv_tcp_bench.py --server relay-host:5577 --concurrency 1,2,4,8 --requests 16
    python3 bench/kv_tcp_bench.py --kv real.npz ...     # KV dumped by a prefill worker (k0..k31, v0..v31)

Per request: push ms (prefill side, all 64 blobs acknowledged by the server) and pull ms (decode side,
all 64 taken), each with GB/s; under concurrency N requests move at once. Every pulled blob is compared
with what was pushed (sha256), so a pass is byte-exact. Prints a JSON line per concurrency level.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rngd_pd.connector import transport as kv                    # noqa: E402
from rngd_pd.connector.tcpkv import TcpKVClient, parse_addr       # noqa: E402

NL, NKV, S, HD = 32, 8, 512, 128


def _layers(path, seed):
    if path:
        z = np.load(path)
        return [(z[f"k{i}"], z[f"v{i}"]) for i in range(NL)]
    rng = np.random.default_rng(seed)
    # bf16-valued float32, as the prefill graphs produce
    f = lambda: (rng.standard_normal((NKV, S, HD)).astype(np.float32).view(np.uint32)   # noqa: E731
                 >> 16 << 16).view(np.float32)
    return [(f(), f()) for _ in range(NL)]


def _blobs(layers):
    out = []
    for i, (k, v) in enumerate(layers):
        out.append((f"layer.{i}/k", kv.ser(k, wire="bf16")))
        out.append((f"layer.{i}/v", kv.ser(v, wire="bf16")))
    return out


def one(req, blobs, digests, push_c, pull_c, res):
    """Push one request's blobs, then take them back as a decode worker would."""
    t0 = time.perf_counter()
    for name, b in blobs:
        push_c.push(f"{req}/{name}", b)
    t1 = time.perf_counter()
    bad = 0
    for name, _b in blobs:
        got = pull_c.take(f"{req}/{name}", timeout=60)
        bad += hashlib.sha256(got).digest() != digests[name]
    t2 = time.perf_counter()
    res.append({"push_ms": (t1 - t0) * 1e3, "pull_ms": (t2 - t1) * 1e3, "bad": bad})


def level(n, total, blobs, digests, addr):
    nbytes = sum(len(b) for _n, b in blobs)
    push_c, pull_c = TcpKVClient(*addr), TcpKVClient(*addr)     # prefill side, decode side
    res, todo, lock = [], list(range(total)), threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop()
            one(f"bench{os.getpid()}-{n}-{i}", blobs, digests, push_c, pull_c, res)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - t0
    push_c.close(); pull_c.close()
    pm = [r["push_ms"] for r in res]
    qm = [r["pull_ms"] for r in res]
    q = lambda v, p: float(np.percentile(v, p))               # noqa: E731
    return {"concurrency": n, "requests": total, "mb_per_request": round(nbytes / 1e6, 1),
            "push_ms_p50": round(q(pm, 50), 1), "push_ms_p90": round(q(pm, 90), 1),
            "pull_ms_p50": round(q(qm, 50), 1), "pull_ms_p90": round(q(qm, 90), 1),
            "push_gb_s_per_request": round(nbytes / statistics.median(pm) / 1e6, 3),
            "pull_gb_s_per_request": round(nbytes / statistics.median(qm) / 1e6, 3),
            "aggregate_gb_s": round(2 * nbytes * total / wall / 1e9, 3),      # both legs
            "byte_exact": all(r["bad"] == 0 for r in res)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", required=True, help="host:port of the TcpKVServer (relay)")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    ap.add_argument("--kv", default="", help="npz with k0..k31, v0..v31 (8 x 512 x 128) instead of random")
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    addr = parse_addr(a.server)
    blobs = _blobs(_layers(a.kv, 0))
    digests = {name: hashlib.sha256(b).digest() for name, b in blobs}
    if a.warmup:
        level(1, a.warmup, blobs, digests, addr)
    crc = os.environ.get("AIRM_KV_TCP_CRC", "1")
    for n in [int(x) for x in a.concurrency.split(",")]:
        r = level(n, max(a.requests, n), blobs, digests, addr)
        r.update(crc=crc != "0", kv="file" if a.kv else "random-bf16")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
