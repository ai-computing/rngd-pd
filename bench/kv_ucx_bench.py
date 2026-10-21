"""Cross-node KV, `nixl-ucx` against `tcp` on the same link: one-way time per request, byte-exact check.

Each request is what a prefill worker hands over at S=512: 32 layers x (K, V) blobs of 8 x 512 x 128 on
the bf16 wire (64 blobs, 67 MB). Two processes, normally on two nodes, both running this file:

    # decode side (driver): runs the descriptor / tcp server, pulls and checks every blob
    UCX_TLS=tcp python3 bench/kv_ucx_bench.py --role pull --port 5577 --transport nixl-ucx,tcp
    # prefill side: pushes on the driver's signal
    UCX_TLS=tcp python3 bench/kv_ucx_bench.py --role push --server decode-host:5577

    python3 bench/kv_ucx_bench.py --role both ...       # both ends in one process (loopback check)

The server sits beside decode, as in the NPU-to-NPU `tcp` layout (rngd-pd#9): with `tcp` the push
crosses the link and the pull is local; with `nixl-ucx` the push stages locally and the pull's NIXL
READ crosses the link (UCX_TLS picks tcp, or rc/ud verbs with an RDMA NIC). Per request the driver
times signal -> all 64 blobs pulled (one way, end to end) and the pusher reports its push time. Every
blob is compared by SHA-256 with what the pusher generated from the same seed. One JSON line per
(transport, concurrency).
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
from rngd_pd.connector.tcpkv import TcpKVClient, TcpKVServer, parse_addr   # noqa: E402

NL, NKV, S, HD = 32, 8, 512, 128


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(NL):
        for side in ("k", "v"):
            a = (rng.standard_normal((NKV, S, HD)).astype(np.float32).view(np.uint32) >> 16 << 16).view(np.float32)
            out.append((f"layer.{i}/{side}", kv.ser(a, wire="bf16")))
    return out


def _transport(name, addr):
    return kv.TcpTransport(remote=addr) if name == "tcp" else kv.NixlUcxTransport(remote=addr)


def pusher(addr, blobs, transports):
    """Serve plans until the driver says stop: per request, wait for its signal, push all blobs."""
    ctl = TcpKVClient(*addr)
    ts = {}
    k = 0
    while True:
        plan = json.loads(bytes(ctl.take(f"plan/{k}", timeout=3600)))
        if plan.get("stop"):
            break
        t = ts.get(plan["transport"]) or ts.setdefault(plan["transport"], _transport(plan["transport"], addr))

        def worker(j):
            for i in range(j, plan["total"], plan["n"]):
                ctl_w = TcpKVClient(*addr)
                ctl_w.take(f"go/{plan['run']}/{i}", timeout=600)
                t0 = time.perf_counter()
                for name, b in blobs:
                    t.push(f"{plan['run']}/{i}/{name}", b)
                ctl_w.push(f"stat/{plan['run']}/{i}", json.dumps({"push_ms": (time.perf_counter() - t0) * 1e3}).encode())
                ctl_w.close()
        th = [threading.Thread(target=worker, args=(j,)) for j in range(plan["n"])]
        [x.start() for x in th]
        [x.join() for x in th]
        k += 1
    # the last READs' notifications may still be in flight: let the slots drain before closing
    for t in ts.values():
        store = getattr(t, "store", None)
        end = time.time() + 30
        while store is not None and store.pending() and time.time() < end:
            time.sleep(0.05)
        t.close()
    ctl.close()


def driver(addr, blobs, transport_names, levels, total, warmup, k0=0):
    digests = {name: hashlib.sha256(b).digest() for name, b in blobs}
    nbytes = sum(len(b) for _n, b in blobs)
    ctl = TcpKVClient(*addr)
    k = k0
    rows = []
    for tname in transport_names:
        t = _transport(tname, addr)
        for n, count, label in [(1, warmup, "warmup")] + [(n, max(total, n), "") for n in levels]:
            if not count:
                continue
            run = f"{tname}-{n}-{k}"
            ctl.push(f"plan/{k}", json.dumps({"run": run, "n": n, "total": count, "transport": tname}).encode())
            k += 1
            res, lock = [], threading.Lock()

            def worker(j):
                c = TcpKVClient(*addr)
                for i in range(j, count, n):
                    t0 = time.perf_counter()
                    c.push(f"go/{run}/{i}", b"1")
                    bad = 0
                    for name, _b in blobs:
                        got = t.pull(f"{run}/{i}/{name}", timeout=600)
                        bad += hashlib.sha256(got).digest() != digests[name]
                    e2e = (time.perf_counter() - t0) * 1e3
                    st = json.loads(bytes(c.take(f"stat/{run}/{i}", timeout=60)))
                    with lock:
                        res.append({"e2e_ms": e2e, "push_ms": st["push_ms"], "bad": bad})
                c.close()
            t0 = time.perf_counter()
            th = [threading.Thread(target=worker, args=(j,)) for j in range(n)]
            [x.start() for x in th]
            [x.join() for x in th]
            wall = time.perf_counter() - t0
            if label == "warmup":
                continue
            e2e = [r["e2e_ms"] for r in res]
            pm = [r["push_ms"] for r in res]
            q = lambda v, p: round(float(np.percentile(v, p)), 1)     # noqa: E731
            row = {"transport": tname, "concurrency": n, "requests": count,
                   "mb_per_request": round(nbytes / 1e6, 1),
                   "one_way_ms_p50": q(e2e, 50), "one_way_ms_p90": q(e2e, 90),
                   "push_ms_p50": q(pm, 50),
                   "gb_s_per_request": round(nbytes / statistics.median(e2e) / 1e6, 3),
                   "aggregate_gb_s": round(nbytes * count / wall / 1e9, 3),
                   "byte_exact": all(r["bad"] == 0 for r in res),
                   "blobs_checked": len(res) * len(blobs),
                   "ucx_tls": os.environ.get("UCX_TLS", "(default)") if tname == "nixl-ucx" else None}
            rows.append(row)
            print(json.dumps(row), flush=True)
        t.close()
    ctl.push(f"plan/{k}", json.dumps({"stop": True}).encode())
    ctl.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--role", choices=("pull", "push", "both"), required=True)
    ap.add_argument("--server", default="", help="push: host:port of the driver's server")
    ap.add_argument("--port", type=int, default=5577, help="pull/both: port of the server to run")
    ap.add_argument("--transport", default="nixl-ucx,tcp")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=8, help="requests per concurrency level")
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    blobs = _blobs(0)
    if a.role == "push":
        pusher(parse_addr(a.server), blobs, a.transport.split(","))
        return
    srv = TcpKVServer(a.port)
    srv.start()
    addr = ("127.0.0.1", srv.wait_ready())
    print(json.dumps({"server_port": addr[1], "ucx_tls": os.environ.get("UCX_TLS", "(default)"),
                      "ucx_net_devices": os.environ.get("UCX_NET_DEVICES", "(default)")}), flush=True)
    if a.role == "both":
        p = threading.Thread(target=pusher, args=(addr, blobs, a.transport.split(",")), daemon=True)
        p.start()
    driver(addr, blobs, a.transport.split(","), [int(x) for x in a.concurrency.split(",")], a.requests, a.warmup)
    if a.role == "both":
        p.join(60)
    srv.stop()


if __name__ == "__main__":
    main()
