"""ai-rm protocol v0.1 adapter: this repo's KV path, workload runner and metrics.

Wraps what is already here; nothing below changes how prefill, decode or the
handoff behave.

  ShmTransport  rngd_pd.connector.transport.ShmTransport (/dev/shm files, bf16
                wire) as an rm-core KVTransport, entry point
                `airm.kv_transports` name `shm`. It is a HOST-STAGING path: KV
                leaves the card, crosses host memory, and is uploaded on the
                other side. It is also still the connector's push/pull
                Transport, so select_transport("shm") hands the serving
                workers exactly the object they used before.
  TcpTransport  rngd_pd.connector.transport.TcpTransport (bf16 wire over TCP to
                a TcpKVServer, connector/tcpkv.py) as an rm-core KVTransport,
                entry point name `tcp`. The same host staging as shm on both
                ends, with the network in between: prefill pushes to the server,
                decode takes from it, and the server can be a relay on a third
                node. Cross-node; CRC-checked end to end.
  NixlUcxTransport  rngd_pd.connector.transport.NixlUcxTransport (connector/ucxkv.py) as an
                rm-core KVTransport, entry point name `nixl-ucx`. Host staging like tcp, but
                the bytes move by a one-sided NIXL READ over the UCX backend between
                NIXL-registered arenas; the TcpKVServer carries only descriptors. UCX_TLS
                picks the wire (tcp today, rc/ud verbs with an RDMA NIC): RDMA is configuration.
  run           workload runner, entry point `airm.workloads` name `rngd-pd`:
                starts one serving role (prefill, decode or gateway) of
                rngd_pd.serving.cluster with the KV path named in its config.
  metrics       ttft_p90_ms, tpot_p90_ms, tok_s and kv_transfer_ms
                (tag kv_transport) under rm-core's names, for the gateway's
                Prometheus endpoint (airm_<name>) and the bench result files.

Configuration comes from keyword arguments and the environment only.
"""
import collections
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from rm_core import (ContractError, KVLayout, Measurement, MemoryTier, TransferHandle,
                     TransferPlan, TransferResult, TransportError, prometheus_name)
from rm_core.metrics import METRICS

from rngd_pd.connector import transport as _kv
from rngd_pd.paths import SHM_ROOT

COMPONENT = "rngd-pd"
WIRE_DTYPES = ("bf16", "fp16")          # what the shm wire can carry, 2 bytes an element


# --------------------------------------------------------------------------
# L2: KV transport
# --------------------------------------------------------------------------
class _KeyedKV:
    """rm-core KVTransport methods over a connector Transport (key -> bytes).

    A subclass sets `name` and `SCOPE`, implements supports() and _detail(), and gives
    _fetch(key, timeout) (the delivered bytes, TimeoutError if absent) and _consume(key) (the same
    bytes once more, then gone).
    """

    def _init_kv(self):
        self._pending = {}                  # token -> (plan, key, started)
        self._delivered = {}                # (request id, tokens) -> key, waited but not taken
        self._lock = threading.Lock()
        self.measurements = collections.deque(maxlen=1024)

    def _sent(self, key):
        """Hook: submit() published `key`."""

    def plan(self, request_id, src, dst, src_layout, dst_layout, tokens):
        if not self.supports(src, dst):
            raise TransportError(f"{self.name}: {self.SCOPE}, "
                                 f"not {src.kind}:{src.name} -> {dst.kind}:{dst.name}")
        src_layout.validate(); dst_layout.validate()
        if src_layout != dst_layout:
            raise TransportError(f"{self.name}: src and dst layouts differ; convert in the model adapter")
        if src_layout.dtype not in WIRE_DTYPES:
            raise TransportError(f"{self.name}: wire carries {WIRE_DTYPES}, not {src_layout.dtype}")
        start, end = tokens
        if not 0 <= start < end:
            raise TransportError(f"{self.name}: bad token range {tokens}")
        return TransferPlan(request_id, src.validate(), dst.validate(), src_layout, dst_layout,
                            (start, end), (end - start) * src_layout.bytes_per_token(),
                            detail=dict(self._detail(), wire=src_layout.dtype))

    def submit(self, plan, kv=None):
        """kv: the token range's KV as an array of (end-start) * bytes_per_token / 2
        elements (e.g. [2, layers, tokens, kv_heads, head_dim]); zeros if omitted."""
        n = plan.nbytes // 2
        if kv is None:
            kv = np.zeros(n, np.float32)
        elif np.asarray(kv).size != n:
            raise ContractError(f"{self.name}: kv has {np.asarray(kv).size} elements, plan needs {n}")
        token = uuid.uuid4().hex
        key = f"{plan.request_id}/{plan.dst.name}/{plan.tokens[0]}-{plan.tokens[1]}.{token[:8]}"
        started = time.perf_counter()
        self.push(key, _kv.ser(np.asarray(kv, np.float32), wire=plan.detail.get("wire", "bf16")))
        with self._lock:
            self._pending[token] = (plan, key, started)
        self._sent(key)
        return TransferHandle(plan.request_id, self.name, token)

    def wait(self, handle, timeout_s):
        with self._lock:
            entry = self._pending.pop(handle.token, None)
        if entry is None:
            raise TransportError(f"{self.name}: unknown or already-waited handle {handle.token}")
        plan, key, started = entry
        try:
            got = _kv.deser(self._fetch(key, timeout_s)).size * 2
        except TimeoutError as e:
            return TransferResult(plan.request_id, False, 0, time.perf_counter() - started, str(e))
        seconds = time.perf_counter() - started
        if got != plan.nbytes:
            return TransferResult(plan.request_id, False, got, seconds,
                                  f"segment holds {got} bytes, plan moved {plan.nbytes}")
        self.measurements.append(Measurement(
            "kv_transfer_ms", seconds * 1e3,
            {"component": COMPONENT, "kv_transport": self.name}).validate())
        with self._lock:
            self._delivered[(plan.request_id, plan.tokens)] = key
        return TransferResult(plan.request_id, True, plan.nbytes, seconds)

    def take(self, plan):
        """The delivered KV for `plan` (float32), removing it."""
        with self._lock:
            key = self._delivered.pop((plan.request_id, plan.tokens), None)
        if key is None:
            raise TransportError(f"{self.name}: nothing delivered for {plan.request_id} {plan.tokens}")
        return _kv.deser(self._consume(key))


class ShmTransport(_KeyedKV, _kv.ShmTransport):
    """KV between two host-memory tiers on one machine, through /dev/shm.

    Keyword arguments: root -- the shared directory (default $PD_SHM_ROOT, else
    /dev/shm/rngd_kv_cluster). Prefill and decode must see the same one, which
    in k8s means a hostPath /dev/shm on both pods.

    submit() writes the segment and publishes it atomically (write, fsync,
    rename), so once wait() has found it the data is visible at the
    destination. wait() never falls back to another path: a missing or short
    segment is ok=False. take() reads the delivered KV back and consumes it.
    """

    name = "shm"
    SCOPE = "only host-dram -> host-dram on one machine"

    def __init__(self, root=None):
        # Not super().__init__: the directory is created on first write, so constructing one
        # (as the conformance runner does) never touches the serving root.
        self.root = str(root or os.environ.get("PD_SHM_ROOT") or SHM_ROOT)
        self._written = set()               # keys this instance published, for close()
        self._init_kv()

    def push(self, key, buf):
        os.makedirs(self.root, exist_ok=True)
        super().push(key, buf)

    # -- rm-core KVTransport --------------------------------------------
    def supports(self, src, dst):
        return src.kind == "host-dram" and dst.kind == "host-dram" and src.name != dst.name

    def _detail(self):
        return {"root": self.root}

    def _sent(self, key):
        with self._lock:
            self._written.add(key)

    def _fetch(self, key, timeout):
        return self.pull(key, timeout=timeout)

    def _consume(self, key):
        buf = self.pull(key, timeout=0)
        self._unlink(key)
        return buf

    def close(self):
        with self._lock:
            keys, self._written = self._written, set()
            self._pending.clear()
            self._delivered.clear()
        for key in keys:
            self._unlink(key)
        if os.path.basename(self.root).startswith("rngd_kv_conformance."):
            try:
                os.rmdir(self.root)
            except OSError:
                pass

    def _unlink(self, key):
        try:
            os.unlink(self._path(key))
        except FileNotFoundError:
            pass
        with self._lock:
            self._written.discard(key)

    def conformance_case(self):
        """Two host tiers on this machine, sharing /dev/shm. No device needed.

        Moves this instance to a private directory next to the serving root, so a
        check never writes into the handoff directory live workers are reading.
        """
        from rm_core.conformance import NoHardware
        self.root = os.path.join(os.path.dirname(self.root.rstrip("/")) or "/dev/shm",
                                 f"rngd_kv_conformance.{os.getuid()}.{os.getpid()}")
        try:
            os.makedirs(self.root, exist_ok=True)
            probe = os.path.join(self.root, f".probe.{os.getpid()}")
            open(probe, "wb").close(); os.unlink(probe)
        except OSError as e:
            raise NoHardware(f"shm root {self.root} is not writable: {e}") from e
        return (MemoryTier("shm-prefill", "host-dram", 1 << 30),
                MemoryTier("shm-decode", "host-dram", 1 << 30),
                KVLayout(num_layers=2, num_kv_heads=2, head_dim=8, dtype="bf16"))


class TcpTransport(_KeyedKV, _kv.TcpTransport):
    """KV between two host-memory tiers on any two machines, over TCP (connector/tcpkv.py).

    Keyword arguments: remote -- (host, port) of the TcpKVServer both sides use (default
    $AIRM_KV_TCP_ADDR), serve_port -- also run that server in this process. Like shm it is a
    HOST-STAGING path: KV leaves the card, crosses host memory and the network, and is uploaded on the
    other side. Every blob carries a CRC-32 checked by the server and by the receiver, so wait() is ok
    only for the bytes submit() sent. The server hands a blob out once, so wait() keeps it for take().
    """

    name = "tcp"
    SCOPE = "only host-dram -> host-dram"

    def __init__(self, remote=None, serve_port=None):
        _kv.TcpTransport.__init__(self, remote=remote, serve_port=serve_port)
        self._held = {}                     # key -> bytes fetched by wait(), until take()
        self._init_kv()

    def supports(self, src, dst):
        return src.kind == "host-dram" and dst.kind == "host-dram" and src.name != dst.name

    def _detail(self):
        return {"server": "%s:%d" % self.remote if self.remote else None}

    def _fetch(self, key, timeout):
        buf = self.pull(key, timeout=timeout)
        with self._lock:
            self._held[key] = buf
        return buf

    def _consume(self, key):
        with self._lock:
            return self._held.pop(key)

    def close(self):
        with self._lock:
            self._pending.clear()
            self._delivered.clear()
            self._held.clear()
        _kv.TcpTransport.close(self)

    def conformance_case(self):
        """Two host tiers and a server on a free loopback port in this process. No device or network."""
        if self.remote is None:
            _kv.TcpTransport.__init__(self, serve_port=0)
        return (MemoryTier("tcp-prefill", "host-dram", 1 << 30),
                MemoryTier("tcp-decode", "host-dram", 1 << 30),
                KVLayout(num_layers=2, num_kv_heads=2, head_dim=8, dtype="bf16"))


class NixlUcxTransport(_KeyedKV, _kv.NixlUcxTransport):
    """KV between two host-memory tiers on any two machines, through NIXL's UCX backend (ucxkv.py).

    Keyword arguments: remote -- (host, port) of the TcpKVServer for descriptors (default
    $AIRM_KV_TCP_ADDR), serve_port -- also run that server in this process, agent -- a nixl_agent with
    UCX (tests pass a fake), pool_bytes -- arena size per side. HOST-STAGING like tcp; the network leg
    is UCX's (UCX_TLS). Every blob carries a CRC-32 checked by the reader, so wait() is ok only for the
    bytes submit() staged. wait() reads the blob and keeps it for take().
    """

    name = "nixl-ucx"
    SCOPE = "only host-dram -> host-dram"

    def __init__(self, remote=None, serve_port=None, agent=None, pool_bytes=None):
        _kv.NixlUcxTransport.__init__(self, remote=remote, serve_port=serve_port, agent=agent,
                                      pool_bytes=pool_bytes)
        self._held = {}
        self._init_kv()

    def supports(self, src, dst):
        return src.kind == "host-dram" and dst.kind == "host-dram" and src.name != dst.name

    def _detail(self):
        return dict(self.store.describe(), server="%s:%d" % self.remote if self.remote else None)

    def wait(self, handle, timeout_s):
        """A READ that failed (UCX error, CRC mismatch, no NIXL) is ok=False, never data."""
        started = time.perf_counter()
        try:
            return super().wait(handle, timeout_s)
        except TransportError:                       # unknown or already-waited handle
            raise
        except (IOError, RuntimeError) as e:
            return TransferResult(handle.request_id, False, 0, time.perf_counter() - started, str(e))

    def _fetch(self, key, timeout):
        buf = self.pull(key, timeout=timeout)
        with self._lock:
            self._held[key] = buf
        return buf

    def _consume(self, key):
        with self._lock:
            return self._held.pop(key)

    def close(self):
        with self._lock:
            self._pending.clear()
            self._delivered.clear()
            self._held.clear()
        _kv.NixlUcxTransport.close(self)

    def conformance_case(self):
        """Two host tiers, a descriptor server on a free loopback port, and this process's NIXL agent
        on both ends (UCX loopback). NoHardware when NIXL with UCX is not installed."""
        from rm_core.conformance import NoHardware
        try:
            self.store._get_agent()
        except RuntimeError as e:
            raise NoHardware(str(e)) from e
        if self.remote is None:
            from rngd_pd.connector.tcpkv import TcpKVClient, TcpKVServer
            self.server = TcpKVServer(0, "127.0.0.1"); self.server.start()
            self.remote = ("127.0.0.1", self.server.wait_ready())
            self._client = self.store.control = TcpKVClient(*self.remote)
        return (MemoryTier("ucx-prefill", "host-dram", 1 << 30),
                MemoryTier("ucx-decode", "host-dram", 1 << 30),
                KVLayout(num_layers=2, num_kv_heads=2, head_dim=8, dtype="bf16"))


# --------------------------------------------------------------------------
# L3: workload runner
# --------------------------------------------------------------------------
# Workload.kind -> cluster.py role and its default backend
ROLES = {"prefill": ("prefill", "rngd-full"), "decode": ("decode", "atom"),
         "serve": ("gateway", None)}


def run(workload, *, kv_transport="shm", port=8000, seq_len=256, backend=None,
        prefill_url=None, decode_url=None, metrics_port=None, args=(), env=None,
        dry_run=False):
    """Start the serving role for `workload` (one pod's worth of work).

    Workload.kind prefill -> RNGD prefill worker, decode -> decode worker
    (ATOM by default), serve -> gateway. Keys: kv_transport (entry-point name
    in airm.kv_transports, passed as AIRM_KV_TRANSPORT), port, seq_len (S the
    graphs were compiled for), backend (cluster.py --backend), prefill_url and
    decode_url (gateway only), metrics_port (gateway: airm_* on /metrics),
    args (extra cluster.py flags), env (extra environment). Returns the
    Popen, or (argv, env) with dry_run=True.
    """
    workload.validate()
    if workload.kind not in ROLES:
        raise ContractError(f"rngd-pd runs prefill, decode or serve, not {workload.kind!r}")
    from rm_core import registry
    if kv_transport not in registry.available("airm.kv_transports") and kv_transport not in _kv._OWN:
        raise KeyError(f"no airm.kv_transports implementation named {kv_transport!r}")
    role, default_backend = ROLES[workload.kind]
    argv = [sys.executable, "-m", "rngd_pd.serving.cluster", "--role", role,
            "--S", str(seq_len), "--port", str(port)]
    if role != "gateway":
        argv += ["--backend", backend or default_backend]
    if workload.batch_hint and role == "decode":
        argv += ["--B", str(workload.batch_hint)]
    if role == "gateway":
        argv += ["--prefill", prefill_url or "http://127.0.0.1:8101",
                 "--decode", decode_url or "http://127.0.0.1:8102"]
    argv += list(args)
    child = dict(os.environ)
    child.update(env or {})
    child.update(AIRM_KV_TRANSPORT=kv_transport, MODEL=workload.model)
    if metrics_port:
        child["AIRM_METRICS_PORT"] = str(metrics_port)
    if dry_run:
        return argv, child
    return subprocess.Popen(argv, env=child)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _pct(vals, q):
    vals = [v for v in vals if v is not None]
    return round(float(np.percentile(vals, q)), 1) if vals else None


def summarize(records, wall_s=None):
    """rm-core-named summary of per-request records.

    A record has ttft_ms, tpot_ms, out_tokens and, when the KV crossed,
    kv_save_ms/kv_load_ms (the gateway's `pd` fields). kv_transfer_ms is the
    median per-request handoff (save + load); tok_s divides generated tokens
    by wall_s, or by the span from first start to last end (t0/t1) if absent.
    """
    recs = [r for r in records if r and "error" not in r]
    if not recs:
        return {}
    if wall_s is None and all("t0" in r and "t1" in r for r in recs):
        wall_s = max(r["t1"] for r in recs) - min(r["t0"] for r in recs)
    kv = [(r.get("kv_save_ms") or 0.0) + (r.get("kv_load_ms") or 0.0) for r in recs
          if r.get("kv_save_ms") is not None or r.get("kv_load_ms") is not None]
    out = {"ttft_p90_ms": _pct([r.get("ttft_ms") for r in recs], 90),
           "tpot_p90_ms": _pct([r.get("tpot_ms") for r in recs], 90),
           "tok_s": round(sum(r.get("out_tokens") or 0 for r in recs) / wall_s, 1)
           if wall_s else None,
           "kv_transfer_ms": _pct(kv, 50)}
    return {k: v for k, v in out.items() if v is not None}


def measurements(summary, **tags):
    """Validated rm-core Measurements for a summarize() result."""
    tags = dict({"component": COMPONENT}, **tags)
    out = []
    for name, value in summary.items():
        t = dict(tags)
        if name != "kv_transfer_ms":
            t.pop("kv_transport", None)
        out.append(Measurement(name, float(value), t).validate())
    return out


def prometheus_text(ms):
    lines, typed = [], set()
    for m in ms:
        series = prometheus_name(m.name)
        if series not in typed:
            lines.append(f"# HELP {series} {METRICS[m.name][2]} ({m.unit})")
            lines.append(f"# TYPE {series} gauge")
            typed.add(series)
        labels = ",".join(f'{k}="{v}"' for k, v in sorted(m.tags.items()))
        lines.append(f"{series}{{{labels}}} {m.value:g}")
    return "\n".join(lines) + "\n"


class Window:
    """The last `size` gateway requests, as airm_* gauges."""

    def __init__(self, size=512):
        self.recs = collections.deque(maxlen=size)
        self.lock = threading.Lock()

    def record(self, t0, t1, resp):
        """One finished request. TPOT is the one a client sees, (wall - TTFT) / (tokens - 1): the
        decode worker's `pd.tpot_ms` is its median graph step, which leaves out the wait for a lane
        and the admission stalls, so it stays flat while the decode side saturates (rm-controller's
        rebalancer reads this gauge to decide when decode needs cards)."""
        pd = resp.get("pd", {}) if isinstance(resp, dict) else {}
        n = (resp.get("usage") or {}).get("completion_tokens", 0) if isinstance(resp, dict) else 0
        ttft = pd.get("ttft_ms")
        wall = (t1 - t0) * 1e3
        tpot = (max(0.0, wall - ttft) / (n - 1) if isinstance(ttft, (int, float)) and n and n > 1
                else pd.get("tpot_ms"))
        rec = {"t0": t0, "t1": t1, "ttft_ms": ttft, "tpot_ms": tpot, "out_tokens": n}
        if pd.get("mode") == "disaggregated-kv-transfer":
            rec.update(kv_save_ms=pd.get("kv_save_ms"), kv_load_ms=pd.get("kv_load_ms"))
        with self.lock:
            self.recs.append(rec)

    def observe(self, fn):
        """Wrap a gateway route so each completed request is recorded."""
        def wrapped(body):
            t0 = time.perf_counter()
            resp = fn(body)
            self.record(t0, time.perf_counter(), resp)
            return resp
        return wrapped

    def text(self):
        with self.lock:
            recs = list(self.recs)
        kv = os.environ.get("AIRM_KV_TRANSPORT") or "shm"
        return prometheus_text(measurements(summarize(recs), kv_transport=kv))


def serve_metrics(window, port, host="0.0.0.0"):
    """airm_* series on http://host:port/metrics, in a daemon thread."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.rstrip("/") != "/metrics":
                self.send_response(404); self.end_headers(); return
            body = window.text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
