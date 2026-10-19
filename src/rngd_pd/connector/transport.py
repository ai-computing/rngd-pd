"""KV transport abstraction for the RNGD KV connector (Phase 1).

One interface, swappable backends — so the vLLM-shaped connector (rngd_kv_connector.py) is
transport-agnostic and a NIXL plugin can drop in later without touching connector logic.

  Transport (ABC): push(key, buf) / pull(key, timeout) / close()
    ├─ InProcTransport : in-process dict          (unit tests, single process)
    ├─ ShmTransport    : /dev/shm file-backed      (cross-process, same host — RNGD default)
    ├─ TcpTransport    : TCP to a TcpKVServer      (cross-node; tcpkv.py, the server can be a relay)
    └─ NixlTransport   : STUB — same interface, raises until NIXL is present (Phase 5)

  select_transport(name) picks the path by name through rm-core's `airm.kv_transports` entry
  points ($AIRM_KV_TRANSPORT, default "shm" = rngd_pd.airm.ShmTransport), so switching to
  hetero-npu-p2p's `p2p` is configuration, not code.

RNGD note: D2D link = host PCIe (no RDMA/NVLink), so shm/TCP is the real transport. NIXL is exposed
as a pluggable backend with the same contract; the handshake hooks (handshake_metadata /
set_remote_handshake) mirror vLLM's NIXL agent exchange so the connector code is future-proof.
"""
from abc import ABC, abstractmethod
import os, time, threading, pickle
import numpy as np

# Wire format: fp16 (default, 2B/elem) or bf16 (2B/elem, BIT-EXACT for bf16-valued
# KV — fp16 collapses subnormals <6e-5 and can flip argmax on long prompts).
WIRE = "fp16"

def ser(arr, wire=None):
    a = np.ascontiguousarray(arr)
    if (wire or WIRE) == "bf16":
        # astype always copies, and the caller already hands us float32 — that
        # was a full extra pass over every layer's cache, 67 MB per request.
        f = a if a.dtype == np.float32 else a.astype(np.float32)
        u = (f.view(np.uint32) >> 16).astype(np.uint16)
        return pickle.dumps({"bf16": u, "shape": a.shape}, protocol=4)
    return pickle.dumps(a.astype(np.float16), protocol=4)

def deser(buf):
    o = pickle.loads(buf)
    if isinstance(o, dict) and "bf16" in o:
        return (o["bf16"].astype(np.uint32) << 16).view(np.float32).reshape(o["shape"])
    return o.astype(np.float32)


class Transport(ABC):
    """Key→bytes KV transport. push is non-blocking-ish (may buffer); pull blocks until present."""
    @abstractmethod
    def push(self, key: str, buf: bytes) -> None: ...
    @abstractmethod
    def pull(self, key: str, timeout: float = 5.0) -> bytes: ...
    def close(self) -> None: ...

    # --- NIXL-parity handshake hooks (no-op for host transports) ---
    def handshake_metadata(self) -> dict:
        """Agent/endpoint info a remote peer needs to reach us (NIXL: agent name + mem descs)."""
        return {}
    def set_remote_handshake(self, meta: dict) -> None:
        """Register a remote peer's handshake metadata (NIXL: add_remote_agent)."""
        pass


class InProcTransport(Transport):
    """In-process shared dict. Single process only (unit tests / same-runtime PD demo)."""
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()
    def push(self, key, buf):
        with self._lock: self._store[key] = buf
    def pull(self, key, timeout=5.0):
        t0 = time.time()
        while True:
            with self._lock:
                if key in self._store: return self._store[key]
            if time.time() - t0 > timeout: raise TimeoutError(f"pull timeout: {key}")
            time.sleep(0.001)
    def close(self): self._store.clear()


class ShmTransport(Transport):
    """/dev/shm file-backed store — cross-process on the same host (RNGD default).

    Genuinely shm-equivalent (tmpfs), so prefill and decode worker PROCESSES on one node share KV
    without a network hop. Atomic publish via write-tmp + rename so pull never sees a partial file.
    """
    def __init__(self, root="/dev/shm/rngd_kv"):
        self.root = root
        os.makedirs(root, exist_ok=True)
    def _path(self, key): return os.path.join(self.root, key.replace("/", "__"))
    def push(self, key, buf):
        p = self._path(key); tmp = p + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(buf); f.flush(); os.fsync(f.fileno())
        os.rename(tmp, p)                                    # atomic publish
    def pull(self, key, timeout=5.0):
        p = self._path(key); t0 = time.time()
        while not os.path.exists(p):
            if time.time() - t0 > timeout: raise TimeoutError(f"pull timeout: {key}")
            time.sleep(0.001)
        with open(p, "rb") as f: return f.read()
    def close(self): pass                                   # files persist for the peer; caller purges


class TcpTransport(Transport):
    """Cross-node KV over TCP (rngd_pd.connector.tcpkv): push to and take from one TcpKVServer.

    remote=(host, port) is that server; without it, $AIRM_KV_TCP_ADDR ("host:port"). Prefill and
    decode name the same server, which can run beside the decode worker (serve_port= starts one in this
    process) or on a third node as a relay; neither worker has to reach the other. Connections are
    pooled and kept open, and every blob carries a CRC-32 checked on arrival and on delivery
    (AIRM_KV_TCP_CRC=0 skips it), so a pull returns the pushed bytes or raises. select_transport("tcp")
    hands every caller in a process one instance (`shared`), so the pool outlives a request.
    """
    shared = True

    def __init__(self, remote=None, serve_port=None):
        from rngd_pd.connector.tcpkv import TcpKVClient, TcpKVServer, parse_addr
        self.server = None
        if serve_port is not None:
            self.server = TcpKVServer(serve_port); self.server.start()
            serve_port = self.server.wait_ready()
        if remote is None and os.environ.get("AIRM_KV_TCP_ADDR"):
            remote = parse_addr(os.environ["AIRM_KV_TCP_ADDR"])
        if remote is None and serve_port is not None:
            remote = ("127.0.0.1", serve_port)
        self.remote = tuple(remote) if remote else None    # (host, port)
        self._client = TcpKVClient(*self.remote) if self.remote else None
    def _need(self):
        if self._client is None:
            raise ValueError("tcp transport needs remote=(host, port) or AIRM_KV_TCP_ADDR=host:port")
        return self._client
    def push(self, key, buf):
        t0 = time.perf_counter()
        self._need().push(key, buf)
        self._log("push", key, buf, t0)
    def pull(self, key, timeout=5.0):
        t0 = time.perf_counter()
        buf = self._need().take(key, timeout=timeout)       # the server waits; no polling
        self._log("pull", key, buf, t0)
        return buf
    @staticmethod
    def _log(op, key, buf, t0):
        """AIRM_KV_TCP_LOG=1: one line per blob (key, bytes, sha256 prefix, ms) for byte-exact audits."""
        if os.environ.get("AIRM_KV_TCP_LOG", "0") not in ("0", ""):
            import hashlib
            print(f"[kv-tcp] {op} {key} {len(buf)} {hashlib.sha256(buf).hexdigest()[:16]} "
                  f"{(time.perf_counter() - t0) * 1e3:.2f}ms", flush=True)
    def handshake_metadata(self):
        return {"tcp": "%s:%d" % self.remote} if self.remote else {}
    def close(self):
        if self._client: self._client.close()
        if self.server: self.server.stop()


class NixlTransport(Transport):
    """STUB — a key->bytes NIXL path for the connector. Not built.

    The NIXL path that exists is rngd_pd.kv_nixl.NixlTransport (AIRM_KV_TRANSPORT=nixl): an rm-core
    tier-to-tier transport over hetero-npu-p2p's HETERO_NPU backend (RNGD -> ATOM DMA-BUF, RNGD <->
    host staging legs), which select_transport() wraps in FencedTransport like `p2p`.

    Kept so the connector can select backend='nixl' unchanged once NIXL lands (Phase 5). RNGD has no
    RDMA (D2D=host PCIe); a real impl would map NIXL to host-staged transfers. Fails LOUD — never a
    silent fallback to shm/tcp (that would mask a misconfigured deployment).
    """
    def __init__(self, *a, **kw):
        try:
            import nixl  # noqa: F401
        except Exception as e:
            raise NotImplementedError(
                "NIXL backend not available (no `nixl` module; RNGD has no RDMA). "
                "Use ShmTransport (same host) or TcpTransport (cross-node). "
                "NIXL integration is scheduled for Phase 5.") from e
        raise NotImplementedError("NixlTransport is a stub: connect/handshake not implemented yet.")
    def push(self, key, buf): raise NotImplementedError
    def pull(self, key, timeout=5.0): raise NotImplementedError


class FencedTransport(Transport):
    """Connector Transport over an rm-core KVTransport that only moves data between tiers.

    A tier-to-tier transport such as hetero-npu-p2p's `p2p` copies device KV itself and has no
    notion of the connector's key -> bytes handoff. push() plans and submits one transfer per key,
    pull() waits on that transfer's completion fence and only then hands back the bytes; a failed
    or missing fence raises rather than returning data the transport never confirmed.

    Limits, by construction: handles are process-local, so prefill and decode must run in one
    process (the hetero-npu-p2p coordinator) -- across pods every pull times out. The bytes the
    connector consumes stay in this process; the transport only fences its own copy between
    `src` and `dst` (host-dram placeholders unless given), using a 2-byte-per-token layout.
    """
    def __init__(self, impl, src=None, dst=None):
        from rm_core import KVLayout, MemoryTier
        self.impl = impl
        self.name = impl.name
        self.src = src or MemoryTier("prefill", "host-dram", 1 << 40)
        self.dst = dst or MemoryTier("decode", "host-dram", 1 << 40)
        self._layout = KVLayout(num_layers=1, num_kv_heads=1, head_dim=1, dtype="int8")   # 2 B/token
        self._pending = {}
        self._lock = threading.Lock()
    def push(self, key, buf):
        n = len(buf)
        plan = self.impl.plan(key, self.src, self.dst, self._layout, self._layout,
                              (0, max(1, (n + 1) // 2)))
        with self._lock:
            if key in self._pending:
                raise ValueError(f"{self.name}: {key} pushed twice before it was pulled")
            self._pending[key] = None                        # reserve before submitting
        handle = self.impl.submit(plan)
        with self._lock: self._pending[key] = (handle, buf)
    def pull(self, key, timeout=5.0):
        from rm_core import TransportError
        t0 = time.time()
        while True:
            with self._lock: entry = self._pending.get(key)
            if entry is not None: break
            if time.time() - t0 > timeout: raise TimeoutError(f"pull timeout: {key}")
            time.sleep(0.001)
        handle, buf = entry
        try:
            res = self.impl.wait(handle, timeout_s=max(0.0, timeout - (time.time() - t0)))
        finally:
            with self._lock: self._pending.pop(key, None)     # a handle is waited at most once
        if not res.ok:
            raise TransportError(f"{self.name}: transfer {key} failed: {res.error}")
        return buf
    def close(self):
        with _LOCK:
            for k in [k for k, v in _FENCED.items() if v is self]: del _FENCED[k]
        self.impl.close()


# Our own adapters, for a checkout run from src/ without `pip install -e .` (the pods put
# /app/src on PYTHONPATH). Same names and objects as the entry points in pyproject.toml.
_OWN = {"shm": "rngd_pd.airm:ShmTransport", "tcp": "rngd_pd.airm:TcpTransport",
        "nixl": "rngd_pd.kv_nixl:NixlTransport"}
_CLASSES = {}          # name -> implementation class, resolved once per process
_FENCED = {}           # name -> the process's one FencedTransport (both sides share handles)
_SHARED = {}           # name -> the process's one instance of a transport that holds connections
_LOCK = threading.Lock()


def _resolve(name, kw):
    """An instance of the implementation registered as `name`; the class is looked up once."""
    cls = _CLASSES.get(name)
    if cls is not None:
        return cls(**kw)
    try:
        from rm_core import registry
    except ImportError:
        # rm-core is not importable (a pod with only src/ on the path). Our own "shm" and "tcp"
        # are the same paths either way; any other name cannot be resolved and says so.
        if name == "tcp":
            _CLASSES[name] = cls = TcpTransport
            return cls(**kw)
        if name != "shm":
            raise KeyError(f"AIRM_KV_TRANSPORT={name!r} needs rm-core installed")
        _CLASSES[name] = cls = ShmTransport
        return cls(**dict({"root": os.environ.get("PD_SHM_ROOT", "/dev/shm/rngd_kv_cluster")}, **kw))
    try:
        impl = registry.load("airm.kv_transports", name, **kw)
    except KeyError:
        if name not in _OWN: raise
        import importlib
        mod, obj = _OWN[name].split(":")
        impl = getattr(importlib.import_module(mod), obj)(**kw)
    _CLASSES[name] = type(impl)
    return impl


def select_transport(name=None, **kw) -> Transport:
    """The connector's KV path, chosen by name from configuration (ai-rm protocol v0.1).

    name falls back to $AIRM_KV_TRANSPORT, then "shm" (the /dev/shm host-staging path the
    serving workers always used). The implementation comes from the `airm.kv_transports` entry
    point of that name (rm_core.registry.load); kw goes to its constructor. One that speaks the
    connector's push/pull is used as is; a tier-to-tier KVTransport is wrapped in FencedTransport.
    An unknown name raises KeyError -- never a fallback to another path.
    """
    name = name or os.environ.get("AIRM_KV_TRANSPORT") or "shm"
    with _LOCK:
        if not kw and name in _FENCED: return _FENCED[name]
        if not kw and name in _SHARED: return _SHARED[name]
        impl = _resolve(name, kw)
        if isinstance(impl, Transport):
            # The workers ask for the path once per request; a connection pool must outlive that.
            if not kw and getattr(impl, "shared", False): _SHARED[name] = impl
            return impl
        t = FencedTransport(impl)
        if not kw: _FENCED[name] = t
        return t


def make_transport(backend="shm", **kw) -> Transport:
    """Factory: backend ∈ {inproc, shm, tcp, nixl}."""
    b = backend.lower()
    if b == "inproc": return InProcTransport()
    if b == "shm":    return ShmTransport(**kw)
    if b == "tcp":    return TcpTransport(**kw)
    if b == "nixl":   return NixlTransport(**kw)
    raise ValueError(f"unknown transport backend: {backend}")


def _selftest():
    print("=== kv_transport self-test ===")
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((8, 16, 128)).astype(np.float32)
    ref16 = arr.astype(np.float16).astype(np.float32)
    for name, tr in [("inproc", InProcTransport()), ("shm", ShmTransport(root="/dev/shm/rngd_kv_test"))]:
        tr.push("req0/layer.0/k", ser(arr))
        got = deser(tr.pull("req0/layer.0/k", timeout=2.0))
        err = np.abs(got - ref16).max()
        print(f"  {name:6s}: round-trip loss {err:.6f}  {'PASS ✅' if err < 1e-6 else 'FAIL ❌'}")
        tr.close()
    # NIXL must fail loud, not silently degrade
    try:
        NixlTransport(); print("  nixl  : FAIL ❌ (should have raised)")
    except NotImplementedError:
        print("  nixl  : PASS ✅ (raises loudly — no silent fallback)")


if __name__ == "__main__":
    _selftest()
