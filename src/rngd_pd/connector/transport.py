"""KV transport abstraction for the RNGD KV connector (Phase 1).

One interface, swappable backends — so the vLLM-shaped connector (rngd_kv_connector.py) is
transport-agnostic and a NIXL plugin can drop in later without touching connector logic.

  Transport (ABC): push(key, buf) / pull(key, timeout) / close()
    ├─ InProcTransport : in-process dict          (unit tests, single process)
    ├─ ShmTransport    : /dev/shm file-backed      (cross-process, same host — RNGD default)
    ├─ TcpTransport    : TCP via kv_connector.py    (cross-node)
    └─ NixlTransport   : STUB — same interface, raises until NIXL is present (Phase 5)

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

def ser(arr):
    a = np.ascontiguousarray(arr)
    if WIRE == "bf16":
        u = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
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
    """Cross-node TCP transport wrapping kv_connector.py's TcpKVServer / tcp_push / tcp_pull.

    remote=(host, port) is the decode-side server this connector pushes to / pulls from.
    Optionally starts a local server (serve_port) so this node can also receive.
    """
    def __init__(self, remote=None, serve_port=None):
        from rngd_pd.connector.base import tcp_push, tcp_pull, TcpKVServer
        self._push, self._pull, self._Server = tcp_push, tcp_pull, TcpKVServer
        self.remote = remote                                # (host, port)
        self.server = None
        if serve_port is not None:
            self.server = TcpKVServer(serve_port); self.server.start()
    def push(self, key, buf):
        assert self.remote, "TcpTransport.push needs remote=(host,port)"
        self._push(self.remote[0], self.remote[1], key, buf)
    def pull(self, key, timeout=5.0):
        host, port = self.remote; t0 = time.time()
        while True:
            data = self._pull(host, port, key)              # server pops; empty until pushed
            if data: return data
            if time.time() - t0 > timeout: raise TimeoutError(f"pull timeout: {key}")
            time.sleep(0.002)
    def close(self):
        if self.server: self.server.stop()


class NixlTransport(Transport):
    """STUB — NIXL (RDMA/UCX) backend, same Transport contract. Not present on RNGD yet.

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
