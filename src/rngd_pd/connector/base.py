"""M4 — KV connector: standard KV block format + pluggable transport (shm | tcp).

Replaces the file-based handoff with a real transfer layer, mirroring llm-d/vLLM's
KVConnector role (NIXL is the GPU/RDMA version; on RNGD the D2D link is host-PCIe, so
we transfer host-side: shared memory for local (zero-copy), TCP for cross-node).

KV block format (contiguous, self-describing):
  header: magic 'KVC1', nL, T, kv_dim, first_token, next_pos  (little-endian uint32/int32)
  body:   float32 [2, nL, T, kv_dim]  (k then v)

  python kv_connector.py            # self-test: shm vs tcp round-trip + timing
"""
import struct, time
from multiprocessing import shared_memory
import numpy as np

MAGIC = b"KVC1"


def serialize_kv(ck, cv, first_token, next_pos):
    nL = len(ck); T = len(ck[0]); kvd = ck[0][0].shape[0]
    k = np.stack([np.stack(ck[L]) for L in range(nL)]).astype(np.float32)   # [nL,T,kvd]
    v = np.stack([np.stack(cv[L]) for L in range(nL)]).astype(np.float32)
    hdr = MAGIC + struct.pack("<5i", nL, T, kvd, first_token, next_pos)
    return hdr + k.tobytes() + v.tobytes()


def deserialize_kv(buf):
    assert buf[:4] == MAGIC, "bad KV block"
    nL, T, kvd, first, pos = struct.unpack("<5i", buf[4:24])
    n = nL * T * kvd
    arr = np.frombuffer(buf[24:], dtype=np.float32)
    k = arr[:n].reshape(nL, T, kvd); v = arr[n:2*n].reshape(nL, T, kvd)
    ck = [[k[L][t].copy() for t in range(T)] for L in range(nL)]
    cv = [[v[L][t].copy() for t in range(T)] for L in range(nL)]
    return ck, cv, first, pos


# ---- local transport: shared memory (zero-copy on same host) ----
class ShmConnector:
    def put(self, key, data: bytes):
        try:
            shared_memory.SharedMemory(name=key).unlink()   # clear stale
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=key, create=True, size=len(data) + 8)
        shm.buf[:8] = struct.pack("<q", len(data))
        shm.buf[8:8 + len(data)] = data
        shm.close()                                          # keep segment alive (not unlinked)
        return f"shm:{key}"

    def get(self, key):
        shm = shared_memory.SharedMemory(name=key)
        n = struct.unpack("<q", bytes(shm.buf[:8]))[0]
        data = bytes(shm.buf[8:8 + n])
        shm.close(); shm.unlink()                            # consume
        return data


# ---- remote transport: TCP (cross-node) ----
# The server and the pooled client live in tcpkv.py (standard library only, so a relay pod needs nothing
# else). These two keep the old one-shot call shape.
from rngd_pd.connector.tcpkv import TcpKVClient, TcpKVServer  # noqa: E402,F401


def tcp_push(host, port, key, data):
    c = TcpKVClient(host, port)
    try:
        c.push(key, data)
    finally:
        c.close()


def tcp_pull(host, port, key):
    """The blob under `key`, taken from the server, or b"" if nothing is there yet."""
    c = TcpKVClient(host, port)
    try:
        return bytes(c.take(key, timeout=0))
    except TimeoutError:
        return b""
    finally:
        c.close()


def _selftest():
    nL, T, kvd = 24, 32, 128
    ck = [[np.random.randn(kvd).astype(np.float32) for _ in range(T)] for _ in range(nL)]
    cv = [[np.random.randn(kvd).astype(np.float32) for _ in range(T)] for _ in range(nL)]
    blob = serialize_kv(ck, cv, first_token=12095, next_pos=T)
    print(f"KV block: {len(blob)/1024:.1f} KB ({nL} layers × {T} × {kvd})\n")

    # shm round-trip
    shm = ShmConnector(); t = time.perf_counter()
    ref = shm.put("kvtest", blob); got = shm.get("kvtest")
    tshm = (time.perf_counter() - t) * 1000
    c2, v2, f2, p2 = deserialize_kv(got)
    ok_shm = (got == blob) and f2 == 12095 and np.allclose(c2[0][0], ck[0][0])

    # tcp round-trip (loopback)
    srv = TcpKVServer(0, "127.0.0.1"); srv.start(); port = srv.wait_ready()
    t = time.perf_counter()
    tcp_push("127.0.0.1", port, "kvtest", blob); gt = tcp_pull("127.0.0.1", port, "kvtest")
    ttcp = (time.perf_counter() - t) * 1000
    srv.stop()
    ok_tcp = (gt == blob)

    print(f"shm  round-trip: {tshm:6.3f} ms  {'PASS ✅' if ok_shm else 'FAIL'}")
    print(f"tcp  round-trip: {ttcp:6.3f} ms  {'PASS ✅' if ok_tcp else 'FAIL'}  (loopback; real = cross-node)")
    print("\n→ same KV block format, two transports: shm (local) / tcp (remote node).")


if __name__ == "__main__":
    _selftest()
