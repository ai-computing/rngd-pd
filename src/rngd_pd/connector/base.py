"""M4 — KV connector: standard KV block format + pluggable transport (shm | tcp).

Replaces the file-based handoff with a real transfer layer, mirroring llm-d/vLLM's
KVConnector role (NIXL is the GPU/RDMA version; on RNGD the D2D link is host-PCIe, so
we transfer host-side: shared memory for local (zero-copy), TCP for cross-node).

KV block format (contiguous, self-describing):
  header: magic 'KVC1', nL, T, kv_dim, first_token, next_pos  (little-endian uint32/int32)
  body:   float32 [2, nL, T, kv_dim]  (k then v)

  python kv_connector.py            # self-test: shm vs tcp round-trip + timing
"""
import socket, struct, threading, time
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
class TcpKVServer(threading.Thread):
    """Holds pushed KV blocks in memory; serves pulls. Runs on the decode node."""
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port; self.store = {}; self._stop = False

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", self.port)); s.listen(8); s.settimeout(0.5)
        while not self._stop:
            try:
                c, _ = s.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()
        s.close()

    def _handle(self, c):
        op = c.recv(1)                                       # b'P' push | b'G' get
        klen = struct.unpack("<i", _recvn(c, 4))[0]
        key = _recvn(c, klen).decode()
        if op == b"P":
            n = struct.unpack("<q", _recvn(c, 8))[0]
            self.store[key] = _recvn(c, n)
            c.sendall(b"OK")
        else:
            data = self.store.pop(key, b"")
            c.sendall(struct.pack("<q", len(data)) + data)
        c.close()

    def stop(self): self._stop = True


def _recvn(c, n):
    buf = b""
    while len(buf) < n:
        p = c.recv(n - len(buf))
        if not p:
            break
        buf += p
    return buf


def tcp_push(host, port, key, data):
    c = socket.create_connection((host, port))
    kb = key.encode()
    c.sendall(b"P" + struct.pack("<i", len(kb)) + kb + struct.pack("<q", len(data)) + data)
    assert c.recv(2) == b"OK"; c.close()


def tcp_pull(host, port, key):
    c = socket.create_connection((host, port))
    kb = key.encode()
    c.sendall(b"G" + struct.pack("<i", len(kb)) + kb)
    n = struct.unpack("<q", _recvn(c, 8))[0]
    data = _recvn(c, n); c.close()
    return data


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
    srv = TcpKVServer(0)
    import socket as _s
    ss = _s.socket(); ss.bind(("127.0.0.1", 0)); port = ss.getsockname()[1]; ss.close()
    srv.port = port; srv.start(); time.sleep(0.2)
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
