"""Cross-node KV over TCP: a key -> bytes store server and a pooled client. Standard library only.

The server holds pushed blobs in memory until one pull takes them. It can run beside the decode worker
or on any other node as a relay (a CPU-only pod is enough): prefill pushes to it, decode pulls from it,
and neither side needs to reach the other directly.

    python3 tcpkv.py --port 5577          # a relay; this file alone, any Python 3.6+

Wire (little-endian), one connection carries any number of requests:

    push  b"P" klen:i32 key  n:i64 crc:u32 data   -> b"OK" | b"ER" (crc mismatch, blob dropped)
    take  b"W" klen:i32 key  wait_ms:i32           -> n:i64 crc:u32 data, n = -1 if absent after wait_ms
    stats b"S"                                    -> n:i64 json

`crc` is zlib.crc32 of the data, computed by the pusher, checked by the server on arrival and by the
puller on delivery, so a pull returns the pushed bytes or raises. AIRM_KV_TCP_CRC=0 skips computing
it (crc 0 = unchecked) where the CPU time matters more than the end-to-end check.
"""
import argparse
import json
import os
import queue
import socket
import struct
import threading
import time
import zlib

_HDR_P = struct.Struct("<qI")          # n, crc
_I32 = struct.Struct("<i")


def _crc_on():
    return os.environ.get("AIRM_KV_TCP_CRC", "1") not in ("0", "false", "no")


def _recv_exact(sock, n):
    """n bytes into one buffer (recv_into: no re-concatenation), or ConnectionError."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError(f"peer closed after {got} of {n} bytes")
        got += k
    return buf


class TcpKVServer(threading.Thread):
    """In-memory key -> bytes store behind a TCP port. Runs on the decode node or a relay node.

    A blob nobody takes within `ttl_s` is dropped (a request that died between prefill and decode would
    otherwise hold its 67 MB forever).
    """

    def __init__(self, port, host="0.0.0.0", ttl_s=None):
        super().__init__(daemon=True)
        self.port, self.host = port, host
        self.ttl_s = float(ttl_s if ttl_s is not None else os.environ.get("AIRM_KV_TCP_TTL_S", "900"))
        self.store = {}                         # key -> (data, crc, t_pushed)
        self.cv = threading.Condition()
        self.stats = {"pushed": 0, "taken": 0, "expired": 0, "crc_errors": 0, "bytes_in": 0, "bytes_out": 0}
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._sock = None

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        self.port = s.getsockname()[1]          # port 0 -> the one the OS picked
        s.listen(64)
        s.settimeout(0.5)
        self._sock = s
        self._ready.set()
        while not self._stop.is_set():
            try:
                c, _ = s.accept()
            except socket.timeout:
                self._expire()
                continue
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()
        s.close()

    def wait_ready(self, timeout=5.0):
        if not self._ready.wait(timeout):
            raise RuntimeError("TcpKVServer did not start")
        return self.port

    def stop(self):
        self._stop.set()

    def _expire(self):
        now = time.monotonic()
        with self.cv:
            old = [k for k, (_d, _c, t) in self.store.items() if now - t > self.ttl_s]
            for k in old:
                del self.store[k]
            self.stats["expired"] += len(old)

    def _serve(self, c):
        try:
            while True:
                op = c.recv(1)
                if not op:
                    return
                if op == b"S":
                    with self.cv:
                        body = json.dumps(dict(self.stats, keys=len(self.store),
                                               bytes_held=sum(len(d) for d, _c, _t in self.store.values()))).encode()
                    c.sendall(struct.pack("<q", len(body)) + body)
                    continue
                key = bytes(_recv_exact(c, _I32.unpack(_recv_exact(c, 4))[0])).decode()
                if op == b"P":
                    n, crc = _HDR_P.unpack(_recv_exact(c, _HDR_P.size))
                    data = _recv_exact(c, n)
                    if crc and zlib.crc32(data) != crc:
                        with self.cv:
                            self.stats["crc_errors"] += 1
                        c.sendall(b"ER")
                        continue
                    with self.cv:
                        self.store[key] = (data, crc, time.monotonic())
                        self.stats["pushed"] += 1
                        self.stats["bytes_in"] += n
                        self.cv.notify_all()
                    c.sendall(b"OK")
                elif op == b"W":
                    deadline = time.monotonic() + _I32.unpack(_recv_exact(c, 4))[0] / 1e3
                    with self.cv:
                        while key not in self.store:
                            left = deadline - time.monotonic()
                            if left <= 0:
                                break
                            self.cv.wait(left)
                        entry = self.store.pop(key, None)
                        if entry is not None:
                            self.stats["taken"] += 1
                            self.stats["bytes_out"] += len(entry[0])
                    if entry is None:
                        c.sendall(_HDR_P.pack(-1, 0))
                    else:
                        c.sendall(_HDR_P.pack(len(entry[0]), entry[1]))
                        c.sendall(entry[0])
                else:
                    return                      # unknown op: drop the connection
        except (ConnectionError, OSError):
            return
        finally:
            c.close()


class TcpKVClient:
    """Pushes to / takes from one TcpKVServer over a small pool of persistent connections.

    Persistent, because a fresh connection per 1 MB layer restarts TCP slow start every time. Safe to
    share between threads: each call borrows its own connection.
    """

    def __init__(self, host, port, connect_timeout=10.0):
        self.addr = (host, int(port))
        self.connect_timeout = connect_timeout
        self._idle = queue.LifoQueue()

    def _conn(self):
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            s = socket.create_connection(self.addr, timeout=self.connect_timeout)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s

    def _call(self, fn):
        s = self._conn()
        try:
            out = fn(s)
        except BaseException:
            s.close()                           # unknown state: never reuse it
            raise
        self._idle.put(s)
        return out

    def push(self, key, data):
        kb = key.encode()
        crc = zlib.crc32(data) if _crc_on() else 0

        def go(s):
            s.sendall(b"P" + _I32.pack(len(kb)) + kb + _HDR_P.pack(len(data), crc))
            s.sendall(data)
            ack = bytes(_recv_exact(s, 2))
            if ack != b"OK":
                raise IOError(f"tcpkv push {key}: server rejected the blob ({ack!r}, crc mismatch)")
        self._call(go)

    def take(self, key, timeout=5.0):
        """The blob pushed under `key` (bytearray), waiting up to `timeout` s for it; TimeoutError if
        it never came. Removes it from the server."""
        kb = key.encode()
        wait_ms = max(0, int(timeout * 1e3))

        def go(s):
            s.sendall(b"W" + _I32.pack(len(kb)) + kb + _I32.pack(wait_ms))
            n, crc = _HDR_P.unpack(_recv_exact(s, _HDR_P.size))
            if n < 0:
                return None
            data = _recv_exact(s, n)
            if crc and _crc_on() and zlib.crc32(data) != crc:
                raise IOError(f"tcpkv take {key}: crc mismatch on delivery")
            return data
        data = self._call(go)
        if data is None:
            raise TimeoutError(f"pull timeout: {key}")
        return data

    def stats(self):
        def go(s):
            s.sendall(b"S")
            n = struct.unpack("<q", bytes(_recv_exact(s, 8)))[0]
            return json.loads(bytes(_recv_exact(s, n)))
        return self._call(go)

    def close(self):
        while True:
            try:
                self._idle.get_nowait().close()
            except queue.Empty:
                return


def parse_addr(text, default_port=5577):
    """'host:port' or 'host' -> (host, port)."""
    host, _, port = text.strip().rpartition(":") if ":" in text else (text.strip(), "", "")
    return host, int(port or default_port)


def main(argv=None):
    ap = argparse.ArgumentParser(description="KV relay: holds pushed KV blobs until a pull takes them")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AIRM_KV_TCP_PORT", "5577")))
    ap.add_argument("--stats-every", type=float, default=60.0, help="print counters every N s (0: never)")
    a = ap.parse_args(argv)
    srv = TcpKVServer(a.port, a.host)
    srv.start()
    print(f"[tcpkv] relay on {a.host}:{srv.wait_ready()}", flush=True)
    last = None
    try:
        while srv.is_alive():
            time.sleep(a.stats_every or 3600)
            if a.stats_every:
                with srv.cv:
                    snap = dict(srv.stats, keys=len(srv.store))
                if snap != last:
                    print(f"[tcpkv] {json.dumps(snap)}", flush=True)
                    last = snap
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
