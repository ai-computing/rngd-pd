"""Cross-node KV over TCP: a key -> bytes store server and a pooled client. Standard library only.

The server holds pushed blobs in memory until one pull takes them. It can run beside the decode worker
or on any other node as a relay (a CPU-only pod is enough): prefill pushes to it, decode pulls from it,
and neither side needs to reach the other directly.

    python3 tcpkv.py --port 5577          # a relay; this file alone, any Python 3.6+

Wire (little-endian), one connection carries any number of requests:

    push  b"P" klen:i32 key  n:i64 crc:u32 data   -> b"OK" | b"ER" (CRC or capacity rejection)
    take  b"W" klen:i32 key  wait_ms:i32           -> n:i64 crc:u32 data, n = -1 if absent after wait_ms
    stats b"S"                                    -> n:i64 json

`crc` is zlib.crc32 of the data, computed by the pusher, checked by the server on arrival and by the
puller on delivery, so a pull returns the pushed bytes or raises. AIRM_KV_TCP_CRC=0 skips computing
it (crc 0 = unchecked) where the CPU time matters more than the end-to-end check.

Limits (AIRM_KV_TCP_ prefix): MAX_BLOB_BYTES=256 MiB, MAX_STORE_BYTES=1 GiB,
MAX_KEY_BYTES=4096, FRAME_TIMEOUT_S=60 and MAX_WAIT_MS=900000. Store admission counts
incoming and outgoing payloads too; max_entries=65536 bounds empty-blob metadata,
and max_connections=128 caps admitted connections before starting their handler threads.
Replacing a key needs temporary room for both blobs. Workers must agree on wire limits.
"""
import argparse
import json
import math
import os
import queue
import select
import socket
import struct
import threading
import time
import zlib

_HDR_P = struct.Struct("<qI")          # n, crc
_I32 = struct.Struct("<i")
_MAX_STATS_BYTES = 64 << 10


def _setting(name, default, cast=int):
    value = cast(os.environ.get("AIRM_KV_TCP_" + name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("AIRM_KV_TCP_" + name + " must be positive and finite")
    return value


def _length(n, maximum):
    if n < 0 or n > maximum:
        raise ValueError("tcpkv length exceeds limit: %s" % n)
    return n


def _remaining(sock, deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("tcpkv operation deadline exceeded")
    sock.settimeout(left)


def _send(sock, data, deadline):
    _remaining(sock, deadline)
    sock.sendall(data)


def _crc_on():
    return os.environ.get("AIRM_KV_TCP_CRC", "1") not in ("0", "false", "no")


def _recv_exact(sock, n, deadline):
    """n bytes into one buffer (recv_into: no re-concatenation), or ConnectionError."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        _remaining(sock, deadline)
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

    def __init__(self, port, host="0.0.0.0", ttl_s=None, max_entries=65536, max_connections=128):
        super().__init__(daemon=True)
        self.port, self.host = port, host
        self.ttl_s = float(ttl_s if ttl_s is not None else os.environ.get("AIRM_KV_TCP_TTL_S", "900"))
        # One full layer is allowed, while stored AND incoming blobs share the relay budget.
        self.max_blob_bytes = _setting("MAX_BLOB_BYTES", 256 << 20)
        self.max_store_bytes = _setting("MAX_STORE_BYTES", 1 << 30)
        self.max_key_bytes = _setting("MAX_KEY_BYTES", 4096)
        self.frame_timeout_s = _setting("FRAME_TIMEOUT_S", 60.0, float)
        self.max_wait_ms = min(_setting("MAX_WAIT_MS", 900000), (1 << 31) - 1)
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        if type(max_connections) is not int or max_connections <= 0:
            raise ValueError("max_connections must be a positive integer")
        self.max_connections = max_connections
        self._bytes_held = 0
        self._bytes_reserved = 0
        self._entries_reserved = 0
        self.store = {}                         # key -> (data, crc, t_pushed)
        self.cv = threading.Condition()
        self._clients = set()
        self.stats = {"pushed": 0, "taken": 0, "expired": 0, "crc_errors": 0, "bytes_in": 0, "bytes_out": 0}
        self._stopping = threading.Event()
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
        while not self._stopping.is_set():
            try:
                c, _ = s.accept()
            except socket.timeout:
                self._expire()
                continue
            with self.cv:
                admitted = not self._stopping.is_set() and len(self._clients) < self.max_connections
                if admitted:
                    self._clients.add(c)
            if not admitted:
                c.close()
                continue
            try:
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=self._serve, args=(c,), daemon=True).start()
            except (OSError, RuntimeError):
                with self.cv:
                    self._clients.discard(c)
                c.close()
        s.close()

    def wait_ready(self, timeout=5.0):
        if not self._ready.wait(timeout):
            raise RuntimeError("TcpKVServer did not start")
        return self.port

    def stop(self):
        self._stopping.set()
        with self.cv:
            clients = list(self._clients)
            self.cv.notify_all()
        for c in clients:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.close()

    def _expire(self):
        now = time.monotonic()
        with self.cv:
            old = [k for k, (_d, _c, t) in self.store.items() if now - t > self.ttl_s]
            for k in old:
                self._bytes_held -= len(self.store[k][0])
                del self.store[k]
            self.stats["expired"] += len(old)

    def _serve(self, c):
        reserved = 0
        entry_reserved = False
        try:
            while True:
                deadline = time.monotonic() + self.frame_timeout_s
                op = bytes(_recv_exact(c, 1, deadline))
                if op not in (b"P", b"W", b"S"):
                    return
                if op == b"S":
                    with self.cv:
                        body = json.dumps(dict(self.stats, keys=len(self.store),
                                               bytes_held=self._bytes_held,
                                               bytes_reserved=self._bytes_reserved,
                                               entries_reserved=self._entries_reserved)).encode()
                    _send(c, struct.pack("<q", len(body)) + body, deadline)
                    continue
                klen = _length(_I32.unpack(_recv_exact(c, 4, deadline))[0], self.max_key_bytes)
                key = bytes(_recv_exact(c, klen, deadline)).decode()
                if op == b"P":
                    n, crc = _HDR_P.unpack(_recv_exact(c, _HDR_P.size, deadline))
                    _length(n, self.max_blob_bytes)
                    with self.cv:
                        self._expire()
                        rejected = (self._bytes_held + self._bytes_reserved + n > self.max_store_bytes or
                                    len(self.store) + self._entries_reserved >= self.max_entries)
                        if not rejected:
                            self._bytes_reserved += n
                            self._entries_reserved += 1
                            reserved = n
                            entry_reserved = True
                    if rejected:
                        _send(c, b"ER", deadline)
                        return
                    data = _recv_exact(c, n, deadline)
                    if crc and zlib.crc32(data) != crc:
                        del data
                        with self.cv:
                            self.stats["crc_errors"] += 1
                            self._bytes_reserved -= reserved
                            self._entries_reserved -= 1
                            reserved = 0
                            entry_reserved = False
                        _send(c, b"ER", deadline)
                        continue
                    with self.cv:
                        old = self.store.get(key)
                        self._bytes_held += n - (len(old[0]) if old else 0)
                        self.store[key] = (data, crc, time.monotonic())
                        del old, data
                        self._bytes_reserved -= reserved
                        self._entries_reserved -= 1
                        reserved = 0
                        entry_reserved = False
                        self.stats["pushed"] += 1
                        self.stats["bytes_in"] += n
                        self.cv.notify_all()
                    _send(c, b"OK", deadline)
                elif op == b"W":
                    wait_ms = _length(_I32.unpack(_recv_exact(c, 4, deadline))[0], self.max_wait_ms)
                    wait_deadline = time.monotonic() + wait_ms / 1e3
                    with self.cv:
                        self._expire()
                        while key not in self.store:
                            left = wait_deadline - time.monotonic()
                            if left <= 0 or self._stopping.is_set():
                                break
                            self.cv.wait(left)
                        entry = self.store.pop(key, None)
                        if entry is not None:
                            self._bytes_held -= len(entry[0])
                            reserved = len(entry[0])
                            self._bytes_reserved += reserved
                            self._entries_reserved += 1
                            entry_reserved = True
                            self.stats["taken"] += 1
                            self.stats["bytes_out"] += len(entry[0])
                    deadline = time.monotonic() + self.frame_timeout_s
                    if entry is None:
                        _send(c, _HDR_P.pack(-1, 0), deadline)
                    else:
                        _send(c, _HDR_P.pack(len(entry[0]), entry[1]), deadline)
                        _send(c, entry[0], deadline)
                        del entry
                        with self.cv:
                            self._bytes_reserved -= reserved
                            self._entries_reserved -= 1
                            reserved = 0
                            entry_reserved = False
        except (ConnectionError, OSError, ValueError, struct.error):
            return
        finally:
            with self.cv:
                if entry_reserved:
                    self._bytes_reserved -= reserved
                    self._entries_reserved -= 1
                self._clients.discard(c)
            c.close()


class TcpKVClient:
    """Pushes to / takes from one TcpKVServer over a small pool of persistent connections.

    Persistent, because a fresh connection per 1 MB layer restarts TCP slow start every time. Safe to
    share between threads: each call borrows its own connection.
    """

    def __init__(self, host, port, connect_timeout=10.0):
        self.addr = (host, int(port))
        self.connect_timeout = connect_timeout
        self.max_blob_bytes = _setting("MAX_BLOB_BYTES", 256 << 20)
        self.max_key_bytes = _setting("MAX_KEY_BYTES", 4096)
        self.operation_timeout_s = _setting("FRAME_TIMEOUT_S", 60.0, float)
        self.max_wait_ms = min(_setting("MAX_WAIT_MS", 900000), (1 << 31) - 1)
        self._idle = queue.LifoQueue()

    def _conn(self, deadline):
        while True:
            try:
                s = self._idle.get_nowait()
            except queue.Empty:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("tcpkv connection deadline exceeded")
                s = socket.create_connection(self.addr, timeout=min(self.connect_timeout, left))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            # A pooled socket must be quiet: readable means EOF or an unexpected old reply.
            if not select.select([s], [], [], 0)[0]:
                return s
            s.close()

    def _call(self, fn, timeout=None):
        deadline = time.monotonic() + (self.operation_timeout_s if timeout is None else timeout)
        s = self._conn(deadline)
        try:
            out = fn(s, deadline)
        except BaseException:
            s.close()                           # unknown state: never reuse it
            raise
        self._idle.put(s)
        return out

    def push(self, key, data):
        kb = key.encode()
        _length(len(kb), self.max_key_bytes)
        _length(len(data), self.max_blob_bytes)
        crc = zlib.crc32(data) if _crc_on() else 0

        def go(s, deadline):
            _send(s, b"P" + _I32.pack(len(kb)) + kb + _HDR_P.pack(len(data), crc), deadline)
            _send(s, data, deadline)
            ack = bytes(_recv_exact(s, 2, deadline))
            if ack != b"OK":
                raise IOError(f"tcpkv push {key}: server rejected the blob ({ack!r}, crc or capacity)")
        self._call(go)

    def take(self, key, timeout=5.0):
        """The blob pushed under `key` (bytearray), waiting up to `timeout` s for it; TimeoutError if
        it never came. Removes it from the server. Connect/send/receive share a deadline of
        timeout + 0.1 s, including a bounded reply allowance for timeout=0 polling."""
        kb = key.encode()
        _length(len(kb), self.max_key_bytes)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("take timeout must be nonnegative and finite")
        wait_ms = _length(int(timeout * 1e3), self.max_wait_ms)

        def go(s, deadline):
            _send(s, b"W" + _I32.pack(len(kb)) + kb + _I32.pack(wait_ms), deadline)
            n, crc = _HDR_P.unpack(_recv_exact(s, _HDR_P.size, deadline))
            if n == -1:
                return None
            _length(n, self.max_blob_bytes)
            data = _recv_exact(s, n, deadline)
            if crc and _crc_on() and zlib.crc32(data) != crc:
                raise IOError(f"tcpkv take {key}: crc mismatch on delivery")
            return data
        # A small bounded reply allowance also permits take(timeout=0) polling.
        data = self._call(go, timeout=timeout + 0.1)
        if data is None:
            raise TimeoutError(f"pull timeout: {key}")
        return data

    def stats(self):
        def go(s, deadline):
            _send(s, b"S", deadline)
            n = struct.unpack("<q", bytes(_recv_exact(s, 8, deadline)))[0]
            return json.loads(bytes(_recv_exact(s, _length(n, _MAX_STATS_BYTES), deadline)))
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
