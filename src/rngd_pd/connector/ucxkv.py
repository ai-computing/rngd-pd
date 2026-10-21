"""Cross-node KV through NIXL's UCX backend: host-staged buffers, one-sided READ, descriptors over TCP.

The data path is NIXL with the UCX backend, so the wire is UCX's choice from its environment:

    UCX_TLS=tcp                 today (no RDMA device on the nodes, ai-rm#46)
    UCX_TLS=rc,ud,sm,self       once an RDMA NIC (RoCE v2 / InfiniBand, or soft-RoCE) is present

and switching to RDMA changes no code. Each side owns one pinned arena registered with NIXL once:

    prefill  push(key, buf): copy buf into its arena, publish a descriptor (agent metadata, address,
             length, CRC-32) under `key` in a TcpKVServer (tcpkv.py: the relay or a server beside decode)
    decode   pull(key):      take the descriptor, add the remote agent once, NIXL READ the bytes into
             its own arena (one-sided: RDMA READ on verbs), copy them out, check the CRC; the READ's
             notification tells prefill to free its slot

Only descriptors (a few KB) cross the TCP store; the KV itself moves over UCX. A slot nobody reads
within AIRM_KV_TCP_TTL_S (900 s) is freed; a READ after that fails its CRC instead of returning
stale bytes. Nothing falls back to another path: no NIXL, no UCX, or a failed READ raises.

Environment: AIRM_KV_UCX_POOL_MB (arena per side, 1024), AIRM_KV_UCX_XFER_TIMEOUT_S (60),
AIRM_KV_TCP_CRC (1), AIRM_KV_TCP_ADDR (the descriptor store), UCX_* (UCX's own, e.g. UCX_TLS,
UCX_NET_DEVICES; UCX_TCP_TX/RX_SEG_SIZE default to 64 KiB here).
"""
import json
import os
import socket
import struct
import threading
import time
import uuid
import zlib

BACKEND = "UCX"
MAGIC = b"AKU1"
_HDR = struct.Struct("<4sQQIii")          # magic, address, n, crc, name len, metadata len
FREE_TAG = b"airm-free:"


def _crc_on():
    return os.environ.get("AIRM_KV_TCP_CRC", "1") not in ("0", "false", "no")


# UCX's tcp transport moves 8 KiB segments by default; 64 KiB took a 1 MiB READ from 0.26 to 0.44 GB/s
# (two processes, one pod, UCX_TLS=tcp). Verbs transports ignore these. The environment wins.
UCX_DEFAULTS = {"UCX_TCP_TX_SEG_SIZE": "65536", "UCX_TCP_RX_SEG_SIZE": "65536"}


def _default_agent(name):
    """A NIXL agent with the UCX backend; UCX reads its UCX_* environment here."""
    for k, v in UCX_DEFAULTS.items():
        os.environ.setdefault(k, v)
    from nixl._api import nixl_agent, nixl_agent_config
    agent = nixl_agent(name, nixl_agent_config(backends=[]))
    agent.create_backend(BACKEND, {})
    return agent


class Arena:
    """One page-aligned host buffer and a first-fit allocator over it (offsets, 64-byte aligned).

    alloc() waits up to `timeout` s for room; free() coalesces. Thread-safe.
    """

    ALIGN = 64

    def __init__(self, nbytes):
        import numpy as np
        self.size = int(nbytes)
        self._raw = np.empty(self.size + 4096, np.uint8)
        base = self._raw.ctypes.data
        start = (-base) % 4096
        self.buf = self._raw[start:start + self.size]
        self.address = base + start
        self._free = [(0, self.size)]           # sorted (offset, length)
        self._cv = threading.Condition()

    def alloc(self, n, timeout=0.0):
        n = max(self.ALIGN, (int(n) + self.ALIGN - 1) // self.ALIGN * self.ALIGN)
        if n > self.size:
            raise ValueError(f"ucxkv: {n} bytes do not fit an arena of {self.size} (AIRM_KV_UCX_POOL_MB)")
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cv:
            while True:
                for i, (off, length) in enumerate(self._free):
                    if length >= n:
                        if length == n:
                            self._free.pop(i)
                        else:
                            self._free[i] = (off + n, length - n)
                        return off, n
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"ucxkv: no {n} free bytes in the arena within {timeout:.1f} s")
                self._cv.wait(left)

    def free(self, off, n):
        with self._cv:
            self._free.append((off, n))
            self._free.sort()
            merged = []
            for o, l in self._free:
                if merged and merged[-1][0] + merged[-1][1] == o:
                    merged[-1] = (merged[-1][0], merged[-1][1] + l)
                else:
                    merged.append((o, l))
            self._free = merged
            self._cv.notify_all()

    def free_bytes(self):
        with self._cv:
            return sum(l for _o, l in self._free)


def pack_desc(name, metadata, address, n, crc):
    nb = name.encode()
    return _HDR.pack(MAGIC, address, n, crc, len(nb), len(metadata)) + nb + metadata


def unpack_desc(blob):
    blob = bytes(blob)
    magic, address, n, crc, nl, ml = _HDR.unpack_from(blob)
    if magic != MAGIC or len(blob) != _HDR.size + nl + ml:
        raise IOError("ucxkv: malformed descriptor")
    name = blob[_HDR.size:_HDR.size + nl].decode()
    return name, blob[_HDR.size + nl:], address, n, crc


class UcxKVStore:
    """push/pull of KV blobs between processes on any hosts, data over NIXL UCX.

    control: a TcpKVClient (or anything with push(key, bytes) / take(key, timeout)) for descriptors.
    agent:   a nixl_agent with UCX created (tests pass a fake); made on first use otherwise.
    One instance serves both roles; each owns one arena per role, registered once.
    """

    def __init__(self, control, agent=None, pool_bytes=None, name=None, xfer_timeout_s=None,
                 ttl_s=None, poll_s=0.0005):
        self.control = control
        self.name = name or getattr(agent, "name", None) or f"airm-kv-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.pool_bytes = int(pool_bytes or int(os.environ.get("AIRM_KV_UCX_POOL_MB", "1024")) << 20)
        self.xfer_timeout_s = float(xfer_timeout_s or os.environ.get("AIRM_KV_UCX_XFER_TIMEOUT_S", "60"))
        self.ttl_s = float(ttl_s if ttl_s is not None else os.environ.get("AIRM_KV_TCP_TTL_S", "900"))
        self.poll_s = poll_s
        self._agent = agent
        self._lock = threading.RLock()            # NIXL calls and the tables below
        self._tx = self._rx = None                # Arena, registered
        self._regs = []
        self._metadata = None
        self._remotes = {}                        # agent name -> metadata loaded
        self._slots = {}                          # key -> (offset, n, t_published)
        self._stop = threading.Event()
        self._reaper = None
        self.stats = {"pushed": 0, "pulled": 0, "freed": 0, "expired": 0, "bytes_out": 0, "bytes_in": 0}

    # -- setup ------------------------------------------------------------------
    def _get_agent(self):
        if self._agent is None:
            try:
                self._agent = _default_agent(self.name)
            except Exception as e:
                raise RuntimeError(f"ucxkv: no NIXL agent with the {BACKEND} backend ({type(e).__name__}: {e}); "
                                   "install nixl (the wheel ships the UCX plugin)") from e
        return self._agent

    def _arena(self, side):
        """The tx (prefill) or rx (decode) arena, allocated and registered on first use."""
        with self._lock:
            a = self._tx if side == "tx" else self._rx
            if a is not None:
                return a
            agent = self._get_agent()
            a = Arena(self.pool_bytes)
            reg = agent.register_memory([(a.address, a.size, 0, "")], "DRAM", [BACKEND])
            self._regs.append(reg)
            if side == "tx":
                self._tx = a
                self._metadata = agent.get_agent_metadata()   # carries the tx registration
                self._reaper = threading.Thread(target=self._reap, daemon=True, name="ucxkv-reaper")
                self._reaper.start()
            else:
                self._rx = a
            return a

    # -- prefill side -------------------------------------------------------------
    def push(self, key, buf):
        """Stage `buf` in the registered arena and publish its descriptor under `key`."""
        mv = memoryview(buf).cast("B")
        n = mv.nbytes
        arena = self._arena("tx")
        off, span = arena.alloc(n, timeout=self.xfer_timeout_s)
        try:
            arena.buf[off:off + n] = mv
            crc = zlib.crc32(mv) if _crc_on() else 0
            with self._lock:
                self._slots[key] = (off, span, time.monotonic())
            self.control.push(key, pack_desc(self.name, self._metadata, arena.address + off, n, crc))
        except BaseException:
            with self._lock:
                self._slots.pop(key, None)
            arena.free(off, span)
            raise
        self.stats["pushed"] += 1
        self.stats["bytes_out"] += n

    def _release(self, key, expired=False):
        with self._lock:
            slot = self._slots.pop(key, None)
        if slot is not None:
            self._tx.free(slot[0], slot[1])
            self.stats["expired" if expired else "freed"] += 1

    def _reap(self):
        """Free a slot once its reader's READ completed (NIXL notification), or after the TTL."""
        agent = self._get_agent()
        while not self._stop.is_set():
            try:
                with self._lock:
                    notifs = agent.get_new_notifs()
            except Exception:
                notifs = {}
            got = False
            for msgs in notifs.values():
                for m in msgs:
                    m = bytes(m)
                    if m.startswith(FREE_TAG):
                        self._release(m[len(FREE_TAG):].decode())
                        got = True
            now = time.monotonic()
            with self._lock:
                old = [k for k, (_o, _n, t) in self._slots.items() if now - t > self.ttl_s]
            for k in old:
                self._release(k, expired=True)
            if not got:
                self._stop.wait(0.002)

    def pending(self):
        """Keys pushed here and not yet read (their slots still held)."""
        with self._lock:
            return sorted(self._slots)

    # -- decode side ----------------------------------------------------------------
    def pull(self, key, timeout=5.0):
        """The bytes pushed under `key` (bytearray), read over UCX; TimeoutError if never published."""
        name, md, address, n, crc = unpack_desc(self.control.take(key, timeout=timeout))
        arena = self._arena("rx")
        agent = self._get_agent()
        off, span = arena.alloc(n, timeout=self.xfer_timeout_s)
        try:
            with self._lock:
                if name not in self._remotes and name != self.name:   # own agent: UCX loopback
                    self._remotes[name] = agent.add_remote_agent(md)
                local = agent.get_xfer_descs([(arena.address + off, n, 0)], "DRAM")
                remote = agent.get_xfer_descs([(address, n, 0)], "DRAM")
                h = agent.initialize_xfer("READ", local, remote, name, FREE_TAG + key.encode(), [BACKEND])
                state = agent.transfer(h)
            deadline = time.monotonic() + self.xfer_timeout_s
            while state == "PROC" and time.monotonic() < deadline:
                if self.poll_s:
                    time.sleep(self.poll_s)
                with self._lock:
                    state = agent.check_xfer_state(h)
            with self._lock:
                h.release()
            if state != "DONE":
                why = "timed out" if state == "PROC" else f"state {state}"
                raise IOError(f"ucxkv pull {key}: NIXL READ from {name} {why}")
            data = bytearray(arena.buf[off:off + n])
        finally:
            arena.free(off, span)
        if crc and _crc_on() and zlib.crc32(data) != crc:
            raise IOError(f"ucxkv pull {key}: crc mismatch (slot freed or overwritten at the source)")
        self.stats["pulled"] += 1
        self.stats["bytes_in"] += n
        return data

    # -- lifecycle --------------------------------------------------------------------
    def describe(self):
        """What a handshake or a log line needs: agent name, backend and the UCX transports asked for."""
        return {"agent": self.name, "backend": BACKEND, "ucx_tls": os.environ.get("UCX_TLS", "(UCX default)"),
                "ucx_net_devices": os.environ.get("UCX_NET_DEVICES", "(UCX default)")}

    def close(self):
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join(1.0)
        agent = self._agent
        if agent is None:
            return
        with self._lock:
            for name in list(self._remotes):
                try:
                    agent.remove_remote_agent(name)
                except Exception:
                    pass
            self._remotes.clear()
            for reg in self._regs:
                try:
                    agent.deregister_memory(reg, [BACKEND])
                except Exception:
                    pass
            self._regs.clear()
            self._slots.clear()


def main(argv=None):
    """Print what this host's NIXL/UCX would use: python -m rngd_pd.connector.ucxkv"""
    s = UcxKVStore(control=None, pool_bytes=1 << 20)
    try:
        s._get_agent()
        print(json.dumps(dict(s.describe(), ok=True)))
    except RuntimeError as e:
        print(json.dumps(dict(s.describe(), ok=False, error=str(e))))
    finally:
        s.close()


if __name__ == "__main__":
    main()
