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

Only descriptors (a few KB) cross the TCP store; the KV itself moves over UCX. An advertised TX
slot is held until its READ completion notification, even after the descriptor store's TTL or a
lost publication ACK. Wall time cannot prove a remote READ stopped. Abandoned slots consume the
finite arena and prevent graceful close; recovery requires process teardown until an SDK-proven
drain/cancel path exists. Nothing falls back: no NIXL, no UCX, or a failed READ raises.

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
        # ttl_s remains accepted for compatibility; it cannot reclaim advertised TX memory.
        self.poll_s = poll_s
        self._agent = agent
        self._lock = threading.RLock()            # NIXL calls and the tables below
        self._tx = self._rx = None                # Arena, registered
        self._regs = []
        self._metadata = None
        self._remotes = {}                        # agent name -> metadata loaded
        self._slots = {}                          # key -> (offset, n, t_published)
        self._publishing = set()                  # includes pushes waiting for arena space
        self._quarantined = []                    # (handle, arena, offset, span), still DMA-owned
        self._active_reads = 0
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
            if self._stop.is_set():
                raise RuntimeError("ucxkv: store is closed")
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
            else:
                self._rx = a
            if self._reaper is None:
                self._reaper = threading.Thread(target=self._reap, daemon=True, name="ucxkv-reaper")
                self._reaper.start()
            return a

    # -- prefill side -------------------------------------------------------------
    def push(self, key, buf):
        """Stage `buf` in the registered arena and publish its descriptor under `key`."""
        mv = memoryview(buf).cast("B")
        n = mv.nbytes
        with self._lock:
            if key in self._slots or key in self._publishing:
                raise ValueError(f"ucxkv: outstanding key {key!r}")
            self._publishing.add(key)
        off, advertised = None, False
        try:
            arena = self._arena("tx")
            off, span = arena.alloc(n, timeout=self.xfer_timeout_s)
            arena.buf[off:off + n] = mv
            crc = zlib.crc32(mv) if _crc_on() else 0
            desc = pack_desc(self.name, self._metadata, arena.address + off, n, crc)
            with self._lock:
                self._slots[key] = (off, span, time.monotonic())
                advertised = True
            self.control.push(key, desc)
        except BaseException:
            # Once push was attempted, an exception may mean only its acknowledgement was lost.
            if off is not None and not advertised:
                arena.free(off, span)
            raise
        finally:
            with self._lock:
                self._publishing.discard(key)
        self.stats["pushed"] += 1
        self.stats["bytes_out"] += n

    def _release(self, key):
        with self._lock:
            slot = self._slots.pop(key, None)
        if slot is not None:
            self._tx.free(slot[0], slot[1])
            self.stats["freed"] += 1

    def _reap(self):
        """Free TX only on READ completion notifications, and poll quarantined RX handles."""
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._drain_quarantined()
                    got = self._drain_notifications()
            except Exception:
                got = False
            if not got:
                self._stop.wait(0.002)

    def _drain_notifications(self):
        """Called under _lock, including during close so confirmed READs can release TX."""
        got = False
        for msgs in self._get_agent().get_new_notifs().values():
            for m in msgs:
                m = bytes(m)
                if m.startswith(FREE_TAG):
                    self._release(m[len(FREE_TAG):].decode())
                    got = True
        return got

    def _drain_quarantined(self):
        """Called under _lock: only DONE proves every READ has completed on the SDK boundary."""
        for entry in list(self._quarantined):
            h, arena, off, span = entry
            try:
                if self._agent.check_xfer_state(h) != "DONE":
                    continue
                h.release()
            except Exception:
                continue
            arena.free(off, span)
            self._quarantined.remove(entry)

    def pending(self):
        """Possibly advertised keys without a confirmed READ completion (their slots still held)."""
        with self._lock:
            return sorted(self._slots)

    # -- decode side ----------------------------------------------------------------
    def pull(self, key, timeout=5.0):
        """The bytes pushed under `key` (bytearray), read over UCX; TimeoutError if never published."""
        name, md, address, n, crc = unpack_desc(self.control.take(key, timeout=timeout))
        arena = self._arena("rx")
        agent = self._get_agent()
        off, span = arena.alloc(n, timeout=self.xfer_timeout_s)
        h, state, active = None, None, False
        try:
            with self._lock:
                if self._stop.is_set():
                    raise RuntimeError("ucxkv: store is closed")
                self._active_reads += 1
                active = True
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
            if state != "DONE":
                why = "timed out" if state == "PROC" else f"state {state}"
                raise IOError(f"ucxkv pull {key}: NIXL READ from {name} {why}")
            data = bytearray(arena.buf[off:off + n])
        finally:
            with self._lock:
                if active:
                    self._active_reads -= 1
                if h is None:
                    arena.free(off, span)
                elif state == "DONE":
                    try:
                        h.release()
                    except BaseException:
                        self._quarantined.append((h, arena, off, span))
                        raise
                    arena.free(off, span)
                else:
                    # Neither ERR nor release() proves outstanding DMA has drained on every SDK.
                    self._quarantined.append((h, arena, off, span))
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
        with self._lock:
            if self._active_reads or self._publishing:
                raise RuntimeError("ucxkv: cannot close with active in-flight operations")
            self._drain_quarantined()
            if self._quarantined:
                # Leave the reaper and its owning references alive until DMA completes.
                raise RuntimeError("ucxkv: cannot close with quarantined in-flight READs; retry after completion")
            if self._slots:
                try:
                    self._drain_notifications()
                except Exception:
                    pass
            if self._slots:
                raise RuntimeError("ucxkv: cannot close with outstanding advertised TX slots; "
                                   "READ completion notifications are required")
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
