"""Host transport trust-boundary regressions; all TCP peers use real loopback sockets."""
from contextlib import contextmanager
import io
import pickle
import socket
import struct
import sys
import threading
import time

import numpy as np
import pytest

from rngd_pd.connector import tcpkv
from rngd_pd.connector import transport as kv


def test_pickle_payload_never_executes(tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (eval, (f"open({str(marker)!r}, 'w').write('unsafe')",))

    try:
        kv.deser(pickle.dumps(Payload()))
    except Exception:
        pass
    assert not marker.exists(), "deserializing untrusted KV executed Python"


@pytest.mark.parametrize("wire, expected", [("fp16", [1.5, -2, 0.0]), ("bf16", [1.5, -2, 2**-100])])
def test_numeric_wire_roundtrip(wire, expected):
    arr = np.array(expected, np.float32).reshape(1, 3)
    blob = kv.ser(arr, wire=wire)
    # A reader can inspect the array without executing pickle or needing connector classes.
    numeric = np.load(io.BytesIO(blob), allow_pickle=False)
    assert numeric.dtype == np.dtype("float16" if wire == "fp16" else "uint16")
    np.testing.assert_array_equal(kv.deser(blob), arr)


def test_wire_rejects_non_numeric_inputs():
    with pytest.raises(ValueError):
        kv.ser(np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        kv.ser(np.ones(2), wire="unknown")


def test_wire_producer_rejects_oversized_views_and_dimensions(monkeypatch):
    monkeypatch.setenv("AIRM_KV_TCP_MAX_BLOB_BYTES", "8192")
    oversized = np.broadcast_to(np.float32(1), (4096,))
    with pytest.raises(ValueError):
        kv.ser(oversized)
    with pytest.raises(ValueError):
        kv.ser(np.zeros((1,) * 9, np.float32))


@pytest.mark.parametrize("dtype, shape", [(object, (1,)), (np.float32, (1,)), (np.float16, (2**40,))])
def test_wire_rejects_unsupported_or_oversized_headers(dtype, shape):
    out = io.BytesIO()
    np.lib.format.write_array_header_1_0(out, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                             "fortran_order": False, "shape": shape})
    with pytest.raises(ValueError):
        kv.deser(out.getvalue())


def test_wire_rejects_truncated_and_trailing_data():
    blob = kv.ser(np.ones((2, 3), np.float32))
    for malformed in (blob[:-1], blob + b"extra"):
        with pytest.raises(ValueError):
            kv.deser(malformed)


def test_cached_shm_fallback_keeps_configured_root(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "rm_core", None)
    monkeypatch.setenv("PD_SHM_ROOT", str(tmp_path))
    monkeypatch.setattr(kv, "_CLASSES", {})
    a, b = kv.select_transport("shm"), kv.select_transport("shm")
    a.push("same-root", b"kv")
    assert b.root == str(tmp_path)
    assert b.pull("same-root", timeout=0.01) == b"kv"


def test_shm_pull_rejects_oversized_files_and_preserves_opaque_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRM_KV_TCP_MAX_BLOB_BYTES", "128")
    tr = kv.ShmTransport(root=str(tmp_path))
    opaque = b"\x00\xff" * 64
    tr.push("opaque", opaque)
    assert tr.pull("opaque") == opaque
    with open(tr._path("oversized"), "wb") as stream:
        stream.truncate(1 << 20)
    with pytest.raises(ValueError):
        tr.pull("oversized")


@contextmanager
def relay(monkeypatch, max_entries=None, max_connections=None, **limits):
    for name, value in limits.items():
        monkeypatch.setenv("AIRM_KV_TCP_" + name, str(value))
    options = {} if max_entries is None else {"max_entries": max_entries}
    if max_connections is not None:
        options["max_connections"] = max_connections
    srv = tcpkv.TcpKVServer(0, "127.0.0.1", **options)
    srv.start()
    port = srv.wait_ready()
    client = tcpkv.TcpKVClient("127.0.0.1", port)
    try:
        yield srv, client, port
    finally:
        client.close()
        srv.stop()


def push_header(n, key=b"k"):
    return b"P" + struct.pack("<i", len(key)) + key + struct.pack("<qI", n, 0)


@pytest.mark.parametrize("header", [b"P" + struct.pack("<i", 4097), push_header(4097)])
def test_relay_rejects_lengths_before_waiting_for_body(monkeypatch, header):
    with relay(monkeypatch, MAX_BLOB_BYTES=4096, MAX_KEY_BYTES=4096) as (_, _, port):
        with socket.create_connection(("127.0.0.1", port), timeout=0.4) as peer:
            peer.sendall(header)
            assert peer.recv(2) in (b"", b"ER")


def test_relay_partial_frame_has_absolute_deadline(monkeypatch):
    with relay(monkeypatch, FRAME_TIMEOUT_S=0.06) as (_, client, port):
        with socket.create_connection(("127.0.0.1", port), timeout=0.4) as peer:
            peer.sendall(push_header(12) + b"x")
            assert peer.recv(2) == b""
        assert client.stats()["bytes_held"] == 0


def test_client_reconnects_after_server_idle_deadline(monkeypatch):
    with relay(monkeypatch, FRAME_TIMEOUT_S=0.05) as (_, client, _):
        client.stats()
        time.sleep(0.1)
        client.push("after-idle", b"kv")
        assert bytes(client.take("after-idle")) == b"kv"


def test_idle_peers_hit_connection_cap_and_disconnect_releases_it(monkeypatch):
    with relay(monkeypatch, max_connections=1) as (_, client, port):
        client.stats()  # Leave the one admitted connection idle in its pool.
        with socket.create_connection(("127.0.0.1", port), timeout=0.3) as overflow:
            assert overflow.recv(1) == b"", "idle peers bypassed the connection cap"
        client.close()
        deadline = time.monotonic() + 0.4
        while True:
            try:
                assert client.stats()["keys"] == 0
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.002)


def test_stop_closes_blocked_receivers_and_releases_reservations(monkeypatch):
    with relay(monkeypatch, max_connections=2) as (srv, client, port):
        with socket.create_connection(("127.0.0.1", port), timeout=0.3) as blocked:
            blocked.sendall(push_header(8) + b"x")
            deadline = time.monotonic() + 0.3
            while client.stats()["bytes_reserved"] != 8 and time.monotonic() < deadline:
                time.sleep(0.002)
            srv.stop()
            assert blocked.recv(1) == b""
            deadline = time.monotonic() + 0.3
            while srv._bytes_reserved and time.monotonic() < deadline:
                time.sleep(0.002)
            assert srv._bytes_reserved == 0


def test_relay_store_limit_counts_inflight_and_releases_timeout(monkeypatch):
    with relay(monkeypatch, max_entries=1, MAX_STORE_BYTES=8, FRAME_TIMEOUT_S=0.12) as (_, client, port):
        with socket.create_connection(("127.0.0.1", port), timeout=0.4) as peer:
            peer.sendall(push_header(8))
            deadline = time.monotonic() + 0.2
            while client.stats().get("bytes_reserved", 0) != 8 and time.monotonic() < deadline:
                time.sleep(0.002)
            with pytest.raises(OSError):
                client.push("other", b"x")
            assert peer.recv(2) == b""
        client.push("freed", b"12345678")
        assert bytes(client.take("freed")) == b"12345678"


def test_relay_store_limit_and_replacement_accounting(monkeypatch):
    with relay(monkeypatch, MAX_STORE_BYTES=12) as (_, client, _):
        client.push("a", b"1234")
        client.push("a", b"12")
        client.push("b", b"3456789")
        with pytest.raises(OSError):
            client.push("overflow", b"xxxx")
        assert client.stats()["bytes_held"] == 9
        assert bytes(client.take("a")) == b"12"
        client.push("fits", b"abcde")


def test_zero_length_blobs_cannot_bypass_entry_limit(monkeypatch):
    with relay(monkeypatch, max_entries=2) as (_, client, _):
        client.push("a", b"")
        client.push("a", b"")  # replacing a key must release its temporary admission slot
        client.push("b", b"")
        with pytest.raises(OSError):
            client.push("overflow", b"")
        assert bytes(client.take("a")) == b""
        client.push("freed", b"")
        assert client.stats()["keys"] == 2


def test_relay_budget_remains_reserved_until_delivery_finishes(monkeypatch):
    size = 8 << 20
    started, release = threading.Event(), threading.Event()
    send = tcpkv._send

    def paused_send(sock, data, deadline):
        if len(data) == size:
            started.set()
            assert release.wait(1)
        return send(sock, data, deadline)

    with relay(monkeypatch, MAX_STORE_BYTES=size, FRAME_TIMEOUT_S=2) as (_, client, port):
        client.push("large", b"x" * size)
        # Windows loopback buffers entire sends; pause delivery to expose the ownership window.
        monkeypatch.setattr(tcpkv, "_send", paused_send)
        with socket.create_connection(("127.0.0.1", port), timeout=0.4) as peer:
            peer.sendall(b"W" + struct.pack("<i", 5) + b"large" + struct.pack("<i", 0))
            try:
                assert started.wait(0.4)
                with pytest.raises(OSError):
                    client.push("overflow", b"x")
            finally:
                release.set()


@pytest.mark.parametrize("response", [b"", struct.pack("<qI", 16, 0) + b"partial"])
def test_take_deadline_includes_silent_and_partial_response(response):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        client = tcpkv.TcpKVClient("127.0.0.1", listener.getsockname()[1], connect_timeout=0.1)
        errors = []

        def take():
            try:
                client.take("k", timeout=0.02)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=take, daemon=True)
        t.start()
        with listener.accept()[0] as peer:
            peer.recv(100)
            if response:
                peer.sendall(response)
            t.join(0.4)
            finished = not t.is_alive()
        t.join(0.4)
        client.close()
        assert finished, "take(timeout) blocked on the peer beyond its deadline"
        assert isinstance(errors[0], TimeoutError)
