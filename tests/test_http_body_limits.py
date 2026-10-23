"""Partial and oversized POST frames cannot hold finite admission indefinitely."""
from contextlib import contextmanager
import http.client
import socket
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import pytest

from rngd_pd.serving import cluster


@contextmanager
def server():
    with patch.dict(cluster.STATE, ready=True):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), cluster.make_handler({"/test": lambda b: {}}))
        srv.daemon_threads = True
        t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
        try:
            yield srv.server_port
        finally:
            srv.shutdown(); srv.server_close(); t.join(2)


def response(port, body):
    with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
        sock.sendall(b"POST /test HTTP/1.1\r\nHost: local\r\nConnection: close\r\n" + body)
        with http.client.HTTPResponse(sock) as r:
            r.begin()
            data = r.read()
            return r.status, data


@pytest.mark.parametrize("header", [b"Content-Length: 10\r\n\r\n{",
                                     b"Transfer-Encoding: chunked\r\n\r\na\r\n{",
                                     b"Transfer-Encoding: chunked\r\n\r\na"])
def test_partial_body_expires_and_releases_admission(monkeypatch, header):
    monkeypatch.setenv("PD_MAX_REQUESTS", "1")
    monkeypatch.setenv("PD_HTTP_BODY_TIMEOUT", ".05")
    with server() as port:
        assert response(port, header)[0] == 408
        assert response(port, b"Content-Length: 2\r\n\r\n{}")[0] == 200


@pytest.mark.parametrize("frame", [b"Content-Length: 1000\r\n\r\n",
                                    b"Transfer-Encoding: chunked\r\n\r\n400\r\n",
                                    b"Transfer-Encoding: chunked\r\n\r\n5\r\n12345\r\n5\r\n"])
def test_oversized_body_is_rejected_before_reading_declared_bytes(monkeypatch, frame):
    monkeypatch.setenv("PD_HTTP_MAX_BODY_BYTES", "8")
    with server() as port:
        assert response(port, frame)[0] == 413


@pytest.mark.parametrize("header", [b"Content-Length: -1\r\n\r\n",
                                     b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
                                     b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n\r\n{}"])
def test_ambiguous_or_negative_lengths_are_rejected(header):
    with server() as port:
        assert response(port, header)[0] == 400


@pytest.mark.parametrize("frame", [b"Content-Length: 2\r\n\r\n{}",
                                    b"Transfer-Encoding: chunked\r\n\r\n1\r\n{\r\n1\r\n}\r\n0\r\n\r\n",
                                    b"Transfer-Encoding: chunked\r\n\r\n2;extension=yes\r\n{}\r\n0\r\nX-Trace: ok\r\n\r\n"])
def test_valid_fixed_and_chunked_requests_keep_working(frame):
    with server() as port:
        assert response(port, frame) == (200, b"{}")


def test_body_deadline_is_absolute_even_when_bytes_keep_arriving(monkeypatch):
    monkeypatch.setenv("PD_HTTP_BODY_TIMEOUT", ".08")
    with server() as port, socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
        sock.sendall(b"POST /test HTTP/1.1\r\nHost: local\r\nConnection: close\r\n"
                     b"Content-Length: 1000\r\n\r\n")
        stop = threading.Event()

        def dribble():
            while not stop.wait(.01):
                try:
                    sock.sendall(b"x")
                except OSError:
                    return

        sender = threading.Thread(target=dribble, daemon=True); sender.start()
        start = time.monotonic()
        try:
            r = http.client.HTTPResponse(sock); r.begin(); r.read()
            assert r.status == 408
            assert time.monotonic() - start < .8
        finally:
            stop.set(); sender.join(1)
