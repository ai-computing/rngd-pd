"""Persistent owner transport, exercised through independent CLI processes."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from rngd_pd.swap import binding_cli
from rngd_pd.swap.binding_cli import create_server


class Owner:
    def __init__(self):
        self.artifact = None
        self.previous = None
        self.closed = False

    def describe(self, artifact):
        return {"schema": "test/artifact", "artifact": artifact}

    def load(self, artifact, manifest):
        self.artifact = artifact
        return {"artifact": artifact, "manifest": manifest}

    def resident(self):
        return {"artifact": self.artifact}

    def plan(self, document):
        if document.get("fail"):
            raise ValueError("invalid switch plan")
        return {"schema": "test/plan", "received": document}

    def apply(self, document, target):
        self.previous, self.artifact = self.artifact, target
        return {"artifact": target, "plan": document}

    def rollback(self):
        self.artifact = self.previous
        return self.resident()

    def close(self):
        self.closed = True


@pytest.fixture
def service():
    owner = Owner()
    with create_server(owner, ("127.0.0.1", 0)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield owner, server
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert not owner.closed


def cli(server, *args, document=None):
    host, port = server.server_address
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "rngd_pd.swap.binding_cli", "--host", host,
         "--port", str(port), *args],
        input=json.dumps(document) if document is not None else "",
        text=True, capture_output=True, env=environment, timeout=15,
    )
    # json.loads rejects accidental logging before/after the one JSON response.
    return result.returncode, json.loads(result.stdout), result.stderr


def request(server, document):
    with socket.create_connection(server.server_address, timeout=5) as connection:
        connection.sendall(json.dumps(document).encode() + b"\n")
        return json.loads(connection.makefile("rb").readline())


def test_cli_operations_share_one_live_owner(service):
    owner, server = service
    assert cli(server, "describe", "base")[1] == {
        "schema": "test/artifact", "artifact": "base"}
    assert cli(server, "conformance-load", "base", "sha256:123")[1] == {
        "artifact": "base", "manifest": "sha256:123"}
    assert cli(server, "resident") == (0, {"artifact": "base"}, "")
    document = {"schema": "switch/1", "changes": [1]}
    assert cli(server, "plan", document=document) == (
        0, {"schema": "test/plan", "received": document}, "")
    assert cli(server, "apply", "--target", "next", document=document) == (
        0, {"artifact": "next", "plan": document}, "")
    assert cli(server, "resident")[1] == {"artifact": "next"}
    assert cli(server, "rollback") == (0, {"artifact": "base"}, "")
    assert owner.artifact == "base"


def test_owner_failure_is_structured_and_does_not_kill_server(service):
    _, server = service
    code, response, stderr = cli(server, "plan", document={"fail": True})
    assert code == 1
    assert response == {"ok": False, "error": {
        "type": "ValueError", "message": "invalid switch plan"}}
    assert stderr == ""
    assert cli(server, "resident")[0] == 0


def test_protocol_cannot_invoke_arbitrary_owner_methods(service):
    owner, server = service
    response = request(server, {"operation": "close", "args": []})
    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"
    assert not owner.closed
    assert cli(server, "resident")[0] == 0


@pytest.mark.parametrize("document", [[], {"operation": "resident", "args": ["extra"]},
                                      {"operation": "plan", "document": []}])
def test_protocol_rejects_malformed_requests(service, document):
    _, server = service
    assert request(server, document)["ok"] is False


def test_cli_unknown_operation_is_json_failure(service):
    _, server = service
    code, response, stderr = cli(server, "close")
    assert code != 0
    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"
    assert stderr == ""


def test_cli_invalid_stdin_is_json_failure(service):
    _, server = service
    code, response, stderr = cli(server, "plan")
    assert code != 0
    assert response["ok"] is False
    assert stderr == ""


def test_server_rejects_oversized_wire_message_and_remains_available(service, monkeypatch):
    _, server = service
    monkeypatch.setattr(binding_cli, "MAX_MESSAGE_BYTES", 256)
    with socket.create_connection(server.server_address, timeout=5) as connection:
        connection.sendall(b" " * 257 + b"\n")
        response = json.loads(connection.makefile("rb").readline())
    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"
    assert request(server, {"operation": "resident"}) == {
        "ok": True, "result": {"artifact": None}}


def test_server_times_out_incomplete_requests_and_remains_available(service, monkeypatch):
    _, server = service
    monkeypatch.setattr(binding_cli, "SOCKET_TIMEOUT", 0.05)
    with socket.create_connection(server.server_address, timeout=5) as connection:
        connection.sendall(b'{"operation":')
        response = json.loads(connection.makefile("rb").readline())
    assert response["ok"] is False
    assert response["error"]["type"] == "TimeoutError"
    assert request(server, {"operation": "resident"})["ok"] is True


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX") or os.name == "nt", reason="Unix sockets")
def test_unix_socket_refuses_existing_paths_and_cleans_only_owned_socket(tmp_path):
    existing = tmp_path / "keep"
    existing.write_text("keep me")
    with pytest.raises(FileExistsError):
        create_server(Owner(), str(existing))
    assert existing.read_text() == "keep me"
    path = tmp_path / "owner.sock"
    with create_server(Owner(), str(path)):
        assert path.stat().st_mode & 0o777 == 0o600
    assert not path.exists()
    with create_server(Owner(), str(path)):
        path.unlink()
        path.write_text("replacement")
    assert path.read_text() == "replacement"


def test_cli_imports_without_native_engines_or_torch():
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'furiosa', 'rebel'}:
            raise RuntimeError('native import: ' + fullname)
sys.meta_path.insert(0, Guard())
import rngd_pd.swap.binding_cli
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
