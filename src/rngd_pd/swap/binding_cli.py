"""Small JSON transport for an already-created, persistent binding owner.

The caller owns the owner's lifecycle and runs ``server.serve_forever()`` in
its own thread. The CLI imports no native engines and prints only JSON.
"""
import argparse
import json
import os
import socket
import socketserver
import stat
import sys


MAX_MESSAGE_BYTES = 4 * 1024 * 1024
SOCKET_TIMEOUT = 10
RESPONSE_TIMEOUT = 300


def _failure(error):
    return {"ok": False, "error": {
        "type": type(error).__name__, "message": str(error)}}


def _read_message(stream):
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if len(line) > MAX_MESSAGE_BYTES:
        raise ValueError("JSON message exceeds size limit")
    if not line.endswith(b"\n"):
        raise ValueError("expected newline-terminated JSON message")
    return json.loads(line)


def _encode_message(document):
    message = json.dumps(document, allow_nan=False).encode("utf-8") + b"\n"
    if len(message) > MAX_MESSAGE_BYTES:
        raise ValueError("JSON message exceeds size limit")
    return message


def _dispatch(owner, request):
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    operation = request.get("operation")
    counts = {"describe": 1, "resident": 0, "plan": 0, "apply": 1,
              "rollback": 0, "conformance-load": 2}
    if not isinstance(operation, str) or operation not in counts:
        raise ValueError("unsupported operation: {!r}".format(operation))
    args = request.get("args", [])
    if (not isinstance(args, list) or len(args) != counts[operation]
            or not all(isinstance(arg, str) for arg in args)):
        raise ValueError("invalid arguments for " + operation)
    document = request.get("document")
    if operation in ("plan", "apply") and not isinstance(document, dict):
        raise ValueError("document must be a JSON object")
    if operation == "describe":
        return owner.describe(args[0])
    if operation == "resident":
        return owner.resident()
    if operation == "plan":
        return owner.plan(document)
    if operation == "apply":
        return owner.apply(document, args[0])
    if operation == "rollback":
        return owner.rollback()
    return owner.load(args[0], args[1])


class _Handler(socketserver.StreamRequestHandler):
    def setup(self):
        self.request.settimeout(SOCKET_TIMEOUT)
        super().setup()

    def handle(self):
        try:
            request = _read_message(self.rfile)
            response = _encode_message({
                "ok": True, "result": _dispatch(self.server.owner, request)})
        except Exception as error:
            response = _encode_message(_failure(error))
        try:
            self.wfile.write(response)
        except OSError:
            pass  # A timed-out or disconnected client cannot receive its error.


class _OwnerServer(socketserver.TCPServer):
    """Serial requests keep every operation on the same in-process owner."""

    def __init__(self, owner, address):
        self.owner = owner
        self._socket_path = None
        self._socket_identity = None
        if not isinstance(address, tuple):
            address = os.fspath(address)
            if os.path.lexists(address):
                raise FileExistsError(address)
            self.address_family = socket.AF_UNIX
            self._socket_path = address
        super().__init__(address, _Handler)

    def server_bind(self):
        super().server_bind()
        if self._socket_path is not None:
            identity = os.lstat(self._socket_path)
            self._socket_identity = identity.st_dev, identity.st_ino
            try:
                os.chmod(self._socket_path, 0o600)
            except OSError:
                self.server_close()
                raise

    def server_close(self):
        super().server_close()
        if self._socket_path is not None and self._socket_identity is not None:
            try:
                identity = os.lstat(self._socket_path)
                if (stat.S_ISSOCK(identity.st_mode)
                        and (identity.st_dev, identity.st_ino) == self._socket_identity):
                    os.unlink(self._socket_path)
            except FileNotFoundError:
                pass
            self._socket_identity = None


def create_server(owner, address):
    """Return a context-managed server for a Unix path or ``(host, port)``.

    ``server.server_address`` contains the bound address (including an assigned
    ephemeral TCP port). Closing the server never closes the supplied owner.
    """
    return _OwnerServer(owner, address)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _arguments(argv):
    parser = _Parser(description=__doc__)
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument("--socket")
    endpoint.add_argument("--host")
    parser.add_argument("--port", type=int)
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("describe").add_argument("artifact")
    commands.add_parser("resident")
    commands.add_parser("plan")
    commands.add_parser("apply").add_argument("--target", required=True)
    commands.add_parser("rollback")
    load = commands.add_parser("conformance-load")
    load.add_argument("artifact")
    load.add_argument("manifest")
    args = parser.parse_args(argv)
    if args.host is not None and args.port is None:
        raise ValueError("--host requires --port")
    if args.socket is not None and args.port is not None:
        raise ValueError("--port requires --host")
    return args


def main(argv=None):
    try:
        args = _arguments(argv)
        request = {"operation": args.operation, "args": []}
        if args.operation in ("describe", "conformance-load"):
            request["args"].append(args.artifact)
        if args.operation == "conformance-load":
            request["args"].append(args.manifest)
        if args.operation in ("plan", "apply"):
            document = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
            if len(document) > MAX_MESSAGE_BYTES:
                raise ValueError("JSON document exceeds size limit")
            request["document"] = json.loads(document)
        if args.operation == "apply":
            request["args"].append(args.target)
        message = _encode_message(request)
        family = socket.AF_UNIX if args.socket is not None else socket.AF_INET
        address = args.socket if args.socket is not None else (args.host, args.port)
        with socket.socket(family, socket.SOCK_STREAM) as connection:
            connection.settimeout(SOCKET_TIMEOUT)
            connection.connect(address)
            connection.sendall(message)
            connection.settimeout(RESPONSE_TIMEOUT)
            with connection.makefile("rb") as stream:
                response = _read_message(stream)
        if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
            raise ValueError("invalid server response")
        output = response["result"] if response["ok"] else response
        print(json.dumps(output, allow_nan=False))
        return 0 if response["ok"] else 1
    except Exception as error:
        print(json.dumps(_failure(error)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
