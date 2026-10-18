# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Local controller for the AI Model Inspector API.

The measurement API can be stopped. This process stays up so the console can
start it again. It binds to 127.0.0.1, requires a local credential, and can
only ask inspector-server.sh to status, start, or stop an Inspector process.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "inspector-server.sh"
TOKEN_PATH = ROOT / ".inspector-runtime" / "controller-token"
CONTROLLER_PORT = 8811
RESERVED_PORTS = {8787, 8799, CONTROLLER_PORT}
SIMULATOR_BUNDLE = "com.lccresearchlab.aimodelinspector"
MAX_BODY = 4096
Runner = Callable[[str, int], dict]


class ControllerError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


def load_or_create_token(path: Path = TOKEN_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            os.chmod(path, 0o600)
            return token
    token = _new_token()
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    os.chmod(path, 0o600)
    return token


def _new_token() -> str:
    import secrets

    return secrets.token_urlsafe(32)


def install_simulator_credential(token: str) -> None:
    """Copy the credential into the booted simulator app, if one is present."""
    try:
        completed = subprocess.run(
            ["xcrun", "simctl", "get_app_container", "booted", SIMULATOR_BUNDLE, "data"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if completed.returncode != 0:
        return
    container = Path(completed.stdout.strip())
    try:
        resolved = container.resolve()
    except OSError:
        return
    if "CoreSimulator" not in resolved.parts:
        return
    documents = resolved / "Documents"
    if not documents.is_dir():
        return
    destination = documents / "inspector-controller-token"
    destination.write_text(token + "\n", encoding="utf-8")
    os.chmod(destination, 0o600)


def run_server_script(command: str, port: int) -> dict:
    if command not in {"status", "start", "stop"}:
        raise ControllerError(400, "rejected", "That action is not available.")
    if port in RESERVED_PORTS:
        raise ControllerError(400, "reserved_port", f"Port {port} is reserved. Nothing was changed.")
    try:
        completed = subprocess.run(
            [str(SCRIPT), command, "--port", str(port), "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180 if command == "start" else 30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ControllerError(504, "timeout", "The server manager did not finish.") from exc
    except OSError as exc:
        raise ControllerError(500, "manager_missing", "The server manager could not be started.") from exc
    payload = _parse_script_json(completed.stdout)
    payload["code"] = completed.returncode
    return payload


def _parse_script_json(stdout: str) -> dict:
    line = ""
    for candidate in reversed(stdout.splitlines()):
        if candidate.strip().startswith("{"):
            line = candidate.strip()
            break
    if not line:
        raise ControllerError(500, "manager_result", "The server manager returned an unexpected result.")
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ControllerError(500, "manager_result", "The server manager returned an unexpected result.") from exc
    state = data.get("state")
    if state not in {"running", "stopped", "foreign", "unresponsive", "error"}:
        raise ControllerError(500, "manager_result", "The server manager returned an unexpected result.")
    pid = data.get("pid")
    if pid is not None and not isinstance(pid, int):
        pid = None
    message = data.get("message")
    if not isinstance(message, str) or not message:
        message = "The server manager finished."
    return {"state": state, "port": data.get("port"), "pid": pid, "message": message}


def local_browser_origin(origin: str) -> bool:
    """A browser on this computer may call the same controller API. Other sites may not."""
    if not origin or len(origin) > 200:
        return False
    parsed = urlparse(origin)
    if parsed.scheme != "http" or parsed.username or parsed.password:
        return False
    return parsed.hostname in {"127.0.0.1", "localhost"}


def _authorized(header: str, token: str) -> bool:
    if not header.startswith("Bearer ") or len(header) > 300:
        return False
    given = header[7:].strip()
    if not given or len(given) > 200:
        return False
    return hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8"))


def _parse_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        else:
            raise ControllerError(400, "bad_port", "Port must be an integer from 1 to 65535.")
    port = int(value)
    if port < 1 or port > 65535:
        raise ControllerError(400, "bad_port", "Port must be an integer from 1 to 65535.")
    if port in RESERVED_PORTS:
        raise ControllerError(400, "reserved_port", f"Port {port} is reserved. Nothing was changed.")
    return port


class Controller:
    def __init__(self, token: str, runner: Runner = run_server_script):
        self.token = token
        self.runner = runner
        self._transition = threading.Lock()

    def handle(self, method: str, path: str, query: str, headers: dict[str, str], body: bytes) -> tuple[int, dict]:
        if path == "/health" and method == "GET":
            return 200, {"ok": True, "service": "ai-model-inspector-controller"}
        if path not in {"/v1/status", "/v1/start", "/v1/stop"}:
            return 404, {"ok": False, "error": "not_found", "message": "That controller action does not exist."}
        if not _authorized(headers.get("authorization", ""), self.token):
            return 401, {
                "ok": False,
                "error": "unauthorized",
                "message": "The controller credential was rejected.",
            }
        try:
            if path == "/v1/status" and method == "GET":
                port = _port_from_query(query)
                result = self.runner("status", port)
                return _http_status(result), _public_result(result)
            if path == "/v1/start" and method == "POST":
                port = _port_from_body(body)
                return self._transition_call("start", port)
            if path == "/v1/stop" and method == "POST":
                port = _port_from_body(body)
                return self._transition_call("stop", port)
        except ControllerError as exc:
            return exc.status, {"ok": False, "error": exc.code, "message": exc.message}
        return 405, {"ok": False, "error": "method", "message": "That method is not available."}

    def _transition_call(self, command: str, port: int) -> tuple[int, dict]:
        if not self._transition.acquire(blocking=False):
            return 409, {
                "ok": False,
                "error": "busy",
                "message": "The controller is already starting or stopping the API.",
            }
        try:
            result = self.runner(command, port)
        finally:
            self._transition.release()
        return _http_status(result), _public_result(result)


def _port_from_query(query: str) -> int:
    from urllib.parse import parse_qs

    found = parse_qs(query, keep_blank_values=False).get("port", [])
    if len(found) != 1:
        raise ControllerError(400, "bad_port", "Provide one port.")
    return _parse_port(found[0])


def _port_from_body(body: bytes) -> int:
    if len(body) > MAX_BODY:
        raise ControllerError(413, "body", "The request is too large.")
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError(400, "body", "The request body must be JSON.") from exc
    if not isinstance(data, dict) or "port" not in data:
        raise ControllerError(400, "bad_port", "Provide one port.")
    return _parse_port(data.get("port"))


def _http_status(result: dict) -> int:
    state = result.get("state")
    if state in {"running", "stopped", "unresponsive"}:
        return 200
    if state == "foreign":
        return 409
    return 500


def _public_result(result: dict) -> dict:
    return {
        "ok": result.get("state") in {"running", "stopped", "unresponsive"},
        "state": result.get("state"),
        "port": result.get("port"),
        "pid": result.get("pid"),
        "message": result.get("message"),
    }


class ControllerHandler(BaseHTTPRequestHandler):
    server: ControllerHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_OPTIONS(self) -> None:  # noqa: N802
        peer = self.client_address[0]
        origin = self.headers.get("Origin", "")
        if peer not in {"127.0.0.1", "::1"} or not local_browser_origin(origin):
            self._send(403, {"ok": False, "error": "forbidden", "message": "The controller accepts local requests only."})
            return
        self.send_response(204)
        self._cors_headers(origin)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return

    def _dispatch(self) -> None:
        peer = self.client_address[0]
        if peer not in {"127.0.0.1", "::1"}:
            self._send(403, {"ok": False, "error": "forbidden", "message": "The controller accepts local requests only."})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > MAX_BODY:
            self._send(413, {"ok": False, "error": "body", "message": "The request is too large."})
            return
        body = self.rfile.read(length) if length else b""
        path, _, query = self.path.partition("?")
        status, payload = self.server.controller.handle(
            self.command,
            path,
            query,
            {"authorization": self.headers.get("Authorization", "")},
            body,
        )
        self._send(status, payload, self.headers.get("Origin", ""))

    def _cors_headers(self, origin: str) -> None:
        if not local_browser_origin(origin):
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def _send(self, status: int, payload: dict, origin: str = "") -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers(origin)
        self.end_headers()
        self.wfile.write(raw)


class ControllerHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], controller: Controller):
        self.controller = controller
        super().__init__(address, ControllerHandler)


def serve(host: str, port: int, token: str) -> None:
    if host != "127.0.0.1":
        raise SystemExit("The controller binds to 127.0.0.1 only.")
    server = ControllerHTTPServer((host, port), Controller(token))
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Model Inspector server controller")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=CONTROLLER_PORT)
    args = parser.parse_args()
    if args.host != "127.0.0.1" or args.port != CONTROLLER_PORT:
        raise SystemExit("The controller listens on 127.0.0.1:8811 only.")
    token = load_or_create_token()
    install_simulator_credential(token)
    try:
        serve(args.host, args.port, token)
    except OSError as exc:
        print(f"The server controller could not listen on 127.0.0.1:{args.port}.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
