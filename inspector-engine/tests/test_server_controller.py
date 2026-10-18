# Authors: LCC Research Lab
# Project: From Document to Evidence

import json
import threading
from http.client import HTTPConnection

import pytest

from inspector_engine.server_controller import (
    TOKEN_PATH,
    Controller,
    ControllerHTTPServer,
    install_simulator_credential,
    load_or_create_token,
)

SECRET = "controller-secret-SHOULD_NOT_LEAK_4411"


def fake_runner(command: str, port: int) -> dict:
    return {
        "state": "stopped" if command == "status" else "running",
        "port": port,
        "pid": 42 if command != "status" else None,
        "message": f"{command} finished",
        "code": 0 if command != "status" else 1,
    }


@pytest.fixture
def server():
    calls: list[tuple[str, int]] = []

    def runner(command: str, port: int) -> dict:
        calls.append((command, port))
        if port == 65535:
            return {"state": "foreign", "port": port, "pid": None, "message": "Port is in use by another program.", "code": 2}
        return fake_runner(command, port)

    httpd = ControllerHTTPServer(("127.0.0.1", 0), Controller(SECRET, runner))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield port, calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def request(port: int, method: str, path: str, token: str | None = None, body: dict | None = None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {}
    raw = None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=raw, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, payload


def test_health_is_public_and_status_requires_the_credential(server) -> None:
    port, calls = server
    status, body = request(port, "GET", "/health")
    assert status == 200
    assert json.loads(body)["service"] == "ai-model-inspector-controller"
    status, body = request(port, "GET", "/v1/status?port=8810")
    assert status == 401
    assert SECRET not in body.decode()
    status, body = request(port, "GET", "/v1/status?port=8810", token="wrong-token")
    assert status == 401
    assert SECRET not in body.decode()
    assert calls == []


def test_status_start_and_stop_use_only_the_port(server) -> None:
    port, calls = server
    status, body = request(port, "GET", "/v1/status?port=8810", token=SECRET)
    assert status == 200
    assert json.loads(body)["state"] == "stopped"
    status, body = request(
        port,
        "POST",
        "/v1/start",
        token=SECRET,
        body={"port": 9000, "command": "rm -rf /", "shell": "curl evil"},
    )
    assert status == 200
    assert json.loads(body)["state"] == "running"
    status, _body = request(port, "POST", "/v1/stop", token=SECRET, body={"port": 9000})
    assert status == 200
    assert calls == [("status", 8810), ("start", 9000), ("stop", 9000)]
    assert SECRET not in json.dumps(calls)


def test_foreign_process_is_not_stopped(server) -> None:
    port, calls = server
    status, body = request(port, "POST", "/v1/stop", token=SECRET, body={"port": 65535})
    assert status == 409
    payload = json.loads(body)
    assert payload["state"] == "foreign"
    assert calls == [("stop", 65535)]


def test_reserved_ports_and_malformed_requests_do_not_call_the_manager(server) -> None:
    port, calls = server
    for reserved in (8787, 8799, 8811):
        status, body = request(port, "POST", "/v1/start", token=SECRET, body={"port": reserved})
        assert status == 400
        assert b"reserved" in body
    status, body = request(port, "POST", "/v1/start", token=SECRET, body={"port": "8810; rm"})
    assert status == 400
    status, body = request(port, "POST", "/v1/start", token=SECRET, body={"command": "stop"})
    assert status == 400
    assert calls == []


def test_token_file_is_reused_and_not_world_readable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "controller-token"
    monkeypatch.setattr("inspector_engine.server_controller.TOKEN_PATH", path)
    first = load_or_create_token(path)
    second = load_or_create_token(path)
    assert first == second
    assert path.stat().st_mode & 0o077 == 0
    assert first


def test_a_local_browser_may_call_the_same_controller(server) -> None:
    port, calls = server
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "OPTIONS",
        "/v1/start",
        headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"},
    )
    response = connection.getresponse()
    response.read()
    connection.close()
    assert response.status == 204
    assert response.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:5173"
    assert "Authorization" in (response.getheader("Access-Control-Allow-Headers") or "")

    status, body = request(port, "GET", "/v1/status?port=8810", token=SECRET)
    assert status == 200
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", "/v1/status?port=8810", headers={"Authorization": f"Bearer {SECRET}", "Origin": "http://127.0.0.1:5173"})
    allowed = connection.getresponse()
    allowed.read()
    connection.close()
    assert allowed.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:5173"

    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("OPTIONS", "/v1/start", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"})
    denied = connection.getresponse()
    denied_body = denied.read()
    connection.close()
    assert denied.status == 403
    assert denied.getheader("Access-Control-Allow-Origin") is None
    assert SECRET.encode() not in denied_body
    assert b"example.com" not in denied_body
    assert calls == [("status", 8810), ("status", 8810)]
    assert SECRET not in body.decode()


def test_simulator_install_refuses_a_path_outside_the_simulator(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "inspector_engine.server_controller.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": str(tmp_path)})(),
    )
    install_simulator_credential(SECRET)
    assert not (tmp_path / "Documents" / "inspector-controller-token").exists()
    assert SECRET not in TOKEN_PATH.as_posix()
