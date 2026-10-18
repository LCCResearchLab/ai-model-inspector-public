# Authors: LCC Research Lab
# Project: From Document to Evidence

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "inspector-server.sh"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, check=False)


def test_closed_port_is_stopped_not_a_foreign_process() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    completed = run_script("status", "--port", str(port), "--json")
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["state"] == "stopped"
    assert payload["port"] == port


def test_stop_refuses_another_programs_port() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        raise AssertionError("the other program did not start listening")
    try:
        completed = run_script("stop", "--port", str(port), "--json")
        assert completed.returncode == 2
        payload = json.loads(completed.stdout)
        assert payload["state"] == "foreign"
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_a_public_host_is_refused() -> None:
    env = os.environ.copy()
    env["INSPECTOR_HOST"] = "0.0.0.0"
    completed = subprocess.run(
        [str(SCRIPT), "status", "--port", "9", "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 1
    assert "127.0.0.1" in completed.stderr
    assert "0.0.0.0" not in completed.stdout


def test_localhost_is_bound_as_the_loopback_address() -> None:
    env = os.environ.copy()
    env["INSPECTOR_HOST"] = "localhost"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    completed = subprocess.run(
        [str(SCRIPT), "status", "--port", str(port), "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["state"] == "stopped"
    assert "listens on 127.0.0.1 only" not in completed.stderr


def test_reserved_ports_are_not_contacted() -> None:
    for port in (8787, 8799, 8811):
        completed = run_script("status", "--port", str(port), "--json")
        assert completed.returncode == 2
        payload = json.loads(completed.stdout)
        assert payload["state"] == "foreign"
        assert "reserved" in payload["message"].lower()
