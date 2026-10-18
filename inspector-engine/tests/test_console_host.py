# Authors: LCC Research Lab
# Project: From Document to Evidence

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inspector_engine.console_host import attach_console


def test_the_console_and_the_api_share_one_application(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>AI Model Inspector</title>", encoding="utf-8")
    (assets / "app.js").write_text("inspector", encoding="utf-8")
    token = tmp_path / "token"
    token.write_text("local-test-credential\n", encoding="utf-8")
    application = FastAPI()

    @application.get("/health")
    def health():
        return {"ok": True, "service": "ai-model-inspector"}

    attach_console(application, dist, token)
    client = TestClient(application)
    assert client.get("/health").json()["ok"] is True
    page = client.get("/")
    assert page.status_code == 200
    assert "AI Model Inspector" in page.text
    assert client.get("/assets/app.js").text == "inspector"
    denied = client.get("/controller-credential", headers={"host": "example.test"})
    assert denied.status_code == 403
    assert denied.text == ""
    allowed = client.get("/controller-credential", headers={"host": "127.0.0.1"})
    assert allowed.status_code == 200
    assert allowed.headers["cache-control"] == "no-store"
    assert allowed.text == "local-test-credential"


def test_a_missing_build_leaves_the_api_available(tmp_path: Path) -> None:
    application = FastAPI()

    @application.get("/health")
    def health():
        return {"ok": True}

    attach_console(application, tmp_path / "missing-dist", tmp_path / "missing-token")
    client = TestClient(application)
    assert client.get("/health").json()["ok"] is True
    body = client.get("/").json()
    assert body["console"] == "not_built"
    assert client.get("/controller-credential", headers={"host": "127.0.0.1"}).status_code == 404
