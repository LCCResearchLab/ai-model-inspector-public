# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Serve the built web console on the measurement API port.

The console is optional. Without a build, the API routes stay available.
The controller credential is returned only to a browser on this computer.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLE_DIST = PROJECT_ROOT / "web" / "dist"
TOKEN_PATH = PROJECT_ROOT / ".inspector-runtime" / "controller-token"


def local_browser(request: Request) -> bool:
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host in {"127.0.0.1", "localhost"}:
        return True
    client = request.client.host if request.client else ""
    return host in {"", "testserver"} and client in {"127.0.0.1", "::1", "testclient"}


def attach_console(application: FastAPI, dist: Path | None = None, token_path: Path | None = None) -> None:
    root = dist if dist is not None else CONSOLE_DIST

    @application.get("/controller-credential")
    def controller_credential(request: Request):
        if not local_browser(request):
            return PlainTextResponse("", status_code=403)
        credential = token_path if token_path is not None else TOKEN_PATH
        try:
            token = credential.read_text(encoding="utf-8").strip()
        except OSError:
            return PlainTextResponse("", status_code=404)
        if not token or len(token) > 200 or "\n" in token:
            return PlainTextResponse("", status_code=404)
        return PlainTextResponse(token, headers={"Cache-Control": "no-store"})

    index = root / "index.html"
    if not index.is_file():
        @application.get("/")
        def console_not_built():
            return JSONResponse(
                {
                    "ok": True,
                    "service": "ai-model-inspector",
                    "console": "not_built",
                    "message": "The web console has not been built. The measurement API is available.",
                }
            )
        return

    @application.get("/")
    def console_index():
        return FileResponse(index)

    assets = root / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="inspector-console-assets")
