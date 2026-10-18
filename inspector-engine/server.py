# Authors: LCC Research Lab
# Project: From Document to Evidence
"""Local API for AI Model Inspector. Bind this process to the desktop Mac."""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from inspector_engine.access_log import install_access_redaction
from inspector_engine.console_host import attach_console
from inspector_engine.inspect_checkpoint import InspectionError, inspect_checkpoint
from inspector_engine.weight_statistics import measure_weight_statistics
from inspector_engine.credentials import (
    CredentialError,
    authorize,
    check_credential,
    list_cloud_models,
    list_credentials,
    remove_credential,
    save_credential,
)
from inspector_engine.model_library import (
    LibraryError,
    cancel_job,
    delete_downloaded,
    discover,
    hold_inspection,
    job_status,
    list_local_models,
    note_library_inspection,
    search_models,
    start_download,
)
from inspector_engine.roots import PathRejected, resolve_checkpoint
from inspector_engine.server_controller import local_browser_origin
from inspector_engine.effective_lora import align_update_factors, factors_for_checkpoint, measure_effective_update
from inspector_engine.research_registry import (
    RegistryError,
    freshman_update_factors,
    list_freshman,
    measure_freshman,
    measure_freshman_effective,
    open_freshman,
    registered_subject_id,
)

DEFAULT_PORT = 8810
API_VERSION = "1.0.0"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def refuse_public_bind(argv: list[str] | None = None) -> None:
    """The supported launcher always passes 127.0.0.1. A public host stops startup."""
    args = list(sys.argv if argv is None else argv)
    host = None
    for index, arg in enumerate(args):
        if arg == "--host" and index + 1 < len(args):
            host = args[index + 1]
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
    if host is None:
        return
    if host.strip().lower() not in LOOPBACK_HOSTS:
        raise SystemExit("The Inspector API listens on 127.0.0.1 only.")


refuse_public_bind()
install_access_redaction()

app = FastAPI(title="AI Model Inspector", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class LocalCredentialGuard(BaseHTTPMiddleware):
    """Credential routes stay on this computer. Another site cannot read their responses."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        credential_route = path == "/credentials" or path.startswith("/credentials/") or path == "/controller-credential"
        origin = request.headers.get("origin")
        if credential_route and origin and not local_browser_origin(origin):
            return JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "error": "not_local",
                    "message": "Credential management is available only on this computer.",
                },
            )
        response = await call_next(request)
        if credential_route and not local_browser_origin(origin or ""):
            if "access-control-allow-origin" in response.headers:
                del response.headers["access-control-allow-origin"]
        return response


app.add_middleware(LocalCredentialGuard)


class CheckpointRequest(BaseModel):
    path: str


class AlignmentRequest(BaseModel):
    paths: list[str]


class DownloadRequest(BaseModel):
    repo: str
    files: list[str] = []
    revision: str | None = None
    mode: str = "selected"


class DeleteRequest(BaseModel):
    repo: str
    file: str | None = None
    scope: str = "file"


def _parse_port(raw: str) -> int | None:
    try:
        port = int(str(raw).strip())
    except ValueError:
        return None
    if 1 <= port <= 65535:
        return port
    return None


def configured_port() -> int:
    """Listening port: uvicorn --port, then INSPECTOR_PORT, then 8810."""
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--port" and index + 1 < len(argv):
            parsed = _parse_port(argv[index + 1])
            if parsed is not None:
                return parsed
        if arg.startswith("--port="):
            parsed = _parse_port(arg.split("=", 1)[1])
            if parsed is not None:
                return parsed
    env = os.environ.get("INSPECTOR_PORT", "")
    parsed = _parse_port(env) if env.strip() else None
    if parsed is not None:
        return parsed
    return DEFAULT_PORT


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "ai-model-inspector",
        "version": API_VERSION,
        "port": configured_port(),
    }


def _library_error(exc: LibraryError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"ok": False, "error": exc.code, "message": exc.message},
    )


@app.get("/models/search")
def search_model_catalog(q: str, cursor: str = "", sort: str = "relevance", task: str = "all", scope: str = "combined"):
    try:
        return search_models(q, cursor=cursor, sort=sort, task=task, scope=scope)
    except LibraryError as exc:
        return _library_error(exc)


@app.get("/models/discover")
def discover_models(q: str):
    try:
        return discover(q)
    except LibraryError as exc:
        return _library_error(exc)


@app.get("/models/local")
def local_models():
    return list_local_models()


@app.post("/models/download")
def download_models(body: DownloadRequest):
    try:
        return start_download(body.repo, body.files, revision=body.revision, mode=body.mode)
    except LibraryError as exc:
        return _library_error(exc)


@app.get("/models/jobs/{job_id}")
def download_status(job_id: str):
    try:
        return job_status(job_id)
    except LibraryError as exc:
        return _library_error(exc)


@app.post("/models/jobs/{job_id}/cancel")
def download_cancel(job_id: str):
    try:
        return cancel_job(job_id)
    except LibraryError as exc:
        return _library_error(exc)


@app.post("/models/library/delete")
def delete_library_files(body: DeleteRequest):
    try:
        return delete_downloaded(body.repo, filename=body.file, scope=body.scope)
    except LibraryError as exc:
        return _library_error(exc)


def _measured(path: str, action):
    try:
        resolved = resolve_checkpoint(path)
    except PathRejected:
        return action()
    with hold_inspection(resolved):
        return action()


def _credential_error(exc: CredentialError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"ok": False, "error": exc.code, "message": exc.message},
    )


def _credential_call(request: Request, action):
    try:
        authorize(request)
        return action()
    except CredentialError as exc:
        return _credential_error(exc)


@app.get("/credentials")
def credential_list(request: Request):
    return _credential_call(request, list_credentials)


@app.post("/credentials/huggingface")
async def credential_save_huggingface(request: Request):
    try:
        authorize(request)
        try:
            payload = await request.json()
        except Exception:
            raise CredentialError("rejected", "The credential was not saved.", 400) from None
        token = payload.get("token") if isinstance(payload, dict) else None
        return save_credential("huggingface", token)
    except CredentialError as exc:
        return _credential_error(exc)


@app.post("/credentials/huggingface/test")
def credential_test_huggingface(request: Request):
    return _credential_call(request, lambda: check_credential("huggingface"))


@app.post("/credentials/huggingface/remove")
def credential_remove_huggingface(request: Request):
    return _credential_call(request, lambda: remove_credential("huggingface"))


@app.post("/credentials/{provider}/test")
def credential_test_other(provider: str, request: Request):
    return _credential_call(request, lambda: check_credential(provider))


@app.post("/credentials/{provider}/remove")
def credential_remove_other(provider: str, request: Request):
    return _credential_call(request, lambda: remove_credential(provider))


@app.get("/credentials/{provider}/models")
def credential_models(provider: str, request: Request):
    return _credential_call(request, lambda: list_cloud_models(provider))


@app.post("/credentials/{provider}")
async def credential_save_other(provider: str, request: Request):
    try:
        authorize(request)
        try:
            payload = await request.json()
        except Exception:
            raise CredentialError("rejected", "The credential was not saved.", 400) from None
        token = payload.get("token") if isinstance(payload, dict) else None
        return save_credential(provider, token)
    except CredentialError as exc:
        return _credential_error(exc)


@app.get("/research/book-three/freshman")
def freshman_research():
    try:
        return list_freshman()
    except RegistryError as exc:
        return JSONResponse(status_code=exc.status, content={"ok": False, "error": exc.code, "message": exc.message})


@app.post("/research/book-three/freshman/{subject_id}/inspect")
def freshman_research_inspect(subject_id: str):
    try:
        return open_freshman(subject_id)
    except (RegistryError, InspectionError) as exc:
        return JSONResponse(status_code=exc.status, content={"ok": False, "error": exc.code, "message": exc.message})


@app.post("/research/book-three/freshman/{subject_id}/weight-statistics")
def freshman_weight_statistics(subject_id: str):
    try:
        return measure_freshman(subject_id)
    except (RegistryError, InspectionError) as exc:
        return JSONResponse(status_code=exc.status, content={"ok": False, "error": exc.code, "message": exc.message})


@app.post("/research/book-three/freshman/{subject_id}/effective-update")
def freshman_effective_update(subject_id: str):
    try:
        return measure_freshman_effective(subject_id)
    except (RegistryError, InspectionError) as exc:
        return JSONResponse(status_code=exc.status, content={"ok": False, "error": exc.code, "message": exc.message})


def _instrument_error(exc: InspectionError | RegistryError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"ok": False, "error": exc.code, "message": exc.message})


@app.post("/inspect/weight-statistics")
def weight_statistics_route(body: CheckpointRequest):
    try:
        return _measured(body.path, lambda: _weight_statistics_body(body))
    except (InspectionError, RegistryError) as exc:
        return _instrument_error(exc)


def _weight_statistics_body(body: CheckpointRequest):
    subject_id = registered_subject_id(body.path)
    if subject_id:
        return measure_freshman(subject_id)
    return measure_weight_statistics(body.path)


@app.post("/inspect/update-alignment")
def update_alignment_route(body: AlignmentRequest):
    try:
        loaded = []
        for path in body.paths:
            subject_id = registered_subject_id(path)
            loaded.append(freshman_update_factors(subject_id) if subject_id else factors_for_checkpoint(path))
        return align_update_factors(loaded)
    except (InspectionError, RegistryError) as exc:
        return _instrument_error(exc)


@app.post("/inspect/effective-update")
def effective_update_route(body: CheckpointRequest):
    try:
        return _measured(body.path, lambda: _effective_update_body(body))
    except (InspectionError, RegistryError) as exc:
        return _instrument_error(exc)


def _effective_update_body(body: CheckpointRequest):
    subject_id = registered_subject_id(body.path)
    if subject_id:
        return measure_freshman_effective(subject_id)
    return measure_effective_update(body.path)


@app.post("/inspect/checkpoint")
def inspect_checkpoint_route(body: CheckpointRequest):
    try:
        return _measured(body.path, lambda: _inspect_checkpoint_body(body))
    except (InspectionError, RegistryError) as exc:
        return _instrument_error(exc)


def _inspect_checkpoint_body(body: CheckpointRequest):
    subject_id = registered_subject_id(body.path)
    if subject_id:
        return open_freshman(subject_id)
    try:
        report = inspect_checkpoint(body.path)
    except InspectionError as exc:
        if exc.code == "malformed_safetensors":
            note_library_inspection(Path(body.path), tensor_count=None, readable=False)
        raise
    checkpoint = report.get("checkpoint") if isinstance(report, dict) else None
    recorded = checkpoint.get("path") if isinstance(checkpoint, dict) else None
    if isinstance(recorded, str):
        note_library_inspection(Path(recorded), tensor_count=report.get("tensor_count"), readable=True)
    return report


attach_console(app)
