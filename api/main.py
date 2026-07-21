"""vGreens Job API — FastAPI service.

Routes:
  POST /jobs                 submit a job -> returns job_id (status=queued)
  GET  /jobs/{id}            fetch one job
  GET  /jobs                 list jobs (filter by status/model, limit)
  POST /jobs/{id}/status     worker callback: report running/succeeded/failed...
  GET  /health               queue + NAS mount check (monitoring scrapes this)

Deployed as systemd unit vgreens-ai-api on gpu-worker-01 (Phase 1 step 1.3).
This file is part of the 1.1 scaffold; the broker (1.2) runs before the API.
"""
from __future__ import annotations

import os
import json
import re
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from pydantic import BaseModel, Field

import db
import broker as jobqueue
import models_registry as reg
import authn
from auth import require_token

COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
MAX_FAILED_LOGINS = int(os.environ.get("MAX_FAILED_LOGINS", "10"))

NAS_JOBS_ROOT = os.environ.get("NAS_JOBS_ROOT", "/mnt/vgreens_storage/ai/jobs")
NAS_MOUNTPOINT = os.environ.get("NAS_MOUNTPOINT", "/mnt/vgreens_storage")
NAS_DATASETS_ROOT = os.environ.get("NAS_DATASETS", "/mnt/vgreens_storage/ai/datasets")

VERSION = "0.2.0"        # API version (bumped for Phase 3 additive endpoints)
API_LEVEL = "v1"         # wire-compat level; bump only on breaking changes


def _git_commit() -> str:
    """Short commit of the deployed code (read-only; 'unknown' if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__) or ".",
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


COMMIT = _git_commit()

app = FastAPI(title="vGreens Job API", version=VERSION)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    authn.seed_admin_if_empty()


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class JobRequest(BaseModel):
    job_type: Literal["training", "processing"]
    model_name: str = Field(..., min_length=1)
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    tenant: Optional[str] = None
    docker_image: str = Field(..., min_length=1)
    config_path: Optional[str] = None
    priority: int = 0


class StatusUpdate(BaseModel):
    status: Literal["running", "succeeded", "failed", "cancelled"]
    worker: Optional[str] = None
    gpu: Optional[int] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    result_version: Optional[str] = None


class LockAcquire(BaseModel):
    holder: str
    job_id: str


class LockRelease(BaseModel):
    job_id: str


class PromoteBody(BaseModel):
    version: str
    reason: Optional[str] = None


class RollbackBody(BaseModel):
    to: Optional[str] = None
    reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _new_job_id(job_type: str) -> str:
    prefix = "trn" if job_type == "training" else "prc"
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}_{day}_{secrets.token_hex(2)}"


def _write_nas_snapshot(job: dict) -> str:
    """Write an immutable job.json snapshot to the NAS job folder.

    Path: <NAS_JOBS_ROOT>/<year>/<job_id>/job.json  (+ logs/ for the worker).
    Returns the job folder path (stored as output_path base).
    """
    year = datetime.now(timezone.utc).strftime("%Y")
    folder = Path(NAS_JOBS_ROOT) / year / job["job_id"]
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps(job, indent=2))
    return str(folder)


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.post("/jobs", status_code=201)
def create_job(req: JobRequest, requested_by: str = Depends(authn.actor("operator"))) -> dict:
    job_id = _new_job_id(req.job_type)
    job = {
        "job_id": job_id,
        "job_type": req.job_type,
        "model_name": req.model_name,
        "dataset_name": req.dataset_name,
        "dataset_version": req.dataset_version,
        "tenant": req.tenant,
        "docker_image": req.docker_image,
        "config_path": req.config_path,
        "status": "queued",
        "worker": None,
        "gpu": None,
        "requested_by": requested_by,
        "priority": req.priority,
        "created_at": db.now_iso(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "output_path": None,
        "result_version": None,
    }

    db.insert_job(job)
    try:
        job["output_path"] = _write_nas_snapshot(job)
        db.update_status(job_id, "queued", output_path=job["output_path"])
    except OSError as exc:
        # NAS write failed — fail loudly, don't queue a job we can't trace.
        db.update_status(job_id, "failed", error=f"NAS snapshot failed: {exc}")
        raise HTTPException(status_code=503, detail=f"NAS unavailable: {exc}")

    try:
        jobqueue.publish(job)
    except Exception as exc:
        db.update_status(job_id, "failed", error=f"publish failed: {exc}")
        raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}")

    db.add_audit(requested_by, "job.submit", job_id, source="api")
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def read_job(job_id: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/events")
def read_job_events(job_id: str, _: str = Depends(authn.actor("viewer"))) -> list[dict]:
    if db.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return db.list_job_events(job_id)


@app.get("/jobs")
def read_jobs(
    response: Response,
    status: Optional[str] = None,
    model: Optional[str] = None,
    job_type: Optional[str] = None,
    tenant: Optional[str] = None,
    worker: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    _: str = Depends(authn.actor("viewer")),
) -> list[dict]:
    # body stays a plain list (backward-compatible with the CLI); total for the
    # web UI's pagination goes in a header.
    response.headers["X-Total-Count"] = str(
        db.count_jobs(status, model, job_type, tenant, worker)
    )
    return db.list_jobs(status, model, job_type, tenant, worker, limit, offset)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, who: str = Depends(authn.actor("operator"))) -> dict:
    """Best-effort cancel: mark the job cancelled and kill its container if running."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] in ("succeeded", "failed", "cancelled"):
        return {"job_id": job_id, "status": job["status"], "note": "already finished"}

    db.update_status(job_id, "cancelled", error=f"cancelled by {who}")
    db.add_audit(who, "job.cancel", job_id, source="api")
    # kill the container if it's running (named vgreens-job-<id>); ignore if absent
    try:
        subprocess.run(["docker", "kill", f"vgreens-job-{job_id}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        pass
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/workers/{name}/heartbeat")
def worker_heartbeat(name: str, _: str = Depends(authn.actor("operator"))) -> dict:
    db.heartbeat(name)
    return {"worker": name, "ok": True}


@app.post("/jobs/{job_id}/status")
def report_status(job_id: str, upd: StatusUpdate, _: str = Depends(authn.actor("operator"))) -> dict:
    if db.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    db.update_status(
        job_id,
        upd.status,
        worker=upd.worker,
        gpu=upd.gpu,
        error=upd.error,
        output_path=upd.output_path,
        result_version=upd.result_version,
    )
    return {"job_id": job_id, "status": upd.status}


@app.post("/locks/{name}/acquire")
def acquire_lock(name: str, body: LockAcquire, _: str = Depends(authn.actor("operator"))) -> dict:
    """Take a named resource lock (e.g. 'training'). Returns whether it was granted."""
    granted = db.acquire_lock(name, body.holder, body.job_id)
    return {"lock": name, "acquired": granted}


@app.post("/locks/{name}/release")
def release_lock(name: str, body: LockRelease, _: str = Depends(authn.actor("operator"))) -> dict:
    released = db.release_lock(name, body.job_id)
    return {"lock": name, "released": released}


@app.get("/locks/{name}")
def read_lock(name: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    return db.get_lock(name) or {"lock": name, "held": False}


# --------------------------------------------------------------------------- #
# model registry — promote / rollback (production control)
# --------------------------------------------------------------------------- #
@app.get("/models")
def list_models(_: str = Depends(authn.actor("viewer"))) -> dict:
    return {"models": reg.list_models()}


@app.get("/models/{model}")
def read_model(model: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    if model not in reg.list_models():
        raise HTTPException(status_code=404, detail="model not found")
    return reg.get_model_summary(model)


@app.get("/models/{model}/versions")
def list_versions(model: str, _: str = Depends(authn.actor("viewer"))) -> list[dict]:
    return reg.list_versions(model)


@app.get("/models/{model}/versions/{version}")
def read_model_version(model: str, version: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    v = reg.get_version(model, version)
    if v is None:
        raise HTTPException(status_code=404, detail="version not found")
    return v


@app.get("/models/{model}/active")
def active_model(model: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    return reg.get_active(model) or {"model": model, "active_version": None}


@app.post("/models/{model}/promote")
def promote_model(model: str, body: PromoteBody, who: str = Depends(authn.actor("admin"))) -> dict:
    try:
        res = reg.promote(model, body.version, who, body.reason)
    except reg.RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.add_audit(who, "model.promote", f"{model}:{body.version}", reason=body.reason, source="api")
    return res


@app.post("/models/{model}/rollback")
def rollback_model(model: str, body: RollbackBody, who: str = Depends(authn.actor("admin"))) -> dict:
    try:
        res = reg.rollback(model, body.to, who, body.reason)
    except reg.RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.add_audit(who, "model.rollback", f"{model}:{res.get('active_version')}",
                 reason=body.reason, source="api")
    return res


@app.get("/audit")
def read_audit(limit: int = 50, _: str = Depends(authn.actor("viewer"))) -> list[dict]:
    return db.list_audit(limit=limit)


# --------------------------------------------------------------------------- #
# authentication (browser sessions). Static bearer tokens still work elsewhere.
# --------------------------------------------------------------------------- #
class LoginBody(BaseModel):
    username: str
    password: str


class ChangePasswordBody(BaseModel):
    old_password: str
    new_password: str


@app.post("/auth/login")
def auth_login(body: LoginBody, response: Response) -> dict:
    user = db.get_user_by_username(body.username)
    if user is None or not user["active"]:
        db.record_login(body.username, False, "web", "unknown or disabled user")
        raise HTTPException(status_code=401, detail="invalid credentials")
    if user["failed_logins"] >= MAX_FAILED_LOGINS:
        db.record_login(body.username, False, "web", "locked")
        raise HTTPException(status_code=429, detail="account temporarily locked — contact an admin")
    if not authn.verify_password(body.password, user["password_hash"]):
        db.note_login_result(user["id"], False)
        db.record_login(body.username, False, "web", "bad password")
        raise HTTPException(status_code=401, detail="invalid credentials")
    db.note_login_result(user["id"], True)
    db.record_login(body.username, True, "web")
    raw = authn.new_session(user["id"])
    response.set_cookie(authn.SESSION_COOKIE, raw, httponly=True, samesite="lax",
                        secure=COOKIE_SECURE, max_age=authn.SESSION_TTL_HOURS * 3600, path="/")
    return db.user_public(user)


@app.post("/auth/logout")
def auth_logout(response: Response, vg_session: str = Cookie(default="")) -> dict:
    if vg_session:
        authn.revoke_session(vg_session)
    response.delete_cookie(authn.SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/me")
def auth_me(ident: Optional[authn.Identity] = Depends(authn.current_identity)) -> dict:
    if ident is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return {"username": ident.username, "role": ident.role, "via": ident.via, "scopes": ident.scopes}


@app.post("/auth/change-password")
def auth_change_password(body: ChangePasswordBody,
                         ident: Optional[authn.Identity] = Depends(authn.current_identity)) -> dict:
    if ident is None or ident.via != "session" or ident.user_id is None:
        raise HTTPException(status_code=401, detail="log in with a user account to change your password")
    user = db.get_user_by_id(ident.user_id)
    if user is None or not authn.verify_password(body.old_password, user["password_hash"]):
        raise HTTPException(status_code=403, detail="current password is incorrect")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=422, detail="new password must be at least 8 characters")
    db.set_user_password(user["id"], authn.hash_password(body.new_password))
    db.add_audit(ident.username, "auth.change_password", ident.username, source="web")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# admin: user + API-token management (admin role only)
# --------------------------------------------------------------------------- #
class CreateUserBody(BaseModel):
    username: str
    password: str
    role: str = "viewer"
    display_name: Optional[str] = None
    email: Optional[str] = None


class UpdateUserBody(BaseModel):
    role: Optional[str] = None
    active: Optional[bool] = None
    display_name: Optional[str] = None
    email: Optional[str] = None


class ResetPasswordBody(BaseModel):
    new_password: str


class CreateTokenBody(BaseModel):
    name: str
    owner: Optional[str] = None
    scopes: Optional[str] = None       # space-separated
    expires_at: Optional[str] = None


@app.get("/admin/users")
def admin_list_users(_: str = Depends(authn.actor("admin"))) -> list[dict]:
    return db.list_users()


@app.post("/admin/users", status_code=201)
def admin_create_user(body: CreateUserBody, who: str = Depends(authn.actor("admin"))) -> dict:
    if body.role not in authn.ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {authn.ROLES}")
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="password must be at least 8 characters")
    if db.get_user_by_username(body.username) is not None:
        raise HTTPException(status_code=409, detail="username already exists")
    uid = db.create_user(body.username, authn.hash_password(body.password), role=body.role,
                         display_name=body.display_name, email=body.email)
    db.add_audit(who, "admin.user.create", body.username, reason=body.role, source="api")
    return db.user_public(db.get_user_by_id(uid))


@app.patch("/admin/users/{user_id}")
def admin_update_user(user_id: int, body: UpdateUserBody, who: str = Depends(authn.actor("admin"))) -> dict:
    if body.role is not None and body.role not in authn.ROLES:
        raise HTTPException(status_code=422, detail=f"role must be one of {authn.ROLES}")
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    db.update_user(user_id, role=body.role,
                   active=(None if body.active is None else (1 if body.active else 0)),
                   display_name=body.display_name, email=body.email)
    db.add_audit(who, "admin.user.update", user["username"], source="api")
    return db.user_public(db.get_user_by_id(user_id))


@app.post("/admin/users/{user_id}/reset-password")
def admin_reset_password(user_id: int, body: ResetPasswordBody,
                         who: str = Depends(authn.actor("admin"))) -> dict:
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=422, detail="password must be at least 8 characters")
    db.set_user_password(user_id, authn.hash_password(body.new_password))
    db.add_audit(who, "admin.user.reset_password", user["username"], source="api")
    return {"ok": True}


@app.get("/admin/tokens")
def admin_list_tokens(_: str = Depends(authn.actor("admin"))) -> list[dict]:
    return db.list_api_tokens()


@app.post("/admin/tokens", status_code=201)
def admin_create_token(body: CreateTokenBody, who: str = Depends(authn.actor("admin"))) -> dict:
    token_id, raw, token_hash = authn.new_api_token()
    db.create_api_token(token_id, body.name, body.owner or who, token_hash,
                        body.scopes or "", body.expires_at)
    db.add_audit(who, "admin.token.create", token_id, reason=body.name, source="api")
    # the raw token is returned exactly once; only its hash is stored
    return {"id": token_id, "name": body.name, "owner": body.owner or who,
            "scopes": body.scopes or "", "token": raw}


@app.delete("/admin/tokens/{token_id}")
def admin_revoke_token(token_id: str, who: str = Depends(authn.actor("admin"))) -> dict:
    if not db.revoke_api_token(token_id):
        raise HTTPException(status_code=404, detail="token not found")
    db.add_audit(who, "admin.token.revoke", token_id, source="api")
    return {"id": token_id, "revoked": True}


# --------------------------------------------------------------------------- #
# Dataset Factory — import workflows (executed by the CPU dataset worker) and
# read-only views of frozen dataset versions. The API is the control plane:
# it writes the import row and publishes to dataset_jobs; the worker consumes.
# --------------------------------------------------------------------------- #
DATASET_SOURCE_TYPES = {"roboflow", "local", "nas", "https", "s3"}
_DATASET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,62}$")


class DatasetImportRequest(BaseModel):
    source_type: str
    dataset_name: str
    source: dict = Field(default_factory=dict)   # adapter-specific coords; NEVER secrets
    requested_version: Optional[str] = None
    idempotency_key: Optional[str] = None


def _new_import_id() -> str:
    return f"imp_{datetime.now(timezone.utc):%Y%m%d}_{secrets.token_hex(2)}"


def _import_public(imp: dict) -> dict:
    """DB row -> API shape: parse source_params JSON back to an object."""
    out = dict(imp)
    try:
        out["source_params"] = json.loads(imp.get("source_params") or "{}")
    except (TypeError, json.JSONDecodeError):
        out["source_params"] = {}
    return out


@app.post("/datasets/imports", status_code=201)
def create_dataset_import(req: DatasetImportRequest, requested_by: str = Depends(authn.actor("operator"))) -> dict:
    if req.source_type not in DATASET_SOURCE_TYPES:
        raise HTTPException(status_code=422, detail=f"unsupported source_type: {req.source_type!r}")
    if not _DATASET_NAME_RE.match(req.dataset_name):
        raise HTTPException(status_code=422,
                            detail="dataset_name must be lowercase [a-z0-9_], start alphanumeric, <=63 chars")

    # idempotency: a repeated key returns the existing workflow, never a duplicate
    if req.idempotency_key:
        existing = db.get_import_by_idempotency(req.idempotency_key)
        if existing:
            return {"import_id": existing["import_id"], "state": existing["state"], "idempotent": True}

    import_id = _new_import_id()
    now = db.now_iso()
    imp = {
        "import_id": import_id,
        "idempotency_key": req.idempotency_key,
        "source_type": req.source_type,
        "source_params": json.dumps(req.source or {}),
        "dataset_name": req.dataset_name,
        "requested_version": req.requested_version,
        "resolved_version": None,
        "state": "submitted",
        "licence_name": None, "licence_status": None, "licence_by": None, "licence_at": None,
        "requested_by": requested_by,
        "error": None,
        "created_at": now, "updated_at": now,
    }
    try:
        db.create_dataset_import(imp)
    except Exception:
        # UNIQUE(idempotency_key) race: another request created it first
        existing = db.get_import_by_idempotency(req.idempotency_key)
        if existing:
            return {"import_id": existing["import_id"], "state": existing["state"], "idempotent": True}
        raise

    try:
        jobqueue.publish_dataset({
            "import_id": import_id,
            "source_type": req.source_type,
            "source_params": req.source or {},
            "dataset_name": req.dataset_name,
            "requested_version": req.requested_version,
        })
    except Exception as exc:
        db.update_dataset_import(import_id, "failed", error=f"publish failed: {exc}")
        raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}")

    db.add_audit(requested_by, "dataset.import.submit", req.dataset_name, source="api")
    return {"import_id": import_id, "state": "submitted"}


@app.get("/datasets/imports")
def list_dataset_imports(response: Response, state: Optional[str] = None,
                         dataset_name: Optional[str] = None, limit: int = 50, offset: int = 0,
                         _: str = Depends(authn.actor("viewer"))) -> list[dict]:
    response.headers["X-Total-Count"] = str(db.count_dataset_imports(state, dataset_name))
    return [_import_public(r) for r in db.list_dataset_imports(state, dataset_name, limit, offset)]


@app.get("/datasets/imports/{import_id}")
def get_dataset_import(import_id: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    imp = db.get_dataset_import(import_id)
    if imp is None:
        raise HTTPException(status_code=404, detail="import not found")
    out = _import_public(imp)
    out["events"] = db.list_dataset_import_events(import_id)
    return out


@app.post("/datasets/imports/{import_id}/cancel")
def cancel_dataset_import(import_id: str, who: str = Depends(authn.actor("operator"))) -> dict:
    imp = db.get_dataset_import(import_id)
    if imp is None:
        raise HTTPException(status_code=404, detail="import not found")
    if imp["state"] in db.DATASET_TERMINAL:
        raise HTTPException(status_code=409, detail=f"import already {imp['state']}")
    db.update_dataset_import(import_id, "cancelled", detail="cancelled via API")
    db.add_audit(who, "dataset.import.cancel", import_id, source="api")
    return {"import_id": import_id, "state": "cancelled"}


DATASET_STATES = {
    "submitted", "downloading", "staging", "quarantined", "auditing",
    "awaiting_license_approval", "normalising", "freezing", "frozen",
    "rejected", "failed", "cancelled",
}


class DatasetStateUpdate(BaseModel):
    state: str
    detail: Optional[str] = None
    error: Optional[str] = None
    resolved_version: Optional[str] = None


@app.post("/datasets/imports/{import_id}/state")
def report_dataset_state(import_id: str, upd: DatasetStateUpdate,
                         who: str = Depends(authn.actor("operator"))) -> dict:
    """Dataset-worker callback. The worker never writes the DB directly — it
    reports transitions here, mirroring the GPU worker's /jobs/{id}/status."""
    if upd.state not in DATASET_STATES:
        raise HTTPException(status_code=422, detail=f"invalid state: {upd.state!r}")
    imp = db.get_dataset_import(import_id)
    if imp is None:
        raise HTTPException(status_code=404, detail="import not found")
    ok = db.update_dataset_import(
        import_id, upd.state,
        detail=upd.detail, error=upd.error, resolved_version=upd.resolved_version,
    )
    return {"import_id": import_id, "state": upd.state if ok else imp["state"], "updated": ok}


class LicenseDecision(BaseModel):
    reason: Optional[str] = None
    licence_name: Optional[str] = None


@app.post("/datasets/imports/{import_id}/approve-license")
def approve_license(import_id: str, body: LicenseDecision, who: str = Depends(authn.actor("admin"))) -> dict:
    """Human gate: approve an external dataset's licence and re-enqueue it so
    the worker resumes and freezes it."""
    imp = db.get_dataset_import(import_id)
    if imp is None:
        raise HTTPException(status_code=404, detail="import not found")
    if imp["state"] in db.DATASET_TERMINAL:
        raise HTTPException(status_code=409, detail=f"import already {imp['state']}")
    db.update_dataset_import(
        import_id, "submitted", licence_status="approved",
        licence_name=body.licence_name or imp.get("licence_name"),
        licence_by=who, licence_at=db.now_iso(), detail="licence approved",
    )
    try:
        jobqueue.publish_dataset({
            "import_id": import_id, "source_type": imp["source_type"],
            "source_params": json.loads(imp.get("source_params") or "{}"),
            "dataset_name": imp["dataset_name"], "requested_version": imp.get("requested_version"),
        })
    except Exception as exc:
        db.update_dataset_import(import_id, "failed", error=f"re-publish failed: {exc}")
        raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}")
    db.add_audit(who, "dataset.license.approve", import_id, reason=body.reason, source="api")
    return {"import_id": import_id, "licence_status": "approved", "state": "submitted"}


@app.post("/datasets/imports/{import_id}/reject-license")
def reject_license(import_id: str, body: LicenseDecision, who: str = Depends(authn.actor("admin"))) -> dict:
    imp = db.get_dataset_import(import_id)
    if imp is None:
        raise HTTPException(status_code=404, detail="import not found")
    if imp["state"] in db.DATASET_TERMINAL:
        raise HTTPException(status_code=409, detail=f"import already {imp['state']}")
    db.update_dataset_import(
        import_id, "rejected", licence_status="rejected", licence_by=who,
        licence_at=db.now_iso(), error=f"licence rejected: {body.reason or 'no reason given'}",
    )
    db.add_audit(who, "dataset.license.reject", import_id, reason=body.reason, source="api")
    return {"import_id": import_id, "licence_status": "rejected", "state": "rejected"}


@app.get("/datasets")
def list_datasets(_: str = Depends(authn.actor("viewer"))) -> dict:
    return {"datasets": db.list_datasets()}


# ---- legacy backfill: index pre-existing frozen datasets on the NAS -------- #
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _dataset_meta_from_dir(ver_dir: Path) -> dict:
    """Prefer MANIFEST.json; otherwise infer counts by scanning the version dir."""
    manifest = ver_dir / "MANIFEST.json"
    if manifest.is_file():
        try:
            m = json.loads(manifest.read_text())
            content = m.get("content") or {}
            return {
                "task": m.get("task"),
                "image_count": content.get("image_count"),
                "class_count": content.get("class_count"),
                "content_digest": content.get("content_digest"),
                "licence_status": (m.get("licence") or {}).get("approval_status"),
                "frozen_at": (m.get("generation") or {}).get("generated_at"),
                "metadata_source": "manifest",
            }
        except (json.JSONDecodeError, OSError):
            pass
    # infer
    n_images = sum(1 for p in ver_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    nc = None
    dy = ver_dir / "data.yaml"
    if dy.is_file():
        try:
            import yaml
            doc = yaml.safe_load(dy.read_text()) or {}
            names = doc.get("names")
            nc = doc.get("nc", len(names) if isinstance(names, list) else None)
        except Exception:  # noqa: BLE001
            pass
    return {"task": None, "image_count": n_images, "class_count": nc,
            "content_digest": None, "licence_status": "legacy",
            "frozen_at": None, "metadata_source": "inferred"}


def _scan_frozen_datasets() -> list[tuple[str, str, Path]]:
    root = Path(NAS_DATASETS_ROOT)
    out: list[tuple[str, str, Path]] = []
    if not root.is_dir():
        return out
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for ver_dir in sorted(ds_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            # a usable frozen version has a completion marker or a data.yaml
            if (ver_dir / ".complete").exists() or (ver_dir / "data.yaml").is_file():
                out.append((ds_dir.name, ver_dir.name, ver_dir))
    return out


@app.post("/datasets/backfill")
def backfill_datasets(dry_run: bool = True, who: str = Depends(authn.actor("admin"))) -> dict:
    """Index pre-existing frozen datasets (legacy) into the version registry.
    Idempotent; dry_run=true (default) reports without writing. Never edits files."""
    report: dict = {"dry_run": dry_run, "scanned": 0, "already_indexed": 0,
                    "added": [], "errors": []}
    for name, version, ver_dir in _scan_frozen_datasets():
        report["scanned"] += 1
        if db.get_dataset_version(name, version):
            report["already_indexed"] += 1
            continue
        try:
            meta = _dataset_meta_from_dir(ver_dir)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append({"dataset": f"{name}:{version}", "error": str(exc)})
            continue
        report["added"].append({
            "dataset_name": name, "version": version,
            "image_count": meta.get("image_count"), "class_count": meta.get("class_count"),
            "metadata_source": meta.get("metadata_source"),
        })
        if not dry_run:
            db.register_backfill_version(name, version, str(ver_dir), meta)
            db.add_audit(who, "dataset.backfill", f"{name}:{version}", source="api")
    return report


class VersionAllocateRequest(BaseModel):
    import_id: str
    requested_version: Optional[str] = None
    task: Optional[str] = None
    image_count: Optional[int] = None
    class_count: Optional[int] = None
    licence_name: Optional[str] = None
    licence_status: Optional[str] = None


class VersionFinalizeRequest(BaseModel):
    import_id: str
    content_digest: str
    image_count: Optional[int] = None
    class_count: Optional[int] = None


@app.post("/datasets/{dataset_name}/versions/allocate")
def allocate_dataset_version(dataset_name: str, req: VersionAllocateRequest,
                             who: str = Depends(authn.actor("operator"))) -> dict:
    """Worker callback: atomically reserve the next version. Returns the version
    string and the NAS path the worker must freeze into."""
    if not _DATASET_NAME_RE.match(dataset_name):
        raise HTTPException(status_code=422, detail="invalid dataset_name")
    try:
        version = db.allocate_dataset_version(
            dataset_name, req.requested_version, req.import_id,
            {"task": req.task, "image_count": req.image_count, "class_count": req.class_count,
             "licence_name": req.licence_name, "licence_status": req.licence_status, "frozen_by": who},
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:                       # UNIQUE race / DB error
        raise HTTPException(status_code=409, detail=f"version allocation failed: {exc}")
    nas_path = str(Path(NAS_DATASETS_ROOT) / dataset_name / version)
    return {"dataset_name": dataset_name, "version": version, "nas_path": nas_path}


@app.post("/datasets/{dataset_name}/versions/{version}/finalize")
def finalize_dataset_version(dataset_name: str, version: str, req: VersionFinalizeRequest,
                             who: str = Depends(authn.actor("operator"))) -> dict:
    """Worker callback: publish a reserved version (make it visible) and flip
    the import to frozen."""
    nas_path = str(Path(NAS_DATASETS_ROOT) / dataset_name / version)
    ok = db.finalize_dataset_version(dataset_name, version, nas_path,
                                     req.content_digest, req.image_count, req.class_count)
    if not ok:
        raise HTTPException(status_code=404, detail="version not allocated")
    db.update_dataset_import(req.import_id, "frozen", resolved_version=version)
    db.add_audit(who, "dataset.freeze", f"{dataset_name}:{version}", source="api")
    return {"dataset_name": dataset_name, "version": version, "state": "frozen"}


@app.get("/datasets/{dataset_name}/versions")
def list_dataset_versions(dataset_name: str, _: str = Depends(authn.actor("viewer"))) -> list[dict]:
    return db.list_dataset_versions(dataset_name)


@app.get("/datasets/{dataset_name}/versions/{version}")
def get_dataset_version(dataset_name: str, version: str, _: str = Depends(authn.actor("viewer"))) -> dict:
    v = db.get_dataset_version(dataset_name, version)
    if v is None:
        raise HTTPException(status_code=404, detail="version not found")
    return v


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics. Unauthenticated on purpose (operational, LAN-only)."""
    reg = CollectorRegistry()
    jobs = Gauge("vgreens_jobs", "Jobs by status", ["status"], registry=reg)
    for status, n in db.count_by_status().items():
        jobs.labels(status=status).set(n)
    nas = Gauge("vgreens_nas_mounted", "NAS mounted (1) or not (0)", registry=reg)
    nas.set(1 if os.path.ismount(NAS_MOUNTPOINT) else 0)

    # seconds since each worker last checked in (drives the worker-stopped alert)
    age = Gauge("vgreens_worker_seconds_since_heartbeat",
                "Seconds since the worker last sent a heartbeat", ["worker"], registry=reg)
    now = datetime.now(timezone.utc)
    for w in db.list_workers():
        try:
            last = datetime.fromisoformat(w["last_seen"])
            age.labels(worker=w["name"]).set((now - last).total_seconds())
        except (TypeError, ValueError):
            pass
    return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)


@app.get("/workers")
def read_workers(_: str = Depends(authn.actor("viewer"))) -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for w in db.list_workers():
        age = None
        try:
            age = (now - datetime.fromisoformat(w["last_seen"])).total_seconds()
        except (TypeError, ValueError):
            pass
        running = db.list_jobs(worker=w["name"], status="running", limit=1)
        out.append({
            "name": w["name"],
            "last_seen": w["last_seen"],
            "seconds_since_heartbeat": round(age, 1) if age is not None else None,
            "online": age is not None and age < 90,
            "current_job": running[0]["job_id"] if running else None,
        })
    return out


def _query_gpus() -> list[dict]:
    """Read GPU state from nvidia-smi on the host (read-only)."""
    fields = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in out.stdout.strip().splitlines():
            idx, name, mu, mt, util, temp = [x.strip() for x in line.split(",")]
            gpus.append({
                "index": int(idx), "name": name,
                "memory_used_mb": int(mu), "memory_total_mb": int(mt),
                "utilization_pct": int(util), "temperature_c": int(temp),
            })
        return gpus
    except Exception:
        return []


@app.get("/gpus")
def read_gpus(_: str = Depends(authn.actor("viewer"))) -> list[dict]:
    return _query_gpus()


@app.get("/version")
def version() -> dict:
    return {"name": "vGreens Job API", "version": VERSION, "api": API_LEVEL, "commit": COMMIT}


@app.get("/health/live")
def health_live() -> dict:
    """Liveness: is the process up? Deliberately cheap — no dependency checks, so a
    NAS/broker blip can't trigger a restart loop."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready(response: Response) -> dict:
    """Readiness: can we safely serve requests? Requires the DB and the broker.
    NAS is reported but does NOT gate readiness (a NAS blip shouldn't take the API out)."""
    db_ok = db.ping()
    queue_ok = jobqueue.check_connection()
    ready = db_ok and queue_ok
    if not ready:
        response.status_code = 503
    return {
        "status": "ready" if ready else "not_ready",
        "db": "ok" if db_ok else "down",
        "queue": "connected" if queue_ok else "unreachable",
        "nas": "mounted" if os.path.ismount(NAS_MOUNTPOINT) else "missing",
    }


@app.get("/health")
def health() -> dict:
    queue_ok = jobqueue.check_connection()
    nas_ok = os.path.ismount(NAS_MOUNTPOINT)
    ok = queue_ok and nas_ok
    return {
        "status": "ok" if ok else "degraded",
        "queue": "connected" if queue_ok else "unreachable",
        "nas": "mounted" if nas_ok else "missing",
    }
