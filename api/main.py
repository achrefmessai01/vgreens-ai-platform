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
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from pydantic import BaseModel, Field

import db
import broker as jobqueue
import models_registry as reg
from auth import require_token

NAS_JOBS_ROOT = os.environ.get("NAS_JOBS_ROOT", "/mnt/vgreens_storage/ai/jobs")
NAS_MOUNTPOINT = os.environ.get("NAS_MOUNTPOINT", "/mnt/vgreens_storage")

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
def create_job(req: JobRequest, requested_by: str = Depends(require_token)) -> dict:
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
def read_job(job_id: str, _: str = Depends(require_token)) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/events")
def read_job_events(job_id: str, _: str = Depends(require_token)) -> list[dict]:
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
    _: str = Depends(require_token),
) -> list[dict]:
    # body stays a plain list (backward-compatible with the CLI); total for the
    # web UI's pagination goes in a header.
    response.headers["X-Total-Count"] = str(
        db.count_jobs(status, model, job_type, tenant, worker)
    )
    return db.list_jobs(status, model, job_type, tenant, worker, limit, offset)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, who: str = Depends(require_token)) -> dict:
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
def worker_heartbeat(name: str, _: str = Depends(require_token)) -> dict:
    db.heartbeat(name)
    return {"worker": name, "ok": True}


@app.post("/jobs/{job_id}/status")
def report_status(job_id: str, upd: StatusUpdate, _: str = Depends(require_token)) -> dict:
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
def acquire_lock(name: str, body: LockAcquire, _: str = Depends(require_token)) -> dict:
    """Take a named resource lock (e.g. 'training'). Returns whether it was granted."""
    granted = db.acquire_lock(name, body.holder, body.job_id)
    return {"lock": name, "acquired": granted}


@app.post("/locks/{name}/release")
def release_lock(name: str, body: LockRelease, _: str = Depends(require_token)) -> dict:
    released = db.release_lock(name, body.job_id)
    return {"lock": name, "released": released}


@app.get("/locks/{name}")
def read_lock(name: str, _: str = Depends(require_token)) -> dict:
    return db.get_lock(name) or {"lock": name, "held": False}


# --------------------------------------------------------------------------- #
# model registry — promote / rollback (production control)
# --------------------------------------------------------------------------- #
@app.get("/models")
def list_models(_: str = Depends(require_token)) -> dict:
    return {"models": reg.list_models()}


@app.get("/models/{model}")
def read_model(model: str, _: str = Depends(require_token)) -> dict:
    if model not in reg.list_models():
        raise HTTPException(status_code=404, detail="model not found")
    return reg.get_model_summary(model)


@app.get("/models/{model}/versions")
def list_versions(model: str, _: str = Depends(require_token)) -> list[dict]:
    return reg.list_versions(model)


@app.get("/models/{model}/versions/{version}")
def read_model_version(model: str, version: str, _: str = Depends(require_token)) -> dict:
    v = reg.get_version(model, version)
    if v is None:
        raise HTTPException(status_code=404, detail="version not found")
    return v


@app.get("/models/{model}/active")
def active_model(model: str, _: str = Depends(require_token)) -> dict:
    return reg.get_active(model) or {"model": model, "active_version": None}


@app.post("/models/{model}/promote")
def promote_model(model: str, body: PromoteBody, who: str = Depends(require_token)) -> dict:
    try:
        res = reg.promote(model, body.version, who, body.reason)
    except reg.RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.add_audit(who, "model.promote", f"{model}:{body.version}", reason=body.reason, source="api")
    return res


@app.post("/models/{model}/rollback")
def rollback_model(model: str, body: RollbackBody, who: str = Depends(require_token)) -> dict:
    try:
        res = reg.rollback(model, body.to, who, body.reason)
    except reg.RegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    db.add_audit(who, "model.rollback", f"{model}:{res.get('active_version')}",
                 reason=body.reason, source="api")
    return res


@app.get("/audit")
def read_audit(limit: int = 50, _: str = Depends(require_token)) -> list[dict]:
    return db.list_audit(limit=limit)


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
def read_workers(_: str = Depends(require_token)) -> list[dict]:
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
def read_gpus(_: str = Depends(require_token)) -> list[dict]:
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
