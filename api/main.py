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
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import db
import queue as jobqueue
from auth import require_token

NAS_JOBS_ROOT = os.environ.get("NAS_JOBS_ROOT", "/mnt/vgreens_storage/ai/jobs")
NAS_MOUNTPOINT = os.environ.get("NAS_MOUNTPOINT", "/mnt/vgreens_storage")

app = FastAPI(title="vGreens Job API", version="0.1.0")


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

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def read_job(job_id: str, _: str = Depends(require_token)) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs")
def read_jobs(
    status: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 50,
    _: str = Depends(require_token),
) -> list[dict]:
    return db.list_jobs(status=status, model=model, limit=limit)


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
