"""SQLite job store.

The database lives on LOCAL disk (JOBS_DB_PATH, default /opt/vgreens-ai/jobs.db).
NEVER put this file on the NAS/SMB share — SQLite over SMB corrupts.

Single `jobs` table, mirroring the design doc schema.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get("JOBS_DB_PATH", "/opt/vgreens-ai/jobs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    job_type        TEXT NOT NULL,              -- training | processing
    model_name      TEXT NOT NULL,
    dataset_name    TEXT,
    dataset_version TEXT,
    tenant          TEXT,
    docker_image    TEXT NOT NULL,
    config_path     TEXT,
    status          TEXT NOT NULL,              -- queued|running|succeeded|failed|cancelled
    worker          TEXT,
    gpu             INTEGER,
    requested_by    TEXT,
    priority        INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    error           TEXT,
    output_path     TEXT,
    result_version  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_model  ON jobs(model_name);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL improves concurrent read/write on local disk.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def insert_job(job: dict) -> None:
    cols = (
        "job_id", "job_type", "model_name", "dataset_name", "dataset_version",
        "tenant", "docker_image", "config_path", "status", "worker", "gpu",
        "requested_by", "priority", "created_at", "started_at", "finished_at",
        "error", "output_path", "result_version",
    )
    values = [job.get(c) for c in cols]
    placeholders = ",".join("?" for _ in cols)
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO jobs ({','.join(cols)}) VALUES ({placeholders})", values
        )


def get_job(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(status: str | None = None, model: str | None = None, limit: int = 50) -> list[dict]:
    clauses, params = [], []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if model:
        clauses.append("model_name = ?")
        params.append(model)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]


def update_status(job_id: str, status: str, **fields) -> bool:
    """Update job status plus optional worker/gpu/error/output_path/result_version.

    Stamps started_at/finished_at automatically on the relevant transitions.
    """
    sets = {"status": status}
    if status == "running":
        sets["started_at"] = now_iso()
    if status in ("succeeded", "failed", "cancelled"):
        sets["finished_at"] = now_iso()
    for key in ("worker", "gpu", "error", "output_path", "result_version"):
        if key in fields and fields[key] is not None:
            sets[key] = fields[key]

    assignments = ",".join(f"{k} = ?" for k in sets)
    params = list(sets.values()) + [job_id]
    with _connect() as conn:
        cur = conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", params)
        return cur.rowcount > 0


def count_by_status() -> dict:
    """For the API /metrics and monitoring."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
