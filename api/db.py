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

CREATE TABLE IF NOT EXISTS locks (
    name        TEXT PRIMARY KEY,   -- e.g. "training"
    holder      TEXT,               -- worker name that holds it
    job_id      TEXT,               -- job currently holding it
    acquired_at TEXT
);

CREATE TABLE IF NOT EXISTS workers (
    name       TEXT PRIMARY KEY,    -- e.g. gpu-worker-01@0
    last_seen  TEXT
);
"""

TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")


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


# Ordered, idempotent migrations applied after the base SCHEMA. Each runs once and
# is recorded in schema_migrations. Add new ones with the next integer version;
# never edit an already-shipped migration.
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "job_events + audit_events", """
        CREATE TABLE IF NOT EXISTS job_events (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id  TEXT NOT NULL,
            ts      TEXT NOT NULL,
            event   TEXT NOT NULL,
            detail  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);

        CREATE TABLE IF NOT EXISTS audit_events (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT NOT NULL,
            actor   TEXT,
            action  TEXT NOT NULL,
            target  TEXT,
            result  TEXT,
            reason  TEXT,
            source  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts);
    """),
]


def _apply_migrations(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)"
    )
    applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    for version, name, sql in MIGRATIONS:
        if version in applied:
            continue
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO schema_migrations(version, name, applied_at) VALUES(?,?,?)",
            (version, name, now_iso()),
        )
        print(f"[db] applied migration {version}: {name}", flush=True)


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)      # base tables (create-if-not-exists)
        _apply_migrations(conn)         # ordered, tracked schema changes


def ping() -> bool:
    """Cheap connectivity check for readiness probes."""
    try:
        with _connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


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
        conn.execute(
            "INSERT INTO job_events(job_id, ts, event, detail) VALUES(?,?,?,?)",
            (job["job_id"], now_iso(), "queued", None),
        )


def get_job(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def _job_filters(status=None, model=None, job_type=None, tenant=None, worker=None):
    clauses, params = [], []
    for col, val in (("status", status), ("model_name", model), ("job_type", job_type),
                     ("tenant", tenant), ("worker", worker)):
        if val:
            clauses.append(f"{col} = ?")
            params.append(val)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def list_jobs(status=None, model=None, job_type=None, tenant=None, worker=None,
              limit: int = 50, offset: int = 0) -> list[dict]:
    where, params = _job_filters(status, model, job_type, tenant, worker)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)],
        ).fetchall()
        return [dict(r) for r in rows]


def count_jobs(status=None, model=None, job_type=None, tenant=None, worker=None) -> int:
    where, params = _job_filters(status, model, job_type, tenant, worker)
    with _connect() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM jobs {where}", params).fetchone()[0]


def update_status(job_id: str, status: str, **fields) -> bool:
    """Update job status plus optional worker/gpu/error/output_path/result_version.

    Stamps started_at/finished_at automatically on the relevant transitions.
    A 'cancelled' job is terminal: later reports (e.g. a worker's 'failed' after
    its container was killed) must not override it.
    """
    with _connect() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        prev_status = row["status"] if row is not None else None
        if prev_status == "cancelled" and status != "cancelled":
            return False

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
        if cur.rowcount > 0 and status != prev_status:   # only log real transitions
            detail = fields.get("error") or fields.get("result_version")
            conn.execute(
                "INSERT INTO job_events(job_id, ts, event, detail) VALUES(?,?,?,?)",
                (job_id, now_iso(), status, detail),
            )
        return cur.rowcount > 0


def count_by_status() -> dict:
    """For the API /metrics and monitoring."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}


# --------------------------------------------------------------------------- #
# resource locks (e.g. the training lock: max one training at a time so a GPU
# always stays free for production)
# --------------------------------------------------------------------------- #
def _lock_conn() -> sqlite3.Connection:
    # autocommit mode (isolation_level=None) so we can drive BEGIN IMMEDIATE
    # ourselves — that takes the write lock up front and makes acquire atomic
    # even when both workers ask at the same moment.
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def acquire_lock(name: str, holder: str, job_id: str) -> bool:
    """Try to take the named lock. Returns True on success.

    A lock is considered free if nobody holds it, or if the previous holder's
    job is no longer active (crashed/finished without releasing) — that makes
    the lock self-healing so a dead worker can't wedge it forever.
    """
    conn = _lock_conn()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        row = conn.execute("SELECT job_id FROM locks WHERE name = ?", (name,)).fetchone()
        if row is not None:
            held = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            if held is not None and held["status"] in ("queued", "running"):
                conn.execute("COMMIT;")
                return False  # someone active still holds it
            # else: stale holder -> fall through and take over
        conn.execute(
            "INSERT INTO locks(name, holder, job_id, acquired_at) VALUES(?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET holder=excluded.holder, "
            "job_id=excluded.job_id, acquired_at=excluded.acquired_at",
            (name, holder, job_id, now_iso()),
        )
        conn.execute("COMMIT;")
        return True
    finally:
        conn.close()


def release_lock(name: str, job_id: str) -> bool:
    """Release the lock only if this job_id still holds it."""
    conn = _lock_conn()
    try:
        cur = conn.execute(
            "DELETE FROM locks WHERE name = ? AND job_id = ?", (name, job_id)
        )
        return cur.rowcount > 0
    finally:
        conn.close()


def get_lock(name: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM locks WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None


# --------------------------------------------------------------------------- #
# worker heartbeats (for the worker-stopped alert)
# --------------------------------------------------------------------------- #
def heartbeat(name: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO workers(name, last_seen) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET last_seen = excluded.last_seen",
            (name, now_iso()),
        )


def list_workers() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT name, last_seen FROM workers").fetchall()
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# job event timeline + audit trail (append-only)
# --------------------------------------------------------------------------- #
def add_job_event(job_id: str, event: str, detail: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO job_events(job_id, ts, event, detail) VALUES(?,?,?,?)",
            (job_id, now_iso(), event, detail),
        )


def list_job_events(job_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, event, detail FROM job_events WHERE job_id = ? ORDER BY id",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_audit(actor: str, action: str, target: str | None = None,
              result: str = "ok", reason: str | None = None, source: str = "api") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_events(ts, actor, action, target, result, reason, source) "
            "VALUES(?,?,?,?,?,?,?)",
            (now_iso(), actor, action, target, result, reason, source),
        )


def list_audit(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, actor, action, target, result, reason, source "
            "FROM audit_events ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
