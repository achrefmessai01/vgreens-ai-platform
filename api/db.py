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
    (2, "dataset factory: imports + events + versions", """
        -- one row per dataset import workflow (Dataset Factory)
        CREATE TABLE IF NOT EXISTS dataset_imports (
            import_id         TEXT PRIMARY KEY,
            idempotency_key   TEXT UNIQUE,          -- retries return the same workflow
            source_type       TEXT NOT NULL,        -- roboflow | local | nas | https | s3
            source_params     TEXT,                 -- JSON; NEVER secrets/signed URLs
            dataset_name      TEXT NOT NULL,
            requested_version TEXT,                 -- optional explicit version
            resolved_version  TEXT,                 -- allocated atomically at freeze
            state             TEXT NOT NULL,        -- submitted|downloading|staging|quarantined|
                                                    -- auditing|awaiting_license_approval|normalising|
                                                    -- freezing|frozen|rejected|failed|cancelled
            licence_name      TEXT,
            licence_status    TEXT,                 -- unknown|approved|rejected|not_required
            licence_by        TEXT,
            licence_at        TEXT,
            requested_by      TEXT,
            error             TEXT,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dsimports_state ON dataset_imports(state);
        CREATE INDEX IF NOT EXISTS idx_dsimports_name  ON dataset_imports(dataset_name);

        -- append-only state-transition log (mirrors job_events)
        CREATE TABLE IF NOT EXISTS dataset_import_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            import_id  TEXT NOT NULL,
            ts         TEXT NOT NULL,
            state      TEXT NOT NULL,
            detail     TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_dsevents_import ON dataset_import_events(import_id);

        -- authoritative index of FROZEN dataset versions. UNIQUE(name,version)
        -- enforces atomic allocation and "never overwrite a frozen version".
        -- The NAS holds the bytes + MANIFEST + .complete marker; a row here is
        -- written only after a successful atomic freeze.
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name   TEXT NOT NULL,
            version        TEXT NOT NULL,
            task           TEXT,
            import_id      TEXT,
            content_digest TEXT,
            image_count    INTEGER,
            class_count    INTEGER,
            licence_name   TEXT,
            licence_status TEXT,
            nas_path       TEXT,
            frozen_at      TEXT NOT NULL,
            frozen_by      TEXT,
            UNIQUE(dataset_name, version)
        );
        CREATE INDEX IF NOT EXISTS idx_dsversions_name ON dataset_versions(dataset_name);
    """),
    (3, "auth: users + sessions + api_tokens + login_events", """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            display_name  TEXT,
            email         TEXT,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'viewer',   -- viewer | operator | admin
            active        INTEGER NOT NULL DEFAULT 1,
            created_at    TEXT NOT NULL,
            last_login    TEXT,
            failed_logins INTEGER NOT NULL DEFAULT 0
        );

        -- browser sessions. The cookie carries an opaque random token; we store
        -- only its sha256 hash. Expiring + revocable.
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            last_used  TEXT,
            revoked    INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

        -- DB-backed API tokens (for CLI/CI), scoped. Raw shown once; store hash.
        CREATE TABLE IF NOT EXISTS api_tokens (
            id         TEXT PRIMARY KEY,
            name       TEXT,
            owner      TEXT,
            token_hash TEXT UNIQUE NOT NULL,
            scopes     TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            last_used  TEXT,
            revoked    INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS login_events (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT NOT NULL,
            username TEXT,
            success  INTEGER NOT NULL,
            source   TEXT,
            detail   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_login_ts ON login_events(ts);
    """),
    (4, "jobs.gpu_count", """
        ALTER TABLE jobs ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1;
    """),
    (5, "job_templates", """
        -- Reusable submit presets. A template is a stored partial job request:
        -- an engineer picks one and only fills in what's left (e.g. dataset).
        CREATE TABLE IF NOT EXISTS job_templates (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            description   TEXT,
            job_type      TEXT NOT NULL,              -- training | processing
            docker_image  TEXT,                       -- optional short ref or full URL
            gpu_count     INTEGER NOT NULL DEFAULT 1,
            dataset_name  TEXT,
            config_yaml   TEXT,                        -- optional generated job.yaml body
            defaults      TEXT,                        -- JSON: extra CreateJob fields
            builtin       INTEGER NOT NULL DEFAULT 0,  -- 1 = seeded starter template
            created_by    TEXT,
            created_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_templates_type ON job_templates(job_type);
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
        "error", "output_path", "result_version", "gpu_count",
    )
    values = [(job.get("gpu_count") or 1) if c == "gpu_count" else job.get(c) for c in cols]
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
        # Higher priority first, then newest — queued work of equal priority is FIFO.
        rows = conn.execute(
            f"SELECT * FROM jobs {where} "
            f"ORDER BY priority DESC, created_at DESC LIMIT ? OFFSET ?",
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
# job templates (reusable submit presets)
# --------------------------------------------------------------------------- #
_TEMPLATE_COLS = (
    "id", "name", "description", "job_type", "docker_image", "gpu_count",
    "dataset_name", "config_yaml", "defaults", "builtin", "created_by", "created_at",
)


def insert_template(tpl: dict) -> None:
    values = [tpl.get(c) for c in _TEMPLATE_COLS]
    placeholders = ",".join("?" for _ in _TEMPLATE_COLS)
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO job_templates ({','.join(_TEMPLATE_COLS)}) VALUES ({placeholders})",
            values,
        )


def list_templates(job_type: str | None = None) -> list[dict]:
    clause = "WHERE job_type = ?" if job_type else ""
    params = [job_type] if job_type else []
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM job_templates {clause} ORDER BY builtin DESC, name ASC", params
        ).fetchall()
        return [dict(r) for r in rows]


def get_template(template_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM job_templates WHERE id = ?", (template_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_template(template_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM job_templates WHERE id = ? AND builtin = 0",
                           (template_id,))
        return cur.rowcount > 0


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


# --------------------------------------------------------------------------- #
# Dataset Factory: import workflows + frozen-version index
# --------------------------------------------------------------------------- #
DATASET_TERMINAL = ("frozen", "rejected", "failed", "cancelled")

_IMPORT_COLS = (
    "import_id", "idempotency_key", "source_type", "source_params", "dataset_name",
    "requested_version", "resolved_version", "state", "licence_name", "licence_status",
    "licence_by", "licence_at", "requested_by", "error", "created_at", "updated_at",
)


def create_dataset_import(imp: dict) -> None:
    """Insert a new import workflow and log its first state event.

    Raises sqlite3.IntegrityError if the idempotency_key already exists."""
    placeholders = ",".join("?" for _ in _IMPORT_COLS)
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO dataset_imports ({','.join(_IMPORT_COLS)}) VALUES ({placeholders})",
            [imp.get(c) for c in _IMPORT_COLS],
        )
        conn.execute(
            "INSERT INTO dataset_import_events(import_id, ts, state, detail) VALUES(?,?,?,?)",
            (imp["import_id"], now_iso(), imp["state"], None),
        )


def get_dataset_import(import_id: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute(
            "SELECT * FROM dataset_imports WHERE import_id = ?", (import_id,)
        ).fetchone()
        return dict(r) if r else None


def get_import_by_idempotency(key: str | None) -> dict | None:
    if not key:
        return None
    with _connect() as conn:
        r = conn.execute(
            "SELECT * FROM dataset_imports WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return dict(r) if r else None


def _import_filters(state=None, dataset_name=None):
    clauses, params = [], []
    for col, val in (("state", state), ("dataset_name", dataset_name)):
        if val:
            clauses.append(f"{col} = ?")
            params.append(val)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def list_dataset_imports(state=None, dataset_name=None, limit: int = 50, offset: int = 0) -> list[dict]:
    where, params = _import_filters(state, dataset_name)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM dataset_imports {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)],
        ).fetchall()
        return [dict(r) for r in rows]


def count_dataset_imports(state=None, dataset_name=None) -> int:
    where, params = _import_filters(state, dataset_name)
    with _connect() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM dataset_imports {where}", params).fetchone()[0]


def add_dataset_import_event(import_id: str, state: str, detail: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO dataset_import_events(import_id, ts, state, detail) VALUES(?,?,?,?)",
            (import_id, now_iso(), state, detail),
        )


def list_dataset_import_events(import_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, state, detail FROM dataset_import_events WHERE import_id = ? ORDER BY id",
            (import_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_dataset_import(import_id: str, state: str, **fields) -> bool:
    """Update an import's state (+ optional fields) and log a transition event.

    Terminal states are never overridden (a cancelled/failed import stays put)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT state FROM dataset_imports WHERE import_id = ?", (import_id,)
        ).fetchone()
        if row is None:
            return False
        prev = row["state"]
        if prev in DATASET_TERMINAL and state != prev:
            return False

        sets = {"state": state, "updated_at": now_iso()}
        for k in ("resolved_version", "licence_name", "licence_status",
                  "licence_by", "licence_at", "error"):
            if fields.get(k) is not None:
                sets[k] = fields[k]
        assignments = ",".join(f"{k} = ?" for k in sets)
        cur = conn.execute(
            f"UPDATE dataset_imports SET {assignments} WHERE import_id = ?",
            list(sets.values()) + [import_id],
        )
        if cur.rowcount > 0 and state != prev:
            conn.execute(
                "INSERT INTO dataset_import_events(import_id, ts, state, detail) VALUES(?,?,?,?)",
                (import_id, now_iso(), state, fields.get("error") or fields.get("detail")),
            )
        return cur.rowcount > 0


# ---- frozen-version index -------------------------------------------------- #
# A row is RESERVED at allocate time (nas_path NULL) and only becomes visible
# once finalize sets nas_path — so a crashed freeze never appears as a real
# version. UNIQUE(dataset_name, version) makes allocation atomic.
import re as _re

_VERSION_RE = _re.compile(r"^v(\d+)$")


def allocate_dataset_version(dataset_name: str, requested: str | None,
                             import_id: str, meta: dict) -> str:
    """Atomically reserve the next version (or an explicit one). Returns the
    version string. Raises ValueError if `requested` already exists."""
    conn = _lock_conn()                       # autocommit; we drive BEGIN IMMEDIATE
    try:
        conn.execute("BEGIN IMMEDIATE;")
        rows = conn.execute(
            "SELECT version FROM dataset_versions WHERE dataset_name = ?", (dataset_name,)
        ).fetchall()
        existing = {r["version"] for r in rows}
        if requested:
            if requested in existing:
                conn.execute("COMMIT;")
                raise ValueError(f"version {requested} already exists")
            version = requested
        else:
            nums = [int(m.group(1)) for v in existing if (m := _VERSION_RE.match(v))]
            version = f"v{(max(nums) + 1) if nums else 1}"
        conn.execute(
            "INSERT INTO dataset_versions(dataset_name, version, task, import_id, "
            "content_digest, image_count, class_count, licence_name, licence_status, "
            "nas_path, frozen_at, frozen_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (dataset_name, version, meta.get("task"), import_id, None,
             meta.get("image_count"), meta.get("class_count"),
             meta.get("licence_name"), meta.get("licence_status"),
             None, now_iso(), meta.get("frozen_by")),
        )
        conn.execute("COMMIT;")
        return version
    finally:
        conn.close()


def finalize_dataset_version(dataset_name: str, version: str, nas_path: str,
                             content_digest: str, image_count: int | None = None,
                             class_count: int | None = None) -> bool:
    """Publish a reserved version: set nas_path (makes it visible) + digest."""
    sets = {"nas_path": nas_path, "content_digest": content_digest}
    if image_count is not None:
        sets["image_count"] = image_count
    if class_count is not None:
        sets["class_count"] = class_count
    assignments = ",".join(f"{k} = ?" for k in sets)
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE dataset_versions SET {assignments} WHERE dataset_name = ? AND version = ?",
            list(sets.values()) + [dataset_name, version],
        )
        return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# auth: users, sessions, api tokens, login events
# --------------------------------------------------------------------------- #
_USER_PUBLIC = ("id", "username", "display_name", "email", "role", "active",
                "created_at", "last_login")


def count_users() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_user(username: str, password_hash: str, role: str = "viewer",
                display_name: str | None = None, email: str | None = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users(username, display_name, email, password_hash, role, "
            "active, created_at) VALUES(?,?,?,?,?,1,?)",
            (username, display_name, email, password_hash, role, now_iso()),
        )
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(r) if r else None


def get_user_by_id(user_id: int) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(r) if r else None


def user_public(user: dict) -> dict:
    return {k: user.get(k) for k in _USER_PUBLIC}


def set_user_password(user_id: int, password_hash: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))


def note_login_result(user_id: int, success: bool) -> None:
    with _connect() as conn:
        if success:
            conn.execute("UPDATE users SET last_login = ?, failed_logins = 0 WHERE id = ?",
                         (now_iso(), user_id))
        else:
            conn.execute("UPDATE users SET failed_logins = failed_logins + 1 WHERE id = ?",
                         (user_id,))


def record_login(username: str | None, success: bool, source: str = "api",
                 detail: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO login_events(ts, username, success, source, detail) VALUES(?,?,?,?,?)",
            (now_iso(), username, 1 if success else 0, source, detail),
        )


def list_users() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY username").fetchall()
        return [user_public(dict(r)) for r in rows]


def update_user(user_id: int, **fields) -> bool:
    allowed = {k: v for k, v in fields.items()
               if k in ("role", "active", "display_name", "email") and v is not None}
    if not allowed:
        return False
    assignments = ",".join(f"{k} = ?" for k in allowed)
    with _connect() as conn:
        cur = conn.execute(f"UPDATE users SET {assignments} WHERE id = ?",
                           list(allowed.values()) + [user_id])
        return cur.rowcount > 0


def count_active_admins() -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]


def revoke_user_sessions(user_id: int) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE sessions SET revoked=1 WHERE user_id=? AND (revoked IS NULL OR revoked=0)",
            (user_id,))
        return cur.rowcount


def count_active_sessions_for_user(user_id: int) -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id=? AND (revoked IS NULL OR revoked=0) "
            "AND expires_at > ?", (user_id, now_iso())).fetchone()[0]


def anonymise_user(user_id: int, anon_username: str, password_hash: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET username=?, display_name=NULL, email=NULL, password_hash=?, "
            "active=0 WHERE id=?", (anon_username, password_hash, user_id))
        return cur.rowcount > 0


def delete_user(user_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        return cur.rowcount > 0


def create_api_token(token_id: str, name: str, owner: str, token_hash: str,
                     scopes: str, expires_at: str | None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO api_tokens(id, name, owner, token_hash, scopes, created_at, expires_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (token_id, name, owner, token_hash, scopes, now_iso(), expires_at),
        )


def list_api_tokens() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, owner, scopes, created_at, expires_at, last_used, revoked "
            "FROM api_tokens ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def revoke_api_token(token_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE api_tokens SET revoked = 1 WHERE id = ?", (token_id,))
        return cur.rowcount > 0


def create_session(token_hash: str, user_id: int, expires_at: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(token_hash, user_id, created_at, expires_at, last_used) "
            "VALUES(?,?,?,?,?)",
            (token_hash, user_id, now_iso(), expires_at, now_iso()),
        )


def get_session(token_hash: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
        return dict(r) if r else None


def touch_session(token_hash: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET last_used = ? WHERE token_hash = ?",
                     (now_iso(), token_hash))


def revoke_session(token_hash: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET revoked = 1 WHERE token_hash = ?", (token_hash,))


def get_api_token(token_hash: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM api_tokens WHERE token_hash = ?", (token_hash,)).fetchone()
        return dict(r) if r else None


def register_backfill_version(dataset_name: str, version: str, nas_path: str, meta: dict) -> bool:
    """Index a pre-existing frozen version (legacy backfill). Inserts a visible
    row (nas_path set) if that (name, version) isn't already indexed. Returns
    True if inserted, False if it already existed. Never touches files."""
    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM dataset_versions WHERE dataset_name = ? AND version = ?",
            (dataset_name, version),
        ).fetchone()
        if exists:
            return False
        conn.execute(
            "INSERT INTO dataset_versions(dataset_name, version, task, import_id, "
            "content_digest, image_count, class_count, licence_name, licence_status, "
            "nas_path, frozen_at, frozen_by) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (dataset_name, version, meta.get("task"), None, meta.get("content_digest"),
             meta.get("image_count"), meta.get("class_count"), meta.get("licence_name"),
             meta.get("licence_status"), nas_path, meta.get("frozen_at") or now_iso(), "backfill"),
        )
        return True


def list_datasets() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT dataset_name, COUNT(*) AS versions, MAX(frozen_at) AS latest_frozen_at "
            "FROM dataset_versions WHERE nas_path IS NOT NULL "
            "GROUP BY dataset_name ORDER BY dataset_name"
        ).fetchall()
        return [dict(r) for r in rows]


def list_dataset_versions(dataset_name: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM dataset_versions WHERE dataset_name = ? AND nas_path IS NOT NULL "
            "ORDER BY frozen_at DESC",
            (dataset_name,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_dataset_version(dataset_name: str, version: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute(
            "SELECT * FROM dataset_versions WHERE dataset_name = ? AND version = ? "
            "AND nas_path IS NOT NULL",
            (dataset_name, version),
        ).fetchone()
        return dict(r) if r else None
