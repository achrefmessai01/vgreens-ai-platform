"""Safety-critical guard tests for the additions in this milestone:

  * safe user management (self-protection, last-admin protection, delete
    conflict codes, disable revokes sessions + blocks login, anonymise),
  * two-GPU / gpu_count validation on job submit,
  * the curated /system/timeseries range allowlist + auth.

These lock down the rules that must never silently regress: you can never
remove the final admin, you can't disable/delete yourself, and the browser
can't smuggle arbitrary PromQL through the timeseries endpoint.
"""
import os
import sys
import tempfile

os.environ.setdefault("JOBS_DB_PATH", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("TOKENS", "svc:statictok")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "adminpass1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # api/

from fastapi.testclient import TestClient  # noqa: E402
import authn  # noqa: E402
import db  # noqa: E402
import main  # noqa: E402

db.init_db()
authn.seed_admin_if_empty()
client = TestClient(main.app)
ADMIN_H = {"Authorization": "Bearer statictok"}


def _mk(username, role, pw="secret123"):
    u = db.get_user_by_username(username)
    if u is None:
        return db.create_user(username, authn.hash_password(pw), role=role)
    return u["id"]


# --------------------------------------------------------------------------- #
# gpu_count / two-GPU gate
# --------------------------------------------------------------------------- #
def test_gpu_count_out_of_range_is_422():
    r = client.post("/jobs", headers=ADMIN_H,
                    json={"job_type": "training", "model_name": "m", "docker_image": "i", "gpu_count": 3})
    assert r.status_code == 422, r.text


def test_two_gpu_requires_admin():
    _mk("oper_gpu", "operator")
    c = TestClient(main.app)
    c.post("/auth/login", json={"username": "oper_gpu", "password": "secret123"})
    r = c.post("/jobs", json={"job_type": "training", "model_name": "m", "docker_image": "i", "gpu_count": 2})
    assert r.status_code == 403, r.text


# --------------------------------------------------------------------------- #
# /system/timeseries — fixed allowlist, no arbitrary PromQL, auth required
# --------------------------------------------------------------------------- #
def test_timeseries_rejects_unknown_range():
    assert client.get("/system/timeseries", headers=ADMIN_H, params={"range": "7d"}).status_code == 422
    assert client.get("/system/timeseries", headers=ADMIN_H, params={"range": "; DROP"}).status_code == 422


def test_timeseries_ok_shape():
    r = client.get("/system/timeseries", headers=ADMIN_H, params={"range": "1h"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["range"] == "1h" and "available" in body and isinstance(body["series"], dict)
    # only the curated series keys are ever present
    assert set(body["series"]).issubset({
        "api_request_rate", "api_p95_seconds", "api_error_ratio",
        "queue_depth", "oldest_queued_seconds", "gpu_workers_online"})


def test_timeseries_requires_auth():
    assert client.get("/system/timeseries", params={"range": "1h"}).status_code == 401


# --------------------------------------------------------------------------- #
# self-protection
# --------------------------------------------------------------------------- #
def test_cannot_disable_or_delete_or_anonymise_self():
    uid = _mk("self_admin", "admin")
    c = TestClient(main.app)
    c.post("/auth/login", json={"username": "self_admin", "password": "secret123"})
    assert c.post(f"/admin/users/{uid}/disable").json()["detail"] == "CANNOT_DISABLE_SELF"
    assert c.request("DELETE", f"/admin/users/{uid}").json()["detail"] == "CANNOT_DELETE_SELF"
    assert c.post(f"/admin/users/{uid}/anonymise").json()["detail"] == "CANNOT_ANONYMISE_SELF"


# --------------------------------------------------------------------------- #
# disable revokes sessions + blocks login
# --------------------------------------------------------------------------- #
def test_disable_revokes_session_and_blocks_login():
    uid = _mk("victim_disable", "viewer")
    victim = TestClient(main.app)
    assert victim.post("/auth/login", json={"username": "victim_disable", "password": "secret123"}).status_code == 200
    assert victim.get("/auth/me").status_code == 200
    assert client.post(f"/admin/users/{uid}/disable", headers=ADMIN_H).status_code == 200
    # existing session is revoked immediately …
    assert victim.get("/auth/me").status_code == 401
    # … and a fresh login is refused. It returns 401 (not 403) on purpose: a disabled
    # account is indistinguishable from bad credentials, so we don't leak that it exists.
    fresh = TestClient(main.app)
    assert fresh.post("/auth/login", json={"username": "victim_disable", "password": "secret123"}).status_code == 401


# --------------------------------------------------------------------------- #
# delete conflict codes
# --------------------------------------------------------------------------- #
def test_delete_blocked_by_active_session_then_succeeds():
    uid = _mk("victim_del", "viewer")
    victim = TestClient(main.app)
    victim.post("/auth/login", json={"username": "victim_del", "password": "secret123"})
    r = client.request("DELETE", f"/admin/users/{uid}", headers=ADMIN_H)
    assert r.status_code == 409 and r.json()["detail"] == "USER_HAS_ACTIVE_SESSIONS"
    assert client.post(f"/admin/users/{uid}/force-logout", headers=ADMIN_H).status_code == 200
    assert client.request("DELETE", f"/admin/users/{uid}", headers=ADMIN_H).status_code == 200
    assert db.get_user_by_id(uid) is None


def test_anonymise_renames_and_blocks_old_login():
    uid = _mk("victim_anon", "viewer")
    assert client.post(f"/admin/users/{uid}/anonymise", headers=ADMIN_H).status_code == 200
    assert db.get_user_by_id(uid)["username"] == f"deleted_user_{uid}"
    assert client.post("/auth/login", json={"username": "victim_anon", "password": "secret123"}).status_code == 401


# --------------------------------------------------------------------------- #
# last-admin protection (must always keep one active admin)
# run last; restores any admin it disables so other suites are unaffected
# --------------------------------------------------------------------------- #
def test_last_admin_cannot_be_removed():
    solo = _mk("solo_admin", "admin")
    others = [u["id"] for u in client.get("/admin/users", headers=ADMIN_H).json()
              if u["role"] == "admin" and (u["active"] in (1, True)) and u["id"] != solo]
    try:
        for oid in others:
            assert client.post(f"/admin/users/{oid}/disable", headers=ADMIN_H).status_code == 200
        # solo_admin is now the only active admin — every removal path must refuse
        assert client.post(f"/admin/users/{solo}/disable", headers=ADMIN_H).json()["detail"] == "USER_IS_FINAL_ADMIN"
        assert client.request("DELETE", f"/admin/users/{solo}", headers=ADMIN_H).json()["detail"] == "USER_IS_FINAL_ADMIN"
        assert client.patch(f"/admin/users/{solo}", headers=ADMIN_H,
                            json={"role": "viewer"}).json()["detail"] == "USER_IS_FINAL_ADMIN"
    finally:
        for oid in others:
            client.post(f"/admin/users/{oid}/enable", headers=ADMIN_H)
