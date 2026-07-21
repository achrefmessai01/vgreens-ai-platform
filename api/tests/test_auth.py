"""Auth backend tests: bcrypt login, session cookie, /auth/me, change-password,
static-token identity, and that existing bearer-token routes still work."""
import os
import sys
import tempfile

os.environ["JOBS_DB_PATH"] = tempfile.mktemp(suffix=".db")
os.environ["TOKENS"] = "svc:statictok"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "adminpass1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # api/

from fastapi.testclient import TestClient  # noqa: E402
import authn  # noqa: E402
import db  # noqa: E402
import main  # noqa: E402

db.init_db()
authn.seed_admin_if_empty()
client = TestClient(main.app)


def test_login_me_logout():
    client.cookies.clear()
    r = client.post("/auth/login", json={"username": "admin", "password": "adminpass1"})
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "admin"
    me = client.get("/auth/me")
    assert me.status_code == 200 and me.json()["via"] == "session" and me.json()["username"] == "admin"
    client.post("/auth/logout")
    client.cookies.clear()
    assert client.get("/auth/me").status_code == 401


def test_bad_password_401():
    client.cookies.clear()
    assert client.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401


def test_static_token_identity():
    client.cookies.clear()
    r = client.get("/auth/me", headers={"Authorization": "Bearer statictok"})
    assert r.status_code == 200, r.text
    assert r.json()["via"] == "static" and r.json()["role"] == "admin"


def test_existing_route_bearer_still_works():
    client.cookies.clear()
    r = client.get("/jobs", headers={"Authorization": "Bearer statictok"}, params={"limit": 1})
    assert r.status_code == 200


def _mkuser(username, role, pw="secret123"):
    if db.get_user_by_username(username) is None:
        db.create_user(username, authn.hash_password(pw), role=role)


def test_rbac_viewer_can_read_not_write():
    _mkuser("vv", "viewer")
    client.cookies.clear()
    client.post("/auth/login", json={"username": "vv", "password": "secret123"})
    assert client.get("/jobs", params={"limit": 1}).status_code == 200          # read ok
    r = client.post("/jobs", json={"job_type": "training", "model_name": "m", "docker_image": "i"})
    assert r.status_code == 403                                                  # cannot submit
    client.cookies.clear()


def test_rbac_operator_submits_but_not_promote():
    _mkuser("oo", "operator")
    client.cookies.clear()
    client.post("/auth/login", json={"username": "oo", "password": "secret123"})
    r = client.post("/jobs", json={"job_type": "training", "model_name": "m", "docker_image": "i"})
    assert r.status_code not in (401, 403)          # auth passed (503 if no broker/NAS is fine)
    assert client.post("/models/m/promote", json={"version": "v1"}).status_code == 403
    client.cookies.clear()


def test_rbac_static_token_is_admin():
    client.cookies.clear()
    r = client.post("/models/nope/promote", json={"version": "v1"},
                    headers={"Authorization": "Bearer statictok"})
    assert r.status_code not in (401, 403)          # admin-equivalent; fails later (404/409), not on auth


ADMIN_H = {"Authorization": "Bearer statictok"}


def test_admin_user_management():
    client.cookies.clear()
    r = client.post("/admin/users", headers=ADMIN_H,
                    json={"username": "webviewer", "password": "viewerpass1", "role": "viewer"})
    assert r.status_code == 201, r.text
    uid = r.json()["id"]
    assert any(u["username"] == "webviewer" for u in client.get("/admin/users", headers=ADMIN_H).json())
    assert client.patch(f"/admin/users/{uid}", headers=ADMIN_H, json={"role": "operator"}).json()["role"] == "operator"
    assert client.post(f"/admin/users/{uid}/reset-password", headers=ADMIN_H,
                       json={"new_password": "newpass123"}).status_code == 200
    client.cookies.clear()
    assert client.post("/auth/login", json={"username": "webviewer", "password": "newpass123"}).status_code == 200
    client.cookies.clear()


def test_admin_token_lifecycle():
    client.cookies.clear()
    r = client.post("/admin/tokens", headers=ADMIN_H, json={"name": "ci", "scopes": "jobs:submit"})
    assert r.status_code == 201, r.text
    raw, tid = r.json()["token"], r.json()["id"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {raw}"})
    assert me.status_code == 200 and me.json()["via"] == "api_token"
    toks = client.get("/admin/tokens", headers=ADMIN_H).json()
    assert any(t["id"] == tid for t in toks) and all("token_hash" not in t for t in toks)
    assert client.delete(f"/admin/tokens/{tid}", headers=ADMIN_H).status_code == 200
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {raw}"}).status_code == 401


def test_admin_requires_admin_role():
    _mkuser("oo2", "operator")
    client.cookies.clear()
    client.post("/auth/login", json={"username": "oo2", "password": "secret123"})
    assert client.get("/admin/users").status_code == 403
    client.cookies.clear()


def test_change_password_roundtrip():
    client.cookies.clear()
    client.post("/auth/login", json={"username": "admin", "password": "adminpass1"})
    r = client.post("/auth/change-password", json={"old_password": "adminpass1", "new_password": "newpass12"})
    assert r.status_code == 200, r.text
    client.post("/auth/logout")
    client.cookies.clear()
    assert client.post("/auth/login", json={"username": "admin", "password": "adminpass1"}).status_code == 401
    assert client.post("/auth/login", json={"username": "admin", "password": "newpass12"}).status_code == 200
    # restore so the suite is order-independent
    client.post("/auth/change-password", json={"old_password": "newpass12", "new_password": "adminpass1"})
    client.cookies.clear()
