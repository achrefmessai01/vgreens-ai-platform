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
