"""Image auto-resolution on job submit (training + processing).

When docker_image is omitted, the API resolves it from the model registry:
  - processing -> the ACTIVE version's image;
  - training   -> the LATEST version's image (or an explicit model_version).
An explicit docker_image always wins; an unresolvable model returns 422.
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# Defaults for a standalone run; setdefault so we never clobber a DB path that an
# earlier test module in the same process already bound at import time.
os.environ.setdefault("JOBS_DB_PATH", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("TOKENS", "svc:statictok")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "adminpass1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # api/

from fastapi.testclient import TestClient  # noqa: E402
import db                # noqa: E402
import authn             # noqa: E402
import broker            # noqa: E402
import models_registry   # noqa: E402
import main              # noqa: E402

# REGISTRY_ROOT / NAS_JOBS_ROOT are read at import time, so patch the module
# globals at RUNTIME — this is robust to test import order in the full suite.
REG = tempfile.mkdtemp()
models_registry.REGISTRY_ROOT = REG
main.NAS_JOBS_ROOT = tempfile.mkdtemp()
broker.publish = lambda job: None          # never touch RabbitMQ in tests

db.init_db()
authn.seed_admin_if_empty()
client = TestClient(main.app)
ADMIN_PW = os.environ["ADMIN_PASSWORD"]


def _seed(model: str, version: str, image: str, active: bool = False, created: str = "2026-01-01T00:00:00+00:00"):
    vdir = Path(REG) / model / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "metadata.json").write_text(json.dumps(
        {"model": model, "version": version, "docker_image": image, "created_at": created}))
    (vdir / "status.json").write_text(json.dumps({"status": "validated"}))
    if active:
        (Path(REG) / model / "active.json").write_text(json.dumps({"active_version": version}))


def _login():
    client.cookies.clear()
    assert client.post("/auth/login", json={"username": "admin", "password": ADMIN_PW}).status_code == 200


def _image_of(job_id: str) -> str:
    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200, r.text
    return r.json()["docker_image"]


def test_training_resolves_latest_image():
    _seed("berry", "v1", "registry.vgreens.lan/vgreens-models/berry:v1", created="2026-01-01T00:00:00+00:00")
    _seed("berry", "v2", "registry.vgreens.lan/vgreens-models/berry:v2", created="2026-02-01T00:00:00+00:00")
    _login()
    r = client.post("/jobs", json={"job_type": "training", "model_name": "berry",
                                   "dataset_name": "demo_berries", "dataset_version": "v1"})
    assert r.status_code == 201, r.text
    assert _image_of(r.json()["job_id"]).endswith("berry:v2")   # latest by created_at


def test_processing_resolves_active_image_not_latest():
    _seed("dis", "v1", "registry.vgreens.lan/vgreens-models/dis:v1", active=True, created="2026-01-01T00:00:00+00:00")
    _seed("dis", "v2", "registry.vgreens.lan/vgreens-models/dis:v2", created="2026-03-01T00:00:00+00:00")
    _login()
    r = client.post("/jobs", json={"job_type": "processing", "model_name": "dis"})
    assert r.status_code == 201, r.text
    assert _image_of(r.json()["job_id"]).endswith("dis:v1")     # active, even though v2 is newer


def test_explicit_image_wins():
    _login()
    r = client.post("/jobs", json={"job_type": "training", "model_name": "berry",
                                   "docker_image": "custom/thing:x", "dataset_name": "d",
                                   "dataset_version": "v1"})
    assert r.status_code == 201, r.text
    assert _image_of(r.json()["job_id"]) == "custom/thing:x"


def test_explicit_model_version_image():
    _login()
    r = client.post("/jobs", json={"job_type": "training", "model_name": "berry",
                                   "model_version": "v1", "dataset_name": "d",
                                   "dataset_version": "v1"})
    assert r.status_code == 201, r.text
    assert _image_of(r.json()["job_id"]).endswith("berry:v1")


def test_unresolvable_returns_422():
    _login()
    r = client.post("/jobs", json={"job_type": "training", "model_name": "brand_new_model",
                                   "dataset_name": "d", "dataset_version": "v1"})
    assert r.status_code == 422, r.text
    assert "docker_image" in r.json()["detail"]


def test_training_blank_version_resolves_latest_frozen():
    # seed two frozen versions; newest by frozen_at wins when the caller omits it
    import db as _db
    _db.register_backfill_version("berry", "v1", "/nas/berry/v1",
                                  {"frozen_at": "2026-01-01T00:00:00+00:00"})
    _db.register_backfill_version("berry", "v2", "/nas/berry/v2",
                                  {"frozen_at": "2026-02-01T00:00:00+00:00"})
    _seed("berry", "v1", "registry.vgreens.lan/vgreens-models/berry:v1")  # image to resolve
    _login()
    r = client.post("/jobs", json={"job_type": "training", "model_name": "berry",
                                   "dataset_name": "berry"})   # NO dataset_version
    assert r.status_code == 201, r.text
    job = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert job["dataset_version"] == "v2"    # latest frozen, resolved server-side


def test_training_no_frozen_version_422():
    _login()
    r = client.post("/jobs", json={"job_type": "training", "model_name": "berry",
                                   "dataset_name": "never_frozen_ds"})   # no versions
    assert r.status_code == 422, r.text
    assert "frozen version" in r.json()["detail"]
