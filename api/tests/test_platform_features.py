"""Tests for the 'implement all the ideas' feature set: templates (#1), job
priority ordering (#20), clone (#10), plan/pre-flight/command-preview/GPU advice
(#3 #6 #14 #15 #18), config generator (#5), progress (#7), explain (#9),
report (#19), compare (#11)."""
import os
import sys
import json
import tempfile

os.environ.setdefault("JOBS_DB_PATH", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "adminpass1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient  # noqa: E402
import db                # noqa: E402
import authn             # noqa: E402
import broker            # noqa: E402
import main              # noqa: E402
from pathlib import Path  # noqa: E402

main.NAS_JOBS_ROOT = tempfile.mkdtemp()
broker.publish = lambda job: None

db.init_db()
authn.seed_admin_if_empty()
client = TestClient(main.app)
ADMIN_PW = os.environ["ADMIN_PASSWORD"]


def _login():
    client.cookies.clear()
    assert client.post("/auth/login",
                       json={"username": "admin", "password": ADMIN_PW}).status_code == 200


# ---- templates (#1) --------------------------------------------------------
def test_template_crud_and_builtin_guard():
    _login()
    r = client.post("/templates", json={
        "name": "Detection training", "job_type": "training",
        "docker_image": "berry:latest", "gpu_count": 1,
        "dataset_name": "knaup", "config_yaml": "train:\n  epochs: 50\n",
        "defaults": {"tenant": "research"}})
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    assert client.get("/templates").json()["templates"]
    got = client.get(f"/templates/{tid}").json()
    assert got["name"] == "Detection training" and got["dataset_name"] == "knaup"
    assert client.delete(f"/templates/{tid}").json()["deleted"] is True
    assert client.get(f"/templates/{tid}").status_code == 404


def test_templates_require_auth():
    client.cookies.clear()
    assert client.post("/templates", json={"name": "x", "job_type": "training"}).status_code in (401, 403)


# ---- config generator (#5) -------------------------------------------------
def test_config_generate():
    _login()
    r = client.post("/config/generate", json={"epochs": 25, "batch_size": 8,
                                               "learning_rate": 0.0005})
    assert r.status_code == 200
    y = r.json()["yaml"]
    assert "epochs: 25" in y and "batch_size: 8" in y


# ---- plan / preflight / command preview / gpu advice (#3 #6 #14 #15 #18) ---
def test_plan_reports_checks_and_preview():
    _login()
    r = client.post("/jobs/plan", json={"job_type": "processing",
                                        "model_name": "berry",
                                        "docker_image": "berry:v2", "gpu_count": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    names = {c["name"] for c in body["checks"]}
    assert {"image", "nas_mounted", "worker_online", "gpu"} <= names
    assert body["resolved_image"].endswith("/berry:v2")
    assert "docker run" in body["command_preview"]
    assert body["recommended_gpu"] == 1  # processing -> 1
    assert "output_path_preview" in body


# ---- priority ordering (#20) ----------------------------------------------
def test_priority_ordering():
    _login()
    # low priority first, then a high-priority job; high should sort first.
    client.post("/jobs", json={"job_type": "processing", "model_name": "m1",
                               "docker_image": "x:1", "priority": 0})
    client.post("/jobs", json={"job_type": "processing", "model_name": "m2",
                               "docker_image": "x:1", "priority": 50})
    jobs = client.get("/jobs?limit=5").json()
    assert jobs[0]["priority"] == 50  # highest priority first


# ---- clone (#10) -----------------------------------------------------------
def test_clone_returns_prefill_body():
    _login()
    sub = client.post("/jobs", json={"job_type": "processing", "model_name": "cl",
                                     "docker_image": "cl:9", "tenant": "t1"})
    jid = sub.json()["job_id"]
    r = client.get(f"/jobs/{jid}/clone")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "cl" and body["tenant"] == "t1"
    assert body["docker_image"].endswith("cl:9")


# ---- progress (#7) + explain (#9) + report (#19) ---------------------------
def _seed_out(jid, files: dict):
    job = db.get_job(jid)
    out = Path(job["output_path"]) / "out"
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def test_progress_parses_metrics_jsonl():
    _login()
    jid = client.post("/jobs", json={"job_type": "training", "model_name": "p",
                                     "docker_image": "p:1", "dataset_name": "d",
                                     "dataset_version": "v1"}).json()["job_id"]
    _seed_out(jid, {"metrics.jsonl":
        '{"epoch":1,"total_epochs":10,"loss":0.9,"ts":"2026-08-07T10:00:00"}\n'
        '{"epoch":4,"total_epochs":10,"loss":0.4,"ts":"2026-08-07T10:03:00"}\n'})
    r = client.get(f"/jobs/{jid}/progress").json()
    assert r["available"] and r["current_epoch"] == 4 and r["total_epochs"] == 10
    assert r["eta_seconds"] and r["eta_seconds"] > 0


def test_explain_matches_known_error():
    _login()
    jid = client.post("/jobs", json={"job_type": "processing", "model_name": "e",
                                     "docker_image": "e:1"}).json()["job_id"]
    db.update_status(jid, "failed", error="RuntimeError: CUDA out of memory")
    r = client.get(f"/jobs/{jid}/explain").json()
    assert r["matched"] and "memory" in r["cause"].lower() and r["suggestions"]


def test_report_assembles_standard_fields():
    _login()
    jid = client.post("/jobs", json={"job_type": "training", "model_name": "r",
                                     "docker_image": "r:1", "dataset_name": "d",
                                     "dataset_version": "v1"}).json()["job_id"]
    _seed_out(jid, {"eval/metrics.json": json.dumps({"mAP50": 0.7})})
    r = client.get(f"/jobs/{jid}/report").json()
    assert r["job_id"] == jid and r["metrics"] == {"mAP50": 0.7}
    assert r["dataset"]["name"] == "d" and "generated_at" in r


# ---- compare (#11) ---------------------------------------------------------
def test_compare_multiple_runs():
    _login()
    ids = []
    for i, mp in enumerate([0.6, 0.8]):
        jid = client.post("/jobs", json={"job_type": "training", "model_name": f"c{i}",
                                         "docker_image": "c:1", "dataset_name": "d",
                                         "dataset_version": "v1"}).json()["job_id"]
        _seed_out(jid, {"eval/metrics.json": json.dumps({"mAP50": mp})})
        ids.append(jid)
    r = client.get("/experiments/compare?ids=" + ",".join(ids)).json()
    assert len(r["runs"]) == 2 and all(x["found"] for x in r["runs"])
    assert r["runs"][1]["metrics"]["mAP50"] == 0.8
