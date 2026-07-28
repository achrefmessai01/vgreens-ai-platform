"""Job results/artifacts/logs endpoints and model delete (archive-to-trash).

Covers:
  - GET /jobs/{id}/result       training metrics + processing report + artifact list
  - GET /jobs/{id}/logs         tail of the container log
  - GET /jobs/{id}/artifacts/download  streams a file, blocks path traversal
  - DELETE /models/{model}      archives to _trash, refuses active unless force
"""
import os
import sys
import json
import tempfile
from pathlib import Path

os.environ.setdefault("JOBS_DB_PATH", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("TOKENS", "svc:statictok")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "adminpass1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # api/

import pytest              # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import db                # noqa: E402
import authn             # noqa: E402
import broker            # noqa: E402
import models_registry   # noqa: E402
import main              # noqa: E402

# Own temp dirs, but re-point the shared module globals PER TEST (not at import),
# so we don't clobber another test module that patches the same globals.
REG = tempfile.mkdtemp()
NAS = tempfile.mkdtemp()
broker.publish = lambda job: None

db.init_db()
authn.seed_admin_if_empty()
client = TestClient(main.app)
ADMIN_PW = os.environ["ADMIN_PASSWORD"]


@pytest.fixture(autouse=True)
def _point_globals():
    models_registry.REGISTRY_ROOT = REG
    main.NAS_JOBS_ROOT = NAS
    yield


def _login():
    client.cookies.clear()
    assert client.post("/auth/login", json={"username": "admin", "password": ADMIN_PW}).status_code == 200


def _make_job(job_id: str, job_type: str, out_files: dict, log_text: str = "") -> str:
    """Insert a job row with an output_path folder and write files into out/."""
    folder = Path(NAS) / "2026" / job_id
    (folder / "out" / "eval").mkdir(parents=True, exist_ok=True)
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    for rel, content in out_files.items():
        p = folder / "out" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    if log_text:
        (folder / "logs" / "container.log").write_text(log_text)
    db.insert_job({
        "job_id": job_id, "job_type": job_type, "model_name": "m", "dataset_name": "d",
        "dataset_version": "v1", "tenant": None, "docker_image": "img:1", "config_path": None,
        "status": "succeeded", "worker": None, "gpu": None, "requested_by": "admin",
        "priority": 0, "created_at": db.now_iso(), "started_at": None, "finished_at": None,
        "error": None, "output_path": str(folder), "result_version": "2026-07-28_001",
        "gpu_count": 1,
    })
    return job_id


def test_training_result_exposes_metrics_and_artifacts():
    _login()
    _make_job("trn_res_1", "training", {
        "eval/metrics.json": json.dumps({"mAP50": 0.95, "precision": 0.94}),
        "evaluation_report.md": "# eval\n\nmAP: 0.95\n",
        "model.pt": "BINARYWEIGHTS",
    }, log_text="line1\nline2\nline3\n")
    r = client.get("/jobs/trn_res_1/result")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["metrics"]["mAP50"] == 0.95
    assert body["result_version"] == "2026-07-28_001"
    paths = {a["path"] for a in body["artifacts"]}
    assert "model.pt" in paths and "eval/metrics.json" in paths
    assert "eval" in body["evaluation_report_md"]


def test_processing_result_exposes_report():
    _login()
    _make_job("prc_res_1", "processing", {
        "report.json": json.dumps({"images": 12, "total_detections": 34,
                                    "by_class": {"ripe": 20, "unripe": 14}, "conf_threshold": 0.25}),
    })
    r = client.get("/jobs/prc_res_1/result")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"]["total_detections"] == 34
    assert body["report"]["by_class"]["ripe"] == 20


def test_logs_tail():
    _login()
    _make_job("trn_log_1", "training", {"model.pt": "x"}, log_text="a\nb\nc\nd\n")
    r = client.get("/jobs/trn_log_1/logs", params={"tail": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["lines"] == ["c", "d"]
    assert body["total_lines"] == 4


def test_download_streams_file():
    _login()
    _make_job("trn_dl_1", "training", {"model.pt": "WEIGHTS123"})
    r = client.get("/jobs/trn_dl_1/artifacts/download", params={"path": "model.pt"})
    assert r.status_code == 200
    assert r.content == b"WEIGHTS123"


def test_download_blocks_path_traversal():
    _login()
    _make_job("trn_dl_2", "training", {"model.pt": "x"})
    r = client.get("/jobs/trn_dl_2/artifacts/download", params={"path": "../../job.json"})
    assert r.status_code == 400, r.text


def test_result_unavailable_when_no_outputs():
    _login()
    db.insert_job({
        "job_id": "prc_empty", "job_type": "processing", "model_name": "m", "dataset_name": None,
        "dataset_version": None, "tenant": None, "docker_image": "img:1", "config_path": None,
        "status": "queued", "worker": None, "gpu": None, "requested_by": "admin", "priority": 0,
        "created_at": db.now_iso(), "started_at": None, "finished_at": None, "error": None,
        "output_path": None, "result_version": None, "gpu_count": 1,
    })
    r = client.get("/jobs/prc_empty/result")
    assert r.status_code == 200
    assert r.json()["available"] is False


def _seed_model(model: str, version: str, active: bool = False):
    vdir = Path(REG) / model / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "metadata.json").write_text(json.dumps({"model": model, "version": version}))
    (vdir / "status.json").write_text(json.dumps({"status": "validated"}))
    if active:
        (Path(REG) / model / "active.json").write_text(json.dumps({"active_version": version}))


def test_delete_model_archives_to_trash():
    _login()
    _seed_model("throwaway", "v1")
    assert "throwaway" in models_registry.list_models()
    r = client.delete("/models/throwaway")
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert "throwaway" not in models_registry.list_models()
    # archived copy exists under _trash
    trash = list((Path(REG) / "_trash").glob("throwaway__*"))
    assert trash and (trash[0] / "versions" / "v1").is_dir()


def test_delete_refuses_active_without_force():
    _login()
    _seed_model("livemodel", "v1", active=True)
    r = client.delete("/models/livemodel")
    assert r.status_code == 409, r.text
    assert "active" in r.json()["detail"]
    assert "livemodel" in models_registry.list_models()   # still there


def test_delete_active_with_force():
    _login()
    _seed_model("livemodel2", "v1", active=True)
    r = client.delete("/models/livemodel2", params={"force": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["forced"] is True
    assert "livemodel2" not in models_registry.list_models()


def test_delete_missing_model_404():
    _login()
    r = client.delete("/models/does_not_exist")
    assert r.status_code == 404, r.text
