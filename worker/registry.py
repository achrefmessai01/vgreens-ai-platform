"""Model registry + validation (Phase 2.2).

On a successful *training* job, publish the model to the NAS registry:

  ai/model_registry/<model>/versions/<YYYY-MM-DD_NNN>/
    <model files...>        (whatever the training wrote to /out)
    eval/metrics.json       (achieved metrics, written by the job)
    evaluation_report.md
    config.yaml             (the job config = thresholds etc.)
    metadata.json           (git commit, image, dataset, metrics, gpu, duration, ...)
    status.json             ({"status":"validated"} | {"status":"rejected","reason":...})

Validation gate: compare the achieved metrics against thresholds from the job
config. `thresholds` = minimums (metric must be >= value); optional
`thresholds_max` = maximums (metric must be <= value, e.g. latency/loss).
No thresholds defined => validated by default (nothing to check against).

Rejected versions are kept forever (never deleted, never served).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

REGISTRY_ROOT = os.environ.get(
    "NAS_MODEL_REGISTRY", "/mnt/vgreens_storage/ai/model_registry"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_version(model_dir: Path) -> str:
    """YYYY-MM-DD_NNN, NNN = count of existing versions today + 1."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    versions = model_dir / "versions"
    n = 1
    if versions.is_dir():
        n = sum(1 for d in versions.iterdir() if d.is_dir() and d.name.startswith(day)) + 1
    return f"{day}_{n:03d}"


def _load_config(config_path: str | None) -> dict:
    if not config_path or not Path(config_path).is_file() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(Path(config_path).read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _validate(metrics: dict, config: dict) -> tuple[str, str | None]:
    """Return (status, reason). status ∈ {validated, rejected}."""
    mins = config.get("thresholds") or {}
    maxs = config.get("thresholds_max") or {}

    if not mins and not maxs:
        return "validated", None  # nothing to check against

    for name, floor in mins.items():
        val = metrics.get(name)
        if val is None:
            return "rejected", f"missing metric {name!r}"
        if float(val) < float(floor):
            return "rejected", f"{name} {val} < {floor}"
    for name, ceil in maxs.items():
        val = metrics.get(name)
        if val is None:
            return "rejected", f"missing metric {name!r}"
        if float(val) > float(ceil):
            return "rejected", f"{name} {val} > {ceil}"
    return "validated", None


def publish(job: dict, out_dir: Path, config_path: str | None,
            gpu_id: int, worker: str, duration_s: float | None) -> dict:
    """Publish /out to the registry and validate. Returns a summary dict."""
    model = job["model_name"]
    model_dir = Path(REGISTRY_ROOT) / model
    version = _next_version(model_dir)
    vdir = model_dir / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)

    # 1. copy everything the training produced
    if out_dir.exists():
        shutil.copytree(out_dir, vdir, dirs_exist_ok=True)

    # 2. keep the job config alongside (thresholds live here)
    config = _load_config(config_path)
    if config_path and Path(config_path).is_file():
        shutil.copyfile(config_path, vdir / "config.yaml")

    # 3. read achieved metrics
    metrics: dict = {}
    metrics_file = vdir / "eval" / "metrics.json"
    if metrics_file.is_file():
        try:
            metrics = json.loads(metrics_file.read_text())
        except json.JSONDecodeError:
            metrics = {}

    # 4. validation gate
    status, reason = _validate(metrics, config)

    # 5. metadata + status (audit trail)
    metadata = {
        "model": model,
        "version": version,
        "job_id": job["job_id"],
        "docker_image": job.get("docker_image"),
        "git_commit": job.get("git_commit"),
        "dataset_name": job.get("dataset_name"),
        "dataset_version": job.get("dataset_version"),
        "metrics": metrics,
        "thresholds": config.get("thresholds") or {},
        "gpu": gpu_id,
        "worker": worker,
        "duration_seconds": round(duration_s, 1) if duration_s is not None else None,
        "created_at": _now(),
        "status": status,
    }
    (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    status_doc = {"status": status}
    if reason:
        status_doc["reason"] = reason
    (vdir / "status.json").write_text(json.dumps(status_doc, indent=2))

    return {"version": version, "status": status, "reason": reason, "path": str(vdir)}
