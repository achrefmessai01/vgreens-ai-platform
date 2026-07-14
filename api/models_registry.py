"""Model registry management — the production-control plane (Phase 2.3).

Reads/writes the file-based registry on the NAS:

  ai/model_registry/<model>/
    versions/<version>/{model files, metadata.json, status.json, ...}
    active.json            atomic pointer to the version serving production
    promotions.jsonl       append-only audit log (who/when/from->to/reason)

Rules:
- Only a `validated` version can be promoted.
- `active.json` is written atomically (temp file + os.replace) — no symlinks on SMB.
- Rollback targets `previous_stable` by default, or an explicit version.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_ROOT = os.environ.get(
    "NAS_MODEL_REGISTRY", "/mnt/vgreens_storage/ai/model_registry"
)


class RegistryError(Exception):
    """Raised for invalid promote/rollback requests (mapped to HTTP 409)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_dir(model: str) -> Path:
    return Path(REGISTRY_ROOT) / model


def _versions_dir(model: str) -> Path:
    return _model_dir(model) / "versions"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def list_models() -> list[str]:
    root = Path(REGISTRY_ROOT)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "versions").is_dir())


def get_active(model: str) -> dict | None:
    p = _model_dir(model) / "active.json"
    return _read_json(p) if p.is_file() else None


def version_status(model: str, version: str) -> dict:
    return _read_json(_versions_dir(model) / version / "status.json")


def list_versions(model: str) -> list[dict]:
    vd = _versions_dir(model)
    if not vd.is_dir():
        return []
    active = (get_active(model) or {}).get("active_version")
    out = []
    for d in sorted(vd.iterdir()):
        if not d.is_dir():
            continue
        st = _read_json(d / "status.json")
        md = _read_json(d / "metadata.json")
        out.append({
            "version": d.name,
            "status": st.get("status"),
            "reason": st.get("reason"),
            "metrics": md.get("metrics", {}),
            "active": d.name == active,
        })
    return out


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic rename on the same filesystem; no symlinks


def _append_audit(model: str, entry: dict) -> None:
    with open(_model_dir(model) / "promotions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def promote(model: str, version: str, by: str, reason: str | None = None) -> dict:
    vdir = _versions_dir(model) / version
    if not vdir.is_dir():
        raise RegistryError(f"version {version!r} not found for model {model!r}")
    status = version_status(model, version).get("status")
    if status != "validated":
        raise RegistryError(
            f"cannot promote {version!r}: status is {status!r}, not 'validated'"
        )
    prev = (get_active(model) or {}).get("active_version")
    active = {
        "active_version": version,
        "previous_stable": prev,
        "updated_at": _now(),
        "updated_by": by,
        "reason": reason,
    }
    _write_atomic(_model_dir(model) / "active.json", active)
    _append_audit(model, {"ts": _now(), "action": "promote", "model": model,
                          "from": prev, "to": version, "by": by, "reason": reason})
    return active


def rollback(model: str, to: str | None, by: str, reason: str | None = None) -> dict:
    current = get_active(model)
    if not current:
        raise RegistryError(f"no active version for {model!r} to roll back from")
    target = to or current.get("previous_stable")
    if not target:
        raise RegistryError("no previous_stable to roll back to — specify --to <version>")
    if not (_versions_dir(model) / target).is_dir():
        raise RegistryError(f"rollback target {target!r} not found")
    prev = current.get("active_version")
    active = {
        "active_version": target,
        "previous_stable": prev,
        "updated_at": _now(),
        "updated_by": by,
        "reason": reason or "rollback",
    }
    _write_atomic(_model_dir(model) / "active.json", active)
    _append_audit(model, {"ts": _now(), "action": "rollback", "model": model,
                          "from": prev, "to": target, "by": by, "reason": reason or "rollback"})
    return active
