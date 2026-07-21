"""model.yaml — the declarative source of truth for model onboarding.

This module validates a spec (schema_version 1) and describes the available
presets. Validation is dependency-light (no pydantic in the CLI): it returns a
list of human-readable errors; an empty list means the spec is valid.
"""
from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = 1
TASKS = ("object_detection", "classification", "segmentation", "custom")
GPU_COUNTS = (1, 2)

# preset -> {task it implements, whether it generates working code}
PRESETS: dict[str, dict] = {
    "ultralytics_detection": {"task": "object_detection", "generates_working_code": True},
    "ultralytics_classification": {"task": "classification", "generates_working_code": True},
    "ultralytics_segmentation": {"task": "segmentation", "generates_working_code": True},
    "custom_python": {"task": "custom", "generates_working_code": False},
}

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,62}$")


def _req(errors: list, cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def validate(spec: Any) -> list[str]:
    """Return a list of validation errors ([] == valid)."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be a mapping (YAML object)"]

    _req(errors, spec.get("schema_version") == SCHEMA_VERSION,
         f"schema_version must be {SCHEMA_VERSION}")

    model = spec.get("model") or {}
    _req(errors, isinstance(model, dict) and bool(model.get("name")),
         "model.name is required")
    if model.get("name") and not _NAME_RE.match(str(model["name"])):
        errors.append("model.name must be lowercase [a-z0-9_], start alphanumeric, <=63 chars")
    task = model.get("task")
    _req(errors, task in TASKS, f"model.task must be one of {TASKS}")

    template = spec.get("template") or {}
    preset = template.get("preset")
    _req(errors, preset in PRESETS, f"template.preset must be one of {tuple(PRESETS)}")
    if preset in PRESETS and task in TASKS:
        want = PRESETS[preset]["task"]
        _req(errors, want == task,
             f"template.preset '{preset}' implements task '{want}', but model.task is '{task}'")

    dataset = spec.get("dataset") or {}
    _req(errors, bool(dataset.get("name")), "dataset.name is required")
    _req(errors, bool(dataset.get("version")),
         "dataset.version is required (a frozen version like 'v1', or 'latest_frozen')")

    training = spec.get("training") or {}
    gpu = training.get("gpu_count", 1)
    _req(errors, gpu in GPU_COUNTS, f"training.gpu_count must be one of {GPU_COUNTS}")

    repo = spec.get("repository") or {}
    _req(errors, bool(repo.get("name")), "repository.name is required")

    return errors


def load_spec(path: str) -> Any:
    """Load a model.yaml from disk. Raises OSError / yaml.YAMLError."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def threshold_state(spec: dict) -> str:
    """Classify a spec's validation thresholds for promotion policy."""
    thresholds = ((spec.get("validation") or {}).get("thresholds")) or {}
    configured = {k: v for k, v in thresholds.items() if v is not None}
    return "configured" if configured else "thresholds_not_configured"
