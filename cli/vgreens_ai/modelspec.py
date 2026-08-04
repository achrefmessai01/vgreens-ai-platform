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
#
# NOTE: the `ultralytics_*` presets are LEGACY. Ultralytics is AGPL-3.0, which is
# commercially risky for a proprietary platform. They are kept only so previously
# generated specs still validate. Prefer the `rfdetr_*` presets (RF-DETR is
# Apache-2.0) for new detection models.
PRESETS: dict[str, dict] = {
    # --- preferred: RF-DETR (Apache-2.0) ---
    "rfdetr_detection": {"task": "object_detection", "generates_working_code": True},
    "rfdetr_segmentation": {"task": "segmentation", "generates_working_code": False},
    "cnn_classification": {"task": "classification", "generates_working_code": False},
    # --- legacy: Ultralytics (AGPL-3.0, commercially risky) ---
    "ultralytics_detection": {"task": "object_detection", "generates_working_code": True},
    "ultralytics_classification": {"task": "classification", "generates_working_code": True},
    "ultralytics_segmentation": {"task": "segmentation", "generates_working_code": True},
    "custom_python": {"task": "custom", "generates_working_code": True},
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

    # Optional batching controls (RF-DETR native batch / grad-accumulation). Not
    # required; when present each must be a positive int. effective_batch =
    # batch_size x grad_accumulation_steps.
    for key in ("batch_size", "grad_accumulation_steps", "effective_batch"):
        if key in training and training[key] is not None:
            val = training[key]
            _req(errors, isinstance(val, int) and not isinstance(val, bool) and val > 0,
                 f"training.{key} must be a positive integer")

    repo = spec.get("repository") or {}
    _req(errors, bool(repo.get("name")), "repository.name is required")

    # Optional extra Python dependencies installed into the model image at build
    # time (a requirements.txt is generated from this list). Must be a list of
    # non-empty pip requirement strings when present.
    deps = spec.get("dependencies")
    if deps is not None:
        if not isinstance(deps, list) or not all(
                isinstance(d, str) and d.strip() for d in deps):
            errors.append("dependencies must be a list of non-empty pip requirement strings")

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
