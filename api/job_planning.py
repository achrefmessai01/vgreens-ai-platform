"""Read-only job planning & diagnostics helpers.

Pure(ish) functions that power:
  * the pre-submit "plan" endpoint — recommended GPU (#15), docker command
    preview (#14), config.yaml generation (#5), image-format validation (#3);
  * post-run diagnostics — live progress parsing from metrics.jsonl (#7),
    rule-based failed-job explanation (#9), and the standard job report (#19).

Everything here is side-effect free and unit-testable without a running stack.
System-state pre-flight checks (NAS mounted, worker online, disk space, queue
reachable) live in main.py because they need live handles; this module holds the
logic that can be tested in isolation.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

# Keep in sync with worker/runner.py defaults.
SHM_SIZE = os.environ.get("SHM_SIZE", "8g")
NAS_DATASETS_ROOT = os.environ.get("NAS_DATASETS", "/mnt/vgreens_storage/ai/datasets")

# Above this many training images we suggest both GPUs (data-parallel pays off).
LARGE_DATASET_IMAGES = int(os.environ.get("VG_LARGE_DATASET_IMAGES", "4000"))


# --------------------------------------------------------------------------- #
# #15 recommended GPU
# --------------------------------------------------------------------------- #
def recommend_gpu(job_type: str, image_count: Optional[int] = None,
                  requested: int = 1) -> dict:
    """Suggest a GPU count with a human reason. Never *forces* a choice."""
    if job_type == "processing":
        rec, reason = 1, "Inference/processing runs on a single GPU."
    elif image_count is not None and image_count >= LARGE_DATASET_IMAGES:
        rec = 2
        reason = (f"Large dataset ({image_count} images) — 2 GPUs (distributed "
                  f"training) can roughly halve wall-clock time.")
    elif image_count is not None:
        rec = 1
        reason = (f"Small/medium dataset ({image_count} images) — 1 GPU is "
                  f"typically enough and keeps the second GPU free for production.")
    else:
        rec = 1
        reason = "Default to 1 GPU; 2 GPUs only help if your training code runs DDP."
    out = {"recommended_gpu": rec, "reason": reason}
    if requested == 2:
        out["note"] = ("2 GPUs are only *used* if the training code runs distributed "
                       "(DDP/torchrun). Onboarded models do this automatically; a "
                       "custom image must launch its own multi-process training.")
    return out


# --------------------------------------------------------------------------- #
# #14 docker command preview
# --------------------------------------------------------------------------- #
def dataset_dir(dataset_name: Optional[str], dataset_version: Optional[str]) -> Optional[str]:
    if not dataset_name:
        return None
    v = dataset_version or "<version>"
    return f"{NAS_DATASETS_ROOT}/{dataset_name}/{v}"


def build_command_preview(job: dict) -> str:
    """Reconstruct the `docker run` the worker will build (worker/runner.py).

    Representative, not executed: real host paths (NVMe workdir, exact mounts)
    are chosen by the worker at run time, but the shape — GPU flags, mounts and
    env — matches so engineers can see and trust what will run.
    """
    gpu_count = int(job.get("gpu_count") or 1)
    ids = "0,1" if gpu_count == 2 else "0"
    gpus_flag = f'--gpus "device={ids}"' if gpu_count == 2 else f"--gpus device={ids}"
    parts = [
        "docker run --rm",
        gpus_flag,
        f"--shm-size={SHM_SIZE}",
        f"--name vgreens-job-{job.get('job_id', '<job_id>')}",
    ]
    env = {
        "JOB_ID": job.get("job_id", "<job_id>"),
        "JOB_TYPE": job.get("job_type", ""),
        "MODEL_NAME": job.get("model_name", ""),
        "MODEL_VERSION": job.get("model_version") or job.get("result_version") or "",
        "DATASET_VERSION": job.get("dataset_version") or "",
        "TENANT": job.get("tenant") or "",
        "GPU_COUNT": str(gpu_count),
    }
    empty = '""'
    for k, v in env.items():
        parts.append(f"-e {k}={v or empty}")
    ddir = dataset_dir(job.get("dataset_name"), job.get("dataset_version"))
    if ddir:
        parts.append(f"-v {ddir}:/data:ro")
    parts.append("-v <job_out>/out:/out")
    parts.append("-v <job_config>/job.yaml:/config/job.yaml:ro")
    if job.get("job_type") == "processing":
        parts.append("-v <active_model_dir>:/model:ro")
    parts.append(job.get("docker_image", "<image>"))
    line = " \\\n  ".join(parts)
    if gpu_count == 2 and job.get("job_type") == "training":
        line += ("\n# entrypoint relaunches training under torchrun "
                 "(--nproc_per_node=2) for DDP")
    return line


# --------------------------------------------------------------------------- #
# #5 config.yaml generator
# --------------------------------------------------------------------------- #
def generate_config_yaml(params: dict) -> str:
    """Emit a job.yaml body from hyperparameters. Hand-rolled YAML so this works
    even where PyYAML isn't installed in the API venv."""
    def num(key, default):
        v = params.get(key, default)
        return v if v is not None else default

    epochs = int(num("epochs", 50))
    batch_size = int(num("batch_size", 16))
    lr = float(num("learning_rate", 0.001))
    image_size = int(num("image_size", 640))
    optimizer = str(params.get("optimizer", "adamw"))
    augment = bool(params.get("augment", True))
    gpus = int(num("gpus", 1))
    arch = params.get("model_architecture")

    lines = ["# Generated by the vGreens config generator", "train:",
             f"  epochs: {epochs}",
             f"  batch_size: {batch_size}",
             f"  learning_rate: {lr}",
             f"  image_size: {image_size}",
             f"  optimizer: {optimizer}",
             f"  augment: {str(augment).lower()}",
             f"  gpus: {gpus}"]
    if arch:
        lines.append(f"  model_architecture: {arch}")
    # Validation gates mirror the model-registry contract (thresholds = minimums).
    thresholds = params.get("thresholds") or {}
    if thresholds:
        lines.append("thresholds:")
        for k, v in thresholds.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# #3 image-ref format validation (existence check is best-effort, elsewhere)
# --------------------------------------------------------------------------- #
def validate_image_ref(full_ref: str) -> dict:
    """Structural validation of a fully-resolved image reference. Existence in
    the registry is NOT checked here (that needs a server-side token and is done
    separately, best-effort). Returns {ok, checked, detail}."""
    ref = (full_ref or "").strip()
    if not ref:
        return {"ok": False, "checked": True, "detail": "empty image reference"}
    # split optional tag/digest off the last path segment
    name = ref.rsplit("/", 1)[-1]
    has_tag = ("@" in name) or (":" in name)
    problems = []
    if " " in ref:
        problems.append("contains whitespace")
    if not has_tag:
        problems.append("no tag or digest (will default to :latest)")
    ok = not any(p != "no tag or digest (will default to :latest)" for p in problems)
    detail = "reference looks valid" if not problems else "; ".join(problems)
    return {"ok": ok, "checked": True, "detail": detail}


# --------------------------------------------------------------------------- #
# #7 live progress from metrics.jsonl
# --------------------------------------------------------------------------- #
def parse_progress(text: str) -> dict:
    """Parse a metrics.jsonl stream (one JSON object per line) into a progress
    summary: the per-epoch series, the latest row, and a naive ETA."""
    rows: list[dict] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    out: dict[str, Any] = {"available": bool(rows), "epochs": rows,
                           "latest": rows[-1] if rows else None,
                           "total_epochs": None, "current_epoch": None,
                           "eta_seconds": None}
    if not rows:
        return out
    latest = rows[-1]
    cur = latest.get("epoch")
    total = latest.get("total_epochs") or latest.get("epochs_total") or latest.get("total")
    out["current_epoch"] = cur
    out["total_epochs"] = total
    # ETA from wall-clock timestamps if present.
    ts = [r.get("ts") or r.get("time") or r.get("timestamp") for r in rows]
    ts = [t for t in ts if t]
    if total and cur and isinstance(cur, (int, float)) and cur > 0 and len(ts) >= 2:
        try:
            t0 = _parse_ts(ts[0])
            t1 = _parse_ts(ts[-1])
            if t0 and t1 and t1 > t0:
                per_epoch = (t1 - t0) / max(cur, 1)
                remaining = max(int(total) - int(cur), 0)
                out["eta_seconds"] = int(per_epoch * remaining)
        except Exception:  # noqa: BLE001 — ETA is best-effort
            pass
    return out


def _parse_ts(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# #9 rule-based failed-job explanation
# --------------------------------------------------------------------------- #
_RULES: list[tuple[tuple[str, ...], str, list[str]]] = [
    (("cuda out of memory", "outofmemoryerror", "cuda oom"),
     "The GPU ran out of memory during training.",
     ["Lower the batch size in your config (e.g. halve batch_size).",
      "Reduce image_size.",
      "Request 2 GPUs if your training code supports distributed training (DDP)."]),
    (("no such file or directory", "dataset path", "filenotfounderror", "does not exist"),
     "A required path (most likely the dataset) was not found inside the container.",
     ["Check the dataset exists and is frozen under the datasets NAS root.",
      "Confirm the dataset name and version on the job match a frozen version.",
      "Verify the NAS is mounted on the worker."]),
    (("manifest unknown", "pull access denied", "not found: manifest", "error pulling image",
      "repository does not exist"),
     "The Docker image could not be pulled from the registry.",
     ["Check the image tag exists in the registry (vgreens-ai images resolve <ref>).",
      "Confirm the worker can authenticate to the registry.",
      "Rebuild/push the image tag via CI if it is missing."]),
    (("cuda error", "cuda driver version", "no kernel image", "device-side assert",
      "cuda capability"),
     "A CUDA/GPU runtime error occurred.",
     ["The image's CUDA version may be incompatible with the host driver.",
      "Ensure the GPU is exposed to the container (the platform sets --gpus).",
      "Check the training script sends tensors to the correct device."]),
    (("no space left on device", "disk quota exceeded"),
     "The worker ran out of disk space while running the job.",
     ["Free space on the worker's NVMe scratch or the NAS.",
      "The platform's LRU eviction may need a larger budget."]),
    (("permission denied", "eacces", "operation not permitted"),
     "A permission error occurred writing outputs or reading inputs.",
     ["Check the output directory is writable by the worker.",
      "Confirm NAS mount permissions."]),
    (("modulenotfounderror", "no module named", "importerror"),
     "A Python dependency is missing inside the image.",
     ["Add the missing package to the image's requirements and rebuild.",
      "Confirm the image was built from the intended commit."]),
    (("yaml", "scannererror", "parsererror", "config"),
     "The job config file could not be parsed.",
     ["Validate the config.yaml — check indentation and types.",
      "Regenerate it with the config generator."]),
    (("timed out", "timeout"),
     "The job exceeded the maximum run time and was stopped.",
     ["Reduce epochs or dataset size, or split the work.",
      "Confirm the training is actually progressing (check logs/metrics)."]),
]


def explain_failure(error: Optional[str], log_tail: Optional[list[str]] = None) -> dict:
    """Best-effort, rule-based explanation of why a job failed. Deterministic;
    an AI-assisted layer can be added later."""
    haystack = " ".join(filter(None, [error or ""] + (log_tail or []))).lower()
    for needles, cause, suggestions in _RULES:
        if any(n in haystack for n in needles):
            return {"matched": True, "cause": cause, "suggestions": suggestions}
    return {
        "matched": False,
        "cause": "The job failed, but no known error pattern matched.",
        "suggestions": ["Open the full logs for the container's error output.",
                        "Re-run with the same config to see if it is reproducible.",
                        "If it persists, share the log tail with the platform team."],
    }


# --------------------------------------------------------------------------- #
# #19 standard job report
# --------------------------------------------------------------------------- #
def assemble_report(job: dict, result: dict) -> dict:
    """Build the standard, traceable job report from a job row + its result."""
    started = job.get("started_at")
    finished = job.get("finished_at")
    duration = None
    if started and finished:
        try:
            duration = int(_parse_ts(finished) - _parse_ts(started))  # type: ignore[operator]
        except Exception:  # noqa: BLE001
            duration = None
    return {
        "job_id": job.get("job_id"),
        "job_type": job.get("job_type"),
        "model_name": job.get("model_name"),
        "dataset": {"name": job.get("dataset_name"), "version": job.get("dataset_version")},
        "docker_image": job.get("docker_image"),
        "gpu_count": job.get("gpu_count"),
        "status": job.get("status"),
        "requested_by": job.get("requested_by"),
        "priority": job.get("priority"),
        "created_at": job.get("created_at"),
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": duration,
        "output_path": job.get("output_path"),
        "result_version": job.get("result_version"),
        "metrics": result.get("metrics"),
        "report": result.get("report"),
        "error": job.get("error"),
        "artifacts": result.get("artifacts", []),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
