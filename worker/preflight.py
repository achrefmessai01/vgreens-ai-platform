"""Preflight checks — run before touching any job.

If any check fails, the worker nacks (requeue) the message and reports the reason,
so a transient outage (NAS down, docker hiccup) doesn't burn the job.
"""
from __future__ import annotations

import os
import shutil
import subprocess

NAS_MARKER = os.environ.get("NAS_MARKER", "/mnt/vgreens_storage/ai/jobs")
NVME_JOBS = os.environ.get("NVME_JOBS", "/nvme/jobs")
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "20"))


class PreflightError(RuntimeError):
    pass


def _run(cmd: list[str]) -> int:
    try:
        return subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20
        ).returncode
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1


def check_nas() -> None:
    # marker path must exist and be statable — proves the SMB mount is live.
    if not os.path.isdir(NAS_MARKER):
        raise PreflightError(f"NAS not available at {NAS_MARKER}")


def check_gpu() -> None:
    if _run(["nvidia-smi"]) != 0:
        raise PreflightError("nvidia-smi failed (GPU/driver not visible)")


def check_docker() -> None:
    if _run(["docker", "info"]) != 0:
        raise PreflightError("docker not available")


def check_disk() -> None:
    os.makedirs(NVME_JOBS, exist_ok=True)
    free_gb = shutil.disk_usage(NVME_JOBS).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        raise PreflightError(f"low disk: {free_gb:.1f} GB free < {MIN_FREE_GB} GB floor")


def run_all() -> None:
    """Raise PreflightError on the first failing check."""
    check_nas()
    check_gpu()
    check_docker()
    check_disk()
