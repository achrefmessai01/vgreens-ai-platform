"""Stage dataset NAS -> NVMe cache before a training job.

rsync with --bwlimit to protect the NAS. A `.complete` marker file signals a
fully-verified cache entry; if present we skip re-copying. The 7-day cleanup
timer (Phase 0) and Phase 2 LRU eviction manage cache size.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

NAS_DATASETS = os.environ.get("NAS_DATASETS", "/mnt/vgreens_storage/ai/datasets")
NVME_DATASETS = os.environ.get("NVME_DATASETS", "/nvme/cache/datasets")
BWLIMIT = os.environ.get("RSYNC_BWLIMIT", "100000")  # KB/s


class StagingError(RuntimeError):
    pass


def stage_dataset(name: str, version: str) -> str:
    """Copy datasets/<name>/<version>/ to the NVMe cache. Returns local path."""
    src = Path(NAS_DATASETS) / name / version
    dst = Path(NVME_DATASETS) / name / version
    marker = dst / ".complete"

    if not src.is_dir():
        raise StagingError(f"dataset not found on NAS: {src}")

    if marker.exists():
        return str(dst)

    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-a", "--delete",
        f"--bwlimit={BWLIMIT}",
        f"{src}/", f"{dst}/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise StagingError(f"rsync failed: {result.stderr.strip()}")

    marker.write_text("ok\n")
    return str(dst)
