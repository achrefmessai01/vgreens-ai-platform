"""Stage dataset NAS -> NVMe cache before a job.

rsync with --bwlimit to protect the NAS. A `.staged_ok` marker signals a fully
staged cache entry; if present we skip re-copying. The cache is the worker's to
own: it is forced writable (the frozen NAS source is read-only) so it can be
re-staged and evicted cleanly. NOTE: the marker is `.staged_ok`, deliberately
distinct from the dataset's own `.complete` freeze marker.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

NAS_DATASETS = os.environ.get("NAS_DATASETS", "/mnt/vgreens_storage/ai/datasets")
NVME_DATASETS = os.environ.get("NVME_DATASETS", "/nvme/cache/datasets")
BWLIMIT = os.environ.get("RSYNC_BWLIMIT", "100000")  # KB/s
STAGE_MARKER = ".staged_ok"


class StagingError(RuntimeError):
    pass


def _make_writable(path: Path) -> None:
    """Frozen datasets are read-only; make a copy owner-writable so it can be
    overwritten/removed."""
    for p in path.rglob("*"):
        try:
            p.chmod(0o755 if p.is_dir() else 0o644)
        except OSError:
            pass
    try:
        path.chmod(0o755)
    except OSError:
        pass


def stage_dataset(name: str, version: str) -> str:
    """Copy datasets/<name>/<version>/ to the NVMe cache. Returns local path."""
    src = Path(NAS_DATASETS) / name / version
    dst = Path(NVME_DATASETS) / name / version
    marker = dst / STAGE_MARKER

    if not src.is_dir():
        raise StagingError(f"dataset not found on NAS: {src}")

    if marker.exists():
        return str(dst)

    # clean any partial/previous copy (may be read-only from a prior rsync)
    if dst.exists():
        _make_writable(dst)
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    # -a preserves structure/times; --chmod forces owner-writable dest perms so
    # the read-only frozen source doesn't produce a read-only, unwritable cache.
    cmd = [
        "rsync", "-a", "--delete", "--chmod=Du+rwx,Fu+rw",
        f"--bwlimit={BWLIMIT}",
        f"{src}/", f"{dst}/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise StagingError(f"rsync failed: {result.stderr.strip()}")

    _make_writable(dst)          # belt-and-braces: ensure the marker + eviction can write
    marker.write_text("ok\n")
    return str(dst)
