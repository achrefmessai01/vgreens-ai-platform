"""Quarantine safety for untrusted dataset sources.

Nothing from an external source is trusted until it passes these checks. We
protect against path traversal, symlink escapes, zip bombs, oversized/too-many
files, and staging outside allowlisted roots. Staging always lands in an
isolated per-import directory on NVMe, never directly in the frozen NAS tree.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_EXTS = IMAGE_EXTS | {".txt", ".yaml", ".yml", ".json", ".md", ".names", ".cache", ""}

DEFAULT_MAX_FILES = 300_000
DEFAULT_MAX_BYTES = 60 * 1024 ** 3        # 60 GB uncompressed
DEFAULT_MAX_RATIO = 200                    # zip-bomb guard: uncompressed / compressed


class SafetyError(Exception):
    """Raised when a source fails a quarantine check -> import is rejected."""


def resolve_within(path: str | Path, roots: list[str]) -> Path:
    """Resolve `path` and require it to live under one of `roots` (realpath).
    Blocks arbitrary-path access and symlink/`..` escapes."""
    rp = Path(path).resolve()
    for root in roots:
        rr = Path(root).resolve()
        try:
            rp.relative_to(rr)
            return rp
        except ValueError:
            continue
    raise SafetyError(f"path is outside allowed roots {roots}: {path}")


def _bad_member(name: str) -> bool:
    p = Path(name)
    return name.startswith("/") or p.is_absolute() or ".." in p.parts


def safe_extract_zip(zip_path: str | Path, dest: str | Path, *,
                     max_files: int = DEFAULT_MAX_FILES,
                     max_bytes: int = DEFAULT_MAX_BYTES,
                     max_ratio: int = DEFAULT_MAX_RATIO) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_real = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise SafetyError(f"archive has too many entries ({len(infos)} > {max_files})")
        total_unc = sum(i.file_size for i in infos)
        total_comp = sum(i.compress_size for i in infos) or 1
        if total_unc > max_bytes:
            raise SafetyError(f"archive too large uncompressed ({total_unc} > {max_bytes})")
        if total_unc / total_comp > max_ratio:
            raise SafetyError(f"possible zip bomb (ratio {total_unc / total_comp:.0f} > {max_ratio})")
        for i in infos:
            if _bad_member(i.filename):
                raise SafetyError(f"unsafe path in archive: {i.filename}")
            if (i.external_attr >> 16) & 0o170000 == 0o120000:      # symlink bit
                raise SafetyError(f"symlink in archive: {i.filename}")
            target = (dest / i.filename).resolve()
            if not str(target).startswith(str(dest_real)):
                raise SafetyError(f"path traversal in archive: {i.filename}")
        zf.extractall(dest)
    return dest


def stage_local_dir(src: str | Path, dest: str | Path, roots: list[str], *,
                    max_files: int = DEFAULT_MAX_FILES,
                    max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    src = resolve_within(src, roots)
    if not src.is_dir():
        raise SafetyError(f"source is not a directory: {src}")
    n_files, total = 0, 0
    for p in src.rglob("*"):
        if p.is_symlink():
            raise SafetyError(f"symlink not allowed in source: {p}")
        if p.is_file():
            n_files += 1
            total += p.stat().st_size
            if n_files > max_files:
                raise SafetyError(f"too many files ({n_files} > {max_files})")
            if total > max_bytes:
                raise SafetyError(f"source too large ({total} > {max_bytes})")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True, symlinks=False)
    return dest
