#!/usr/bin/env python3
"""Pull a Roboflow object-detection dataset, audit it, and freeze it immutably
on the NAS so training runs are reproducible.

Design rules (match the platform):
  - Datasets are frozen, versioned, read-only. A frozen version is NEVER
    overwritten — if the target exists, this script refuses.
  - Nothing secret is passed on the command line. The Roboflow API key is read
    from the ROBOFLOW_API_KEY environment variable only.
  - The freeze is atomic-ish: build in a temp dir on the same filesystem, then
    os.replace() into place, so a half-written dataset never appears frozen.
  - A MANIFEST.json records provenance (source, roboflow version, class map,
    per-split counts, a content hash) for audit.

Typical use (on the GPU server, which has the NAS mounted):

    export ROBOFLOW_API_KEY=...            # do NOT paste the key into a command
    python3 freeze_roboflow.py \
        --workspace strawberry-disease \
        --project   strawberry-disease-detection-dataset \
        --rf-version 1 \
        --name strawberry_disease \
        --version v1

Audit only (download to a temp dir and print the report, freeze nothing):

    python3 freeze_roboflow.py ... --audit-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAS_DATASETS_ROOT = os.environ.get("NAS_DATASETS_ROOT", "/mnt/vgreens_storage/ai/datasets")
NAS_MOUNTPOINT = os.environ.get("NAS_MOUNTPOINT", "/mnt/vgreens_storage")


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def info(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def download_roboflow(workspace: str, project: str, rf_version: int, dest: Path) -> Path:
    """Download a YOLOv8 export of the given Roboflow project version into `dest`.

    Returns the path to the extracted dataset directory (contains data.yaml).
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        die("ROBOFLOW_API_KEY is not set. `export ROBOFLOW_API_KEY=...` first "
            "(do not put the key on the command line).")

    try:
        from roboflow import Roboflow
    except ImportError:
        die("the 'roboflow' package is not installed. Run: "
            "pip install roboflow --break-system-packages")

    info(f"→ downloading {workspace}/{project} v{rf_version} (yolov8) …")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    version = proj.version(rf_version)
    ds = version.download("yolov8", location=str(dest / "download"), overwrite=True)
    root = Path(ds.location)
    if not (root / "data.yaml").is_file():
        die(f"downloaded dataset has no data.yaml at {root}")
    return root


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _read_data_yaml(root: Path) -> dict:
    import yaml
    with open(root / "data.yaml") as fh:
        return yaml.safe_load(fh)


def _split_dirs(root: Path, data: dict) -> dict[str, Path]:
    """Map split -> images dir, tolerating the common Roboflow layouts."""
    out: dict[str, Path] = {}
    for split in ("train", "valid", "val", "test"):
        # data.yaml usually points at .../images
        p = data.get(split)
        cand: list[Path] = []
        if isinstance(p, str):
            cand.append((root / p).resolve())
        cand.append(root / split / "images")
        for c in cand:
            if c.is_dir():
                out["valid" if split == "val" else split] = c
                break
    return out


def audit(root: Path) -> dict:
    data = _read_data_yaml(root)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    nc = data.get("nc", len(names) if names else 0)

    splits = _split_dirs(root, data)
    if not splits:
        die("could not locate any train/valid/test image directories")

    report: dict = {"classes": names, "num_classes": nc, "splits": {}, "problems": []}
    class_instances = {i: 0 for i in range(nc)}

    for split, imgdir in splits.items():
        labeldir = imgdir.parent / "labels"
        images = [p for p in imgdir.iterdir() if p.suffix.lower() in IMG_EXTS]
        n_img = len(images)
        n_lbl = n_empty = n_missing = n_bad = 0
        for img in images:
            lbl = labeldir / (img.stem + ".txt")
            if not lbl.is_file():
                n_missing += 1
                continue
            n_lbl += 1
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            if not lines:
                n_empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    n_bad += 1
                    continue
                try:
                    ci = int(float(parts[0]))
                    class_instances[ci] = class_instances.get(ci, 0) + 1
                except ValueError:
                    n_bad += 1
        report["splits"][split] = {
            "images": n_img,
            "labels": n_lbl,
            "missing_labels": n_missing,
            "empty_labels": n_empty,
            "malformed_lines": n_bad,
        }
        if n_img == 0:
            report["problems"].append(f"{split}: no images")
        if n_missing:
            report["problems"].append(f"{split}: {n_missing} images without a label file")
        if n_bad:
            report["problems"].append(f"{split}: {n_bad} malformed label lines")

    report["instances_per_class"] = {
        (names[i] if names and i < len(names) else str(i)): class_instances.get(i, 0)
        for i in range(nc)
    }
    empty_classes = [c for c, n in report["instances_per_class"].items() if n == 0]
    if empty_classes:
        report["problems"].append(f"classes with 0 instances: {', '.join(empty_classes)}")
    return report


def print_report(report: dict) -> None:
    info("\n===== dataset audit =====")
    info(f"classes ({report['num_classes']}): {report['classes']}")
    for split, s in report["splits"].items():
        info(f"  {split:6s} images={s['images']:5d} labels={s['labels']:5d} "
             f"missing={s['missing_labels']} empty={s['empty_labels']} bad={s['malformed_lines']}")
    info("  instances/class:")
    for c, n in report["instances_per_class"].items():
        info(f"    {c:28s} {n}")
    if report["problems"]:
        info("  PROBLEMS:")
        for p in report["problems"]:
            info(f"    ! {p}")
    else:
        info("  no problems detected")
    info("=========================\n")


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def _hash_tree(root: Path) -> str:
    """Stable content hash over relative paths + file bytes."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _write_training_yaml(frozen: Path, report: dict) -> None:
    """Write a data.yaml with ABSOLUTE paths so training works regardless of cwd."""
    import yaml
    doc = {
        "path": str(frozen),
        "train": "train/images",
        "val": "valid/images" if (frozen / "valid").is_dir() else "test/images",
        "names": report["classes"],
        "nc": report["num_classes"],
    }
    if (frozen / "test").is_dir():
        doc["test"] = "test/images"
    (frozen / "data.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))


def freeze(src: Path, name: str, version: str, report: dict, source_meta: dict) -> Path:
    if not os.path.ismount(NAS_MOUNTPOINT):
        die(f"NAS not mounted at {NAS_MOUNTPOINT}")
    target = Path(NAS_DATASETS_ROOT) / name / version
    if target.exists():
        die(f"{target} already exists — frozen datasets are immutable, refusing to overwrite")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}_{version}_", dir=str(target.parent)))
    try:
        for split in ("train", "valid", "test"):
            s = src / split
            if s.is_dir():
                shutil.copytree(s, tmp / split)
        _write_training_yaml(tmp, report)
        content_hash = _hash_tree(tmp)
        manifest = {
            "name": name,
            "version": version,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "frozen_by": os.environ.get("USER", "unknown"),
            "source": source_meta,
            "classes": report["classes"],
            "num_classes": report["num_classes"],
            "splits": report["splits"],
            "instances_per_class": report["instances_per_class"],
            "content_sha256": content_hash,
        }
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    # best-effort read-only
    for p in target.rglob("*"):
        try:
            p.chmod(0o444 if p.is_file() else 0o555)
        except OSError:
            pass
    info(f"✓ frozen at {target}")
    info(f"  content_sha256 = {content_hash}")
    return target


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Audit + freeze a Roboflow detection dataset on the NAS.")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--rf-version", type=int, required=True, help="Roboflow dataset version number")
    ap.add_argument("--name", required=True, help="frozen dataset name (registry key), e.g. strawberry_disease")
    ap.add_argument("--version", required=True, help="frozen version tag, e.g. v1")
    ap.add_argument("--audit-only", action="store_true", help="download to a temp dir, audit, freeze nothing")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="rf_pull_"))
    try:
        root = download_roboflow(args.workspace, args.project, args.rf_version, workdir)
        report = audit(root)
        print_report(report)

        if args.audit_only:
            info("audit-only: nothing frozen.")
            return
        if report["problems"]:
            info("Audit found problems (listed above). Freeze anyway? [y/N] ")
            if input().strip().lower() not in ("y", "yes"):
                die("aborted by user", code=2)

        source_meta = {
            "provider": "roboflow",
            "workspace": args.workspace,
            "project": args.project,
            "rf_version": args.rf_version,
            "format": "yolov8",
        }
        freeze(root, args.name, args.version, report, source_meta)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
