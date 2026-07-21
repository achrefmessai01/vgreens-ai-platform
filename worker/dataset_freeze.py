"""Normalise a staged YOLO dataset into the canonical layout, then freeze it
atomically onto the NAS.

Canonical layout:
    images/{train,val,test}/  labels/{train,val,test}/
    data.yaml  MANIFEST.json  CHECKSUMS.sha256  audit/  source/  .complete

Atomicity: build into a temp dir on the SAME filesystem as the final path
(the NAS), then os.replace() it into place — the platform ignores any version
dir without a .complete marker, so a partial freeze never looks valid.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def normalise(raw: Path, out: Path, names: list[str], nc: int) -> dict:
    """Copy raw <split>/images|labels into canonical images/<split>, labels/<split>.
    Returns {image_count, class_count, splits}."""
    if out.exists():
        shutil.rmtree(out)
    image_count = 0
    present: list[str] = []
    for src_split, dst in _SPLIT_MAP.items():
        src_img = raw / src_split / "images"
        src_lbl = raw / src_split / "labels"
        if not src_img.is_dir():
            continue
        (out / "images" / dst).mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_img, out / "images" / dst, dirs_exist_ok=True)
        image_count += sum(1 for p in (out / "images" / dst).iterdir()
                           if p.suffix.lower() in IMG_EXTS)
        if src_lbl.is_dir():
            (out / "labels" / dst).mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_lbl, out / "labels" / dst, dirs_exist_ok=True)
        if dst not in present:
            present.append(dst)

    doc: dict = {"path": ".", "names": names, "nc": nc}
    doc["train"] = "images/train" if "train" in present else "images/" + present[0]
    doc["val"] = "images/val" if "val" in present else doc["train"]
    if "test" in present:
        doc["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return {"image_count": image_count, "class_count": nc, "splits": present}


def hash_tree(root: Path) -> tuple[str, str]:
    """Return (content_digest, checksums_text) over all files under root."""
    h = hashlib.sha256()
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            fh = hashlib.sha256(b).hexdigest()
            rel = p.relative_to(root)
            lines.append(f"{fh}  {rel}")
            h.update(str(rel).encode())
            h.update(b)
    return h.hexdigest(), "\n".join(lines) + "\n"


def freeze(normalised: Path, final_path: str, manifest: dict, checksums_text: str,
           audit_report: dict, source_meta: dict) -> None:
    """Atomically publish `normalised` to `final_path` on the NAS."""
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"{final} already exists — refusing to overwrite a frozen version")

    tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}.incoming_", dir=str(final.parent)))
    try:
        # data (images/, labels/, data.yaml)
        for item in normalised.iterdir():
            dst = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        # metadata
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        (tmp / "CHECKSUMS.sha256").write_text(checksums_text)
        adir = tmp / "audit"
        adir.mkdir(exist_ok=True)
        (adir / "report.json").write_text(json.dumps(audit_report, indent=2))
        (adir / "issues.json").write_text(json.dumps(audit_report.get("findings", []), indent=2))
        sdir = tmp / "source"
        sdir.mkdir(exist_ok=True)
        (sdir / "source.json").write_text(json.dumps(source_meta, indent=2))
        (tmp / ".complete").write_text(manifest.get("generated_at", ""))
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    for p in final.rglob("*"):        # best-effort read-only
        try:
            p.chmod(0o444 if p.is_file() else 0o555)
        except OSError:
            pass
