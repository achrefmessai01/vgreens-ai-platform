"""Dataset audit for YOLO detection datasets.

Inspects a staged dataset directory (data.yaml + train/valid/test splits) and
classifies findings by a configurable policy into fail / warn / allow. Any
"fail" finding makes the whole audit fatal -> the import is rejected.

Corrupt-image and dimension checks use Pillow if present; if not, they are
skipped and reported as informational (never a silent pass of a broken image).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:   # noqa: BLE001
    _HAVE_PIL = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_POLICY = {
    "corrupt_images": "fail",
    "invalid_labels": "fail",     # malformed lines or class id out of range (unfixable)
    "degenerate_boxes": "warn",   # zero/negative size — normalise drops these lines
    "oob_boxes": "warn",          # box extends past the frame edge (YOLO clips these)
    "split_leakage": "fail",      # same image in >1 split
    "missing_labels": "warn",
    "duplicate_images": "warn",
    "empty_labels": "allow",
}

# tolerance for coordinates just past [0,1] due to annotation rounding
_BOX_EPS = 0.01


def _read_yaml(root: Path) -> dict:
    p = root / "data.yaml"
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names or []


def _find_splits(root: Path, data: dict) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split in ("train", "valid", "val", "test"):
        p = data.get(split)
        cands = []
        if isinstance(p, str):
            cands.append((root / p).resolve())
        cands.append(root / split / "images")
        for c in cands:
            if c.is_dir():
                out["val" if split in ("val", "valid") else split] = c
                break
    return out


def audit(root: Path, policy: dict | None = None) -> dict:
    policy = {**DEFAULT_POLICY, **(policy or {})}
    data = _read_yaml(root)
    names = _names(data)
    nc = data.get("nc", len(names))
    splits = _find_splits(root, data)

    findings: list[dict] = []

    def add(code: str, msg: str):
        level = policy.get(code, "warn")
        if level != "allow":
            findings.append({"level": level, "code": code, "msg": msg})

    report: dict = {"classes": names, "num_classes": nc, "splits": {}, "findings": findings}
    class_instances: dict[int, int] = {}
    hashes_by_split: dict[str, set] = {}

    if not splits:
        findings.append({"level": "fail", "code": "no_data", "msg": "no train/val/test image dirs found"})
        report["fatal"] = True
        return report

    if not _HAVE_PIL:
        findings.append({"level": "info", "code": "corrupt_images",
                         "msg": "Pillow not installed — corrupt-image/dimension checks skipped"})

    for split, imgdir in splits.items():
        labeldir = imgdir.parent / "labels"
        images = [p for p in imgdir.iterdir() if p.suffix.lower() in IMG_EXTS]
        hashes = set()
        n_missing = n_empty = n_bad = n_badbox = n_corrupt = 0
        for img in images:
            h = hashlib.sha256(img.read_bytes()).hexdigest()
            if h in hashes:
                add("duplicate_images", f"{split}: duplicate image {img.name}")
            hashes.add(h)
            if _HAVE_PIL:
                try:
                    with Image.open(img) as im:
                        im.verify()
                except Exception:   # noqa: BLE001
                    n_corrupt += 1
                    add("corrupt_images", f"{split}: corrupt image {img.name}")
            lbl = labeldir / (img.stem + ".txt")
            if not lbl.is_file():
                n_missing += 1
                add("missing_labels", f"{split}: no label for {img.name}")
                continue
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            if not lines:
                n_empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    n_bad += 1
                    add("invalid_labels", f"{split}: malformed line in {lbl.name}")
                    continue
                try:
                    ci = int(float(parts[0]))
                    x, y, w, hh = (float(v) for v in parts[1:5])
                except ValueError:
                    n_bad += 1
                    add("invalid_labels", f"{split}: unparseable numeric line in {lbl.name}")
                    continue
                if ci < 0 or (nc and ci >= nc):
                    n_bad += 1
                    add("invalid_labels", f"{split}: class id {ci} out of range in {lbl.name}")
                    continue
                class_instances[ci] = class_instances.get(ci, 0) + 1
                if w <= 0 or hh <= 0:                     # degenerate -> normalise drops it
                    n_badbox += 1
                    add("degenerate_boxes", f"{split}: zero/negative-size box in {lbl.name}")
                elif (x < -_BOX_EPS or x > 1 + _BOX_EPS or y < -_BOX_EPS or y > 1 + _BOX_EPS
                      or w > 1 + _BOX_EPS or hh > 1 + _BOX_EPS):   # past the edge -> warn (clipped)
                    n_badbox += 1
                    add("oob_boxes", f"{split}: box extends past frame in {lbl.name}")
        hashes_by_split[split] = hashes
        report["splits"][split] = {
            "images": len(images), "missing_labels": n_missing, "empty_labels": n_empty,
            "malformed_lines": n_bad, "invalid_boxes": n_badbox, "corrupt_images": n_corrupt,
        }
        if len(images) == 0:
            add("no_data", f"{split}: no images")

    # split leakage: same image content in more than one split
    seen: dict[str, str] = {}
    for split, hs in hashes_by_split.items():
        for h in hs:
            if h in seen and seen[h] != split:
                add("split_leakage", f"image shared between {seen[h]} and {split}")
            seen[h] = split

    report["instances_per_class"] = {
        (names[i] if names and i < len(names) else str(i)): class_instances.get(i, 0)
        for i in range(nc)
    }
    report["fatal"] = any(f["level"] == "fail" for f in findings)
    return report
