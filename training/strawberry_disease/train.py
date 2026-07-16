"""YOLO object-detection training job — contract-compliant.

Fine-tunes a baked-in Ultralytics backbone on a frozen dataset mounted at /data,
and writes exactly what the platform's registry/validation expects into /out:

  model.pt              the trained weights (Ultralytics best.pt)
  eval/metrics.json     achieved metrics: mAP50, mAP50_95, precision, recall
  evaluation_report.md  human-readable summary + per-class mAP
  classes.json          class names (inference convenience)
  runs/                 full Ultralytics artifacts (plots, confusion matrix)

Contract:  /data (ro) dataset · /config/job.yaml (ro) config · /out (rw) · exit 0.

Config (from /config/job.yaml):
  train:
    model: /app/yolo11n.pt   # backbone (baked into the image)
    epochs: 50
    imgsz: 640
    batch: 16                # int, or -1 for Ultralytics auto-batch
    patience: 20
The `thresholds:` block is read by the platform (worker), NOT here — this job
only reports the metrics it achieved, like any real model.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

DATA = Path("/data")
OUT = Path("/out")
CONFIG = Path("/config/job.yaml")


def load_cfg() -> dict:
    if CONFIG.is_file():
        try:
            return yaml.safe_load(CONFIG.read_text()) or {}
        except yaml.YAMLError as exc:
            sys.stderr.write(f"[train] bad job.yaml: {exc}\n")
    return {}


def build_runtime_data_yaml(dst: Path) -> dict:
    """The frozen data.yaml has an absolute NAS `path:`; rewrite it to /data
    (where the platform mounts the staged dataset copy)."""
    src = DATA / "data.yaml"
    doc = yaml.safe_load(src.read_text()) if src.is_file() else {}
    if not isinstance(doc, dict):
        doc = {}
    doc["path"] = "/data"
    doc.setdefault("train", "train/images")
    doc.setdefault("val", "valid/images" if (DATA / "valid").is_dir() else "test/images")
    if (DATA / "test").is_dir():
        doc.setdefault("test", "test/images")
    dst.write_text(yaml.safe_dump(doc, sort_keys=False))
    return doc


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        sys.stderr.write("[train] CUDA not available inside container\n")
        return 1

    from ultralytics import YOLO

    cfg = load_cfg()
    t = cfg.get("train") or {}
    backbone = str(t.get("model", "/app/yolo11n.pt"))
    epochs = int(t.get("epochs", 50))
    imgsz = int(t.get("imgsz", 640))
    batch = t.get("batch", 16)
    patience = int(t.get("patience", 20))

    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    data_yaml = Path("/tmp/data.yaml")
    doc = build_runtime_data_yaml(data_yaml)
    names = doc.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    job_id = os.environ.get("JOB_ID", "unknown")
    model_name = os.environ.get("MODEL_NAME", "strawberry_disease")
    dataset_ver = os.environ.get("DATASET_VERSION", "?")
    gpu = torch.cuda.get_device_name(0)
    print(f"[train] job={job_id} model={model_name} gpu={gpu} backbone={backbone} "
          f"epochs={epochs} imgsz={imgsz} batch={batch}", flush=True)

    model = YOLO(backbone)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        patience=patience, device=0, project=str(OUT / "runs"), name="train",
        exist_ok=True, verbose=True,
    )

    best = OUT / "runs" / "train" / "weights" / "best.pt"
    if not best.is_file():
        sys.stderr.write("[train] no best.pt produced\n")
        return 1
    shutil.copy(best, OUT / "model.pt")

    # clean metrics on the validation split
    metrics = model.val(data=str(data_yaml), split="val", device=0,
                         project=str(OUT / "runs"), name="val", exist_ok=True)
    box = metrics.box
    m = {
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }
    (OUT / "eval" / "metrics.json").write_text(json.dumps(m, indent=2))

    per_class = []
    try:
        for i, ap in enumerate(box.maps):
            per_class.append((names[i] if i < len(names) else str(i), round(float(ap), 4)))
    except Exception:  # noqa: BLE001 - per-class is best-effort
        pass

    report = [
        f"# Evaluation — {model_name}", "",
        f"- job: `{job_id}`",
        f"- dataset: `{model_name}:{dataset_ver}`",
        f"- backbone: `{backbone}`",
        f"- epochs: {epochs} · imgsz: {imgsz} · batch: {batch}", "",
        "## Metrics", "",
        f"- mAP@0.5: **{m['mAP50']}**",
        f"- mAP@0.5:0.95: {m['mAP50_95']}",
        f"- precision: {m['precision']}",
        f"- recall: {m['recall']}", "",
    ]
    if per_class:
        report += ["## Per-class mAP@0.5:0.95", ""]
        report += [f"- {n}: {ap}" for n, ap in per_class]
    (OUT / "evaluation_report.md").write_text("\n".join(report) + "\n")
    (OUT / "classes.json").write_text(json.dumps(names, indent=2))

    print(f"[train] done: mAP50={m['mAP50']} mAP50-95={m['mAP50_95']} "
          f"P={m['precision']} R={m['recall']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
