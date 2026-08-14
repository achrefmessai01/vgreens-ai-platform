"""Model repository generator.

Renders a complete, ready-to-push model repo from a validated model.yaml.
Detection uses RF-DETR (Apache-2.0); classification/segmentation and any other
task use the runnable custom_python reference. Ultralytics/YOLO is not used
(AGPL-3.0). Only config/CI/README are templated per spec.

custom_python emits a RUNNABLE reference (train.py/process.py train a trivial
baseline and honour the contract) so the pipeline is green from commit 1; the
engineer replaces the marked sections with their real model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# --- shared contract-compliant entrypoint (all presets) --------------------- #
_ENTRYPOINT_PY = '''\
"""Container entrypoint. Dispatches on JOB_TYPE: training -> train.py, processing -> process.py.

Multi-GPU: when the platform requests >1 GPU it sets GPU_COUNT and exposes that
many devices to the container. For TRAINING we then relaunch train.py under
torchrun with one process per GPU (DistributedDataParallel). Single-GPU and
processing jobs run in-process exactly as before. If torchrun is unavailable we
fall back to a single process (single-GPU) rather than fail.
"""
import os
import runpy
import subprocess
import sys

mode = os.environ.get("JOB_TYPE", "training")
if mode == "processing":
    print("[entrypoint] JOB_TYPE=processing -> /app/process.py", flush=True)
    runpy.run_path("/app/process.py", run_name="__main__")
    raise SystemExit(0)

gpu_count = int(os.environ.get("GPU_COUNT", "1") or 1)
already = os.environ.get("VG_DDP_LAUNCHED") == "1"
if gpu_count > 1 and not already:
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--standalone", f"--nproc_per_node={gpu_count}", "/app/train.py"]
    print(f"[entrypoint] multi-GPU training on {gpu_count} GPUs via torchrun: "
          f"{' '.join(cmd)}", flush=True)
    env = dict(os.environ, VG_DDP_LAUNCHED="1")
    rc = subprocess.call(cmd, env=env)
    if rc == 0:
        raise SystemExit(0)
    # torchrun missing / failed to start -> fall back to single-process training.
    print(f"[entrypoint] torchrun exited {rc}; falling back to single-GPU", flush=True)

print("[entrypoint] JOB_TYPE=training -> /app/train.py", flush=True)
runpy.run_path("/app/train.py", run_name="__main__")
'''

_RFDETR_TRAIN = '''\
# NOTE: validate on a GPU host before first production use — metrics extraction
# (supervision mAP) and the EMA checkpoint filename are best-effort and may need
# a one-line adjustment after the first real run.
"""RF-DETR detection training — platform contract entrypoint (Apache-2.0).

reads  /data (ro), /config/job.yaml ; writes /out/model.pt, /out/eval/metrics.json,
/out/evaluation_report.md ; exit 0 on success. /data is copied to a writable dir
because RF-DETR writes caches next to the labels. RF-DETR reads YOLO-format
datasets directly via dataset_file="yolo" (no COCO conversion). batch_size and
grad_accum_steps are native params; effective batch = batch_size x grad_accum_steps.
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Force MKL's GNU threading layer BEFORE importing torch/rfdetr. The conda base
# image's Intel MKL otherwise selects MKL_THREADING_LAYER=INTEL, which clashes
# with libgomp.so.1 in the DDP subprocesses spawned for multi-GPU training and
# aborts multi-GPU runs. Must precede the first torch import.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import yaml

OUT = Path("/out")
CONFIG = Path("/config/job.yaml")
WORK = Path("/tmp/ds")


def _progress(**row):
    """Append one JSON line to /out/metrics.jsonl (platform live-progress contract).

    The API reads this file to show epoch/loss/mAP/ETA while the job runs. Each
    line is a standalone JSON object; include an 'epoch' and 'total_epochs' so the
    dashboard can draw a progress bar. Best-effort — never crashes training.
    """
    import time as _t
    row.setdefault("ts", _t.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        with open(OUT / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\\n")
    except OSError:
        pass


# model_variant -> RF-DETR class name
_VARIANTS = {
    "base": "RFDETRBase",
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _model_class(variant):
    import rfdetr
    name = _VARIANTS.get(str(variant).lower(), "RFDETRBase")
    return getattr(rfdetr, name, rfdetr.RFDETRBase)


def _versions():
    """Library versions this run used — recorded in report.json so results are
    reproducible (e.g. an RF-DETR version bump can change detections)."""
    import importlib.metadata as _md
    out = {}
    for pkg in ("rfdetr", "torch", "torchvision", "supervision", "transformers"):
        try:
            out[pkg] = _md.version(pkg)
        except Exception:  # noqa: BLE001
            pass
    return out


def _dataset_classes(ds_root):
    """Class names + count from the dataset's data.yaml, so the index->name map
    travels with the result (YOLO is 0-indexed; don't trust bare numeric ids)."""
    try:
        doc = yaml.safe_load((ds_root / "data.yaml").read_text()) or {}
        names = doc.get("names")
        if isinstance(names, dict):            # {0: 'a', 1: 'b'} -> ordered list
            names = [names[k] for k in sorted(names)]
        return {"names": names, "num_classes": doc.get("nc"),
                "dataset_format": "yolo"}
    except Exception:  # noqa: BLE001
        return {"names": None, "num_classes": None, "dataset_format": "yolo"}


def _yolo_ground_truth(label_path, w, h):
    """Build an sv.Detections from a YOLO label file (normalized cx cy bw bh)."""
    import numpy as np
    import supervision as sv
    boxes, classes = [], []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = parts[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            classes.append(int(float(c)))
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(xyxy=np.array(boxes, dtype=float),
                         class_id=np.array(classes, dtype=int))


def _metrics_from_csv():
    """Read the val mAP RF-DETR already logged to /out/runs/metrics.csv and
    return the best epoch's {mAP50, mAP50-95}. This is far more reliable than
    recomputing, and avoids rebuilding the model (which can mismatch a
    fine-tuned checkpoint's patch_size/num_classes). Returns None if unavailable.
    """
    import csv
    csv_path = OUT / "runs" / "metrics.csv"
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    if not rows:
        return None

    def _get(row, *needles):
        for k, v in row.items():
            kl = (k or "").lower().replace(" ", "").replace("-", "_")
            if all(n in kl for n in needles) and v not in (None, "", "nan", "NaN"):
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    scored = [(_get(r, "map", "50", "95"), r) for r in rows]
    scored = [(s, r) for s, r in scored if s is not None]
    if not scored:
        return None
    _, best = max(scored, key=lambda t: t[0])
    m5095 = _get(best, "map", "50", "95")
    m50 = None
    for k, v in best.items():
        kl = (k or "").lower().replace(" ", "").replace("-", "_")
        if "map" in kl and "50" in kl and "95" not in kl and v not in (None, "", "nan", "NaN"):
            try:
                m50 = float(v)
                break
            except ValueError:
                pass
    if m5095 is None and m50 is None:
        return None
    return {"mAP50": round(m50, 4) if m50 is not None else 0.0,
            "mAP50-95": round(m5095, 4) if m5095 is not None else 0.0,
            "source": "rfdetr/runs/metrics.csv"}


def _compute_metrics(ckpt, ds_root, variant):
    """Val mAP for the trained model. Prefers the metrics RF-DETR itself logged
    to runs/metrics.csv; if that is unavailable, falls back to a supervision
    recompute over the val split. Never crashes — returns zeros + a note only if
    both paths fail.
    """
    try:
        csv_metrics = _metrics_from_csv()
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[train] reading runs/metrics.csv failed: {exc}\\n")
        csv_metrics = None
    if csv_metrics:
        print(f"[train] metrics from RF-DETR log: {csv_metrics}", flush=True)
        return csv_metrics
    try:
        from PIL import Image
        from supervision.metrics import MeanAveragePrecision

        cls = _model_class(variant)
        model = cls(pretrain_weights=str(ckpt))

        # Locate the val images. Trust the dataset's own data.yaml `val:` entry
        # (that is exactly what RF-DETR used during training), then fall back to
        # the common YOLO folder layouts. Reading data.yaml fixes the frequent
        # "no val split found" false-zero when the split is not literally at
        # valid/images (e.g. val/, images/val, or a data.yaml `path:` root).
        val_dir = None
        data_yaml = ds_root / "data.yaml"
        if data_yaml.is_file():
            doc = yaml.safe_load(data_yaml.read_text()) or {}
            root = (ds_root / str(doc.get("path", "."))).resolve()
            val = doc.get("val")
            for cand in ([val] if isinstance(val, str) else (val or [])):
                p = (root / str(cand)).resolve()
                if (p / "images").is_dir():
                    val_dir = p / "images"
                    break
                if p.is_dir():
                    val_dir = p
                    break
        if val_dir is None:
            for cand in ("valid/images", "val/images", "validation/images"):
                if (ds_root / cand).is_dir():
                    val_dir = ds_root / cand
                    break
        if val_dir is None:
            return {"mAP50": 0.0, "mAP50-95": 0.0,
                    "note": "no val split found — check the dataset's data.yaml `val:` path"}

        # labels sit beside images in YOLO layout (.../images -> .../labels)
        labels_dir = val_dir.parent / "labels"
        images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
                        for p in val_dir.glob(ext))
        preds, targets = [], []
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            preds.append(model.predict(image, threshold=0.5))
            targets.append(_yolo_ground_truth(labels_dir / (img_path.stem + ".txt"), w, h))

        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        result = metric.compute()
        return {"mAP50": round(float(result.map50), 4),
                "mAP50-95": round(float(result.map50_95), 4)}
    except Exception as exc:  # noqa: BLE001 -- best-effort: never crash on metrics
        sys.stderr.write(f"[train] metrics extraction failed (best-effort): {exc}\\n")
        return {"mAP50": 0.0, "mAP50-95": 0.0,
                "note": f"metrics extraction failed on this host: {exc}"}


def main() -> int:
    # Standard output structure (platform contract): eval/ metrics, artifacts/,
    # predictions/. metrics.jsonl (progress) + report.json are written below.
    for d in ("eval", "artifacts", "predictions"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text()) if CONFIG.is_file() else {}
    t = (cfg or {}).get("train", {})

    variant = t.get("model_variant", "base")
    epochs = int(t.get("epochs", 10))
    image_size = int(t.get("image_size", t.get("imgsz", 512)))
    batch_size = int(t.get("batch_size", t.get("batch", 4)))

    # grad-accum: explicit steps win; else derive from effective_batch; else 4.
    gas = t.get("grad_accum_steps", t.get("grad_accumulation_steps"))
    eff = t.get("effective_batch")
    if gas:
        grad_accum = max(1, int(gas))
    elif eff:
        grad_accum = max(1, math.ceil(int(eff) / max(1, batch_size)))
    else:
        grad_accum = 4
    effective = batch_size * grad_accum

    # Distributed context (set by torchrun when the platform requests >1 GPU).
    # Single-GPU runs have no RANK/WORLD_SIZE -> rank 0, world 1 -> unchanged path.
    import time
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    is_main = rank == 0

    # Stage /data once. Under DDP only rank 0 copies; other ranks wait for the
    # sentinel so all processes read the same staged dataset without racing.
    if is_main:
        if WORK.exists():
            shutil.rmtree(WORK)
        shutil.copytree("/data", WORK)
        (WORK / ".vg_ready").write_text("ok")
    else:
        for _ in range(1800):  # up to ~30 min for rank 0 to finish staging
            if (WORK / ".vg_ready").exists():
                break
            time.sleep(1)

    print(f"[train] rank={rank}/{world} variant={variant} epochs={epochs} "
          f"image_size={image_size} batch_size={batch_size} "
          f"grad_accum_steps={grad_accum} effective_batch={effective}", flush=True)

    if is_main:
        _progress(epoch=0, total_epochs=epochs, stage="start", world_size=world)

    import torch  # imported here, after the MKL env var is set above
    cls = _model_class(variant)
    model = cls()

    try:
        model.train(dataset_dir=str(WORK), dataset_file="yolo", epochs=epochs,
                    batch_size=batch_size, grad_accum_steps=grad_accum,
                    lr=1e-4, output_dir=str(OUT / "runs"), early_stopping=True)
    except Exception as exc:
        oom = isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ()))
        if not oom and isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            oom = True
        if oom:
            sys.stderr.write(
                "[train] training ran out of GPU memory (VRAM): reduce "
                "training.batch_size or training.image_size, or lower effective "
                "batch; effective batch = batch_size x grad_accum_steps "
                f"(was {batch_size} x {grad_accum} = {effective})\\n")
            return 2
        raise

    # Under DDP, every rank trains but only rank 0 owns the artefacts. Non-main
    # ranks stop here so the checkpoint, metrics and report are written exactly once.
    if not is_main:
        print(f"[train] rank {rank}/{world} finished (rank 0 writes outputs)", flush=True)
        return 0

    # Pick the produced checkpoint: prefer an EMA/best *.pth (the smoothed one),
    # else the newest .pth. Filename may need a one-line tweak after the first run.
    ckpts = sorted((OUT / "runs").rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        sys.stderr.write("[train] no .pth checkpoint produced under /out/runs\\n")
        return 1
    ema = [p for p in ckpts if "ema" in p.name.lower()]
    best = [p for p in ckpts if "best" in p.name.lower()]
    chosen = (ema or best or ckpts)[-1]
    shutil.copyfile(chosen, OUT / "model.pt")
    print(f"[train] selected checkpoint: {chosen.name}", flush=True)

    m = _compute_metrics(chosen, WORK, variant)
    (OUT / "eval" / "metrics.json").write_text(json.dumps(m, indent=2))

    lines = [f"# {os.environ.get('MODEL_NAME', 'model')} — evaluation\\n",
             f"- variant: {variant}  epochs: {epochs}  image_size: {image_size}",
             (f"- batch_size: {batch_size}  grad_accum_steps: {grad_accum}  "
              f"effective_batch: {effective}"),
             f"- mAP@0.5: {m.get('mAP50')}",
             f"- mAP@0.5:0.95: {m.get('mAP50-95')}"]
    if m.get("note"):
        lines.append(f"- note: {m['note']}")
    (OUT / "evaluation_report.md").write_text("\\n".join(lines) + "\\n")

    # Final progress line + standard report.json (traceability contract).
    _progress(epoch=epochs, total_epochs=epochs, stage="done",
              **{k: v for k, v in m.items() if isinstance(v, (int, float))})
    classes = _dataset_classes(WORK)
    report = {
        "model": os.environ.get("MODEL_NAME", "model"),
        "job_type": "training",
        "variant": variant,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "effective_batch": effective,
        "checkpoint": chosen.name,
        "metrics": m,
        # Traceability: exactly what produced this result, so runs stay
        # reproducible/comparable (versions, class map, and the config used).
        "versions": _versions(),
        "dataset_version": os.environ.get("DATASET_VERSION") or None,
        "dataset_format": classes.get("dataset_format"),
        "num_classes": classes.get("num_classes"),
        "class_names": classes.get("names"),
        "config_used": cfg,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    # Also drop the exact config that ran, as its own artifact, for auditing.
    try:
        (OUT / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:  # noqa: BLE001 -- never fail the run over an audit artifact
        pass
    # The container runs as root; RF-DETR/Lightning checkpoints can be written
    # with restrictive perms that block the non-root worker from copying /out to
    # the NAS. Relax /out so every artefact is readable/traversable by the worker.
    subprocess.run(["chmod", "-R", "a+rwX", "/out"], check=False)
    print(f"[train] done: {m}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_RFDETR_PROCESS = '''\
"""RF-DETR processing/inference — serves the ACTIVE promoted model (/model/model.pt).

reads /data (images) + /config/job.yaml ; writes /out/report.json with images,
total_detections, by_class (class name -> count), conf_threshold ; exit 0.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

OUT = Path("/out")
DATA = Path("/data")
MODEL = Path("/model/model.pt")
CONFIG = Path("/config/job.yaml")

_VARIANTS = {
    "base": "RFDETRBase",
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _model_class(variant):
    import rfdetr
    name = _VARIANTS.get(str(variant).lower(), "RFDETRBase")
    return getattr(rfdetr, name, rfdetr.RFDETRBase)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        sys.stderr.write("no model at /model/model.pt — is a version promoted?\\n")
        return 1

    cfg = {}
    if yaml and CONFIG.is_file():
        cfg = yaml.safe_load(CONFIG.read_text()) or {}
    conf = float((cfg.get("process") or {}).get("confidence_threshold",
                 (cfg.get("process") or {}).get("conf", 0.25)))
    variant = (cfg.get("process") or {}).get("model_variant",
              (cfg.get("train") or {}).get("model_variant", "base"))

    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
                    for p in DATA.rglob(ext))
    if not images:
        sys.stderr.write("no images under /data\\n")
        return 1

    from PIL import Image
    cls = _model_class(variant)
    model = cls(pretrain_weights=str(MODEL))
    names = getattr(model, "class_names", None)

    per_class, total = {}, 0
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        det = model.predict(image, threshold=conf)
        class_ids = getattr(det, "class_id", None)
        if class_ids is None:
            continue
        total += len(class_ids)
        for c in class_ids.tolist() if hasattr(class_ids, "tolist") else list(class_ids):
            per_class[int(c)] = per_class.get(int(c), 0) + 1

    def _name(k):
        if isinstance(names, dict):
            return names.get(k, str(k))
        if isinstance(names, (list, tuple)) and 0 <= k < len(names):
            return names[k]
        return str(k)

    report = {"model": os.environ.get("MODEL_NAME", ""),
              "model_version": os.environ.get("MODEL_VERSION", ""),
              "images": len(images), "total_detections": total,
              "by_class": {_name(k): v for k, v in sorted(per_class.items())},
              "conf_threshold": conf}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[process] done: {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_CONTRACT_TEST = '''\
"""CPU-only contract checks for CI (no GPU, no platform)."""
import py_compile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_config_has_train_and_thresholds():
    cfgs = list((ROOT / "configs").glob("*.yaml"))
    assert cfgs, "no config file"
    cfg = yaml.safe_load(cfgs[0].read_text())
    assert "train" in cfg and "thresholds" in cfg


def test_scripts_compile():
    for name in ("train.py", "process.py", "entrypoint.py"):
        py_compile.compile(str(ROOT / name), doraise=True)
'''

_CUSTOM_TRAIN = '''\
"""Custom training — RUNNABLE REFERENCE (edit the marked sections).

Honours the platform contract: reads /data (ro) + /config/job.yaml (ro), writes
/out/model.pt and /out/eval/metrics.json, exits 0. As shipped it trains a trivial
baseline so the pipeline is GREEN from commit 1 — replace the marked block with
your real model + training loop (any framework).
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import yaml

OUT = Path("/out")
DATA = Path("/data")
CONFIG = Path("/config/job.yaml")


def _progress(**row):
    """Append one JSON line to /out/metrics.jsonl (live-progress contract).
    The dashboard reads epoch/total_epochs/metrics from here while training runs.
    """
    import time as _t
    row.setdefault("ts", _t.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        with open(OUT / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\\n")
    except OSError:
        pass


def main() -> int:
    # Standard output structure (platform contract).
    for d in ("eval", "artifacts", "predictions"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    cfg = (yaml.safe_load(CONFIG.read_text()) if CONFIG.is_file() else {}) or {}
    epochs = int(cfg.get("train", {}).get("epochs", 5))
    samples = sorted(p for p in DATA.rglob("*") if p.is_file())
    print(f"[train] epochs={epochs} samples={len(samples)}", flush=True)

    # ===== REPLACE: build + train your model on the data in /data =====
    # Emit one metrics.jsonl line per epoch so the platform shows live progress.
    accuracy = 0.0
    for ep in range(1, epochs + 1):
        accuracy = (1.0 if samples else 0.0) * (ep / epochs)   # placeholder curve
        _progress(epoch=ep, total_epochs=epochs, accuracy=round(float(accuracy), 4))
    model = {"kind": "baseline", "n_samples": len(samples), "epochs": epochs}
    # ==================================================================

    with open(OUT / "model.pt", "wb") as fh:        # required artifact
        pickle.dump(model, fh)
    # metric keys here must include whatever you set under `thresholds` in the config
    metrics = {"accuracy": round(float(accuracy), 4)}
    (OUT / "eval" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "report.json").write_text(json.dumps({
        "model": __import__("os").environ.get("MODEL_NAME", "model"),
        "job_type": "training", "epochs": epochs,
        "n_samples": len(samples), "metrics": metrics,
    }, indent=2))
    # Container runs as root; relax /out so the non-root worker can copy it to NAS.
    subprocess.run(["chmod", "-R", "a+rwX", "/out"], check=False)
    print(f"[train] wrote model.pt + metrics.json (accuracy={accuracy})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_CUSTOM_PROCESS = '''\
"""Custom processing/inference — RUNNABLE REFERENCE (edit the marked section).

Loads the promoted model at /model/model.pt, runs over /data, writes
/out/report.json, exits 0. Replace the marked block with your real inference.
"""
import json
import os
import pickle
import sys
from pathlib import Path

OUT = Path("/out")
DATA = Path("/data")
MODEL = Path("/model/model.pt")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        sys.stderr.write("no model at /model/model.pt — is a version promoted?\\n")
        return 1
    with open(MODEL, "rb") as fh:
        model = pickle.load(fh)
    inputs = sorted(p for p in DATA.rglob("*") if p.is_file())

    # ===== REPLACE: run real inference with `model` over `inputs` =====
    kind = model.get("kind", "custom") if isinstance(model, dict) else "custom"
    results = len(inputs)
    # =================================================================

    report = {"model": os.environ.get("MODEL_NAME", ""),
              "model_version": os.environ.get("MODEL_VERSION", ""),
              "model_kind": kind,
              "inputs": len(inputs), "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[process] done: {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\nruns/\nout/\n*.pt\n.DS_Store\n"



def _dockerfile_custom() -> str:
    # Lightweight base for the pure-Python custom reference. No CUDA/apt, so no
    # mirror dependency. Swap the base + pip installs for your framework
    # (torch, tensorflow, scikit-learn, ...) when you implement a real model.
    return """\
# Lightweight custom-model image. The runnable reference needs only PyYAML;
# your model's own libraries (torch, tensorflow, timm, scikit-learn, ...) are
# declared in model.yaml (`dependencies:`) or requirements.txt and installed here.
FROM python:3.11-slim
RUN pip install --no-cache-dir pyyaml
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
WORKDIR /app
COPY train.py process.py entrypoint.py /app/
ENTRYPOINT ["python", "/app/entrypoint.py"]
"""


def _dockerfile_rfdetr() -> str:
    # RF-DETR (Apache-2.0) detection image. RF-DETR reads YOLO-format datasets
    # directly at train time via dataset_file="yolo" (no COCO conversion).
    return """\
# RF-DETR (Apache-2.0) detection image. RF-DETR pulls transformers 5.x, which
# imports torch.distributed.tensor.DTensor — only present in PyTorch >= 2.5 — so
# this image uses a torch-2.5 base and works with numpy 2 (no numpy<2 pin needed).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# Multi-GPU (DDP) spawns training subprocesses; the conda image's Intel MKL
# otherwise picks MKL_THREADING_LAYER=INTEL, which clashes with libgomp.so.1
# (GNU OpenMP) in the child and aborts multi-GPU runs. Force the GNU layer so it
# matches. Harmless for single-GPU. See torch.distributed / DDP.
ENV MKL_THREADING_LAYER=GNU
# Pin the whole RF-DETR stack to a known-consistent set. Unpinned, pip pulls the
# newest transformers, which outran the torch base (transformers 5.x needs
# torch.distributed.tensor.DTensor -> torch>=2.5). These versions are proven to
# resolve together on the 2.5.1 base; torch/torchvision come from the base image.
# The [train,loggers] extra pulls the training deps (pytorch_lightning, etc.);
# a plain `pip install rfdetr` gives only inference deps and training aborts with
# "RF-DETR training dependencies are missing".
RUN pip install --no-cache-dir "rfdetr[train,loggers]==1.9.1" "transformers==5.14.1" "supervision==0.30.0" pyyaml
# Build-time self-check: importing rfdetr + a training dep exercises the
# torch/transformers/lightning ABI, so a mismatched or missing combo fails the
# BUILD here rather than a GPU run.
RUN python -c "import rfdetr, supervision, torch, pytorch_lightning; print('env-check OK rfdetr', torch.__version__)"
# Extra dependencies declared in model.yaml (`dependencies:`) or added to
# requirements.txt by hand — installed on top of the preset's own stack.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
WORKDIR /app
COPY train.py process.py entrypoint.py /app/
ENTRYPOINT ["python", "/app/entrypoint.py"]
"""


def _config_yaml(spec: dict) -> str:
    import yaml
    t = spec.get("training", {})
    v = spec.get("validation", {})
    p = spec.get("processing", {})
    train = {"epochs": t.get("epochs", 50)}
    # Pass through RF-DETR native batching controls + variant when present so the
    # generated train.py can read them from /config/job.yaml.
    if t.get("model_variant"):
        train["model_variant"] = t["model_variant"]
    if isinstance(t.get("image_size"), int):
        train["image_size"] = t["image_size"]
    if isinstance(t.get("batch_size"), int):
        train["batch_size"] = t["batch_size"]
    for key in ("grad_accum_steps", "grad_accumulation_steps", "effective_batch"):
        if isinstance(t.get(key), int):
            train[key] = t[key]
    doc = {
        "train": train,
        "thresholds": {k: val for k, val in (v.get("thresholds") or {}).items()},
        "thresholds_max": v.get("thresholds_max", {}),
        "process": {"conf": p.get("confidence_threshold", 0.25),
                    "confidence_threshold": p.get("confidence_threshold", 0.25)},
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _gitlab_ci(spec: dict) -> str:
    m = spec["model"]["name"]
    ds = spec["dataset"]
    ns = spec.get("repository", {}).get("namespace", "vgreens-models")
    return f"""\
include:
  - project: '{ns}/vgreens-model-template'
    ref: main
    file: '/ci/model-ci.yml'

variables:
  MODEL_NAME: {m}
  DATASET_NAME: {ds['name']}
  DATASET_VERSION: {ds.get('version', 'v1')}
  CONFIG_PATH: /mnt/vgreens_storage/ai/job-configs/{m}.yaml
"""


def _readme(spec: dict, preset: str) -> str:
    m = spec["model"]
    ds = spec["dataset"]
    return f"""\
# {spec.get('repository', {}).get('name', m['name'])}

{m.get('display_name', m['name'])} — task: {m.get('task')} · preset: `{preset}`.
Generated by `vgreens-ai model onboard`.

Dataset: `{ds['name']}:{ds.get('version', 'v1')}` (frozen in the registry).

## Deploy this model

1. Publish the job-config to the platform (no SSH to the GPU box):
   ```bash
   vgreens-ai config publish {m['name']} --file configs/{m['name']}.yaml
   ```
2. Push this repo. CI (on the self-hosted runner) runs: lint → test →
   build+scan+push image → manual training submission through the Job API.

Or drive it entirely from the CLI once the image is built:
```bash
vgreens-ai job train --model {m['name']} --dataset {ds['name']}:{ds.get('version', 'v1')} \\
  --image $CI_REGISTRY_IMAGE:latest --config /mnt/vgreens_storage/ai/job-configs/{m['name']}.yaml
vgreens-ai model show {m['name']}
vgreens-ai model promote {m['name']} <version>
```

## Dependencies

Extra Python packages this model needs (e.g. a custom CNN's `torch`, `timm`,
`scikit-learn`) are declared once in `model.yaml` under `dependencies:` and
generated into `requirements.txt`, which the image installs at build time — no
need to hand-edit the Dockerfile. You can also add pip lines to
`requirements.txt` directly.
"""


def _requirements_txt(spec: dict) -> str:
    """Generate requirements.txt from the spec's optional `dependencies:` list.

    Extra Python packages the model needs (e.g. a custom CNN's torch/timm) are
    declared once in model.yaml and installed into the image at build time, so
    the user never has to hand-edit the Dockerfile. The file is always emitted
    (empty except for a header when no dependencies are declared) so the
    Dockerfile's `pip install -r requirements.txt` layer is preset-independent.
    """
    deps = spec.get("dependencies") or []
    header = (
        "# Extra Python dependencies for this model, installed into the image at\n"
        "# build time on top of the preset's own stack. Declared in model.yaml\n"
        "# under `dependencies:`; you may also add pip requirement lines by hand.\n"
    )
    lines = [str(d).strip() for d in deps if str(d).strip()]
    body = ("\n".join(lines) + "\n") if lines else ""
    return header + body



def _render_rfdetr(spec: dict) -> dict:
    m = spec["model"]["name"]
    preset = spec["template"]["preset"]
    return {
        "entrypoint.py": _ENTRYPOINT_PY,
        "train.py": _RFDETR_TRAIN,
        "process.py": _RFDETR_PROCESS,
        "Dockerfile": _dockerfile_rfdetr(),
        "requirements.txt": _requirements_txt(spec),
        f"configs/{m}.yaml": _config_yaml(spec),
        ".gitlab-ci.yml": _gitlab_ci(spec),
        "tests/test_contract.py": _CONTRACT_TEST,
        "README.md": _readme(spec, preset),
        ".gitignore": _GITIGNORE,
    }


def _render_custom(spec: dict) -> dict:
    m = spec["model"]["name"]
    return {
        "entrypoint.py": _ENTRYPOINT_PY,
        "train.py": _CUSTOM_TRAIN,
        "process.py": _CUSTOM_PROCESS,
        "Dockerfile": _dockerfile_custom(),
        "requirements.txt": _requirements_txt(spec),
        f"configs/{m}.yaml": _config_yaml(spec),
        ".gitlab-ci.yml": _gitlab_ci(spec),
        "tests/test_contract.py": _CONTRACT_TEST,
        "README.md": _readme(spec, "custom_python"),
        ".gitignore": _GITIGNORE,
    }


def render(spec: dict) -> dict:
    preset = (spec.get("template") or {}).get("preset")
    if preset == "rfdetr_detection":
        return _render_rfdetr(spec)
    if preset == "custom_python":
        return _render_custom(spec)
    # rfdetr_segmentation / cnn_classification are declared presets but not yet
    # implemented as real working code — emit the runnable custom scaffold so
    # onboarding stays green; the engineer fills in the marked sections.
    if preset in ("rfdetr_segmentation", "cnn_classification"):
        return _render_custom(spec)
    raise ValueError(f"generator for preset '{preset}' is not implemented yet "
                     "(available: rfdetr_detection, rfdetr_segmentation, "
                     "cnn_classification, custom_python)")


def generate(spec: Any, out_dir: str, dry_run: bool = False, force: bool = False) -> list[str]:
    files = render(spec)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{out} exists and is not empty — use --force to overwrite")
    if not dry_run:
        for rel, content in files.items():
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    return sorted(files)
