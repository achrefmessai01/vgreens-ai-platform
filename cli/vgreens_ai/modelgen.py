"""Model repository generator.

Renders a complete, ready-to-push model repo from a validated model.yaml.
The runnable code (entrypoint/train/process/Dockerfile) is generic across
Ultralytics detection models — the same contract-compliant code proven by
vgreens-model-strawberry-disease. Only config/CI/README are templated per spec.

custom_python emits a skeleton whose contract test FAILS until the engineer
implements train()/process() — never placeholder code that silently succeeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# --- constant, generic Ultralytics-detection code (contract-compliant) ------ #
_ENTRYPOINT_PY = '''\
"""Container entrypoint. Dispatches on JOB_TYPE: training -> train.py, processing -> process.py."""
import os
import runpy

mode = os.environ.get("JOB_TYPE", "training")
script = "/app/process.py" if mode == "processing" else "/app/train.py"
print(f"[entrypoint] JOB_TYPE={mode} -> {script}", flush=True)
runpy.run_path(script, run_name="__main__")
'''

_TRAIN_PY = '''\
"""YOLO detection training — platform contract entrypoint.

reads  /data (ro), /config/job.yaml ; writes /out/model.pt, /out/eval/metrics.json,
/out/evaluation_report.md ; exit 0 on success. /data is copied to a writable dir
because ultralytics writes a label cache next to the labels.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

OUT = Path("/out")
CONFIG = Path("/config/job.yaml")
WORK = Path("/tmp/ds")


def main() -> int:
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text()) if CONFIG.is_file() else {}
    t = (cfg or {}).get("train", {})
    model_name = t.get("model", "yolo11n.pt")
    epochs = int(t.get("epochs", 50))
    imgsz = int(t.get("imgsz", 640))
    batch = t.get("batch", 16)

    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree("/data", WORK)
    data_yaml = WORK / "data.yaml"
    dy = yaml.safe_load(data_yaml.read_text())
    dy["path"] = str(WORK)
    data_yaml.write_text(yaml.dump(dy))

    print(f"[train] model={model_name} epochs={epochs} imgsz={imgsz} batch={batch}", flush=True)
    model = YOLO(model_name)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
                project=str(OUT / "runs"), name="train", exist_ok=True, plots=True, verbose=True)

    metrics = model.val(data=str(data_yaml), device=0, project=str(OUT / "runs"),
                        name="val", exist_ok=True)
    m = {"mAP50": round(float(metrics.box.map50), 4), "mAP50-95": round(float(metrics.box.map), 4),
         "precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4)}
    (OUT / "eval" / "metrics.json").write_text(json.dumps(m, indent=2))

    best = OUT / "runs" / "train" / "weights" / "best.pt"
    if not best.exists():
        sys.stderr.write("[train] no best.pt produced\\n")
        return 1
    shutil.copyfile(best, OUT / "model.pt")
    (OUT / "evaluation_report.md").write_text(
        f"# {os.environ.get('MODEL_NAME', 'model')} — evaluation\\n\\n"
        f"- epochs: {epochs} imgsz: {imgsz} batch: {batch}\\n\\n"
        f"- mAP@0.5: {m['mAP50']}\\n- mAP@0.5:0.95: {m['mAP50-95']}\\n"
        f"- precision: {m['precision']}\\n- recall: {m['recall']}\\n")
    print(f"[train] done: {m}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_PROCESS_PY = '''\
"""Processing/inference — serves the ACTIVE promoted model (/model/model.pt)."""
import json
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

OUT = Path("/out")
DATA = Path("/data")
MODEL = Path("/model/model.pt")
CONFIG = Path("/config/job.yaml")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        sys.stderr.write("no model at /model/model.pt — is a version promoted?\\n")
        return 1
    conf = 0.25
    if yaml and CONFIG.is_file():
        conf = float((yaml.safe_load(CONFIG.read_text()) or {}).get("process", {}).get("conf", conf))
    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG") for p in DATA.rglob(ext))
    if not images:
        sys.stderr.write("no images under /data\\n")
        return 1
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names or []))
    per_class, total = {}, 0
    for i in range(0, len(images), 64):
        for r in model.predict(source=[str(p) for p in images[i:i + 64]], device=device, conf=conf, verbose=False):
            total += int(len(r.boxes))
            for c in r.boxes.cls.tolist():
                per_class[int(c)] = per_class.get(int(c), 0) + 1
    report = {"model": os.environ.get("MODEL_NAME", ""), "model_version": os.environ.get("MODEL_VERSION", ""),
              "images": len(images), "total_detections": total,
              "by_class": {names.get(k, str(k)): v for k, v in sorted(per_class.items())},
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
"""Custom training. IMPLEMENT THIS: read /data + /config/job.yaml, train, and write
/out/model.pt + /out/eval/metrics.json. Must exit 0 on success."""
import sys


def main() -> int:
    raise NotImplementedError("implement training for this custom model")


if __name__ == "__main__":
    sys.exit(main())
'''

_CUSTOM_CONTRACT = '''\
"""Contract test — FAILS until train()/process() are implemented."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


@pytest.mark.parametrize("script", ["train.py", "process.py"])
def test_not_placeholder(script):
    # A generated custom model must be implemented before it can pass CI.
    with pytest.raises(NotImplementedError):
        _run(script)()
    pytest.fail(f"{script} still raises NotImplementedError — implement it")
'''

_GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\nruns/\nout/\n*.pt\n.DS_Store\n"


def _dockerfile(base_model: str) -> str:
    return f"""\
# {base_model} detection image (ultralytics). PyTorch CUDA base proven on this host.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "ultralytics==8.3.0" pyyaml
RUN python -c "from ultralytics import YOLO; YOLO('{base_model}')"
WORKDIR /app
COPY train.py process.py entrypoint.py /app/
ENTRYPOINT ["python", "/app/entrypoint.py"]
"""


def _config_yaml(spec: dict) -> str:
    import yaml
    t = spec.get("training", {})
    v = spec.get("validation", {})
    p = spec.get("processing", {})
    doc = {
        "train": {"model": t.get("base_model", "yolo11n.pt"),
                  "epochs": t.get("epochs", 50), "imgsz": t.get("image_size", 640),
                  "batch": t.get("batch_size", 16) if isinstance(t.get("batch_size"), int) else 16},
        "thresholds": {k: val for k, val in (v.get("thresholds") or {}).items()},
        "thresholds_max": v.get("thresholds_max", {}),
        "process": {"conf": p.get("confidence_threshold", 0.25)},
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _gitlab_ci(spec: dict) -> str:
    m = spec["model"]["name"]
    ds = spec["dataset"]
    ns = spec.get("repository", {}).get("namespace", "achrefmessai")
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

## Local build + run (on the GPU server)

```bash
docker build -t vgreens/{m['name']}:v1 .
sudo cp configs/{m['name']}.yaml /mnt/vgreens_storage/ai/job-configs/
vgreens-ai job train --model {m['name']} --dataset {ds['name']}:{ds.get('version', 'v1')} \\
  --image vgreens/{m['name']}:v1 --config /mnt/vgreens_storage/ai/job-configs/{m['name']}.yaml
```

CI (once a GitLab Runner is registered): push → lint → test → build+scan+push image →
manual training submission through the Job API.
"""


def _render_ultra(spec: dict) -> dict:
    m = spec["model"]["name"]
    base = spec.get("training", {}).get("base_model", "yolo11n.pt")
    preset = spec["template"]["preset"]
    return {
        "entrypoint.py": _ENTRYPOINT_PY,
        "train.py": _TRAIN_PY,
        "process.py": _PROCESS_PY,
        "Dockerfile": _dockerfile(base),
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
        "process.py": _CUSTOM_TRAIN.replace("training", "processing"),
        "Dockerfile": _dockerfile("yolo11n.pt"),
        f"configs/{m}.yaml": _config_yaml(spec),
        ".gitlab-ci.yml": _gitlab_ci(spec),
        "tests/test_contract.py": _CUSTOM_CONTRACT,
        "README.md": _readme(spec, "custom_python") + "\n> TODO: implement train.py and process.py.\n",
        ".gitignore": _GITIGNORE,
    }


def render(spec: dict) -> dict:
    preset = (spec.get("template") or {}).get("preset")
    if preset in ("ultralytics_detection",):
        return _render_ultra(spec)
    if preset == "custom_python":
        return _render_custom(spec)
    raise ValueError(f"generator for preset '{preset}' is not implemented yet "
                     "(available: ultralytics_detection, custom_python)")


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
