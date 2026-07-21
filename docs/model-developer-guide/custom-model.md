# Bring your own model (custom_python)

Use this when your model isn't a standard preset task — a bespoke architecture,
a different framework (TensorFlow, scikit-learn, plain PyTorch), or a non-YOLO
task. The platform scaffolds everything around your model; you implement the two
functions that honour the [contract](model-contract.md).

## What you write vs what you get

**You write:** `train.py`, `process.py` (your model + training/inference), and
the `Dockerfile` dependencies.
**The platform gives you:** dataset freezing, the job queue, GPU allocation, the
model registry, versioning, metric-agnostic validation, promotion/rollback,
inference serving, monitoring, the web UI, and CI scaffolding.

## Step by step

### 1. Freeze a dataset (if you need one)
See `../datasets/`. Your `train.py` reads `/data` in whatever layout you froze.
(The audit is currently YOLO-detection oriented; non-YOLO data can still be
staged via a local import — your code decides how to read it.)

### 2. Write model.yaml with the custom preset
```yaml
schema_version: 1
model: { name: my_model, task: custom, owner: you, description: "..." }
template: { preset: custom_python, template_version: v1.0.0 }
dataset: { name: my_data, version: v1 }
training: { base_model: "", epochs: 20, image_size: 224, batch_size: 32, gpu_count: 1 }
validation: { thresholds: { accuracy: 0.90 } }   # your metric, your bar
processing: { confidence_threshold: 0.5 }
repository: { namespace: achrefmessai, name: vgreens-model-my-model }
```
```bash
vgreens-ai model validate-spec model.yaml
vgreens-ai model onboard --spec model.yaml
```
The skeleton's `train.py`/`process.py` raise `NotImplementedError` and the
contract test fails on purpose — that's your TODO.

### 3. Implement train.py
Read `/config` + `/data`, train with **any** framework, write `/out/model.pt`
and `/out/eval/metrics.json`, exit 0. Minimal working example:
```python
import json, sys
from pathlib import Path
import yaml

OUT, DATA, CONFIG = Path("/out"), Path("/data"), Path("/config/job.yaml")

def main() -> int:
    cfg = (yaml.safe_load(CONFIG.read_text()) if CONFIG.is_file() else {}).get("train", {})
    epochs = int(cfg.get("epochs", 5))

    # ---- YOUR model + training loop reading from /data ----
    # model = build(); train(model, DATA, epochs); acc = evaluate(model, DATA)
    acc = 0.93

    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    # save_model(model, OUT / "model.pt")            # <-- required artifact
    (OUT / "model.pt").write_text("...your serialized model...")
    (OUT / "eval" / "metrics.json").write_text(json.dumps({"accuracy": round(acc, 4)}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
```
The metric keys in `metrics.json` must include whatever you set in `thresholds`.

### 4. Implement process.py
Load `/model/model.pt` (the promoted model), run inference over `/data`, write
`/out/report.json`, exit 0.

### 5. Edit the Dockerfile
The generated base is PyTorch+ultralytics; for a custom model swap it for what
you need and `pip install` your libraries. Bake any weights so runtime needs no
internet. Example for a light, pure-Python model:
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir pyyaml numpy scikit-learn
WORKDIR /app
COPY train.py process.py entrypoint.py /app/
ENTRYPOINT ["python", "/app/entrypoint.py"]
```

### 6. Make the tests real
Replace the placeholder `tests/test_contract.py` with a real check (scripts
compile / a tiny smoke). `pytest tests/ -q` must pass.

### 7. Build, train, promote — same as any model
```bash
# push the repo, then on gpu-worker-01:
docker build -t vgreens/my-model:v1 .
sudo cp configs/my_model.yaml /mnt/vgreens_storage/ai/job-configs/
# from your machine:
vgreens-ai job train --model my_model --dataset my_data:v1 \
  --image vgreens/my-model:v1 --config /mnt/vgreens_storage/ai/job-configs/my_model.yaml
vgreens-ai model show my_model
vgreens-ai model promote my_model <version>
```

## If this recurs, make it a preset
The `custom_python` path is the escape hatch for one-offs. If your team keeps
building the same kind of custom model, ask the platform maintainer to add a
preset for it — then those models are back to the one-command golden path.
