# Bring your own model (custom_python)

Use this when your model isn't a standard preset task — a bespoke architecture,
a different framework (TensorFlow, scikit-learn, plain PyTorch), or a non-YOLO
task. The platform scaffolds everything around your model; you implement the two
functions that honour the [contract](model-contract.md).

## What you write vs what you get

**You write:** the body of `train.py` / `process.py` (your model + training/inference)
and the `Dockerfile` dependencies.
**The platform gives you:** dataset freezing, the job queue, GPU allocation, the
model registry, versioning, metric-agnostic validation, promotion/rollback,
inference serving, monitoring, the web UI, and CI.

Unlike before, the generated `custom_python` skeleton is **runnable from commit 1**:
`train.py`/`process.py` train a trivial baseline, write the required artifacts, and
pass CI green. You then replace the clearly-marked sections with your real model —
you're editing working code, not filling in blanks behind a red pipeline.

## Fast path

```bash
vgreens-ai model init                       # interactive: writes a valid model.yaml
vgreens-ai model onboard --spec model.yaml --create-repo
```

`model init` prompts for name/task/preset/dataset/thresholds/repo and writes the
spec — no hand-editing YAML. `onboard --create-repo` generates the repo, creates
the GitLab project **inside the `vgreens-models` group** (so it inherits the
runner, CI variables, and shared CI template), commits, and pushes — the first
pipeline starts automatically. (`--create-repo` needs `GITLAB_TOKEN`, a PAT with
`api` scope, in your environment.)

Without `--create-repo`, onboard just writes the files and you push yourself.

## Step by step

### 1. Freeze a dataset (if you need one)
See `../datasets/`. Your `train.py` reads `/data` in whatever layout you froze.
(The audit is currently YOLO-detection oriented; non-YOLO data can still be
staged via a local import — your code decides how to read it.)

### 2. Write the spec
Either `vgreens-ai model init` (recommended) or by hand:
```yaml
schema_version: 1
model: { name: my_model, task: custom, owner: you, description: "..." }
template: { preset: custom_python }
dataset: { name: my_data, version: v1 }
training: { epochs: 20, image_size: 224, batch_size: 32, gpu_count: 1 }
validation: { thresholds: { accuracy: 0.90 } }   # your metric, your bar
processing: { confidence_threshold: 0.5 }
repository: { namespace: vgreens-models, name: vgreens-model-my-model }
```
```bash
vgreens-ai model validate-spec model.yaml
vgreens-ai model onboard --spec model.yaml
```

### 3. Implement train.py
Replace the marked block. Read `/config/job.yaml` + `/data`, train with **any**
framework, write `/out/model.pt` and `/out/eval/metrics.json`, exit 0. The metric
keys in `metrics.json` must include whatever you set under `thresholds`.

The shipped reference already does the plumbing (reads config, writes a pickled
model + metrics) — swap the baseline for your real model and loss.

### 4. Implement process.py
Load `/model/model.pt` (the promoted model), run inference over `/data`, write
`/out/report.json`, exit 0. Again, replace only the marked block.

### 5. Edit the Dockerfile
The generated base is `python:3.11-slim` with just PyYAML — enough for the
pure-Python reference. Swap the base + `pip install` for your framework
(`torch`, `tensorflow`, `scikit-learn`, ...). Bake any weights so runtime needs
no internet. Prefer headless wheels (e.g. `opencv-python-headless`) so the build
doesn't depend on system GUI libraries or the OS package mirror.

### 6. Keep the tests meaningful
The generated `tests/test_contract.py` compiles the scripts and checks the config
has `train` + `thresholds`. `pytest tests/ -q` must pass — add a real smoke test
for your model when you can.

### 7. Publish config, build, train, promote
```bash
# publish the job-config to the platform (no SSH/cp on the GPU box):
vgreens-ai config publish my_model --file configs/my_model.yaml

# push the repo — CI builds+scans+pushes the image and offers a manual
# submit_training. Or drive it from the CLI once an image exists:
vgreens-ai job train --model my_model --dataset my_data:v1 \
  --image registry.gitlab.com/vgreens-models/vgreens-model-my-model:latest \
  --config /mnt/vgreens_storage/ai/job-configs/my_model.yaml
vgreens-ai model show my_model
vgreens-ai model promote my_model <version>
```

## If this recurs, make it a preset
The `custom_python` path is the escape hatch for one-offs. If your team keeps
building the same kind of custom model, add a preset for it in
`cli/vgreens_ai/modelspec.py` (`PRESETS`) plus a `_render_*` in `modelgen.py` that
emits working code — then those models are back to the one-command golden path,
exactly like `rfdetr_detection`.
