# Onboarding a model — quick start

Onboarding turns **one small file** into a complete, ready-to-train model on the
platform. You describe the model; the platform generates the code, builds a
container image, and gets it ready — you don't hand-write any training plumbing.

## The one file you write: `model.yaml`

```yaml
schema_version: 1
model:      { name: my_detector, task: object_detection }
template:   { preset: rfdetr_detection }     # RF-DETR (Apache-2.0)
dataset:    { name: my_dataset, version: v1 } # must already be frozen
training:   { epochs: 20, image_size: 560, batch_size: 4, effective_batch: 16, gpu_count: 1 }
validation: { thresholds: { mAP50: 0.30 } }   # promotion gate
repository: { namespace: vgreens-models, name: my-detector }
```

That's the whole spec. A few notes:

- **`preset`** picks the recipe (see the table below).
- **`dataset`** must point at a **frozen** version — import and freeze it first.
- **`effective_batch`** is what matters for learning; the platform derives the
  gradient-accumulation steps so it fits the GPU.
- **`thresholds`** is the bar a trained version must clear to be promotable.
- Need extra libraries (e.g. a custom model)? add `dependencies: [timm==1.0.9, ...]`
  and they're installed into the image automatically.

## Presets

| Preset | Task | What you get |
|---|---|---|
| `rfdetr_detection` | object detection | working training + inference code — no edits |
| `rfdetr_segmentation` | segmentation | scaffold (start from custom_python) |
| `cnn_classification` | classification | scaffold (bring your own CNN) |
| `custom_python` | anything | runnable skeleton you fill in |

*(Ultralytics/YOLO is intentionally not offered — it's AGPL-3.0, not free for
commercial use. RF-DETR is Apache-2.0.)*

## The four steps

1. **Write the spec** — the `model.yaml` above (or scaffold it: `vgreens-ai model init`).
2. **Onboard** — one command generates the repo and pushes it:
   ```bash
   vgreens-ai model onboard --spec model.yaml --create-repo
   ```
   It generates a full repo: `Dockerfile`, `train.py`, `process.py`, `requirements.txt`,
   tests, and CI config.
3. **CI builds it** — on push, GitLab CI validates, builds the image (kaniko),
   scans it (Trivy), inventories it (SBOM/Syft) and pushes it to the registry.
   Nothing runs on a GPU yet.
4. **Train** — submit through the platform (web app *Submit*, or CLI):
   ```bash
   vgreens-ai job train --model my_detector --dataset my_dataset:v1
   vgreens-ai job watch <job_id>
   vgreens-ai job result <job_id>
   ```

By the time you train, everything the run needs already exists: a **frozen
dataset**, a **validated spec**, and a **built, scanned image**.

## After training

A successful run registers a new **model version** with its metrics. A human then
**promotes** a validated version to be the active/production model
(`vgreens-ai model promote <model> <version>`), and processing jobs serve that
active model. You can see metrics, artifacts and logs on the job's page, or with
`vgreens-ai job result <id>` / `job logs <id>`.

## Common gotchas

- **Dataset not frozen** → import it first; training points at a frozen version.
- **RF-DETR image size** → keep it divisible by 56 (e.g. 560).
- **Out of GPU memory** → lower `batch_size` (and raise accumulation) or `image_size`;
  the run fails with a clear VRAM message telling you which to change.
- **1 vs 2 GPU** → 1 GPU leaves the other free for production; 2-GPU is faster but
  admin-only and uses both.
