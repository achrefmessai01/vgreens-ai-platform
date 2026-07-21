# model.yaml reference

The single declarative source of truth for a model. Validate with
`vgreens-ai model validate-spec model.yaml`.

```yaml
schema_version: 1                    # spec format version (currently 1)

model:
  name: strawberry_disease           # registry key. lowercase [a-z0-9_], starts alnum, <=63
  display_name: Strawberry Disease   # human label (UI)
  task: object_detection             # object_detection | classification | segmentation | custom
  owner: ai-team
  description: One-line description.

template:
  preset: ultralytics_detection      # see presets.md; must match model.task
  template_version: v1.0.0

dataset:
  name: strawberry_disease_auto      # MUST be a frozen dataset (vgreens-ai dataset list)
  version: v1                        # a frozen version, or 'latest_frozen'

training:
  base_model: yolo11n.pt             # pretrained backbone to fine-tune (preset models)
  epochs: 50
  image_size: 640
  batch_size: 16                     # int, or 'auto' for supported presets
  gpu_count: 1                       # 1 or 2 (2 needs the jobs:two_gpu permission)
  trigger: on_main                   # CI: auto-submit training when merged to main

validation:
  thresholds:                        # PROMOTION GATE — achieved metric must be >= each
    mAP50: 0.50
  thresholds_max: {}                 # optional upper bounds (lower-is-better metrics)

processing:
  confidence_threshold: 0.30         # inference confidence (detection)

repository:
  namespace: achrefmessai            # GitLab namespace (configurable)
  name: vgreens-model-strawberry-disease
```

## Field notes

- **model.task ↔ template.preset must agree.** `validate-spec` enforces it.
- **dataset** must reference something frozen; the version pins reproducibility.
- **validation.thresholds** is the promotion bar and is **metric-agnostic**: any
  key here is compared against the same key your container writes in
  `metrics.json`. If thresholds are empty/`null`, a trained version is marked
  `baseline_only` and **cannot be promoted** until you set thresholds and
  revalidate — this is intentional (no accidental promotion of an unmeasured model).
- **training.gpu_count: 2** requires the two-GPU permission and shows a capacity
  warning; default is 1.
- **repository.namespace** is configurable — don't assume a personal namespace.

## Validation states of a trained version

| State | Meaning |
|-------|---------|
| `validated` | met all thresholds — promotable |
| `rejected` | missed a threshold |
| `baseline_only` / `thresholds_not_configured` | no thresholds set — not promotable |
