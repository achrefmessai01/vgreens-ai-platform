# Presets

A preset decides what code `vgreens-ai model onboard` generates. List them:

```bash
vgreens-ai model presets
```

| Preset | Task | Generates working code? |
|--------|------|-------------------------|
| `ultralytics_detection` | object_detection | yes — ready to train, no edits |
| `ultralytics_classification` | classification | (planned) |
| `ultralytics_segmentation` | segmentation | (planned) |
| `custom_python` | custom | skeleton only — you implement it |

## Which to use

- **Your task is standard object detection?** Use `ultralytics_detection`. You
  write only `model.yaml`; the generated `train.py`/`process.py` are complete and
  identical to the proven reference model. → [golden-path.md](golden-path.md)

- **Your model is anything else** (a bespoke architecture, another framework, a
  non-YOLO task)? Use `custom_python`. The generator scaffolds the repo, contract,
  Dockerfile, CI and tests; you implement `train.py`/`process.py`. →
  [custom-model.md](custom-model.md)

## Adding a new preset

If a custom pattern recurs (e.g. your team keeps building PyTorch classifiers),
it's worth turning it into a preset so those models become one-command again.
Presets are platform-owned templates in the CLI's generator (`modelgen.py`);
adding one is a platform change, not per-model work. Ask the platform maintainer.
