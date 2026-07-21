# Model developer guide

How to add and ship a model on the vGreens AI platform. Start here.

- **[golden-path.md](golden-path.md)** — the five-minute path for a supported
  model (Ultralytics detection). Start here if your model is a standard task.
- **[model-spec.md](model-spec.md)** — every field of `model.yaml`.
- **[presets.md](presets.md)** — the available presets and when to use each.
- **[custom-model.md](custom-model.md)** — bring your own model (any framework)
  with the `custom_python` preset.
- **[model-contract.md](model-contract.md)** — the container contract every
  model image must honour (the one hard rule).

## The mental model

You author one file — `model.yaml` — and the platform turns it into a running,
versioned, promotable model. Two things are always true:

1. Your model runs as a **Docker container** that obeys the **contract**
   (`/data` + `/config` in → `/out/model.pt` + `/out/eval/metrics.json` out,
   exit 0). See `model-contract.md`.
2. Only **two actions are manual by design**: approving a dataset licence, and
   promoting a trained version to production. Everything else is automated.

## Prerequisites

- Platform access (`vgreens-ai config init`, or the web app login).
- A role of **operator** (submit training) or **admin** (promote).
- A **frozen dataset** to train on — see `../datasets/overview.md`.
