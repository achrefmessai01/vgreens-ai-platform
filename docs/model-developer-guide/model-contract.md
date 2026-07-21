# The container contract

Every model image — preset or custom — must obey this. If it does, the platform
runs it, versions it, validates it, serves it, and shows it in the UI. The
platform does **not** care what framework or code is inside.

## What the container receives

| Path | Mode | When | Meaning |
|------|------|------|---------|
| `/data` | read-only | always | the staged dataset version |
| `/config/job.yaml` | read-only | always | hyperparameters + thresholds (your `configs/<name>.yaml`) |
| `/out` | read-write | always | where you write results |
| `/model/model.pt` | read-only | processing only | the ACTIVE promoted model |

Environment variables: `JOB_ID`, `JOB_TYPE` (`training`|`processing`),
`MODEL_NAME`, `MODEL_VERSION`, `DATASET_VERSION`, `TENANT`.

`entrypoint.py` dispatches on `JOB_TYPE`: `training` → `train.py`,
`processing` → `process.py`.

## What the container must produce

**Training** (`JOB_TYPE=training`):
- `/out/model.pt` — your trained artifact (any format; the registry stores it).
- `/out/eval/metrics.json` — a flat JSON object of achieved metrics, e.g.
  `{"mAP50": 0.71, "precision": 0.68}` or `{"accuracy": 0.93}`.
- exit code `0` on success (non-zero → the job is marked failed).

**Processing** (`JOB_TYPE=processing`):
- `/out/report.json` — your inference summary.
- exit code `0`.

## How validation uses metrics.json

Metric-agnostic: for each key in your config's `thresholds`, the registry looks
up the **same key** in `metrics.json` and requires `achieved >= threshold`
(and `<= ` for `thresholds_max`). Miss any → `rejected`. No thresholds → the
version is `baseline_only` and can't be promoted.

So the keys must line up:
```yaml
# configs/<name>.yaml
thresholds: { accuracy: 0.90 }
```
```python
# your train.py writes
json.dump({"accuracy": 0.93}, open("/out/eval/metrics.json", "w"))
```

## Rules of thumb

- Read everything from `/data` and `/config`; never hard-code paths or reach out
  to the NAS/registry/queue directly.
- Never require internet at runtime — bake weights/deps into the image at build.
- Write outputs only under `/out`.
- Fail loudly (non-zero exit) if something's wrong — a silent success that wrote
  no model is worse than a clean failure.
