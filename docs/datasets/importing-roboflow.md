# Importing from Roboflow

## Prerequisite (one-time, server side)

The Roboflow API key lives on the platform, never in a request. Set it in
`/opt/vgreens-ai/.env` as `ROBOFLOW_API_KEY=…` and restart the dataset worker:

```bash
sudo systemctl restart vgreens-dataset-worker
```

The key is used only by the worker to fetch the export; it is never logged,
stored in manifests, or returned in API output. (Error messages are redacted.)

## CLI

```bash
vgreens-ai dataset import roboflow \
  --workspace strawberry-disease \
  --project strawberry-disease-detection-dataset \
  --source-version 1 \
  --name strawberry_disease_auto \
  --wait
```

`--wait` streams the state transitions. External sources pause at
`awaiting_license_approval` — the command returns there and prints the approval
hint. After an admin approves, the import resumes and freezes.

Options: `--version` (explicit vN; default auto), `--idempotency-key`,
`--output json|yaml`, `--wait-timeout`.

## Web

Datasets → **New import** → *roboflow* → workspace / project / version / name →
**Start import**. The workflow page shows progress and, for external sources, an
**Approve / Reject** panel once it reaches the licence gate.

## Finding workspace/project/version

On the dataset's Roboflow Universe page, the download URL is
`universe.roboflow.com/<workspace>/<project>/dataset/<version>`. Use those three
values. A wrong slug returns a clean `not found (404)` (no key leaked).
