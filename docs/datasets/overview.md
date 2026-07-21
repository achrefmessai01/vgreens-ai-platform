# Dataset Factory — overview

The Dataset Factory turns an external or local source into a **frozen, audited,
versioned, immutable** dataset on the NAS — driven entirely through the Job API.
No SSH, no manual NAS writes, no hand-written manifests.

## Pipeline

```
submit → downloading → staging → quarantined → auditing
       → [awaiting_license_approval] → normalising → freezing → frozen
       (or → rejected / failed / cancelled)
```

- **staging / quarantine** — the source is downloaded/copied into an isolated
  per-import dir on NVMe and checked for path traversal, symlink escape, zip
  bombs, and size/file-count limits before anything approaches the NAS.
- **audit** — geometry + integrity checks against a configurable policy
  (see `audit-policy.md`). Fatal findings reject the import.
- **licence gate** — external sources pause at `awaiting_license_approval`
  until an admin approves (see `licence-approval.md`). Local/NAS data is treated
  as internal (`not_required`).
- **normalise** — canonical layout (`images/{train,val,test}`,
  `labels/…`, `data.yaml`), with unusable label lines dropped and counted.
- **freeze** — build in a NAS temp dir, write `MANIFEST.json` + `CHECKSUMS.sha256`
  + `.complete`, then atomic `os.replace` into
  `…/datasets/<name>/<version>/`. Only completed freezes are visible.

## Components

| Piece | Role |
|-------|------|
| API `/datasets/*` | control plane: submit, list, approve, allocate/finalize versions |
| `dataset_jobs` queue | RabbitMQ queue for import work |
| `vgreens-dataset-worker` | CPU-only systemd worker (no GPU) that runs the pipeline |
| `dataset_imports` / `dataset_versions` (SQLite) | workflow state + frozen-version index |
| NAS `…/ai/datasets/<name>/<version>/` | the frozen bytes + manifest |

## Interfaces

- CLI: `vgreens-ai dataset …` (see `../cli/reference.md`).
- Web: the **Datasets** section (list, import wizard, workflow detail, licence approval).
- Both call the API only — never the DB, queue, or filesystem directly.
