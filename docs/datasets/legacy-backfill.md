# Legacy dataset backfill

Datasets frozen before the version index existed (e.g. `strawberry_detection`,
`strawberry_disease`, `test`) live on the NAS but aren't in the `dataset_versions`
table, so they don't appear in `GET /datasets` or the web UI. The backfill indexes
them **without copying or modifying any files**.

## Command

Dry-run first (default — reports, writes nothing):

```bash
vgreens-ai dataset backfill
```

Apply:

```bash
vgreens-ai dataset backfill --apply
```

## Behaviour

- Scans `…/ai/datasets/<name>/<version>/`; a version counts as frozen if it has
  a `.complete` marker or a `data.yaml`.
- Reads `MANIFEST.json` when present (`metadata_source: manifest`); otherwise
  infers image/class counts by scanning (`metadata_source: inferred`,
  `licence_status: legacy`).
- **Idempotent** — already-indexed versions are skipped, never duplicated.
- Never overwrites dataset files; only inserts index rows (`frozen_by: backfill`).
- Produces a report: `scanned`, `already_indexed`, `added[]`, `errors[]`.

## Notes

- Inferred metadata is clearly marked; re-freezing through the Dataset Factory
  would produce a full manifest + content digest if that's ever needed.
- The operation is auditable (`dataset.backfill` events).
