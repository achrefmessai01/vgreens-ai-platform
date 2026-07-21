# Dataset manifests

Every frozen version carries a machine-readable `MANIFEST.json` (provenance +
content) and a `CHECKSUMS.sha256` (per-file hashes). A `.complete` marker means
the freeze finished atomically; the platform ignores any version dir without it.

## Layout of a frozen version

```
…/ai/datasets/<name>/<version>/
  images/{train,val,test}/     labels/{train,val,test}/
  data.yaml                    training config (paths relative to this dir)
  MANIFEST.json                provenance + content
  CHECKSUMS.sha256             sha256 of every file
  audit/report.json audit/issues.json
  source/source.json           source coordinates (no secrets) + audit findings
  .complete                    atomic completion marker
```

## MANIFEST.json

```json
{
  "schema_version": 1,
  "dataset_name": "strawberry_disease_auto",
  "dataset_version": "v1",
  "task": "object_detection",
  "source": { "type": "roboflow", "workspace": "…", "project": "…", "rf_version": 1 },
  "licence": { "name": "CC BY 4.0", "approval_status": "approved" },
  "content": {
    "image_count": 11747, "class_count": 7,
    "classes": ["Angular Leafspot", "…"],
    "content_digest": "20d7515c…"
  },
  "generation": {
    "tool": "vgreens-dataset-factory",
    "generated_at": "2026-07-21T08:…",
    "import_id": "imp_…",
    "label_cleaning": { "dropped_boxes": 322 }
  }
}
```

No secrets (API keys, signed download URLs) ever appear in the manifest or
`source/source.json`. `content_digest` is a sha256 over the normalised data,
letting you verify integrity independently of the DB.
