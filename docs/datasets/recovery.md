# Dataset import recovery

The Dataset Factory is designed to tolerate restarts and retries.

## Guarantees

- **Idempotent submit** — a repeated import with the same `--idempotency-key`
  returns the existing workflow, never a duplicate.
- **Atomic freeze** — a version is built in a NAS temp dir and `os.replace`d into
  place with a `.complete` marker. A crash mid-freeze leaves a hidden `.incoming`
  dir and no visible version; the DB row (reserved, `nas_path` NULL) stays hidden.
- **No overwrite** — `UNIQUE(dataset_name, version)` means a frozen version is
  never overwritten.
- **Redelivery-safe** — the worker acks a message only after handling it; a crash
  before ack re-delivers, and the worker re-stages from scratch.

## Common situations

**Import stuck in `staging`/`freezing` for a while**
Large datasets are slow over CIFS (copy + checksum + per-file perms). Check the
worker log and the growing temp dir:
```bash
journalctl -u vgreens-dataset-worker -n 30
du -sh /mnt/vgreens_storage/ai/datasets/<name>/.<version>.incoming_* 2>/dev/null
```

**Import `failed` (transient, e.g. NAS blip)**
Re-submit it (same source). Failed/rejected/cancelled are terminal; a new submit
creates a fresh workflow. Clear leftover staging if needed:
```bash
rm -rf /nvme/dataset-staging/<import-id>
```

**Import `rejected` by audit**
Read the `error` (and `audit/report.json` if a partial exists). Fix the source or
adjust expectations, then re-import. Degenerate boxes are auto-cleaned, so a
reject is usually a real structural problem (leakage, corrupt images, bad classes).

**Reserved-but-orphaned version rows**
A version row with `nas_path` NULL (worker died between allocate and finalize) is
invisible and harmless. It can be left; a future cleanup can reap them.

**Leftover `.incoming` dirs on the NAS**
Safe to delete — they are never read (no `.complete`).
