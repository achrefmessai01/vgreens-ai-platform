# Dataset audit policy

Every import is audited before freezing. Findings are classified `fail`
(fatal — rejects the import), `warn` (recorded, non-blocking), or `allow`
(ignored).

## Default policy (Balanced)

| Finding | Level | Notes |
|---------|-------|-------|
| `corrupt_images` | fail | unreadable image (Pillow verify) |
| `invalid_labels` | fail | malformed line or class id out of range — unfixable |
| `degenerate_boxes` | warn | zero/negative-size box — **dropped during normalise** |
| `oob_boxes` | warn | box past the frame edge (±0.01 tolerance); YOLO clips |
| `split_leakage` | fail | identical image content in >1 split (content hash) |
| `missing_labels` | warn | image with no label file |
| `duplicate_images` | warn | identical image within a split |
| `empty_labels` | allow | image with an empty label file (background) |

## What the audit inspects

Class map + counts, per-split image/label counts, missing/empty/malformed
labels, box geometry (degenerate, out-of-frame), duplicate images, and
train/val/test leakage. Corrupt-image checks require Pillow on the worker; if
absent they are skipped and reported as informational (never a silent pass).

## Reports

Written into the frozen dataset under `audit/`:

- `report.json` — full structured findings + per-split stats + per-class counts
- `issues.json` — the findings list (level/code/message)

## Cleaning

`degenerate_boxes` are auto-dropped during normalisation and the count is
recorded in `MANIFEST.json` under `generation.label_cleaning.dropped_boxes`.
The original source is never modified.

The policy is intended to be overridable per-import in a later iteration;
today the Balanced defaults above apply.
