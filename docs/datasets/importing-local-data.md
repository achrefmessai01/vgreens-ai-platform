# Importing local / NAS data

For datasets already on disk (a YOLO-format directory or a `.zip`) under an
**allowlisted root** (default `/mnt/vgreens_storage`; configured via
`DATASET_IMPORT_ROOTS` on the worker).

## CLI

```bash
vgreens-ai dataset import local \
  --path /mnt/vgreens_storage/ai/import-test/my_dataset \
  --name my_dataset --wait
```

`import nas` is identical but tags the source type `nas`. Local/NAS data is
treated as internal (`licence_status: not_required`) — no approval gate.

## Expected layout

A YOLO detection dataset:

```
<root>/
  train/images/*.jpg   train/labels/*.txt
  valid/images/*.jpg   valid/labels/*.txt
  test/images/*.jpg    test/labels/*.txt   (optional)
  data.yaml            (names, nc)
```

The factory normalises `valid`→`val` and rewrites `data.yaml` paths.

## Safety

Paths are resolved and must stay within an allowlisted root (no `..` escapes,
no symlink escapes). Zips are checked for traversal, symlinks, zip-bombs, and
size/file-count limits before extraction. A source outside the allowlist is
rejected with a safety error.
