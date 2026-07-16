#!/usr/bin/env bash
# Consistent online backup of the local jobs.db to the NAS (the one allowed
# exception to "SQLite off the NAS" — a *copy* is fine). Keeps the last 14.
set -euo pipefail

DB="${JOBS_DB_PATH:-/opt/vgreens-ai/jobs.db}"
DEST="${DB_BACKUP_DIR:-/mnt/vgreens_storage/ai/backups}"
PY="${VENV_PY:-/opt/vgreens-ai/venv/bin/python}"

mkdir -p "$DEST"
STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="$DEST/jobs.db.$STAMP"

# sqlite's backup API takes a consistent snapshot even while the API is writing
"$PY" - "$DB" "$OUT" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
s = sqlite3.connect(src); d = sqlite3.connect(dst)
with d:
    s.backup(d)
d.close(); s.close()
print(f"backed up {src} -> {dst}")
PY

# retention: keep the 14 most recent
ls -1t "$DEST"/jobs.db.* 2>/dev/null | tail -n +15 | xargs -r rm -f
