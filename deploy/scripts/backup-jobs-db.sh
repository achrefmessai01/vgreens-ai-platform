#!/usr/bin/env bash
# Consistent online backup of the local jobs.db.
#
# Snapshot LOCALLY first (fast, reliable), verify it's a valid non-empty SQLite
# DB, compress, then copy to the NAS. We never run sqlite's .backup directly
# onto the NAS mount: a NAS hiccup there hangs the job forever and leaves
# 0-byte files (which is exactly what happened before). Keeps the last 14.
set -euo pipefail

DB="${JOBS_DB_PATH:-/opt/vgreens-ai/jobs.db}"
DEST="${DB_BACKUP_DIR:-/mnt/vgreens_storage/ai/backups}"
NAS_MOUNT="${NAS_MOUNTPOINT:-/mnt/vgreens_storage}"
PY="${VENV_PY:-/opt/vgreens-ai/venv/bin/python}"
STAMP="$(date +%Y-%m-%d_%H%M)"

TMP="$(mktemp -t jobsdb.XXXXXX)"
trap 'rm -f "$TMP" "$TMP.gz"' EXIT

# 1. consistent snapshot to a LOCAL temp file (sqlite backup API is safe while
#    the API writes)
"$PY" - "$DB" "$TMP" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
s = sqlite3.connect(src); d = sqlite3.connect(dst)
with d:
    s.backup(d)
d.close(); s.close()
PY

# 2. verify: non-empty + passes integrity check
[ -s "$TMP" ] || { echo "backup snapshot is empty" >&2; exit 1; }
ok="$("$PY" - "$TMP" <<'PY'
import sqlite3, sys
print(sqlite3.connect(sys.argv[1]).execute("PRAGMA integrity_check").fetchone()[0])
PY
)"
[ "$ok" = "ok" ] || { echo "integrity check failed: $ok" >&2; exit 1; }

# 3. compress the verified snapshot
gzip -f "$TMP"          # -> $TMP.gz

# 4. NAS must be mounted, else fail fast (don't hang like before)
mountpoint -q "$NAS_MOUNT" || { echo "NAS not mounted at $NAS_MOUNT" >&2; exit 1; }
mkdir -p "$DEST"

# 5. atomic publish to the NAS (write .partial, then rename)
OUT="$DEST/jobs.db.$STAMP.gz"
cp "$TMP.gz" "$OUT.partial"
mv -f "$OUT.partial" "$OUT"
echo "backed up $DB -> $OUT ($(du -h "$OUT" | cut -f1))"

# 6. retention: keep the 14 most recent (both old uncompressed + new .gz names)
ls -1t "$DEST"/jobs.db.* 2>/dev/null | tail -n +15 | xargs -r rm -f
