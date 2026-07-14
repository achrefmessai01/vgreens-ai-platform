#!/usr/bin/env bash
# LRU eviction for the NVMe cache: when free space drops below the floor, delete
# the least-recently-modified cache entries (dataset/model/job dirs) until back
# above it. Complements the 7-day time-based cleanup timer.
set -euo pipefail

MNT="${NVME_MNT:-/nvme}"
MIN_FREE_PCT="${MIN_FREE_PCT:-15}"          # keep at least this % free
ROOTS=(/nvme/cache/datasets /nvme/cache/models /nvme/jobs)

used_pct() { df --output=pcent "$MNT" | tail -1 | tr -dc '0-9'; }

THRESH=$((100 - MIN_FREE_PCT))              # evict while used% > THRESH
used="$(used_pct)"
if [ "$used" -le "$THRESH" ]; then
  echo "nvme used ${used}% <= ${THRESH}% — nothing to evict"
  exit 0
fi

# oldest-first list of cache entries (version/job dirs)
mapfile -t entries < <(
  for r in "${ROOTS[@]}"; do
    [ -d "$r" ] && find "$r" -mindepth 1 -maxdepth 2 -type d -printf '%T@ %p\n'
  done | sort -n
)

for e in "${entries[@]}"; do
  used="$(used_pct)"
  [ "$used" -le "$THRESH" ] && break
  path="${e#* }"
  echo "evicting (used ${used}%): $path"
  rm -rf "$path"
done

echo "nvme used $(used_pct)% (floor ${MIN_FREE_PCT}% free)"
