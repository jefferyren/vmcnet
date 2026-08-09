#!/bin/bash
# Upload offline wandb runs. RUN THIS ON A LOGIN NODE — compute nodes have no
# internet, which is why every sbatch here sets WANDB_MODE=offline.
#
# `wandb sync` drops a .synced marker in each run directory, so this is idempotent:
# re-running only uploads what is new. Safe to run while jobs are still going; the
# runs that have finished will upload and the rest will be picked up next time.
#
# Usage (from the vmcnet repo root, or anywhere):
#     bash slurm/sync_wandb.sh                 # sync everything not yet uploaded
#     bash slurm/sync_wandb.sh e7              # only dirs whose run name matches e7
#     DRY=1 bash slurm/sync_wandb.sh           # list what would upload, do nothing
set -euo pipefail

export PATH="/global/scratch/users/$USER/envs/vmcnet/bin:$PATH"
WANDB_DIR="${WANDB_DIR:-/global/scratch/users/$USER/wandb}"
ROOT="$WANDB_DIR/wandb"
FILTER="${1:-}"
DRY="${DRY:-0}"

if [[ ! -d "$ROOT" ]]; then
  echo "No wandb directory at $ROOT — nothing to sync." >&2
  exit 0
fi

# `wandb login` must have been done once on a login node; without it sync fails
# with an auth error rather than doing anything destructive.
if ! wandb status >/dev/null 2>&1; then
  echo "wandb does not appear to be logged in. Run 'wandb login' first." >&2
  exit 1
fi

mapfile -t DIRS < <(find "$ROOT" -maxdepth 1 -type d -name 'offline-run-*' | sort)
todo=()
for d in "${DIRS[@]}"; do
  [[ -e "$d/.synced" ]] && continue
  if [[ -n "$FILTER" ]]; then
    # Match the directory name, the recorded run name, or the config. FAIL OPEN:
    # if none of those files are readable we sync anyway. An earlier version
    # matched only wandb-metadata.json, which is often absent — that silently
    # skipped 23 of 24 runs and looked like a successful sync.
    hay="$d"
    for f in "$d"/files/wandb-metadata.json "$d"/files/config.yaml; do
      [[ -r "$f" ]] && hay+=$(tr -d '\n' < "$f")
    done
    if [[ "$hay" == "$d" ]]; then
      echo "  (no metadata in $(basename "$d"); syncing it rather than risk skipping)"
    elif [[ "$hay" != *"$FILTER"* ]]; then
      continue
    fi
  fi
  todo+=("$d")
done

echo "${#DIRS[@]} offline run(s) present, ${#todo[@]} to upload${FILTER:+ (filter: $FILTER)}."
if [[ ${#todo[@]} -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

if [[ "$DRY" == "1" ]]; then
  printf '  would sync: %s\n' "${todo[@]}"
  exit 0
fi

failed=0
for d in "${todo[@]}"; do
  echo "--- syncing $(basename "$d")"
  if ! wandb sync "$d"; then
    echo "    FAILED: $d" >&2
    failed=$((failed + 1))
  fi
done

echo
if [[ $failed -gt 0 ]]; then
  echo "$failed run(s) failed to sync; re-run to retry just those." >&2
  exit 1
fi
echo "Synced ${#todo[@]} run(s). View at https://wandb.ai/$(wandb status 2>/dev/null \
  | grep -o 'Logged in as [^ ]*' | awk '{print $4}')"
