#!/bin/bash
# dev_sync.sh — fast lane for the local-edit / robot-test loop.
#
# Push the current branch straight into the Jetson's working tree over SSH
# (git remote "jetson", with receive.denyCurrentBranch=updateInstead). GitHub
# (origin) stays the source of truth; this just saves the round-trip.
#
# Usage:
#   scripts/dev_sync.sh                    push current branch to the Jetson
#   scripts/dev_sync.sh --test             push, then run pytest on the Jetson
#   scripts/dev_sync.sh --discard-results  revert tracked bench/results on the
#                                          Jetson first (throwaway bench output)
#
# One-time setup (already done on this machine):
#   ssh jetson 'cd ~/repos/rover-assistant && \
#     git config receive.denyCurrentBranch updateInstead'
#   git remote add jetson jetson:repos/rover-assistant
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=jetson
BRANCH=$(git rev-parse --abbrev-ref HEAD)
DISCARD_RESULTS=0
RUN_TEST=0
for arg in "$@"; do
  case "$arg" in
    --discard-results) DISCARD_RESULTS=1 ;;
    --test) RUN_TEST=1 ;;
    -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "error: git remote '$REMOTE' is not configured. See the header of this script." >&2
  exit 1
fi

if [ "$DISCARD_RESULTS" = 1 ]; then
  echo "== reverting tracked bench results on $REMOTE (regenerated on-robot artifacts) =="
  ssh "$REMOTE" "cd repos/rover-assistant && git checkout -- bench/results 2>/dev/null || true"
fi

echo "== pushing $BRANCH -> $REMOTE =="
if ! git push "$REMOTE" "$BRANCH"; then
  echo
  echo "push refused: the Jetson working tree is dirty and updateInstead will not clobber it." >&2
  echo "current status on the Jetson:" >&2
  ssh "$REMOTE" "cd repos/rover-assistant && git status --short" >&2 || true
  echo
  echo "If those are throwaway bench outputs, re-run with --discard-results." >&2
  echo "If they are results worth keeping, commit them on the Jetson and push to origin." >&2
  exit 1
fi

if [ "$RUN_TEST" = 1 ]; then
  echo "== pytest on $REMOTE =="
  ssh "$REMOTE" "cd repos/rover-assistant && .venv/bin/pytest -q"
fi
