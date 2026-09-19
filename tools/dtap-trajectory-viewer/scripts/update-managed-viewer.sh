#!/usr/bin/env bash
set -euo pipefail

: "${DTAP_VIEWER_SOURCE_REPO:?set DTAP_VIEWER_SOURCE_REPO}"
: "${DTAP_VIEWER_REPO_ROOT:?set DTAP_VIEWER_REPO_ROOT}"

git -C "${DTAP_VIEWER_SOURCE_REPO}" fetch --quiet origin main
target="$(git -C "${DTAP_VIEWER_SOURCE_REPO}" rev-parse origin/main)"
current="$(git -C "${DTAP_VIEWER_REPO_ROOT}" rev-parse HEAD)"
if [[ "${current}" == "${target}" ]]; then
  exit 0
fi
if ! git -C "${DTAP_VIEWER_REPO_ROOT}" diff --quiet || \
   ! git -C "${DTAP_VIEWER_REPO_ROOT}" diff --cached --quiet; then
  echo "refusing to update a dirty viewer deployment worktree" >&2
  exit 1
fi
git -C "${DTAP_VIEWER_REPO_ROOT}" checkout --quiet --detach "${target}"
systemctl --user restart dtap-trajectory-viewer.service
