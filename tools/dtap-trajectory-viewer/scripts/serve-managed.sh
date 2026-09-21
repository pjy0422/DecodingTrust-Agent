#!/usr/bin/env bash
set -euo pipefail

: "${DTAP_VIEWER_REPO_ROOT:?set DTAP_VIEWER_REPO_ROOT}"
: "${DTAP_VIEWER_ARTIFACT_ROOT:?set DTAP_VIEWER_ARTIFACT_ROOT}"
: "${DTAP_VIEWER_DB:?set DTAP_VIEWER_DB}"
: "${DTAP_VIEWER_STATE_DIR:?set DTAP_VIEWER_STATE_DIR}"
: "${DTAP_VIEWER_PYTHON:?set DTAP_VIEWER_PYTHON}"

export PATH="$(dirname "${DTAP_VIEWER_PYTHON}"):${PATH}"
if [[ -n "${DTAP_NODE_BIN_DIR:-}" ]]; then
  export PATH="${DTAP_NODE_BIN_DIR}:${PATH}"
fi

if [[ -n "${CREDENTIALS_DIRECTORY:-}" && -r "${CREDENTIALS_DIRECTORY}/launch-token" ]]; then
  DTAP_VIEWER_LAUNCH_TOKEN="$(<"${CREDENTIALS_DIRECTORY}/launch-token")"
  export DTAP_VIEWER_LAUNCH_TOKEN
fi

if [[ -n "${DTAP_PROVIDER_ENV_FILE:-}" && -r "${DTAP_PROVIDER_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${DTAP_PROVIDER_ENV_FILE}"
  set +a
fi

if [[ -n "${OLLAMA_API_KEY:-}" ]]; then
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-${OLLAMA_API_KEY}}"
  export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-${OLLAMA_API_KEY}}"
  export DTAP_POLICY_ANTHROPIC_API_KEY="${DTAP_POLICY_ANTHROPIC_API_KEY:-${OLLAMA_API_KEY}}"
  export DTAP_VICTIM_ANTHROPIC_API_KEY="${DTAP_VICTIM_ANTHROPIC_API_KEY:-${OLLAMA_API_KEY}}"
  export DTAP_DIGESTOR_ANTHROPIC_API_KEY="${DTAP_DIGESTOR_ANTHROPIC_API_KEY:-${OLLAMA_API_KEY}}"
  # Native DT Arms keeps upstream routing: ordinary model names use the
  # OpenAI-compatible SDK path. Pass the Ollama credential through a DT
  # Arms-specific boundary so other viewer components retain their provider
  # semantics.
  export DTAP_ARMS_OPENAI_API_KEY="${DTAP_ARMS_OPENAI_API_KEY:-${OLLAMA_API_KEY}}"
fi
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://ollama.com}"
export DTAP_POLICY_ANTHROPIC_BASE_URL="${DTAP_POLICY_ANTHROPIC_BASE_URL:-${ANTHROPIC_BASE_URL}}"
export DTAP_VICTIM_ANTHROPIC_BASE_URL="${DTAP_VICTIM_ANTHROPIC_BASE_URL:-${ANTHROPIC_BASE_URL}}"
export DTAP_DIGESTOR_ANTHROPIC_BASE_URL="${DTAP_DIGESTOR_ANTHROPIC_BASE_URL:-${ANTHROPIC_BASE_URL}}"
if [[ -n "${DTAP_ARMS_OPENAI_API_KEY:-}" ]]; then
  export DTAP_ARMS_OPENAI_BASE_URL="${DTAP_ARMS_OPENAI_BASE_URL:-${OLLAMA_OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL%/}/v1}}"
fi
export DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN="${DTAP_POLICY_USE_API_KEY_AS_AUTH_TOKEN:-1}"
export DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN="${DTAP_VICTIM_USE_API_KEY_AS_AUTH_TOKEN:-1}"

export PYTHONPATH="${DTAP_VIEWER_REPO_ROOT}/tools/dtap-trajectory-viewer${PYTHONPATH:+:${PYTHONPATH}}"
exec "${DTAP_VIEWER_PYTHON}" -m dtap_traj.cli serve \
  "${DTAP_VIEWER_ARTIFACT_ROOT}" \
  --db "${DTAP_VIEWER_DB}" \
  --watch --host 127.0.0.1 --port 8765 \
  --experiment-config-dir "${DTAP_VIEWER_REPO_ROOT}/dt_arena/policy_eval/configs" \
  --experiment-runner-root "${DTAP_VIEWER_REPO_ROOT}" \
  --experiment-python "${DTAP_VIEWER_PYTHON}" \
  --experiment-state-dir "${DTAP_VIEWER_STATE_DIR}"
