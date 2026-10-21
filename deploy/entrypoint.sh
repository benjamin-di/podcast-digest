#!/usr/bin/env bash
# Ensure models exist, then exec the CLI.
#
# We don't fail the container if OpenVINO encoder conversion fails — whisper.cpp
# falls back to CPU automatically when the encoder files aren't present, so the
# pipeline still works (just slower).
set -euo pipefail

MODEL_DIR="${WHISPER_MODEL_DIR:-${PODCAST_DIGEST_DATA_DIR:-/data}/models}"

mkdir -p "${MODEL_DIR}"

# --- SSH key for the command summarizer (PODCAST_DIGEST_SSH_HOST set) ---------
# The private key + optional pinned known_hosts can arrive as base64 env vars
# (handy with secret managers that inject env). Materialize them to files the
# ssh client reads and point the app at them. Single-line base64 dodges the
# multi-line quoting trouble of carrying a PEM through env/compose.
KEY_B64="${PODCAST_DIGEST_SSH_KEY_B64:-}"
if [[ -n "${KEY_B64}" ]]; then
    KEY_PATH="${PODCAST_DIGEST_SSH_KEY:-/root/.ssh/id_podcast_digest}"
    mkdir -p "$(dirname "${KEY_PATH}")"
    chmod 700 "$(dirname "${KEY_PATH}")"
    printf '%s' "${KEY_B64}" | base64 -d > "${KEY_PATH}"
    chmod 600 "${KEY_PATH}"
    export PODCAST_DIGEST_SSH_KEY="${KEY_PATH}"
    echo "[entrypoint] wrote ssh key to ${KEY_PATH}"
fi
KNOWN_B64="${PODCAST_DIGEST_SSH_KNOWN_HOSTS_B64:-}"
if [[ -n "${KNOWN_B64}" ]]; then
    KNOWN_PATH="${PODCAST_DIGEST_SSH_KNOWN_HOSTS:-/root/.ssh/known_hosts_podcast_digest}"
    mkdir -p "$(dirname "${KNOWN_PATH}")"
    printf '%s' "${KNOWN_B64}" | base64 -d > "${KNOWN_PATH}"
    chmod 644 "${KNOWN_PATH}"
    export PODCAST_DIGEST_SSH_KNOWN_HOSTS="${KNOWN_PATH}"
    echo "[entrypoint] wrote known_hosts to ${KNOWN_PATH}"
fi

# Always run prepare-models.sh: it is internally idempotent (skips the GGML
# download if present, skips the OpenVINO encoder conversion if the IR already
# exists). Running it unconditionally lets a *failed* OpenVINO conversion
# self-heal on the next restart once the cause is fixed — the previous
# "only if the .bin is missing" gate left a failed conversion stuck on CPU
# forever, because the model file was already downloaded.
echo "[entrypoint] ensuring models (GGML + OpenVINO encoder; idempotent)..."
/usr/local/bin/prepare-models.sh || {
    echo "[entrypoint] WARNING: model preparation failed; pipeline will retry on each tick." >&2
}

# NOTE: We deliberately do NOT source /opt/intel/openvino/setupvars.sh here.
# It prepends OpenVINO's runtime libs (libtbb/libgomp/etc.) to LD_LIBRARY_PATH
# for the whole process tree, which crashes the long-running Python scheduler
# (`podcast-digest serve` dies silently before its first log line). whisper-cli
# instead gets the OpenVINO + whisper shared libs via a scoped LD_LIBRARY_PATH
# applied only to its own subprocess — see transcribe.py / the
# PODCAST_DIGEST_WHISPER_LD_PATH env var set in the Dockerfile.

exec "$@"
