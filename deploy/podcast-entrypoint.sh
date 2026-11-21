#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PODCAST_DIGEST_PARAKEET_MODEL_DIR:-}" ]]; then
    if ! /usr/local/bin/prepare-onnx-model.sh parakeet "$PODCAST_DIGEST_PARAKEET_MODEL_DIR"; then
        echo '[entrypoint] Parakeet preparation failed. Using Whisper for this start.' >&2
        unset PODCAST_DIGEST_PARAKEET_MODEL_DIR
    fi
fi
if [[ -n "${PODCAST_DIGEST_QWEN_MODEL_DIR:-}" ]]; then
    if ! /usr/local/bin/prepare-onnx-model.sh qwen "$PODCAST_DIGEST_QWEN_MODEL_DIR"; then
        echo '[entrypoint] Qwen preparation failed. Using Whisper for Chinese on this start.' >&2
        unset PODCAST_DIGEST_QWEN_MODEL_DIR
    fi
fi
exec /usr/local/bin/entrypoint.sh "$@"
