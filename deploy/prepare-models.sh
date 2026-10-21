#!/usr/bin/env bash
# One-time model setup, per model in PODCAST_DIGEST_WHISPER_MODELS:
#   1. Download ggml-<name>.bin via the whisper.cpp helper script (HF mirror).
#   2. Convert the encoder to OpenVINO IR format so whisper.cpp can use an
#      Intel iGPU — only when one is present (/dev/dri render node) or
#      PODCAST_DIGEST_OPENVINO=on. Idempotent — skips both steps if outputs
#      already exist.
#
# PODCAST_DIGEST_WHISPER_MODELS is a space-separated list of whisper.cpp model
# names; it defaults to English-only. Deployments
# with non-English feeds (feeds.toml `language`) add the multilingual model,
# e.g. PODCAST_DIGEST_WHISPER_MODELS="medium.en large-v3-turbo". Footprint
# caveats for the large models: large-v3-turbo is ~1.6 GB (large-v3 ~3 GB),
# the OpenVINO conversion needs more RAM/time than medium, and iGPU encode is
# slower per minute of audio — budget first-boot time and disk accordingly.
# If the OpenVINO conversion fails for a model (e.g. openai-whisper alias
# mismatch for large-v3-turbo), transcribe.py CPU-falls-back when no encoder
# sits beside the ggml file — correctness holds, just slower.
#
# Runs from the entrypoint on first container start, after which the data
# volume persists everything.
set -euo pipefail

MODEL_DIR="${WHISPER_MODEL_DIR:-${PODCAST_DIGEST_DATA_DIR:-/data}/models}"
MODELS="${WHISPER_MODELS:-${PODCAST_DIGEST_WHISPER_MODELS:-medium.en}}"

mkdir -p "${MODEL_DIR}"

prepare_model() {
    local model_name="$1"
    local ggml_file="${MODEL_DIR}/ggml-${model_name}.bin"
    local encoder_bin="${MODEL_DIR}/ggml-${model_name}-encoder-openvino.bin"
    local encoder_xml="${MODEL_DIR}/ggml-${model_name}-encoder-openvino.xml"

    # ---- 1. GGML model -------------------------------------------------------
    if [[ -f "${ggml_file}" ]]; then
        echo "[prepare-models] ggml model present: ${ggml_file}"
    else
        echo "[prepare-models] downloading ggml-${model_name}.bin..."
        /usr/local/bin/download-ggml-model.sh "${model_name}" "${MODEL_DIR}"
        if [[ ! -f "${ggml_file}" ]]; then
            echo "[prepare-models] ERROR: download script returned 0 but ${ggml_file} is missing" >&2
            exit 1
        fi
    fi

    # ---- 2. OpenVINO encoder -------------------------------------------------
    if [[ -f "${encoder_bin}" && -f "${encoder_xml}" ]]; then
        echo "[prepare-models] OpenVINO encoder present for ${model_name}"
        return 0
    fi

    # The encoder is only useful with an Intel GPU (OpenVINO). Without one the
    # conversion costs a ~1.5 GB checkpoint download and minutes of CPU for
    # nothing, so it runs only when a render node is present — or when
    # PODCAST_DIGEST_OPENVINO=on forces it (=off skips it everywhere).
    case "${PODCAST_DIGEST_OPENVINO:-auto}" in
        off) echo "[prepare-models] PODCAST_DIGEST_OPENVINO=off: skipping the encoder"; return 0 ;;
        on) ;;
        *)
            if ! compgen -G "/dev/dri/renderD*" >/dev/null; then
                echo "[prepare-models] no GPU render node (/dev/dri): skipping the OpenVINO encoder; whisper.cpp runs on the CPU"
                return 0
            fi
            ;;
    esac

    echo "[prepare-models] converting ${model_name} encoder to OpenVINO IR (one-time)..."
    # The upstream convert script writes outputs alongside itself (via
    # `os.path.dirname(__file__)`), NOT cwd. So we copy the script into
    # MODEL_DIR and run it from there — that way __file__ resolves to MODEL_DIR
    # and the encoder files land where whisper.cpp expects to find them.
    local script_in_model_dir="${MODEL_DIR}/convert-whisper-to-openvino.py"
    cp /usr/local/bin/convert-whisper-to-openvino.py "${script_in_model_dir}"
    (
        cd "${MODEL_DIR}"
        /opt/venv/bin/python "${script_in_model_dir}" --model "${model_name}"
    )
    # The script copy was needed only for the run; remove it so the model dir
    # stays minimal.
    rm -f "${script_in_model_dir}"

    # Safety net: if a future script version reverts to cwd-based output, sweep
    # the working dirs we know about and move anything matching the expected
    # pattern back to MODEL_DIR.
    local src f
    for src in /usr/local/bin "${MODEL_DIR}"; do
        for f in "${src}/ggml-${model_name}-encoder-openvino."{bin,xml}; do
            if [[ -f "${f}" && "${f}" != "${MODEL_DIR}/$(basename "${f}")" ]]; then
                echo "[prepare-models] moving stray output: ${f} -> ${MODEL_DIR}/"
                mv -f "${f}" "${MODEL_DIR}/"
            fi
        done
    done

    if [[ -f "${encoder_bin}" && -f "${encoder_xml}" ]]; then
        echo "[prepare-models] encoder ready: ${encoder_bin}"
    else
        echo "[prepare-models] WARNING: conversion ran but expected outputs missing" >&2
        ls -la "${MODEL_DIR}" /usr/local/bin 2>&1 | grep -i openvino >&2 || true
        exit 1
    fi
}

for m in ${MODELS}; do
    prepare_model "${m}"
done
