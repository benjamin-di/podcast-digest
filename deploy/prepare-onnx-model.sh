#!/usr/bin/env bash
# Pin the offline INT8 ONNX exports consumed by the podcast image.
set -euo pipefail

engine=$1
model_dir=$2
case "$engine" in
    parakeet)
        source=https://huggingface.co/csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/resolve/8c3a10fb13408c7a7054f6898958bf1c64a8d6c7
        assets=$(cat <<'MODELS'
6716910b7a0833997fec7a410494c995d70124001a0e9b66d6370d6aced577e0 encoder.int8.onnx
a5e223392c90e75f8144cdb5eb95af7625db389e39edef2bd1a9c872b3298fe6 decoder.int8.onnx
869f43f7d24595c55581ad3bf249a935fb8a71389fbdaa7504b9f46f93140f8a joiner.int8.onnx
dc0b4584ab2e4ddbf888425c076c61b736e7356a015250db7d307e6f1a8188ff tokens.txt
MODELS
        )
        ;;
    qwen)
        source=https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/68818b2313fe77bd06f6a7c5068ff3ef59d02b8a
        assets=$(cat <<'MODELS'
d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e conv_frontend.onnx
60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9 encoder.int8.onnx
4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771 decoder.int8.onnx
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 tokenizer/merges.txt
4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c tokenizer/tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 tokenizer/vocab.json
MODELS
        )
        ;;
    *) exit 64 ;;
esac
mkdir -p -- "$model_dir"

# A second container can share this cache during a manual rehearsal.
exec 9>"$model_dir/.prepare.lock"
flock --exclusive --wait 30 9
staged=$(mktemp -d "$model_dir/.prepare.XXXXXX")
trap 'rm -rf -- "$staged"' EXIT

while read -r hash name; do
    if printf '%s  %s\n' "$hash" "$model_dir/$name" | sha256sum --check --status; then
        continue
    fi
    mkdir -p -- "$(dirname -- "$staged/$name")" "$(dirname -- "$model_dir/$name")"
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --connect-timeout 10 --max-time 210 --retry 1 --retry-max-time 240 \
        "$source/$name" --output "$staged/$name"
    printf '%s  %s\n' "$hash" "$staged/$name" | sha256sum --check
    chmod 0644 "$staged/$name"
    mv -fT -- "$staged/$name" "$model_dir/$name"
done <<< "$assets"

printf '[prepare-%s] Verified offline INT8 model assets\n' "$engine"
