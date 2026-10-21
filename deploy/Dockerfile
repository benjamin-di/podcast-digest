# ============================================================================
# podcast-digest
#
# Two-stage build:
#   1. whisper-builder: builds whisper.cpp's `whisper-cli` with OpenVINO support
#      so an Intel iGPU (if passed through) accelerates the encoder; without one
#      whisper.cpp runs on the CPU.
#   2. runtime: OpenVINO runtime + Python 3.12 + ffmpeg + our package + the
#      whisper binary.
#
# OpenVINO encoder conversion (~3 GB of downloads on first boot) happens in
# the entrypoint so the image itself doesn't bake in model weights — they
# land in /data/models, which should be a persistent volume.
# ============================================================================

ARG OPENVINO_VERSION=2024.6.0
ARG SOURCE_REVISION=unknown

# ----------------------------------------------------------------------------
# Stage 1: build whisper-cli
# ----------------------------------------------------------------------------
FROM openvino/ubuntu24_runtime:${OPENVINO_VERSION} AS whisper-builder
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG WHISPER_CPP_REF=v1.7.4
WORKDIR /src
RUN git clone --depth 1 --branch ${WHISPER_CPP_REF} \
        https://github.com/ggerganov/whisper.cpp.git

WORKDIR /src/whisper.cpp
# OpenVINO runtime images source setupvars.sh on shell init; we do it here too
# so cmake finds OpenVINOConfig.cmake.
#
# GGML_NATIVE=OFF is critical: ggml defaults to -march=native, which bakes the
# *build host's* CPU instruction set into libggml-cpu.so. Built on GitHub
# Actions, the runner's CPU varies between builds — a runner exposing an
# AVX-512 subset that many desktop CPUs lack produced kernels that SIGILL
# (rc=-4) for certain audio shapes. Pin a portable AVX2/FMA/F16C baseline
# (any x86-64 CPU since ~2013; with an iGPU the heavy encode runs via OpenVINO,
# so the CPU backend loses nothing) to make every rebuild deterministic.
RUN bash -lc '. /opt/intel/openvino/setupvars.sh && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_OPENVINO=1 \
          -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON && \
    cmake --build build -j"$(nproc)" --config Release && \
    mkdir -p /whisperlibs && \
    find build \( -name "*.so" -o -name "*.so.*" \) -exec cp -P {} /whisperlibs/ \;'

# ----------------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------------
FROM openvino/ubuntu24_runtime:${OPENVINO_VERSION} AS transcriber-runtime
USER root

ARG WHISPER_CPP_REF=v1.7.4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_CPP_VERSION=${WHISPER_CPP_REF}

# Intel GPU compute runtime so OpenVINO can actually use the iGPU:
#   intel-opencl-icd  — OpenCL NEO driver
#   libze1            — oneAPI Level Zero loader
#   libze-intel-gpu1  — Intel Level Zero GPU driver
# Without these, OpenVINO's GPU plugin reports "no supported devices found" and
# whisper.cpp silently falls back to CPU despite -oved GPU + /dev/dri.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ffmpeg curl ca-certificates tini \
        openssh-client \
        intel-opencl-icd libze1 libze-intel-gpu1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# whisper.cpp artifacts
COPY --from=whisper-builder /src/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper-builder /src/whisper.cpp/models/download-ggml-model.sh /usr/local/bin/download-ggml-model.sh
# Vendored converter (drops whisper.cpp's dead `openvino.tools.mo` import, which
# 404s on OpenVINO 2024.x) so the iGPU OpenVINO encoder IR is produced.
COPY deploy/convert-whisper-to-openvino.py /usr/local/bin/convert-whisper-to-openvino.py
# whisper-cli is dynamically linked against libwhisper.so / libggml*.so (built
# in stage 1). Ship them and register the dir with the loader. The OpenVINO
# runtime libs whisper-cli also needs are NOT added system-wide (they conflict
# with the Python process); they're injected only for the whisper subprocess
# via PODCAST_DIGEST_WHISPER_LD_PATH (see below + transcribe.py).
COPY --from=whisper-builder /whisperlibs/ /usr/local/lib/whisper/
RUN chmod +x /usr/local/bin/whisper-cli /usr/local/bin/download-ggml-model.sh \
 && echo "/usr/local/lib/whisper" > /etc/ld.so.conf.d/whisper.conf \
 && ldconfig

# App venv (PEP 668 makes /usr/bin/pip refuse, so we always use the venv).
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# OpenVINO encoder conversion deps (torch + openai-whisper + openvino). Pre-
# installed so first-run conversion is just a CPU/disk job, not a pip install.
# torch CPU wheel is much smaller than the default CUDA one.
#
# openai-whisper==20240930's setup.py does `import pkg_resources`, which current
# setuptools (>=81) no longer ships. Seed the venv with setuptools<81 + wheel and
# build whisper with --no-build-isolation so it uses that setuptools instead of
# pip pulling a newer one into an isolated build env.
#
# `onnx` is required by the first-boot OpenVINO conversion: the converter does
# torch model -> ONNX (torch.onnx.export, which needs `onnx`) -> OpenVINO IR.
# Without it the conversion fails ("Module onnx is not installed!") and
# whisper.cpp silently falls back to CPU instead of the Intel iGPU.
RUN /opt/venv/bin/pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1 \
 && /opt/venv/bin/pip install \
        torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && /opt/venv/bin/pip install --no-build-isolation \
        openai-whisper==20240930 \
 && /opt/venv/bin/pip install \
        openvino==2024.6.0 \
        onnx==1.17.0

# Reusable transcriber package. This stage is also published directly as the
# local-transcriber image, for other jobs that need local transcription.
WORKDIR /app
COPY packages/local-transcriber /app/packages/local-transcriber
RUN /opt/venv/bin/pip install -e ./packages/local-transcriber

# Model provisioning is shared by the generic runtime and the podcast image.
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY deploy/prepare-models.sh /usr/local/bin/prepare-models.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/prepare-models.sh

# Commit metadata must not invalidate apt, Whisper compilation or Torch.
ARG SOURCE_REVISION
ENV WHISPER_MODEL_DIR=/models \
    WHISPER_BIN=/usr/local/bin/whisper-cli \
    WHISPER_LD_PATH=/usr/local/lib/whisper:/opt/intel/openvino/runtime/lib/intel64 \
    LOCAL_TRANSCRIBER_SOURCE_REV=${SOURCE_REVISION}

# Licences of what this image redistributes: this project (LICENSE, NOTICE) and
# whisper.cpp (MIT). Distro packages keep theirs under /usr/share/doc; the
# OpenVINO base image documents its own. Copied last to keep the layers above
# cacheable.
COPY --from=whisper-builder /src/whisper.cpp/LICENSE /usr/share/licenses/whisper.cpp/LICENSE
COPY LICENSE NOTICE /usr/share/licenses/podcast-digest/

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["local-transcribe", "--help"]

# ----------------------------------------------------------------------------
# Podcast application
# ----------------------------------------------------------------------------
FROM transcriber-runtime AS podcast-runtime

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN /opt/venv/bin/pip install -e .

# Example feeds.toml (mount your own over it).
COPY feeds.toml /app/feeds.toml

# The quickstart compose file mounts a data volume at /data and points
# PODCAST_DIGEST_CONFIG at its config directory instead of this example.
ENV WHISPER_MODEL_DIR=/data/models \
    PODCAST_DIGEST_DATA_DIR=/data \
    PODCAST_DIGEST_CONFIG=/app/feeds.toml \
    PODCAST_DIGEST_WHISPER_BIN=/usr/local/bin/whisper-cli \
    PODCAST_DIGEST_WHISPER_MODEL=/data/models/ggml-medium.en.bin \
    PODCAST_DIGEST_WHISPER_LD_PATH=/usr/local/lib/whisper:/opt/intel/openvino/runtime/lib/intel64

# Healthcheck: scheduler is a long-lived Python process running the `serve`
# subcommand. We match on the precise argv pattern to avoid matching
# bootstrap scripts (prepare-models.sh path contains "podcast-digest" too).
# `pgrep -f` excludes its own PID by default; the literal command shows up
# in /proc/<pid>/cmdline as space-joined argv, hence the regex.
HEALTHCHECK --interval=5m --timeout=10s --start-period=10m --start-interval=2s --retries=3 \
    CMD pgrep -f "podcast_digest.*serve|podcast-digest serve" >/dev/null || exit 1

CMD ["podcast-digest", "serve"]
