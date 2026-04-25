# syntax=docker/dockerfile:1.7
# stl-seed Phase 2 RunPod image.
#
# Layered for cache efficiency:
#   1. base CUDA / cuDNN / Python 3.11 from PyTorch's official image
#   2. system deps (git, build tools needed for any source builds)
#   3. uv (lockfile-driven Python dependency resolution)
#   4. Python dependencies (from pyproject.toml + uv.lock — `uv sync --frozen`)
#   5. project source code (volatile; rebuilds on every code change only)
#
# Pin policy:
#   * The base image is pinned to a specific PyTorch + CUDA + cuDNN tuple so
#     that bitsandbytes' precompiled NF4 kernels are guaranteed to match the
#     CUDA runtime ABI. As of 2026-04, bitsandbytes 0.49.x ships wheels for
#     CUDA 12.1 / 12.4. We use 12.4 to match the host driver typically shipped
#     by RunPod 4090 / Ada-Lovelace pods (driver 550+ supports CUDA 12.4).
#   * uv is pinned to 0.11.7 (the user's local version, see pyproject.toml).
#   * All Python deps come from uv.lock — no floating versions in this image.
#
# Build:
#   docker build -t stl-seed:phase2 -f docker/Dockerfile .
#
# Smoke-test on a CUDA host:
#   docker run --gpus all --rm -it stl-seed:phase2 \
#       bash docker/runpod_bootstrap.sh

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS base

# Avoid interactive tzdata prompts during apt installs.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Place HF cache on a persistent path so users can mount a RunPod
    # network volume here and avoid re-downloading the base model on each
    # pod start.
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    # bitsandbytes: silence the "could not find any cuda" sniffer when
    # importing on a build host without CUDA visible.
    BNB_CUDA_VERSION=124

# ---------------------------------------------------------------------------
# Layer 2: system deps.
# ---------------------------------------------------------------------------
# git: required by `uv sync` for any VCS-pinned deps and for cloning the repo.
# ca-certificates: TLS roots for HF Hub / GitHub.
# curl: bootstrap script + uv installer fallback.
# build-essential: in case any wheel needs to build from source on this base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Layer 3: uv.
# ---------------------------------------------------------------------------
# Install uv via its official standalone installer at a pinned version. We
# install to /usr/local/bin so it is on PATH for every subsequent layer and
# for the runtime user.
ARG UV_VERSION=0.11.7
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh \
        -o /tmp/uv-install.sh \
    && UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh \
    && rm /tmp/uv-install.sh \
    && uv --version

WORKDIR /workspace/stl-seed

# ---------------------------------------------------------------------------
# Layer 4: Python deps.
# ---------------------------------------------------------------------------
# Copy ONLY the dependency-defining files first. This way, code changes
# (Layer 5) do not invalidate the heavy `uv sync` layer.
COPY pyproject.toml uv.lock README.md ./

# `uv sync --frozen` resolves exactly the versions in uv.lock — no upgrades.
# We install the cuda + dev extras here. The mlx extra is intentionally
# omitted: the mlx wheel only ships for darwin-arm64 and would fail to
# resolve on this linux-x86_64 image.
RUN uv sync --frozen --extra cuda --extra dev \
    && uv cache prune --ci

# ---------------------------------------------------------------------------
# Layer 5: project source.
# ---------------------------------------------------------------------------
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY paper ./paper
COPY docker ./docker
COPY Makefile ./Makefile

# Re-run `uv sync` so the project itself is installed editable on top of the
# pre-built dep environment. This is cheap because deps are already cached.
RUN uv sync --frozen --extra cuda --extra dev

# ---------------------------------------------------------------------------
# Default entrypoint.
# ---------------------------------------------------------------------------
# We do NOT auto-run training. The bootstrap script verifies the env and
# prints a banner; the user invokes the canonical sweep manually.
CMD ["/bin/bash"]
