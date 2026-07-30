#!/bin/bash
# ---------------------------------------------------------------------------
# env.sh — environment for serving MOSS-TTS-Local 4.55B with SGLang-Omni
#          on JUPITER (JSC, aarch64 Grace + GH200 120GB, Slurm + Apptainer).
#
#   source env.sh
#
# Every value can be overridden by exporting it BEFORE sourcing this file.
# The defaults are the exact values that produced the measured numbers in
# README.md, so they double as documentation of a known-good configuration.
#
# NOTHING SECRET BELONGS IN THIS FILE. Tokens come from the environment
# (HF_TOKEN) and are only needed on a login node for the initial download.
# ---------------------------------------------------------------------------

# --- Where everything lives -------------------------------------------------
# On JUPITER the /p (JUST) filesystems are NOT mounted on booster COMPUTE
# nodes — only /e (ExaSTORE) is. Venv, caches and model snapshots must all
# live under a path the compute nodes can see.
: "${SGLO_BASE:=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox}"
: "${SGLO_VENV:=$SGLO_BASE/env_sglang}"
: "${SGLO_SRC:=$SGLO_BASE/moss/sglang-omni}"
: "${SGLO_LOGS:=$SGLO_BASE/logs}"
export SGLO_BASE SGLO_VENV SGLO_SRC SGLO_LOGS

# --- CUDA -------------------------------------------------------------------
# MUST be CUDA 13. torch in this stack is 2.11.0+cu130 and the system
# compiler is GCC 14.3. CUDA 12.6's nvcc rejects GCC > 13 outright, and
# deep_gemm JIT-compiles against whatever CUDA_HOME points at.
: "${SGLO_CUDA_HOME:=/e/software/default/stages/2026/software/CUDA/13}"
export CUDA_HOME="$SGLO_CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH"
# Belt and braces: lets nvcc accept GCC 14.3 instead of erroring out.
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

# --- HuggingFace caches -----------------------------------------------------
# Point at the cache that actually CONTAINS the model repos. Getting this
# wrong is failure mode #4 in the README: HF_HOME/hub and the real cache
# were two different directories.
: "${SGLO_HF_CACHE:=$SGLO_BASE/hfcache/.cache/dramabox}"
export HF_HOME="${HF_HOME:-$SGLO_BASE/hfcache}"
export HF_HUB_CACHE="$SGLO_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$SGLO_HF_CACHE"
export TRANSFORMERS_CACHE="$SGLO_HF_CACHE"

# Compute nodes have no outbound network -> run fully offline.
# Set SGLO_OFFLINE=0 on a login node when you still need to download.
: "${SGLO_OFFLINE:=1}"
export HF_HUB_OFFLINE="$SGLO_OFFLINE"
export TRANSFORMERS_OFFLINE="$SGLO_OFFLINE"

# --- Scratch / caches that would otherwise blow up the HOME quota ----------
# failure mode #2: uv's default cache is ~/.cache/uv and the sglang-omni
# dependency tree (torch, flash-attn-4, sglang kernels, ...) overruns a
# typical 20 GB HOME quota with "Disk quota exceeded (os error 122)".
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SGLO_BASE/uvcache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SGLO_BASE/uvpython}"
export TMPDIR="${TMPDIR:-$SGLO_BASE/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SGLO_BASE/hfcache/xdg}"
export TORCH_HOME="${TORCH_HOME:-$SGLO_BASE/hfcache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SGLO_BASE/hfcache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SGLO_BASE/hfcache/inductor}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SGLO_LOGS" 2>/dev/null || true

# A crashing sgl-omni writes multi-GB core files into $PWD (we collected
# 11 GB of them in one afternoon). Disable them.
ulimit -c 0 2>/dev/null || true

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Model ------------------------------------------------------------------
# CRITICAL (failure mode #3): pass sgl-omni a LOCAL SNAPSHOT PATH, never the
# HuggingFace repo id. SGLang resolves the architecture with
# AutoConfig.from_pretrained() WITHOUT trust_remote_code; the MOSS config
# only declares its architecture through `auto_map`, so the AutoConfig route
# fails and the repo-id fallback needs network access we do not have.
# With a local path SGLang falls back to try_resolve_arch_from_raw_config(),
# which reads config.json directly and succeeds.
: "${SGLO_MODEL_REPO:=models--laion--moss-tts-local-transformer-4.55b-voice-acting-v2}"
: "${SGLO_MODEL_REVISION:=ae8f25442b8e5176f74d4c544ff7ec5fb9ec1d8b}"
: "${SGLO_MODEL_PATH:=$SGLO_HF_CACHE/$SGLO_MODEL_REPO/snapshots/$SGLO_MODEL_REVISION}"
export SGLO_MODEL_PATH
export MODEL="$SGLO_MODEL_PATH"   # sgl_bench.py reads $MODEL as the API model name

# --- Server -----------------------------------------------------------------
: "${SGLO_HOST:=127.0.0.1}"
: "${SGLO_PORT:=31711}"
export SGLO_HOST SGLO_PORT

# --- Activate the venv ------------------------------------------------------
if [ -f "$SGLO_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SGLO_VENV/bin/activate"
else
    echo "env.sh: WARNING venv not found at $SGLO_VENV — run 'make install'" >&2
fi

if [ -n "${SGLO_VERBOSE:-}" ]; then
    echo "SGLO_BASE       = $SGLO_BASE"
    echo "SGLO_VENV       = $SGLO_VENV"
    echo "CUDA_HOME       = $CUDA_HOME"
    echo "HF_HUB_CACHE    = $HF_HUB_CACHE"
    echo "SGLO_MODEL_PATH = $SGLO_MODEL_PATH"
    echo "endpoint        = http://$SGLO_HOST:$SGLO_PORT"
fi
