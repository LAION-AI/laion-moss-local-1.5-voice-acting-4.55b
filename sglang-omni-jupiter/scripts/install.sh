#!/bin/bash
# ---------------------------------------------------------------------------
# install.sh — build the SGLang-Omni venv on a JUPITER LOGIN node.
#
# Run this on a login node (it needs outbound network). It takes ~30-45 min
# the first time; most of that is compiling/downloading the torch + sglang +
# flash-attn-4 stack for aarch64.
#
#   SGLO_OFFLINE=0 ./scripts/install.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SGLO_OFFLINE="${SGLO_OFFLINE:-0}"        # need the network here
# shellcheck disable=SC1091
source "$HERE/env.sh"

: "${SGLO_SGLANG_OMNI_GIT:=https://github.com/sgl-project/sglang-omni}"
: "${SGLO_SGLANG_OMNI_REF:=main}"
: "${SGLO_PYTHON:=3.12}"

command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

# --- 1. source checkout -----------------------------------------------------
if [ ! -d "$SGLO_SRC/.git" ]; then
  echo "==> cloning sglang-omni into $SGLO_SRC"
  mkdir -p "$(dirname "$SGLO_SRC")"
  git clone "$SGLO_SGLANG_OMNI_GIT" "$SGLO_SRC"
fi
git -C "$SGLO_SRC" fetch --all --quiet || true
git -C "$SGLO_SRC" checkout "$SGLO_SGLANG_OMNI_REF"

# --- 2. patch out nemo_text_processing (failure mode #1) --------------------
# nemo_text_processing -> pynini -> OpenFST, which does not build on aarch64
# ("fatal error: fst/util.h: No such file or directory"). It is only used by
# the ZONOS2 model path; MOSS-TTS-Local does not need it.
PY="$SGLO_SRC/pyproject.toml"
if grep -qE '^\s*"nemo_text_processing' "$PY"; then
  echo "==> commenting out nemo_text_processing in pyproject.toml (aarch64)"
  cp "$PY" "$PY.bak"
  sed -i -E 's|^(\s*)("nemo_text_processing[^,]*",)|\1# \2  # removed: pynini/OpenFST unbuildable on aarch64; ZONOS2-only, not needed for MOSS|' "$PY"
fi

# --- 3. venv + install ------------------------------------------------------
if [ ! -d "$SGLO_VENV" ]; then
  echo "==> creating venv at $SGLO_VENV (python $SGLO_PYTHON)"
  uv venv --python "$SGLO_PYTHON" "$SGLO_VENV"
fi
# shellcheck disable=SC1091
source "$SGLO_VENV/bin/activate"

echo "==> uv pip install -e $SGLO_SRC   (this is the long part)"
uv pip install --python "$SGLO_VENV/bin/python" -e "$SGLO_SRC" 2>&1 \
  | tee "$SGLO_LOGS/sglang_install.log"

# --- 4. pre-download the model into the cache the JOB will read -------------
# Compute nodes are offline; everything must be in $HF_HUB_CACHE beforehand.
echo "==> pre-fetching model into $HF_HUB_CACHE"
HF_HUB_OFFLINE=0 python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("SGLO_HF_REPO_ID", "laion/moss-tts-local-transformer-4.55b-voice-acting-v2")
codec = os.environ.get("SGLO_CODEC_REPO_ID", "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
for r in (repo, codec):
    p = snapshot_download(r, token=os.environ.get("HF_TOKEN"))
    print(f"{r} -> {p}")
PY

echo
echo "==> done. Verify:"
echo "    python -c 'import sglang, sglang_omni; print(sglang.__version__)'"
echo "    ls \$SGLO_MODEL_PATH"
echo "Remember: SGLO_MODEL_PATH must point at the *snapshot* dir printed above."
