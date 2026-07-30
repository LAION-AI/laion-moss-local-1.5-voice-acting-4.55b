#!/bin/bash
# ---------------------------------------------------------------------------
# serve.sh — start sgl-omni in the foreground with the JUPITER environment.
# Use inside an salloc/srun shell, or as the payload of an sbatch script.
#
#   source ../env.sh && ./serve.sh
#   SGLO_PORT=31711 ./serve.sh
#
# Extra sgl-omni flags can be appended:  ./serve.sh --tp-size 1
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

if [ ! -d "$SGLO_MODEL_PATH" ]; then
  echo "SGLO_MODEL_PATH does not exist: $SGLO_MODEL_PATH" >&2
  echo "It MUST be a local snapshot directory, not a HuggingFace repo id." >&2
  exit 1
fi

echo "host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec sgl-omni serve \
  --model-path "$SGLO_MODEL_PATH" \
  --host "$SGLO_HOST" \
  --port "$SGLO_PORT" \
  "$@"
