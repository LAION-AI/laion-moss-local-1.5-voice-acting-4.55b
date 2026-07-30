#!/bin/bash
# ---------------------------------------------------------------------------
# sgl_bench.sbatch — start sgl-omni on one GH200 and run the throughput sweep.
#
#   sbatch --account=<your-account> scripts/sgl_bench.sbatch
#
# Override anything via --export, e.g.
#   sbatch --export=ALL,SGLO_CONCURRENCIES=32:96,64:192 scripts/sgl_bench.sbatch
#
# Defaults below are the JUPITER booster values that produced the numbers in
# README.md. Change --account and the log paths for your project.
# ---------------------------------------------------------------------------
#SBATCH --job-name=sgl-bench
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --output=logs/sgl_bench_%j.out
#SBATCH --error=logs/sgl_bench_%j.out

set -x

# Directory this script was submitted from (repo root of sglang-omni-jupiter/).
HERE="${SGLO_REPO_DIR:-$SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source "$HERE/env.sh"

SERVER_LOG="$SGLO_LOGS/sgl_server_${SLURM_JOB_ID}.log"

echo "host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- launch server ----------------------------------------------------------
# NOTE: sgl-omni forks a coordinator + one process per pipeline stage.
# Do NOT use `kill -0 $SRV` as a readiness/liveness probe (see
# scripts/wait_for_server.sh). Poll HTTP.
sgl-omni serve \
  --model-path "$SGLO_MODEL_PATH" \
  --host "$SGLO_HOST" \
  --port "$SGLO_PORT" \
  > "$SERVER_LOG" 2>&1 &
SRV=$!
echo "server pid=$SRV  log=$SERVER_LOG"

if ! bash "$HERE/scripts/wait_for_server.sh" \
        "$SGLO_HOST" "$SGLO_PORT" "${SGLO_STARTUP_TIMEOUT:-1200}" "$SERVER_LOG"; then
  kill "$SRV" 2>/dev/null
  exit 1
fi

# --- benchmark --------------------------------------------------------------
python "$HERE/scripts/sgl_bench.py" \
  --host "$SGLO_HOST" \
  --port "$SGLO_PORT" \
  --levels "${SGLO_CONCURRENCIES:-1:4,8:24,32:64,64:192,96:288,128:384}" \
  --out "$SGLO_LOGS/sgl_bench_results_${SLURM_JOB_ID}.json"
echo "BENCH_EXIT=$?"

kill "$SRV" 2>/dev/null
wait 2>/dev/null
echo "DONE $(date)"
