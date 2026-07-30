#!/bin/bash
# ---------------------------------------------------------------------------
# wait_for_server.sh HOST PORT [TIMEOUT_S] [SERVER_LOG]
#
# Block until the sgl-omni HTTP endpoint answers, or fail loudly.
#
# WHY THIS EXISTS: `kill -0 $SERVER_PID` is NOT a usable readiness or
# liveness check for sgl-omni. It forks a coordinator plus one process per
# pipeline stage; the PID you backgrounded is not the thing that ends up
# serving HTTP, and its lifetime does not track the server's. Poll the
# endpoint, and only the endpoint.
#
# Cold start on 1x GH200 is 4-8 minutes: ~10 s weight load, ~135 s AR
# CUDA-graph capture, then ~25 vocoder CUDA graphs, then a
# torch.compile(max-autotune-no-cudagraphs) of the frame sampler.
# Do not set the timeout below ~15 min.
# ---------------------------------------------------------------------------
set -uo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-31711}"
TIMEOUT="${3:-1200}"
SERVER_LOG="${4:-}"

deadline=$(( $(date +%s) + TIMEOUT ))
i=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if curl -sf "http://$HOST:$PORT/health"    >/dev/null 2>&1 || \
     curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "SERVER READY after ${i}s"
    exit 0
  fi
  sleep 5
  i=$(( i + 5 ))
done

echo "=== SERVER NOT READY after ${TIMEOUT}s ==="
if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ]; then
  echo "--- last 80 lines of $SERVER_LOG ---"
  tail -80 "$SERVER_LOG"
fi
exit 1
