# ---------------------------------------------------------------------------
# Zero -> serving endpoint on JUPITER.
#
#   make install          (login node, needs network; ~30-45 min)
#   make check            (verify env + model snapshot are where we think)
#   make submit           (sbatch the server+benchmark job)
#   make serve            (foreground server; run inside salloc/srun)
#   make bench PORT=31711 (point the client at an already-running server)
#   make container        (build the Apptainer SIF — needs root/fakeroot)
#
# Override anything on the command line, e.g.
#   make submit ACCOUNT=myproj SGLO_BASE=/e/.../my/base
# ---------------------------------------------------------------------------
SHELL   := /bin/bash
HERE    := $(shell cd $(dir $(lastword $(MAKEFILE_LIST))) && pwd)
ACCOUNT ?=
PORT    ?= 31711
HOST    ?= 127.0.0.1
LEVELS  ?= 1:4,8:24,32:64,64:192,96:288,128:384

.PHONY: help install check serve submit bench container clean

help:
	@sed -n '2,20p' $(firstword $(MAKEFILE_LIST))

install:
	SGLO_OFFLINE=0 bash $(HERE)/scripts/install.sh

check:
	@set -euo pipefail; \
	export SGLO_VERBOSE=1; \
	source $(HERE)/env.sh; \
	echo; \
	python -c "import sglang, sglang_omni; print('sglang', sglang.__version__)"; \
	python -c "import torch; print('torch', torch.__version__)"; \
	test -d "$$SGLO_MODEL_PATH" && echo "model snapshot OK" || { echo "MISSING: $$SGLO_MODEL_PATH"; exit 1; }; \
	test -f "$$SGLO_MODEL_PATH/config.json" && echo "config.json OK" || echo "WARNING: no config.json in snapshot"; \
	command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "WARNING: nvcc not on PATH (CUDA_HOME=$$CUDA_HOME)"

serve:
	bash $(HERE)/scripts/serve.sh

submit:
	@test -n "$(ACCOUNT)" || { echo "set ACCOUNT=<slurm account>"; exit 1; }
	mkdir -p logs
	sbatch --account=$(ACCOUNT) \
	       --export=ALL,SGLO_REPO_DIR=$(HERE),SGLO_CONCURRENCIES=$(LEVELS) \
	       $(HERE)/scripts/sgl_bench.sbatch

bench:
	source $(HERE)/env.sh && \
	python $(HERE)/scripts/sgl_bench.py --host $(HOST) --port $(PORT) \
	       --levels $(LEVELS) --out $(HERE)/results/sgl_bench_results_latest.json

container:
	apptainer build --fakeroot \
	  $(HERE)/container/sglang-omni-jupiter.sif \
	  $(HERE)/container/sglang-omni-jupiter.def

clean:
	rm -f $(HERE)/results/sgl_bench_results_latest.json
