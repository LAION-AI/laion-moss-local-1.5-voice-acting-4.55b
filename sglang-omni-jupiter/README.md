# MOSS-TTS-Local 4.55B on SGLang-Omni — JUPITER (aarch64 / GH200) runbook

Everything needed to get **[MOSS-TTS-Local-Transformer 4.55B voice-acting](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)**
served through **[SGLang-Omni](https://github.com/sgl-project/sglang-omni)** on the
**JUPITER** supercomputer at Jülich — an ARM (NVIDIA Grace) system with **GH200 120 GB**
GPUs, Slurm, no Docker, and no outbound network on compute nodes.

Getting there took **five failed attempts, each with a different error**. All five are
documented below with the exact symptom string, the real cause, and the fix. If you landed
here from a traceback, jump to [Troubleshooting by error string](#troubleshooting-by-error-string).

> **Scope.** This is an *inference/serving* runbook. It does not cover training or
> fine-tuning. It was written on aarch64; most of it applies to x86 SGLang-Omni too, but
> failure mode #1 is aarch64-specific.

---

## TL;DR

```bash
# once, on a LOGIN node (needs network, ~30-45 min)
make install

# then, per job
make submit ACCOUNT=<your-slurm-account>
```

The whole configuration lives in [`env.sh`](env.sh). Every value is overridable; the
defaults are the exact values that produced the measurements below.

Non-negotiable bits, if you only read four lines:

```bash
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13   # 13, NOT 12.6
export HF_HUB_CACHE=/path/to/the/cache/that/actually/has/the/model
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
sgl-omni serve --model-path /abs/path/to/snapshots/<sha>  # a LOCAL PATH, not a repo id
```

---

## Why SGLang-Omni at all?

The reference way to run MOSS-TTS-Local is `transformers` with a batched
`model.generate()` loop. That works and it is fast enough to be useful, but:

* batching is manual — you pad a batch, wait for the slowest sequence, and repeat;
* the audio codec decode is serial with generation unless you hand-pipeline it onto a
  second CUDA stream (we did; it is worth ~1.2x);
* there is no server, so every consumer of the model re-implements the same loop.

SGLang-Omni gives continuous batching, a staged pipeline
(preprocessing → AR TTS engine → vocoder, each in its own process), CUDA graphs for both
the autoregressive step and the vocoder, and an **OpenAI-compatible HTTP API**
(`POST /v1/audio/speech`). For bulk dataset generation, "point N workers at one endpoint"
is a much better shape than "N copies of a Python loop".

Measured result on 1x GH200: **1.47x the throughput of our tuned `transformers` baseline**
at concurrency 32, and saturation was not reached. See
[Measured performance](#measured-performance) — including why the headline number is
smaller than the "~35x realtime on A100" figure that circulates for this stack.

---

## What JUPITER makes hard

| constraint | consequence |
|---|---|
| **aarch64** (NVIDIA Grace) | some wheels do not exist; `pynini`/OpenFST does not build (failure #1) |
| **GH200 120 GB, GCC 14.3, torch `2.11.0+cu130`** | you must use CUDA **13**; CUDA 12.6's `nvcc` rejects GCC > 13 (failure #5) |
| **No Docker daemon, no root** | container route is Apptainer/Singularity, not `docker run` |
| **Compute nodes have no network** | everything runs `HF_HUB_OFFLINE=1`; the model must be in the cache first (failures #3, #4) |
| **`/p` (JUST) is not mounted on booster compute nodes** | venv, caches and weights must all live under `/e` (ExaSTORE) |
| **HOME quota ~20 GB** | uv's default cache overruns it while resolving this dependency tree (failure #2) |
| **Cold start 4–8 min** | ~10 s weight load + **135 s** AR CUDA-graph capture + ~25 vocoder graphs + a `torch.compile` — plan job time accordingly |

---

## The five failure modes

Each row is a real job that died. Grep your traceback in the "symptom" column.

| # | symptom (exact string) | cause | fix |
|---|---|---|---|
| 1 | `fatal error: fst/util.h: No such file or directory` while building `pynini` | `sglang-omni` depends on `nemo_text_processing` → `pynini` → OpenFST, which has no aarch64 wheel and no bundled headers | Comment `"nemo_text_processing==1.2.0"` out of `pyproject.toml`. It is used **only by the ZONOS2 model path** — MOSS-TTS-Local never touches it. `scripts/install.sh` patches this automatically. |
| 2 | `Disk quota exceeded (os error 122)` while extracting a wheel into `~/.cache/uv/...` | uv caches into `$HOME/.cache/uv`; the torch + sglang + flashinfer-cubin + cutlass tree blows past a 20 GB HOME quota | Put `UV_CACHE_DIR` **and** `TMPDIR` on scratch/project storage before installing (set in `env.sh`) |
| 3 | `ValueError: Could not resolve model architecture for <repo-id>` | SGLang calls `AutoConfig.from_pretrained()` **without `trust_remote_code`**. MOSS's `config.json` declares its class only through `auto_map`, so `AutoConfig` cannot resolve it, and the repo-id fallback path needs network — which compute nodes do not have | Pass the **local snapshot directory** to `--model-path`, not the HF repo id. With a filesystem path SGLang falls back to `try_resolve_arch_from_raw_config()`, which reads `config.json` directly and succeeds |
| 4 | `huggingface_hub.errors.OfflineModeIsEnabled` / `LocalEntryNotFoundError` | `HUGGINGFACE_HUB_CACHE` pointed at `$HF_HOME/hub` — a *different, mostly empty* cache from the one holding the downloaded repos | Point **both** `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` at the cache that actually contains the model. Verify with `ls $HF_HUB_CACHE/models--*` before submitting |
| 5 | `deep_gemm/__init__.py ... _find_cuda_home() ... assert cuda_home is not None` → after setting CUDA 12.6: `#error -- unsupported GNU version! gcc versions later than 13 are not supported!` | `deep_gemm` JIT-compiles at import and needs `CUDA_HOME`. Setting it to **CUDA 12.6** made it worse: torch here is `2.11.0+cu130` and the system compiler is GCC 14.3, which 12.6's `nvcc` refuses outright | Use **CUDA 13** (`/e/software/default/stages/2026/software/CUDA/13`). Matches `cu130` and accepts GCC 14.3. Also export `NVCC_APPEND_FLAGS=-allow-unsupported-compiler` as a safety net |

Raw log excerpts for each are in [`logs_excerpt/`](logs_excerpt/).

### Further pitfalls (not fatal, but they cost time)

* **`kill -0 $SERVER_PID` is useless as a readiness or liveness check.** `sgl-omni` forks a
  coordinator plus one process per pipeline stage; the PID you backgrounded is not the
  process that serves HTTP. **Poll the endpoint** (`/health`, falling back to `/v1/models`)
  — that is all `scripts/wait_for_server.sh` does.
* **Cold start is 4–8 minutes.** Do not set a readiness timeout below ~15 min. Breakdown
  from a real run: weight load 10.5 s → AR CUDA graph capture **135 s** (batch sizes
  `[1, 2, 4, 8, 12, 16]`) → ~25 vocoder CUDA graphs → `torch.compile(mode="max-autotune-no-cudagraphs")`
  of the frame sampler (~3 min).
* **The vocoder CUDA graphs are captured at `B=16`** (25 T-buckets, `T=1..25` →
  `audio (16, 2, 96000)`). That is a plausible ceiling on vocoder batching efficiency and
  is our leading hypothesis for why throughput does not keep scaling with concurrency.
* **Isolated HTTP 500s under load.** At concurrency 96, exactly **1 of 192** requests came
  back `500` with *no* server-side traceback. In the original client this exception
  propagated out of `ThreadPoolExecutor.map()` and destroyed the whole concurrency level.
  Benchmark and production clients must tolerate and count single failures — see the
  `_safe()` wrapper in [`scripts/sgl_bench.py`](scripts/sgl_bench.py).
* **`Bus error (core dumped)` during startup, intermittently.** One job died this way after
  CUDA-graph capture completed. It did not reproduce on resubmit. Note that each crash
  wrote a multi-GB core file into `$PWD` (we collected ~11 GB in one afternoon) — `env.sh`
  sets `ulimit -c 0`.
* **`Do you wish to run the custom code? [y/N]`** appears in the server log. It is
  `transformers` prompting for `trust_remote_code`; in a batch job stdin is not a TTY, and
  in our runs the server proceeded to load anyway. Harmless, but noisy — and worth knowing
  it is not the cause of a later failure.
* **`Failed to import nixl: No module named 'nixl'`** is a benign warning for single-node
  serving (`NixlRelay` is for multi-node KV transfer).
* **`[c10d] The client socket cannot be initialized to connect to [localhost]:...
  (errno: 97 - Address family not supported by protocol)`** — IPv6 probe failing, benign.

---

## Working environment

The authoritative copy is [`env.sh`](env.sh). Inlined here so it is greppable:

```bash
# --- CUDA: MUST be 13. torch is 2.11.0+cu130, system GCC is 14.3 ------------
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler

# --- offline; the model must already be in the cache ------------------------
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# --- the cache that ACTUALLY contains the model (not $HF_HOME/hub) ----------
BASE=/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox
export HF_HUB_CACHE=$BASE/hfcache/.cache/dramabox
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE

# --- keep caches off the HOME quota -----------------------------------------
export UV_CACHE_DIR=$BASE/uvcache
export TMPDIR=$BASE/tmp
ulimit -c 0            # sgl-omni crashes write multi-GB core files

source $BASE/env_sglang/bin/activate

# --- LOCAL SNAPSHOT PATH, never the repo id ---------------------------------
SNAP=$HF_HUB_CACHE/models--laion--moss-tts-local-transformer-4.55b-voice-acting-v2/snapshots/ae8f25442b8e5176f74d4c544ff7ec5fb9ec1d8b
sgl-omni serve --model-path $SNAP --host 127.0.0.1 --port 31711
```

Slurm resources that worked: **1 node, 1 GPU, 64 CPUs, 90 min** (`--partition=booster`).
Verified stack: Python 3.12.13, uv 0.11.31, `torch 2.11.0+cu130`, `sglang 0.5.12.post1`,
`transformers 5.6.0`, `sglang-omni 0.1.0` (upstream `main`), CUDA 13, GCC 14.3,
NVIDIA GH200 120 GB (97871 MiB reported).

### Calling the server

```bash
curl -s http://127.0.0.1:31711/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$SNAP\", \"input\": \"Bitte geh nicht. Nicht so.\", \"response_format\": \"wav\"}" \
  -o out.wav
```

`model` must be the same string the server was launched with — i.e. the snapshot path.

---

## Measured performance

**1x GH200 120 GB, `sglang-omni` @ upstream `main`, job 1113953.**
Workload: 8 short German voice-acting lines, ~5 s of audio each.

| concurrency | x realtime | ms/clip | p50 latency | p95 latency |
|---|---|---|---|---|
| 1 | 7.8 | 572 | 0.55 s | 0.62 s |
| 8 | 18.3 | 307 | 1.34 s | 1.82 s |
| **32** | **27.7** | 182 | 4.79 s | 6.0 s |

Baseline: `transformers` batched `.generate()` on the same GPU —
**18.8x realtime** with codec decode pipelined onto a second CUDA stream,
15.4x without.

**→ SGLang-Omni is 1.47x faster than the tuned `transformers` baseline at concurrency 32,
and throughput was still climbing.** Saturation was not reached; a follow-up sweep at
32/64/96/128 is what `make submit` runs by default.

### Reading the numbers

Three honest caveats, because this is easy to oversell:

1. **Only `x realtime` is comparable across the two systems.** It normalises by the
   duration of the audio actually produced. **`ms/clip` is not comparable here**: the
   SGLang benchmark sentences yield ~5 s clips, while the `transformers` measurement used
   ~11.4 s clips. The apparent ~3.3x advantage in ms/clip (182 ms vs 605 ms) is an
   **artifact of clip length**, not a real speedup. Quote 1.47x, not 3.3x.
2. **We did not reproduce the advertised "~35x realtime on an A100" figure.** Our best is
   27.7x on a GH200. Different clip lengths, different batching, different hardware, and
   possibly a different concurrency sweet spot. We are reporting what we measured.
3. **Latency degrades sharply with concurrency.** p95 goes 0.62 s → 6.0 s from concurrency
   1 to 32. These settings are tuned for *offline bulk generation*, where throughput is
   what matters. For interactive use, stay at low concurrency and expect ~4.5x realtime
   single-stream behaviour rather than 27.7x.

Raw numbers: [`results/sgl_bench_results_1113953.json`](results/sgl_bench_results_1113953.json).

### Re-running the benchmark

```bash
make submit ACCOUNT=<slurm-account>                       # default sweep 1..128
make submit ACCOUNT=<acct> LEVELS=32:96,64:192,96:288,128:384
```

or, against a server you already have running:

```bash
source env.sh
python scripts/sgl_bench.py --port 31711 --levels 32:96,64:192 --out results/mine.json
```

The client writes one JSON object per concurrency level, so adding a new row to the table
above is a copy-paste. `mean_clip_s` is included in the output specifically so that anyone
comparing against another system can check caveat #1 for themselves.

---

## Reproducible packaging

### Why there is no Docker image

JUPITER — like essentially every HPC system — **does not run a Docker daemon** and does not
give users root on compute nodes. `docker run` is not an option. Upstream `sglang-omni`
ships a `docker/Dockerfile`; it is fine on a cloud VM and unusable here. Its base image
(`lmsysorg/sglang:dev`) is also x86_64-first, so on Grace you would be rebuilding it anyway.

The two routes that *do* work on JUPITER:

**(a) Bare-metal venv — this is what we used and measured.**
```bash
make install     # login node, needs network
make check       # sanity-check env, model snapshot, nvcc
make submit ACCOUNT=...
```

**(b) Apptainer/Singularity — [`container/sglang-omni-jupiter.def`](container/sglang-omni-jupiter.def).**
> **This definition file has NOT been built or tested.** It encodes the same fixes
> (CUDA 13 base, `nemo_text_processing` patched out, uv cache off-quota) in container form.
> Build it where you have root or `--fakeroot`, copy the `.sif` over, and run it with
> `apptainer exec --nv --bind /e:/e`. Model weights are deliberately not baked in —
> bind-mount the HF cache. PRs with corrections welcome.

---

## Troubleshooting by error string

| you see | go to |
|---|---|
| `fatal error: fst/util.h: No such file or directory` | failure #1 — comment out `nemo_text_processing` |
| `pynini` ... `_pywrapfst` ... `command '...c++' failed` | failure #1 |
| `Disk quota exceeded (os error 122)` | failure #2 — move `UV_CACHE_DIR` and `TMPDIR` off HOME |
| `Failed to download 'flashinfer-cubin...'` / `I/O operation failed during extraction` | failure #2 (same root cause) |
| `ValueError: Could not resolve model architecture` | failure #3 — use the local snapshot path, not the repo id |
| `OfflineModeIsEnabled: Cannot reach ...: offline mode is enabled` | failure #4 — `HF_HUB_CACHE` points at the wrong cache |
| `LocalEntryNotFoundError: Cannot find an appropriate ...` | failure #4 |
| `assert cuda_home is not None` (in `deep_gemm/__init__.py`) | failure #5 — export `CUDA_HOME` |
| `#error -- unsupported GNU version! gcc versions later than 13 are not supported!` | failure #5 — you picked CUDA 12.x; use CUDA 13 |
| `nvcc: unsupported host compiler` | failure #5 |
| server never becomes ready, but the process is "alive" | you are using `kill -0`; poll HTTP instead |
| `Bus error (core dumped)` during startup | intermittent, did not reproduce; resubmit. Set `ulimit -c 0` first |
| `HTTP Error 500` on a small fraction of requests under load | known; tolerate it client-side (`_safe()` in `sgl_bench.py`) |
| `Failed to import nixl: No module named 'nixl'` | benign on single node |
| `[c10d] The client socket cannot be initialized ... errno: 97` | benign IPv6 probe |
| `Do you wish to run the custom code? [y/N]` in the server log | benign in batch mode |
| job FAILS after ~9 s with no log output | JUPITER node-health preemption (check `sinfo -R`), not your code — resubmit |

---

## Layout

```
sglang-omni-jupiter/
├── README.md                        this file
├── env.sh                           the whole configuration, overridable
├── Makefile                         install / check / serve / submit / bench / container
├── scripts/
│   ├── install.sh                   login-node build: clone, patch, venv, prefetch model
│   ├── serve.sh                     foreground server with the right env
│   ├── sgl_bench.sbatch             Slurm: server + readiness poll + sweep
│   ├── sgl_bench.py                 OpenAI-compatible client + throughput harness
│   └── wait_for_server.sh           HTTP readiness poll (do not use kill -0)
├── container/
│   └── sglang-omni-jupiter.def      Apptainer recipe — UNBUILT, UNTESTED
├── results/
│   └── sgl_bench_results_1113953.json
└── logs_excerpt/                    one raw excerpt per failure mode
```

## Links

* SGLang-Omni — https://github.com/sgl-project/sglang-omni
* SGLang — https://github.com/sgl-project/sglang
* Model card — https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting
* Upstream MOSS-TTS-Local — https://github.com/OpenMOSS/MOSS-TTSD
* Audio tokenizer — https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2
* Earlier A100 serving notes (different hardware, different numbers) —
  https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/sglang.html
* JUPITER / JSC — https://www.fz-juelich.de/en/ias/jsc/systems/supercomputers/jupiter
