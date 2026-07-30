#!/usr/bin/env python
"""Throughput benchmark for MOSS-TTS-Local 4.55B served by SGLang-Omni.

Speaks the OpenAI-compatible `POST /v1/audio/speech` API, so it works against
any sgl-omni server (and, with --model, against OpenAI itself).

The headline metric is **x realtime** = audio-seconds produced per wall-second.
That is the only number that is comparable across systems, because it is
normalised by the duration of the audio that was actually generated.
ms/clip is NOT comparable unless both systems synthesise clips of the same
length -- see README.md, "Reading the numbers".

Usage:
    python sgl_bench.py --port 31711
    python sgl_bench.py --port 31711 --levels 32:96,64:192,96:288,128:384
"""
import argparse
import io
import json
import os
import statistics
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor

# The server identifies the model by the path/id it was launched with.
# env.sh exports MODEL=$SGLO_MODEL_PATH for exactly this reason.
DEFAULT_MODEL = os.environ.get("MODEL", "moss")

# Eight short German voice-acting lines, ~5 s of audio each.
# Swap these for your own workload -- but then re-measure the baseline too,
# because clip length changes ms/clip (though not x realtime).
TEXTS = [
    "Ich kann nicht glauben, dass du das wirklich getan hast.",
    "Es tut mir so leid, ich wollte dich nicht verletzen.",
    "Pass auf! Da kommt ein Auto!",
    "Das ist die beste Nachricht, die ich seit Wochen gehoert habe.",
    "Warum sagst du mir das erst jetzt?",
    "Ich habe die ganze Nacht wach gelegen und nachgedacht.",
    "Bitte geh nicht. Nicht so.",
    "Wir haben es tatsaechlich geschafft!",
]


def synth(url, model, text, timeout=600):
    """One /v1/audio/speech request. Returns (latency_s, audio_s, bytes)."""
    req = urllib.request.Request(
        url,
        data=json.dumps({
            "model": model,
            "input": text,
            "response_format": "wav",
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    dt = time.time() - t0
    try:
        w = wave.open(io.BytesIO(raw))
        secs = w.getnframes() / w.getframerate()
    except Exception:
        # fall back to a 48 kHz / 16-bit mono assumption
        secs = max(0.0, (len(raw) - 44) / (48000 * 2))
    return dt, secs, len(raw)


def _safe(url, model, text):
    """Never raises.

    At concurrency 96 we saw exactly 1 of 192 requests come back as HTTP 500
    with no server-side traceback. Without this wrapper that single failure
    propagated out of ThreadPoolExecutor.map() and killed the entire
    concurrency level, losing the measurement. Tolerate isolated failures,
    count them, and report them.
    """
    try:
        return synth(url, model, text)
    except Exception as e:  # noqa: BLE001
        return (None, 0.0, 0, f"{type(e).__name__}: {str(e)[:120]}")


def run(url, model, conc, n):
    jobs = [TEXTS[i % len(TEXTS)] for i in range(n)]
    t0 = time.time()
    lat, audio, fails = [], 0.0, []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for r in ex.map(lambda t: _safe(url, model, t), jobs):
            if len(r) == 4:
                fails.append(r[3])
                continue
            dt, secs, _ = r
            lat.append(dt)
            audio += secs
    wall = time.time() - t0
    ok = len(lat)
    if ok == 0:
        return dict(concurrency=conc, n=n, ok=0, failed=len(fails),
                    err=fails[0] if fails else "all failed")
    return dict(
        concurrency=conc, n=n, ok=ok, failed=len(fails),
        wall_s=round(wall, 2),
        audio_s=round(audio, 1),
        realtime_x=round(audio / wall, 1) if wall else 0,
        clips_per_s=round(ok / wall, 2),
        ms_per_clip=round(1000 * wall / ok, 1),
        mean_clip_s=round(audio / ok, 2),
        lat_p50_s=round(statistics.median(lat), 2),
        lat_p95_s=round(sorted(lat)[max(0, int(0.95 * len(lat)) - 1)], 2),
        err=fails[0] if fails else None,
    )


def parse_levels(s):
    out = []
    for part in s.split(","):
        conc, _, n = part.partition(":")
        conc = int(conc)
        out.append((conc, int(n) if n else conc * 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model id sent in the request body; defaults to $MODEL")
    ap.add_argument("--levels", default="1:4,8:24,32:64,64:192,96:288,128:384",
                    help="comma-separated concurrency:num_requests pairs")
    ap.add_argument("--baseline-rt", type=float, default=18.8,
                    help="reference x-realtime to print alongside (transformers batched .generate())")
    ap.add_argument("--out", default="sgl_bench_results.json")
    a = ap.parse_args()

    url = f"http://{a.host}:{a.port}/v1/audio/speech"

    print("=== warmup ===", flush=True)
    try:
        print(synth(url, a.model, TEXTS[0]), flush=True)
    except Exception as e:  # noqa: BLE001
        print("WARMUP FAILED:", type(e).__name__, str(e)[:300], flush=True)
        raise SystemExit(1)

    results = []
    for conc, n in parse_levels(a.levels):
        print(f"=== concurrency {conc}, n={n} ===", flush=True)
        r = run(url, a.model, conc, n)
        results.append(r)
        print(json.dumps(r), flush=True)

    print(f"\n=== SUMMARY (baseline transformers: {a.baseline_rt}x realtime) ===")
    for r in [x for x in results if x.get("ok")]:
        print(f"  conc={r['concurrency']:4d}  {r['realtime_x']:7.1f}x RT  "
              f"{r['ms_per_clip']:7.1f} ms/clip  mean_clip={r['mean_clip_s']}s  "
              f"p50={r['lat_p50_s']}s p95={r['lat_p95_s']}s  ok={r['ok']}/{r['n']}")
    print("\nNOTE: compare x realtime, NOT ms/clip -- ms/clip depends on how long "
          "the generated clips are and is meaningless across differing workloads.")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
