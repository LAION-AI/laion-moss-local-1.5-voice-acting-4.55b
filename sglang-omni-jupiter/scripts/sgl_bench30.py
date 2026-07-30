#!/usr/bin/env python
"""MOSS/SGLang-Omni throughput with ~30 s clips, measured (not derived).

Fixes vs the earlier batch benchmark:
  * response items carry base64 under results[].audio_data - the old parser looked
    for other keys and silently reported audio_s = 0, so x-realtime was derived
    from an assumed clip length instead of measured.
  * texts are ~70 German words (~30 s of speech) instead of one-liners (~5 s), so
    per-clip fixed overhead stops dominating and x-realtime is comparable to the
    transformers baseline, which used ~11.4 s clips.

Env: MODEL (served model id/path), TAG (label for the results file).
"""
import argparse, base64, io, json, os, statistics, time, wave
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("MODEL", "moss")
TAG = os.environ.get("TAG", "base")

# ~70 words each -> ~30 s of German speech
TEXTS = [
    "Ich habe die ganze Nacht wach gelegen und über das nachgedacht, was du gesagt hast. "
    "Am Anfang war ich wütend, dann traurig, und irgendwann gegen vier Uhr morgens habe ich "
    "verstanden, dass du eigentlich recht hattest. Das zuzugeben fällt mir schwer, weil ich "
    "mich jahrelang darauf verlassen habe, dass ich in solchen Dingen der Vernünftigere bin. "
    "Vielleicht war genau das mein Fehler, und vielleicht sollten wir noch einmal ganz von "
    "vorne anfangen, ohne die alten Vorwürfe.",

    "Also pass auf, ich erkläre dir das jetzt ein letztes Mal, und danach möchte ich nicht "
    "mehr darüber reden. Wir haben damals eine Entscheidung getroffen, gemeinsam, und niemand "
    "hat dich dazu gezwungen. Dass es am Ende anders gekommen ist, als wir uns das vorgestellt "
    "hatten, tut mir genauso leid wie dir. Aber ich lasse mir nicht länger einreden, dass ich "
    "allein die Verantwortung dafür trage, was in diesem Sommer passiert ist.",

    "Es war ein ganz gewöhnlicher Dienstagmorgen, der Nebel hing noch zwischen den Häusern, "
    "und die Straßenbahn war so voll wie immer um diese Zeit. Ich stand an der Tür und dachte "
    "an nichts Bestimmtes, als mein Telefon klingelte. Danach war nichts mehr gewöhnlich an "
    "diesem Tag. Ich weiß noch genau, wie das Licht auf den nassen Schienen lag und wie "
    "seltsam ruhig meine eigene Stimme klang, als ich antwortete.",

    "Wir haben es tatsächlich geschafft, kannst du das glauben? Nach achtzehn Monaten, nach "
    "all den Nächten, in denen wir dachten, wir müssten aufgeben, nach diesem furchtbaren "
    "Februar, in dem wirklich alles gleichzeitig schiefgegangen ist. Und jetzt sitze ich hier "
    "und halte das Ergebnis in der Hand und weiß gar nicht, ob ich lachen oder heulen soll. "
    "Wahrscheinlich beides. Komm her, wir haben uns das verdient, alle beide.",

    "Bitte hör mir einen Moment zu, ohne mich zu unterbrechen, das ist alles, worum ich dich "
    "bitte. Ich habe nicht vor, mich zu rechtfertigen, und ich erwarte auch nicht, dass du mir "
    "verzeihst. Ich will nur, dass du weißt, wie es aus meiner Sicht war, damit du später nicht "
    "denkst, es sei dir gleichgültig gewesen. Danach kannst du gehen, und ich werde dich nicht "
    "aufhalten, das verspreche ich dir.",

    "Der Bericht liegt Ihnen seit drei Wochen vor, und ich gehe davon aus, dass Sie ihn "
    "gelesen haben. Die Zahlen sprechen eine deutliche Sprache: Wir haben im dritten Quartal "
    "einen Rückgang von siebzehn Prozent, und die Prognose für das kommende Jahr sieht nicht "
    "besser aus. Ich bin nicht hier, um Schuldige zu suchen. Ich bin hier, weil wir gemeinsam "
    "eine Entscheidung treffen müssen, und zwar heute, nicht irgendwann im nächsten Frühjahr.",

    "Weißt du, was das Verrückteste daran ist? Ich habe ihn sofort erkannt, nach dreiundzwanzig "
    "Jahren, mitten in dieser vollen Bahnhofshalle. Er stand da mit seinem Koffer und sah genauso "
    "verloren aus wie damals am letzten Schultag. Und für einen winzigen Augenblick war ich "
    "wieder sechzehn und hatte überhaupt keine Ahnung, was ich sagen sollte. Am Ende habe ich "
    "einfach seinen Namen gerufen, quer durch die ganze Halle.",

    "Nein, das ist kein Missverständnis, und ich möchte auch nicht, dass du es als eines "
    "abtust. Ich habe genau gehört, was du gesagt hast, und ich habe es zum ersten Mal wirklich "
    "verstanden. Es ist nicht das eine Wort gewesen, es ist die Selbstverständlichkeit, mit der "
    "du es ausgesprochen hast. Genau die macht mir Angst. Und deshalb werde ich morgen früh "
    "meine Sachen packen, bevor du aufwachst.",
]

def _secs(raw: bytes) -> float:
    try:
        w = wave.open(io.BytesIO(raw))
        return w.getnframes() / w.getframerate()
    except Exception:
        return max(0.0, (len(raw) - 44) / (48000 * 2))

def _post(port, path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def single(port, text):
    t0 = time.time()
    raw = _post(port, "/v1/audio/speech",
                {"model": MODEL, "input": text, "response_format": "wav"})
    return time.time() - t0, _secs(raw), 1

def batch(port, texts):
    t0 = time.time()
    raw = _post(port, "/v1/audio/speech/batch",
                {"model": MODEL, "items": [{"input": t} for t in texts],
                 "response_format": "wav"})
    dt = time.time() - t0
    obj = json.loads(raw)
    audio, n = 0.0, 0
    for it in obj.get("results", []):
        b64 = it.get("audio_data")
        if it.get("status") in (None, "ok", "success", "succeeded") and b64:
            try:
                audio += _secs(base64.b64decode(b64)); n += 1
            except Exception:
                pass
    return dt, audio, n

def run(port, mode, bsz, conc, total):
    if mode == "single":
        jobs = [TEXTS[i % len(TEXTS)] for i in range(total)]
        fn = lambda t: single(port, t)
    else:
        rounds = max(1, total // bsz)
        jobs = [[TEXTS[(i * bsz + j) % len(TEXTS)] for j in range(bsz)] for i in range(rounds)]
        fn = lambda ts: batch(port, ts)
    def safe(j):
        try:
            return fn(j)
        except Exception as e:
            return (None, 0.0, 0)
    t0 = time.time(); audio = 0.0; n = 0; lat = []; fails = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for dt, secs, k in ex.map(safe, jobs):
            if dt is None:
                fails += 1; continue
            lat.append(dt); audio += secs; n += k
    wall = time.time() - t0
    return dict(tag=TAG, mode=mode, batch=bsz, concurrency=conc, clips=n, failed=fails,
                wall_s=round(wall, 2), audio_s=round(audio, 1),
                realtime_x=round(audio / wall, 1) if wall else 0,
                clips_per_s=round(n / wall, 2) if wall else 0,
                mean_clip_s=round(audio / n, 2) if n else 0,
                lat_p50_s=round(statistics.median(lat), 2) if lat else None)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()

    print("=== warmup (checks 30s clip length + audio parsing) ===", flush=True)
    dt, secs, _ = single(a.port, TEXTS[0])
    print(f"  single: {dt:.1f}s wall -> {secs:.1f}s audio", flush=True)
    dt, secs, n = batch(a.port, TEXTS[:2])
    print(f"  batch2: {dt:.1f}s wall -> {secs:.1f}s audio over {n} items", flush=True)
    if secs <= 0:
        print("  !! batch audio still unparsed", flush=True)

    grid = [("single", 1, 32, 32), ("single", 1, 64, 64), ("single", 1, 128, 128),
            ("batch", 16, 2, 64), ("batch", 32, 1, 64), ("batch", 32, 2, 128),
            ("batch", 64, 1, 128)]
    out = []
    for mode, bsz, conc, total in grid:
        print(f"=== {mode} batch={bsz} conc={conc} n={total} ===", flush=True)
        r = run(a.port, mode, bsz, conc, total)
        out.append(r); print(json.dumps(r), flush=True)

    print(f"\n=== SUMMARY [{TAG}] 30s clips (5s-clip baseline: 6.97 clips/s / ~34x RT) ===")
    for r in sorted(out, key=lambda x: -x["realtime_x"]):
        print(f"  {r['mode']:6s} b={r['batch']:3d} c={r['concurrency']:3d}  "
              f"{r['realtime_x']:7.1f}x RT  {r['clips_per_s']:5.2f} clips/s  "
              f"mean_clip={r['mean_clip_s']}s  p50={r['lat_p50_s']}s  fail={r['failed']}")
    p = f"/e/data1/datasets/playground/mmlaion/schuhmann1/dramabox/logs/sgl30_{TAG}.json"
    json.dump(out, open(p, "w"), indent=2)
    print("WROTE", p)
