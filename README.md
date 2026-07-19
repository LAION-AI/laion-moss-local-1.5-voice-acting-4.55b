# LAION MOSS Local-Transformer 1.5 — Voice Acting (4.55B, 48 kHz)

The public home of **LAION's expressive voice-acting model** built on the MOSS-TTS-Local-Transformer-v1.5
architecture: fantasy characters (orc / dragon / fairy / goblin), shouting, whispering, and **vocal
bursts** (laughs, gasps, sighs, giggles, moans) driven by natural-language director instructions,
in **English and German**, at **native 48 kHz**.

> **Model checkpoint (merged, off-the-shelf):**
> **[`laion/moss-tts-local-transformer-4.55b-voice-acting`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)**
> — Apache-2.0, no adapter handling needed.

> ## 🏆🎭 **THE FINAL SHOWCASE — [13 Character Voices, Directed by Prompt](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/voices.html)**
> The **capstone** of the character voice-acting study: the round-5 **best-of-64 champions** for all **13 voices**
> (zombie · ork · goblin · dragon · evil-ghost · mouse · fairy · ranting · pain-scream · ASMR man/woman · grieving woman/man),
> each with its **exact reusable prompt**, listenable champion audio + full VoiceNet profile, the **complete 13-principle
> prompting playbook** distilled from all five rounds (with a copy-paste template and an effect→words lookup), and the
> **round-1→5 improvement arc** — all **zero-shot, no reference audio**.
> **→ [projects.laion.ai/.../voices.html](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/voices.html)**

🎧 **Hear it:** [production best-of-64 grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/production_best50.html)
— top-3 takes per group across 40 emotions with full quality scores, generated with this model.

🎭 **Reinterpretations:** [original vs top-3 side-by-side](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/dramabox_reinterpretations_preview.html)
— 30 full DramaBox performances, each voice-cloned and re-performed 64×; the 3 best takes (ranked by
(blend + genuineness + 1.25× 42-dim Empathic-Insight emotion-profile similarity) × invWER) next to the
original recording. Full 2,953-group dataset: [`laion/moss-local-dramabox-full-reinterpretations-64`](https://huggingface.co/datasets/laion/moss-local-dramabox-full-reinterpretations-64).

🏆 **Top-3-by-reward previews** (reward = (blend + genuineness + target-emotion EI) × invWER):
[v3 grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/v3_preview_grid.html)
— 40 groups (one per emotion), top 3 of 64 takes each, voice-cloned from a unique reference speaker with
double-length scripts ([dataset](https://huggingface.co/datasets/laion/moss-local-voice-acting-v3-refs-64x100)) ·
[v4 “emorant” grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/v4_preview_grid.html)
— maximum-intensity delivery prompting, 40 groups from the first emotions of the (still running) 512k-clip run
([dataset](https://huggingface.co/datasets/laion/moss-local-voice-acting-v4-emorant-64x200)).


📚 **Read it:** [full experiment overview](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/overview.html)
— every temperature/scaling, best-of-N, reward-variant (V1–V6), VoiceCLAP-small blend-scorer, FAIR-eval, character/emotion and voice-cloning study we ran with this model, with results and links to the live grids.

🚀 **Serving on SGLang (A100) — setup, traps & speed:** [for-dummies guide](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/sglang.html)
— exact `sgl-omni` install + serve command, the full `/v1/audio/speech` API (reference-less / voice-clone / streaming), every startup trap with its fix, and a measured tuning chapter: **~0.74 s TTFA**, **~4.5× single-stream**, and **~35× realtime (≈35 audio-hours per A100-hour) at concurrency 96** for offline dataset generation.

🎭 **Character archetype prompt study:** [evolving director instructions (no reference audio)](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes.html)
— an evolutionary search for the natural-language *performance instruction* that best embodies each of **10 character/style archetypes** (zombie, orc, fairy, mouse, dragon, goblin, ASMR man/woman, ranting, pain-scream), scored by a VoiceNet 57-dim reward (blend + genuineness + 1.5·character). ~60 listenable takes with per-clip voice profiles, the full reward method, and a consolidated **lessons playbook** (describe acoustics not labels, balance beats char-maxing, one director-burst max, escalation phrasing).

🎭🔥 **Round 2 — emotion-aware reward:** [directing character voices *with emotion*](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes_round2.html)
— the same 10 archetypes re-optimized with an **EmoNet** (Empathic-Insight-Voice-Plus) emotion score added to the reward (`0.5·blend + 0.5·genu + 2.0·char_vn + 1.5·char_emo`; blend/genu halved, character pushed harder, prompts more extreme). Shows the **round-1 → round-2** jump per archetype (e.g. the **ranting** fix: char 0.65 → 0.90 with words that still land; **goblin** gains Amusement 3.5 / Malice 2.1; **zombie** gains Helplessness/Fatigue/Pain) and a new emotion-layer lessons playbook.

🎭🎚️ **Round 3 — resonance · pitch · dominance:** [directing character voices with *body*](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes_round3.html)
— the same 10 archetypes re-optimized once more, now exploiting three further families of VoiceNet dimensions: **resonance** (chest/throat/nasal/head), **pitch & register** (how high/low, how wide the pitch swings) and **dominance** (commanding vs meek). Prompts learn to name a *hollow chest groan*, *violent pitch swings*, *low flat menace* or *meek and powerless*. Shows the **round-1 → 2 → 3** char progression, the newly-rewarded dim values per take, and a resonance/pitch/dominance lessons playbook (e.g. the rant's dominance framing lifts char to 0.88; zombie 0.75 → 0.81; goblin 0.66 → 0.74).

🎭🎯 **Round 4 — weighted dims & emotions, + 3 new voices:** [tuning the reward and building new characters](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes_round4.html)
— introduces **per-dimension** (`vn_w`) and **per-emotion** (`emo_w`) reward weights so the search chases the two or three signals that define a character. Four archetypes are re-tuned (**goblin** made *more evil* — Malevolence ×2.5, char_emo 0.28 → 0.67; **mouse** pushed to an *extreme high pitch* — RANG ×2.5, char 0.76 → 0.87; **pain-scream** with *real screams written into the text* — "Aaah! … Aaargh!" — and Arousal ×2.5, char_emo 0.81 → 0.94; **ranting** with arousal up-weighted), and **three brand-new voices** are built from scratch: a **breathy evil ghost** (airy hollow whispered menace), a **crying woman** and a **crying man** (grief split by a single gender phrase; Sadness ×2.5, char_emo 0.92 / 0.89). Per-take weighted-dim values, firing emotions, and a weighting-behaviour lessons playbook.

🎭🏆 **Round 5 — the FINAL showcase (all 13 voices):** [**13 Character Voices, Directed by Prompt**](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/voices.html)
— the **definitive capstone** page. The round-5 **best-of-64 champions** for every one of the **13 characters**, each with its
**exact reusable director instruction**, champion audio + a freshly recomputed 57-dim VoiceNet profile and procedural caption,
its char / char_emo / blend / genu / invWER scores, signature dimensions and firing emotions. Adds a **complete "how to prompt any
character voice" playbook** (13 principles, reusable template, effect→words lookup, per-character round-5 recipes), the full
**round-1→5 improvement arc** table, and the compute/method footer. Round 5 introduced **per-character** blend/genu weights on top of
the `vn_w`/`emo_w` levers and pushed every voice to its extreme (ork hatred char_emo 0.85→0.93; grieving man overt sobbing 0.89→0.99;
goblin caught cleaner *and* more malevolent; ASMR-man serenity 0.67→0.88). Everything **zero-shot, no reference audio.**

🎭🎬 **Character groups — best-of-64 grid (live snapshot):** [**one champion instruction × many spoken lines**](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/character_groups.html)
— a browsable grid built from the growing [`laion/moss-character-voices-bestof64`](https://huggingface.co/datasets/laion/moss-character-voices-bestof64) dataset. Each **group** holds a character's fixed champion instruction paired with a different Gemma-authored spoken **text**; the model samples **64 takes** per group, reward-ranked *within* the group, and each card plays the **top-3 of 64** (① Champion / ② Runner-up / ③ Third) with its reward / char / char_emo / blend / genu / invWER. This is a **snapshot of the first characters** generated so far (dragon + the two ASMR voices, EN&nbsp;+&nbsp;DE); more creatures, the intense voices and the grief voices land as generation continues.

📘 **Practical cheat-sheet:** [**`PROMPTING_GUIDE.md`**](PROMPTING_GUIDE.md) — a for-dummies, per-archetype guide distilling all three rounds: copy-paste champion instructions, a short recipe for each character (which acoustic + resonance + pitch + dominance + emotion words, which bursts, what to avoid), a general principles list, and a reusable template.

🧩 **Fine-tuning prompt-format proposal:** [**sentence-level caption format**](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/finetuning_prompt_format.html)
— a design document for a fine-tuning format that gives **both** a global speaker/scene description (in `instruction`) **and** per-sentence delivery control interleaved inline in `text` as `(caption) words [pause Xs] …`, using MOSS's verified inline markup (`[pause Xs]`, `${token:N}`). The per-sentence `(captions)` are **derived automatically from audio** via the four LAION scoring models (Empathic-Insight 42 emotions · VoiceNet 57 dims · genuineness · vocal-burst blend) and a deviation-from-baseline (z-score) recipe — with **real captions** produced by the existing `voicenet_v3grid/caption.py` proof-of-concept, worked examples, and design recommendations.

🎬 **DramaBox best-of-4 TTS benchmark:** [vanilla two-scene `CUT TO:` benchmark](https://projects.laion.ai/Voice-Acting-Pipeline/dramabox.html)
— 4 reward-ranked seed variants per two-scene `CUT TO:` prompt with listenable takes, the best-of-k quality/compute trade-off, measured timing, and the full k=1..32 seed-scaling walltime table. *(Now lives in the [Voice-Acting-Pipeline](https://github.com/LAION-AI/Voice-Acting-Pipeline) repo.)*

Everything in this repository is **Apache-2.0** (code) / **CC-BY-4.0** (docs & sample pages).

---

## The two LAION MOSS voice-acting models

| | **this model — 4.55B local transformer** | sibling — [8B delay](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting) |
|---|---|---|
| architecture | `moss_tts_local` (local transformer, 12-codebook RVQ) | `moss_tts_delay` (delay pattern, 32 codebooks) |
| audio tokenizer | [`MOSS-Audio-Tokenizer-v2`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) — **48 kHz** | [`MOSS-Audio-Tokenizer`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) — 24 kHz |
| output | **native 48 kHz — use raw, no post-processing** | 24 kHz (optionally bandwidth-extended) |
| word accuracy (identical prompts, best-of-64 study) | **invWER 0.97** | invWER 0.83 |
| streaming | local-transformer family is the **streaming-oriented** MOSS design | batch-oriented |
| recency | newer training recipe (1M-sample voice-acting stage) | earlier |

In our side-by-side listening and scoring, **this 4.55B model gives subjectively better results** —
higher bandwidth out of the box, cleaner word accuracy, comparable emotional range — while being
smaller and faster. Use the 8B sibling when you need its broader multilingual coverage.

## Training data

DramaBox-style voice-acting distillation, Gemini TTS distillation (audio-matched fixed prompts with
MOSS RVQ tokens), German **Emolia** emotional speech, and podcast snippets from publicly available
podcasts on the web. Vocal-burst / strong-emotion filtered; rank-256 LoRA (α=256) over all attention
and MLP projections on top of a full DramaBox fine-tune, then merged into the released weights.

## Quick start

```bash
pip install "transformers>=4.45" torch torchaudio safetensors huggingface_hub
```

```python
import torch, torchaudio
from transformers import AutoProcessor, AutoModel

REPO = "laion/moss-tts-local-transformer-4.55b-voice-acting"
DEVICE = "cuda"

proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True,
                                     codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2")
proc.audio_tokenizer = proc.audio_tokenizer.to(DEVICE).eval()
model = AutoModel.from_pretrained(REPO, trust_remote_code=True,
                                  dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to(DEVICE).eval()

msg = proc.build_user_message(
    text="You dare enter my domain? Then you shall not leave alive!",
    instruction="As a mighty orc warrior with a deep, guttural, growling voice, "
                "bellow these words with brutal force.",
    language="English",
    tokens=120,                       # ≈ words × 6 codec frames (12.5 Hz) — pacing dial
)
batch = proc([[msg]], mode="generation")
with torch.no_grad():
    out = model.generate(input_ids=batch["input_ids"].to(DEVICE),
                         attention_mask=batch["attention_mask"].to(DEVICE),
                         max_new_tokens=600, do_sample=True,
                         audio_temperature=0.8, audio_top_p=0.95,
                         audio_top_k=25, audio_repetition_penalty=1.1)
wav = proc.decode(out)[0].audio_codes_list[0]     # (channels, T) @ 48 kHz
torchaudio.save("orc.wav", wav.cpu().float(), 48000)
```

**Prompt format (DramaBox style):** `instruction` = the performance description / stage directions
(vocal bursts as director notes like `(gasping)`); `text` = the words to speak.
**Voice cloning:** pass `reference=[codes]` from `proc.encode_audios_from_path([...], n_vq=12)`; an
*empty* instruction gives the highest speaker similarity (measured).

## Fast inference — what actually matters (all measured on A100-80GB)

1. **Batch by repetition.** Repeat the prompt's `input_ids` n× and call `generate` once.
   Batch 64 → **0.87 s per ~15 s clip**; batch 128 → 0.83 s. Best-of-N groups get this for free
   (batch = group ⇒ identical lengths ⇒ zero padding waste).
2. **SDPA + bf16.** Flash-attention 2.x is **incompatible** with this checkpoint's remote-code
   attention (reshape mismatch in the packed-QKV path) — SDPA is the supported fast path.
3. **Token budget = pacing dial.** `tokens ≈ words × 6` (12.5 Hz frames ≈ 2.1 words/s) avoids
   mid-sentence cut-offs on slow emotional delivery; smaller budgets force a rushed pace.
   `max_new_tokens ≈ 2.2 × tokens + 300`.
4. **Keep the output raw.** The model emits native 48 kHz; restoration/enhancement passes
   (e.g. 48→16→48 kHz vocoders) audibly soften it.
5. **Score in the same process** if you rank takes: batched bf16 ASR (Parakeet, batch 32 →
   0.0097 s/clip, 134× vs naive) + one VoiceCLAP encode reused for several quality heads. A fused
   generate→score worker sustains ~1.1–1.3 s/clip/GPU end-to-end.

Full write-up with every measurement: [docs/fast_inference_insights.html](docs/fast_inference_insights.html)
([rendered](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/fast_inference_insights.html)).
Bulk-generation recipe (prompt sources, rewards, throughput planning): [SYNTHETIC_DATA_RECIPE.md](SYNTHETIC_DATA_RECIPE.md).

## Going faster: SGLang, and streaming

The upstream [OpenMOSS MOSS-TTS repository](https://github.com/OpenMOSS/MOSS-TTS) ships an
[SGLang backend](https://github.com/OpenMOSS/sglang) for the **`MossTTSDelay`** architecture —
a fused TTS+codec serving path with continuous batching, reported **~3× generation throughput**.
That applies directly to our [8B delay sibling](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting).

For **this `MossTTSLocal` model** there is no official SGLang integration yet, but the port is
well-scoped: the OpenMOSS fork already contains `moss_tts_delay.py` / `moss_tts_delay_with_codec.py`
model files to use as templates; the local-transformer head predicts the 12 RVQ codebooks per frame
with a small per-frame transformer on top of the backbone, which fits SGLang's model interface —
the backbone gets continuous batching + radix caching, the local head runs per decoded frame.
Expected gains mirror the delay backend (~2–3× over HF `generate`).

**Streaming:** the local-transformer family is OpenMOSS's *streaming-oriented* design — audio
frames are complete as they are emitted (12 codebooks per 80 ms frame), so playback can start
after the first few frames: decode incrementally with chunked `MOSS-Audio-Tokenizer-v2` calls
(e.g. 12–25-frame windows ≈ 1–2 s of audio with a 1-frame overlap) instead of waiting for the
full sequence. For fully interactive agents, see also OpenMOSS's dedicated low-latency models
(MOSS-TTS-Realtime, ~180 ms TTFB, and the 100M MOSS-TTS-Nano with CPU streaming at 48 kHz) —
this repo's model targets *quality-first batch and near-real-time* use.

## Evaluation corpus

40 emotions × 100 prompts × 64 takes (256k clips) generated with this model, scored with
Parakeet WER, VoiceCLAP blend/genuineness and Empathic-Insight-Plus emotion heads:
[`laion/moss-local-voice-acting-64x100`](https://huggingface.co/datasets/laion/moss-local-voice-acting-64x100)
— browse the best takes in the [audio grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/production_best50.html).

## Links

- **Model:** [`laion/moss-tts-local-transformer-4.55b-voice-acting`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)
- **8B sibling:** [`laion/moss-tts-v1.5-8b-voice-acting`](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting)
- **Upstream architecture & tooling:** [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) · [OpenMOSS/sglang](https://github.com/OpenMOSS/sglang)
- **Audio tokenizer:** [`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)

## License

Code: **Apache-2.0**. Documentation and sample pages: **CC-BY-4.0**.
Model weights: Apache-2.0 (see the model card).
