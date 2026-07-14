# LAION MOSS Local-Transformer 1.5 — Voice Acting (4.55B, 48 kHz)

The public home of **LAION's expressive voice-acting model** built on the MOSS-TTS-Local-Transformer-v1.5
architecture: fantasy characters (orc / dragon / fairy / goblin), shouting, whispering, and **vocal
bursts** (laughs, gasps, sighs, giggles, moans) driven by natural-language director instructions,
in **English and German**, at **native 48 kHz**.

> **Model checkpoint (merged, off-the-shelf):**
> **[`laion/moss-tts-local-transformer-4.55b-voice-acting`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)**
> — Apache-2.0, no adapter handling needed.

🎧 **Hear it:** [production best-of-64 grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/production_best50.html)
— top-3 takes per group across 40 emotions with full quality scores, generated with this model.

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
[`laion/moss-local-voice-acting-64x100-part1..4`](https://huggingface.co/datasets/laion/moss-local-voice-acting-64x100-part1)
— browse the best takes in the [audio grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/production_best50.html).

## Links

- **Model:** [`laion/moss-tts-local-transformer-4.55b-voice-acting`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)
- **8B sibling:** [`laion/moss-tts-v1.5-8b-voice-acting`](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting)
- **Upstream architecture & tooling:** [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) · [OpenMOSS/sglang](https://github.com/OpenMOSS/sglang)
- **Audio tokenizer:** [`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)

## License

Code: **Apache-2.0**. Documentation and sample pages: **CC-BY-4.0**.
Model weights: Apache-2.0 (see the model card).
