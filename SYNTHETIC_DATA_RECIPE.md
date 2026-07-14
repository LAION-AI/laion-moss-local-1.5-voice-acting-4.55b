# Synthetic voice-acting data at scale — best-of-N generation, Sidon restoration, multi-reward ranking

A standalone recipe for generating ranked synthetic voice-acting corpora with the two MOSS
voice-acting models. Everything here was measured on an 8×A100-80GB box; all numbers in this
document are real, not estimates, unless marked.

Works identically for **both** models — the prompt construction, Sidon restoration, scoring stack
and rewards are model-independent; only the model loading differs (§3).

| | 8B "delay" | 4.55B "local transformer" |
|---|---|---|
| model | [`laion/moss-tts-v1.5-8b-voice-acting`](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting) | [`TTS-AGI/moss-dramabox-ft`](https://huggingface.co/TTS-AGI/moss-dramabox-ft) (`checkpoint-last`) + LoRA [`TTS-AGI/moss-local-transformer-voice-acting`](https://huggingface.co/TTS-AGI/moss-local-transformer-voice-acting) (`checkpoint-1m`) |
| architecture | `moss_tts_delay`, 32 codebooks | `moss_tts_local`, 12 codebooks |
| audio codec | [`OpenMOSS-Team/MOSS-Audio-Tokenizer`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) (**v1**, 24 kHz mono, 12.5 Hz) | [`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) (**v2**, 48 kHz, 12.5 Hz) |
| output | 24 kHz | 48 kHz |

---

## 1 · Prompt sources

Three interchangeable sources; all reduce to the same `(text, instruction)` pair.

### 1a. DramaBox prompts — [Voice-Acting-Pipeline `data/`](https://github.com/LAION-AI/Voice-Acting-Pipeline/tree/main/data)
79,087 pre-generated prompts (75,593 usable), 9 subsets (EN + DE): `cca_voicenet`,
`cc2c_archetype`, `accc_acting_challenge`, `sit_situation` (multi-lingual incl. FR/ES),
`extreme_physical`. Most are two-scene "CUT TO:" (character-consistent); ~10 % single-emotion.

**Parsing (procedural, no LLM):**
- `text` = all `"double-quoted"` segments joined (`re.findall(r'"([^"]*)"', prompt)`); skip prompts
  with < 4 dialogue words (~4 %).
- `instruction` = the full DramaBox prompt **including the quoted dialogue** (best word accuracy),
  or with dialogue stripped (slightly higher blend/genu reward, worse WER). Measured trade-off:
  invWER 0.686 vs 0.624; best-16 reward 0.921 vs 0.947. Default: keep the dialogue.

**For a 1,000-group run split evenly:** 200 prompts per EN subset (or 111/subset over all 9
subsets incl. DE) — sample with a fixed RNG seed for reproducibility.

### 1b. Emotion-labelled prompts — [`laion/voice-acting-prompts`](https://huggingface.co/datasets/laion/voice-acting-prompts)
16,262 files structured `speaker/emotion/vocal_burst.json`; the 40 emotion dirs map **1:1** to the
Empathic-Insight-Plus scoring heads (3 renames: `Hope_Optimism→Hope_Enthusiasm_Optimism`,
`Intoxication_Altered_States→…_of_Consciousness`, `Jealousy_Envy→Jealousy_&_Envy`). ~1,700 usable
texts per emotion.
- `text` = prompt with `[bracket tags]` stripped.
- `instruction` (all lowercase) = `"extremely {emotion synonyms from the dir name}. starts already
  emotional and escalates line by line until the feeling completely takes over. includes this vocal
  burst, performed fully: {vocal_burst}. loud, intense, high quality recording."`
  (escalation-arc + loudness are the two amplifiers that measurably increase emotion/arousal —
  from a 6-emotion × 20-style ablation.)

### 1c. Casting instructions — [`laion/voice-acting-instructions`](https://huggingface.co/datasets/laion/voice-acting-instructions)
124,349 CSV rows with ready-made `instruction` + `text` columns (text contains inline burst cues
like `(SIGH)`). No emotion labels — use when emotion-targeted ranking is not needed.

**Prompting rules that matter (measured):**
- Token budget is a **pacing dial**: pass `tokens = clamp(words × 6.0, 96, 1400)` codec-frames
  (12.5 Hz ⇒ ~2.1 words/s). `words × 4.2` forces a rushed pace and truncates slow emotional takes
  mid-sentence (96.9 % of takes hit that ceiling). `max_new_tokens = 2.2 × tokens + 300`.
- Record a `finished` flag per take (last-3-words-in-ASR-tail check) to catch residual truncations.
- For reference-voice cloning: pass reference codes and an **empty instruction** — every wording
  ("clone the voice exactly…") measurably *reduces* speaker similarity.
- Don't ask for genuineness ("be authentic…") — plain instructions score higher on both
  genuineness judges.

## 2 · Group design: 1,000 × 128 vs 5,000 × 16

Measured best-of-k behaviour (reward = invWER × (norm blend + norm genu)):
gains flatten around **k ≈ 8–16** for typical use; expanding 16 → 128 still lifts the best take by
**+14–16 %** and finds a better take in ~85–90 % of groups; at n=1000 reward keeps rising (no
plateau; knee ~k = 300–500).

| config | clips | best for | selection |
|---|---|---|---|
| **5,000 groups × 16** | 80k | broad coverage, more distinct prompts | top-1..3/group (WER-gated) |
| **1,000 groups × 128** | 128k | premium quality per prompt | top-3/group by several rewards, dedup |

Save **all** raw takes + scores; ranking is a re-runnable post-pass.

## 3 · Model loading

**8B delay:** `AutoProcessor/AutoModel.from_pretrained("laion/moss-tts-v1.5-8b-voice-acting",
trust_remote_code=True)`, bf16, `attn_implementation="sdpa"`. The processor now resolves the
correct **v1** codec (a former misconfiguration pointing at v2 was fixed on the Hub — if in doubt
pass `codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer"`).

**4.55B local (simplest):** use the pre-merged off-the-shelf release —
[`laion/moss-tts-local-transformer-4.55b-voice-acting`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)
(`AutoModel.from_pretrained(..., trust_remote_code=True)` + `codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"`,
no LoRA handling needed). Manual alternative: load base `TTS-AGI/moss-dramabox-ft` subfolder `checkpoint-last` with
`codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"`, then **merge the rank-256 LoRA into the
weights** (252 deltas, ~30 s; see the model card's inference snippet). bf16 + **SDPA only** —
flash-attn 2.x fails in this checkpoint's remote-code attention (reshape mismatch).

Generation params (both): `text_temperature 0.7, audio_top_p 0.95, audio_top_k 25,
audio_repetition_penalty 1.1`; `audio_temperature` **0.8 without** reference audio, **1.0 with**.
Batch = repeat the prompt's `input_ids` n times and call `generate` once.

## 4 · Post-processing policy: keep RAW for 48 kHz-native models; Sidon only for 24 kHz

[Sidon](https://github.com/sarulab-speech/Sidon) ([weights: `sarulab-speech/sidon_raw_weight`](https://huggingface.co/sarulab-speech/sidon_raw_weight)):
LoRA-adapted [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0) encoder
(8 hidden layers) + DAC decoder (`input_channel=1024, channels=1536, rates=[8,5,4,3,2]`),
16 kHz in → 48 kHz out.

- Resample gen output to 16 kHz mono, peak-normalise to 0.9, batch ≈ 8 through the encoder+decoder,
  **trim each output to `true_frames × 960` samples** using the feature-extractor attention mask
  (never trust padded length). Measured **0.34 s/clip** batched (DAC-decode-bound; batching beyond
  8 doesn't help).
- Store the restored audio as **FLAC 48 kHz** (≈ 0.8 MB per 15 s clip; ~55 % of WAV) and discard
  the raw take.
- Caveat from a 12.9k-clip paired ablation (8B model): Sidon-before-scoring is **neutral for
  selection** (Δreward ≈ 0) — its value is output audio quality, not better ranking.
- **Policy (final, from listening tests + spectra):** the 4.55B local model outputs native 48 kHz —
  running it through Sidon is a 48k→16k→48k round-trip that removes real 8–16 kHz detail and adds a
  vocoder sheen. **Store the raw output for this model; do not apply Sidon.** For the 24 kHz 8B model
  Sidon remains a genuine 16→48 kHz bandwidth upgrade (optional). Skipping Sidon also saves
  0.34 s/clip.

## 5 · Scoring stack (per take; every model linked)

| signal | model | cost (A100, measured) |
|---|---|---|
| WER / invWER | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (`ParakeetForTDT`, transformers ≥ 5.6, **bf16, batch 32**, cast `input_features` to model dtype) | **0.0097 s** (134× vs fp32 unbatched) |
| blend (vocal-burst blending) + genuineness | [`laion/voiceclap-commercial`](https://huggingface.co/laion/voiceclap-commercial) — **one** `encode_waveform` per clip, reused by both MLP heads (blend head standardised-raw, genu head L2-normalised-standardised) | 0.027 s total |
| prompt adherence (`prompt_sim`) | same VoiceCLAP embedding · cosine vs `encode_text(direction-only instruction)` (text embedding cached per prompt) | ~0 (reuses encode) |
| 40 emotions + arousal/valence/authenticity | [`laion/Empathic-Insight-Voice-Plus`](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) heads on [`laion/BUD-E-Whisper`](https://huggingface.co/laion/BUD-E-Whisper) encoder — heads take the **flattened** 1500×768 encoder states (1,152,000-d input, `Linear(→64)` + MLP; strip `_orig_mod.` from checkpoints) | 0.24 s (one encoder pass, all heads) |
| volume | mean-RMS dBFS + peak dBFS | ~0 |

WER normalisation: strip `(parentheticals)`, lowercase, remove punctuation before `jiwer.wer`.

## 6 · Rewards (define per use-case; compute at export time)

Store raw scores; min-max normalise **globally over the whole run** at export
(`norm(x) = (x−min)/(max−min)`), then:

```
reward_orig = invWER × ( norm(blend) + norm(genu) )                       # quality default
reward_ps   = prompt_sim × invWER                                          # direction adherence
reward_all  = invWER × ( norm(prompt_sim) + norm(blend) + norm(genu) )     # equal weights
reward_emo  = invWER × ( w·norm(target_emotion) + norm(blend) + norm(genu) )  # emotion-targeted, w≈1.5
```

Selection: top-1/2/3 per group per reward; **dedup** takes that win multiple rewards (in practice
~25 % overlap); optional hard gate `WER < 0.1` for intelligibility-critical corpora (reward alone
can prefer expressive-but-garbled takes — measured on chris-voice groups). Note `reward_orig` and
`prompt_sim` are nearly uncorrelated (r ≈ −0.04): if adherence matters it **must** be in the reward.

## 7 · Measured throughput & cost (per A100; Sidon + full scoring included)

Per-clip (≈ 15 s audio, batch = group, SDPA bf16):

| stage | 8B delay | 4.55B local |
|---|---|---|
| generation | 0.67 s (batch 100) / 0.87 s (batch 16) | 0.87 s (batch 64) / 0.83 s (batch 128) |
| Sidon (batch 8) | 0.34 s | 0.34 s |
| scoring (ASR+CLAP+volume) | 0.04 s | 0.04 s |
| + EI-Plus emotion heads | +0.24 s | +0.24 s |
| **fused end-to-end (with Sidon)** | ≈ 1.2–1.3 s/clip | ≈ 1.1–1.75 s/clip (measured) |
| **fused end-to-end (RAW, no Sidon — recommended for 4.55B)** | ≈ 0.9–1.0 s/clip | **≈ 1.0–1.2 s/clip (production-measured, early-run mean 1.21)** |

Workload table (fused, EI included; halve the EI column out if not emotion-ranking):

| workload | clips | 4.55B local (8×A100) | 8B delay (8×A100) | 1×A100 (local) |
|---|---:|---:|---:|---:|
| 5,000 × 16 | 80k | **~3.3 h** | ~3.7 h | ~26 h |
| 1,000 × 128 | 128k | **~5.3 h** | ~5.9 h | ~42 h |
| 40 emo × 100 × 64 | 256k | **~9–11 h raw** (measured) | ~11.5 h | ~86 h |

Storage: FLAC 48 kHz ≈ 0.8 MB / 15 s → 128k clips ≈ 105 GB. Tar per bucket (emotion/subset) with a
`scores.parquet` + `cells.jsonl` inside; upload-and-delete keeps local disk bounded.

## 8 · Making it fast on A100s (checklist)

1. **Batch the ASR** (bf16, batch 32) — the single biggest win in the whole pipeline (134×).
2. **Batch = group size** (same prompt repeated): zero padding waste, no length bucketing needed.
   Bucketing only matters if you batch across different prompts.
3. **SDPA, not flash-attn**, for both MOSS checkpoints (FA2 path broken in the remote code);
   flash-attn also gave no speedup on the 8B decoding in earlier tests.
4. **Fuse gen → Sidon → score in one process per GPU**; audio never hits disk twice; shard groups
   `cells[i::N]` over N GPUs; resume markers per group.
5. **Merge LoRA into weights** at load; **cache reference codes** and **direction-text embeddings**.
6. One VoiceCLAP encode per clip → blend + genu + prompt_sim.
7. Token budget ×6.0 frames/word (quality) — accept the ~40 % audio-length cost vs ×4.2; budget is
   the biggest single knob trading cost against truncation rate.
8. Keep EI-Plus heads off unless you rank by emotion (+0.24 s/clip); heads are fp32, ~300 MB each.
9. Don't bother: Sidon batch > 8 (DAC-bound), torch.compile (variable lengths), fp8/int8
   (unvalidated audio-quality risk). vLLM-style continuous batching would give ~2× but needs custom
   integration for the MOSS audio heads — only worth it at ≥ 1M-clip scale.

## 9 · Running on a fresh GPU pod (bootstrap checklist)

1. **Env** (Python 3.10+, CUDA 12.x): `pip install torch torchaudio --index-url
   https://download.pytorch.org/whl/cu128` then `pip install "transformers==5.13.1"
   huggingface_hub accelerate sentencepiece librosa soundfile numpy pandas pyarrow "jiwer>=3.0"
   peft descript-audio-codec safetensors` . If `torchaudio.load` complains about torchcodec,
   monkey-patch load/save to use `soundfile` (all pipeline audio is WAV/FLAC).
2. **Auth**: `export HF_TOKEN=...` (needed for the private base `TTS-AGI/moss-dramabox-ft` and any
   private prompt repos).
3. **Prompts** (pick one source, §1):
   - DramaBox: `git clone https://github.com/LAION-AI/Voice-Acting-Pipeline` → `data/dramabox_*.json`
   - 40-emotion prompts: `huggingface-cli download laion/voice-acting-prompts
     geminittsprompts-output-en.tar --repo-type dataset` (emotion+burst labelled)
   - casting CSVs: `laion/voice-acting-instructions`
4. **Models** (auto-downloaded on first run): generator (§3 — 8B: `laion/moss-tts-v1.5-8b-voice-acting`;
   4.55B: `TTS-AGI/moss-dramabox-ft` + LoRA `TTS-AGI/moss-local-transformer-voice-acting@checkpoint-1m`),
   scorers `nvidia/parakeet-tdt-0.6b-v3`, `laion/voiceclap-commercial`, `laion/BUD-E-Whisper` +
   `laion/Empathic-Insight-Voice-Plus` heads; optional Sidon `sarulab-speech/sidon_raw_weight`
   + `facebook/w2v-bert-2.0` (24 kHz 8B model only, see §4).
5. **Storage**: raw FLAC 48 kHz ≈ 0.8 MB / 15 s; plan ~1 GB per 1,200 clips; tar per bucket and
   upload-and-delete to bound disk.
6. **Sanity gates before a long run**: (a) one-group smoke end-to-end, (b) ASR hypothesis on 4 takes
   (WER≈0 on an easy prompt proves the codec pairing is right), (c) confirm SDPA is active.
7. **Example output corpus / reference**: [`laion/moss-local-voice-acting-64x100-part1..4`](https://huggingface.co/datasets/laion/moss-local-voice-acting-64x100-part1)
   (raw 48 kHz FLAC + per-take scores + prompt metadata per emotion tar).

## 10 · Reference implementation

Working scripts (this study's box): manifest builder, fused worker (gen+Sidon+score→FLAC+parquet),
uploader with per-bucket tar-and-delete, and rank/export with all four rewards — see
`bulk_pipeline/` and the production scripts described in the study pages at
<https://projects.laion.ai/moss-8b-voice-acting/> (fast-inference write-up:
[fast_inference_insights.html](https://projects.laion.ai/moss-8b-voice-acting/fast_inference_insights.html)).
Example output corpus: [`laion/moss-local-voice-acting-64x100-part1..4`](https://huggingface.co/datasets/laion/moss-local-voice-acting-64x100-part1)
(40 emotions × 100 groups × 64 Sidon-restored takes with full scores).
