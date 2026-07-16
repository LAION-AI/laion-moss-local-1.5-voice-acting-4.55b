# Character-Voice Prompting Guide — MOSS-TTS-Local 4.55B (for dummies)

A practical, plain-English cheat-sheet for writing **director instructions** that make
[`laion/moss-tts-local-transformer-4.55b-voice-acting`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting)
*become* a character — **zero-shot, no reference audio**. It distills three rounds of an
evolutionary prompt search (scored by a VoiceNet 57-dim reward + an EmoNet emotion reward).

- **Round 1** — describe the acoustics. → [`docs/archetypes.html`](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes.html)
- **Round 2** — also name the emotion. → [`docs/archetypes_round2.html`](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes_round2.html)
- **Round 3** — also name resonance, pitch/register and dominance. → [`docs/archetypes_round3.html`](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes_round3.html)

The instruction goes in the **`instruction`** field of `build_user_message(...)`; the words
to be spoken go in **`text`**. Bursts like `(guttural growl)` go **in the instruction**, in
round brackets — not inside the spoken text.

---

## Part 1 — Principles (read this first)

1. **Describe concrete acoustics, not the label.** "cartoon mouse voice" scores worse than
   "extremely high, thin, pinched, buzzing nasal, bright piercing head voice." The model
   listens to the *description*, so spell out how it sounds.

2. **Name the emotion explicitly.** A perfect zombie rasp that sounds bored is a bad zombie.
   Add the feeling in words — *helpless, murderous glee, seething contempt, giddy delight,
   drowning in pain* — matched to the character. This was round 2's single biggest lever.

3. **Say WHERE the voice resonates (resonance).** Not just "deep" or "bright" — name the
   cavity: **chest** (R_CHST, big/hollow/warm), **throat** (R_THRT, guttural/raspy),
   **nose** (R_NASL, nasal/whiny/buzzing), **head** (R_HEAD, bright/glassy/light).
   Resonances **stack** — a zombie can have a hollow *chest* groan *and* a wet *throat* rasp;
   an ASMR whisper can have a soft warm *chest* underneath.

4. **Control pitch with two separate words: register and range.**
   - **Register** = how high or low the voice sits. Say "deep subterranean bass register"
     (low) or "sitting way up in a sparkling high register" (high).
   - **Range** = how *far* the pitch moves. Say "flat and unwavering, never leaping" (narrow)
     or "the pitch swooping/lurching/skittering wildly up and down" (wide).
   - Wide swings read as **mania / fury / excitement**; flat narrow pitch reads as
     **menace / authority / deadness**.

5. **Set dominance: commanding vs meek.** "Commanding, dominant, immovable, planting each
   word with authority" pushes S_AUTH/STNC **up** (ork, dragon, rant). "Feeble, powerless,
   meek, yielding, no authority whatsoever" pushes them **down** (zombie, goblin, fairy,
   ASMR). **Meekness is a real, rewardable direction** — negating dominance is as useful as
   asserting it.

6. **Dominance/emotion/resonance words must ride ON TOP of the character core — never
   replace it.** Naming authority alone ("an orc warlord, commanding") produced a clean
   commander, not a monster (character collapsed ~0.30). Keep the guttural/nasal/whisper
   timbre description and *append* the new layer.

7. **Bursts: one by default.** A single emotion-matched burst — `(agonized moan)`,
   `(delighted giggle)`, `(terrified scream)` — reliably helps. A second burst usually hurts
   **except for very high-character archetypes** (ranting uses two: `(enraged shout)(seething
   snarl)`). The higher the character weight, the more a second affect-burst pays off.

8. **Negate the penalized dims out loud.** "no chest, no warmth, no fullness" (goblin),
   "no brightness, no high notes" (dragon), "zero brightness" (ork) directly raise reward.

9. **Balance beats character-maxing.** Pushing one dimension to the extreme ("ear-splitting
   volume", "impossibly high") garbles words or thins the voice. Keep a safety clause like
   **"roaring but the words still land"** / **"whispered but every word clear"** to protect
   intelligibility on the WER-gated archetypes.

10. **Some dims are already implied — check first.** Pain-scream's "each cry cracking higher"
    already encoded pitch range, so adding explicit octave-leap wording only added variance.
    If the target property already fires, spend the words elsewhere.

11. **Pick champions in same-call playoffs.** The reward is min-max normalized *within* each
    best-of-N run, so a "2.4" in one run ≠ a "2.4" in another. Compare finalists by putting
    them in one run together, and generate **best-of-N** (N=8 for search, N=64 for the final)
    — the best single take is much better than the average.

12. **The ranting fix (a worked example of 1–11).** Round 1 failed because "maximum volume"
    made the model garble. The fix: reframe rage as **shredding, not louder** ("voice
    shredding and cracking"), name the **contempt/hatred**, add **dominance** ("talking down,
    refusing to yield"), add **violent pitch swings**, use **two** bursts, and keep "the words
    still land." Result: furious *and* intelligible.

---

## Part 2 — Per-archetype recipes

Each entry: the **round-3 champion instruction** (copy-paste ready), then a short **recipe**
of which words to use, which burst, and what to avoid.

### 🧟 Zombie
> A pitiful half-dead zombie, utterly powerless and feeble, weak, trembling and vulnerable, helpless and abandoned, delirious with fever and dull aching pain. A hollow groan from a sunken caved-in chest dragging up through a wet gurgling guttural throat, phlegmy rasp, decayed. Low, dark, toneless and meek, submissive, no authority, no brightness, slurred and drained. (feeble whimper)

- **Acoustic:** wet gurgling guttural throat rasp, phlegmy, decayed, slurred.
- **Resonance:** a **hollow / sunken / caved-in chest** groan *plus* the throat rasp (stack them).
- **Pitch:** low, dark, toneless (low register, near-flat).
- **Dominance:** **meek, powerless, submissive, no authority** (this is what makes it pitiful, not a monster).
- **Emotion:** helpless, feeble, vulnerable, drained, delirious, in dull aching pain.
- **Burst:** one — `(agonized moan)` / `(feeble whimper)`.
- **Avoid:** a *booming* chest (reads as powerful); brightness; a second burst.

### 👹 Ork
> An enormous brutish orc, his voice a deep chest-driven guttural bass, resonating from a massive barrel chest and gravelled throat, thick and rough, full and cavernous. Commanding, dominant and forceful, planting each word with authority. Low, flat and narrow, never leaping in pitch. Seething hatred and sneering contempt, dark triumphant menace, zero brightness. Growled but every word lands. (guttural growl)

- **Acoustic:** cavernous guttural bass, thick gravel rasp, full.
- **Resonance:** **deep chest-driven bass from a massive barrel chest** + gravelled throat.
- **Pitch:** low, **flat and narrow, never leaping** (low register + narrow range = menace).
- **Dominance:** **commanding, dominant, immovable, planting each word with authority.**
- **Emotion:** seething hatred, sneering contempt, dark triumphant menace.
- **Burst:** one — `(guttural growl)` up front.
- **Avoid:** naming authority *without* the growl/chest (makes a clean commander, character collapses); "roaring" (garbles words).

### 🗯️ Ranting Fury
> A seething furious rant, aggressive and confrontational, leaning in and refusing to yield, dominating and talking down with withering contempt. Voice shredding and cracking with rage, spitting venom, harsh, tense and volatile, explosive consonants, pounding emphasis, the pitch swinging violently high to low, escalating louder and more vicious. Roaring but the words land. (enraged shout)(seething snarl)

- **Acoustic:** voice **shredding and cracking** (not just loud), harsh, tense, volatile, explosive consonants, pounding emphasis, escalating.
- **Pitch:** **swinging violently high to low** (wide range = fury).
- **Dominance:** **domineering, confrontational, talking down with contempt, refusing to yield** — the biggest round-3 win.
- **Emotion:** rage, vicious contempt, spitting venom.
- **Burst:** **two** — `(enraged shout)(seething snarl)` (the one archetype where two help).
- **Avoid:** "maximum volume / ear-splitting" (garbles); drop the "words still land" clause and intelligibility craters.

### 👺 Goblin
> A nasty little goblin voice, extremely nasal and pinched, buzzing high through the nose with a grating scratchy rasp, whiny and reedy. The pitch swoops and cracks wildly higher and shriller with gleeful spite, giggling, cackling, taunting and jeering, gloating and mean. A weak, cowardly underling, no authority whatsoever. No chest, no warmth, no fullness, zero fullness.

- **Acoustic:** extremely **nasal**, pinched, buzzing through the nose, grating/scratchy, whiny, reedy.
- **Pitch:** **swoops and cracks wildly higher and shriller** (wide, manic range).
- **Dominance:** **a weak, cowardly underling, no authority** (sniveling, not commanding).
- **Emotion:** gleeful spite, cackling/giggling, taunting, jeering, gloating, mean.
- **Negate:** "no chest, no warmth, no fullness."
- **Burst:** one optional `(wicked cackle)`.
- **Note:** hardest to balance — nasal high-character takes tend to have lower emotion; the soft WER gate caps reward because the buzz slurs.

### 🐉 Dragon
> An ancient dragon proclaiming judgement, voice in a cavernous subterranean bass register, flat and unwavering in pitch, slow and thunderous. Enormous chest resonance, deep guttural throat rumble, dark and full. Utterly dominant and immovable, planting every word with crushing authority, wrathful, prideful, contemptuous and menacing. No brightness, no high notes. (deep rumble)

- **Acoustic:** slow, thunderous, dark, full.
- **Resonance:** **enormous chest resonance + deep guttural throat rumble.**
- **Pitch:** **subterranean bass register, flat and unwavering** (lowest register + narrowest range).
- **Dominance:** **utterly dominant and immovable, crushing authority.**
- **Emotion:** wrathful, prideful, contemptuous, menacing.
- **Burst:** one — `(deep rumble)`.
- **Avoid:** any "high notes" or brightness; fast tempo.

### 😱 Pain Scream
> Shrieking in raw physical agony and terror, voice torn and guttural, harsh grating rasp shredding through, each cry cracking higher and more desperate. Panic-stricken and helpless, distressed and trembling with fear, drowning in pain. Extreme arousal, extreme tension, violently volatile and frantic. (terrified scream)

- **Acoustic:** torn, guttural, harsh grating rasp, shredding; extreme arousal/tension/volatility.
- **Pitch:** **each cry cracking higher** (this already encodes wide range — don't over-specify octave leaps or you lose intelligibility).
- **Emotion:** agony, terror, panic, helplessness, drowning in pain.
- **Burst:** one — `(terrified scream)`.
- **Avoid:** explicit "octave leaps" wording (adds variance, costs WER); it's already the highest-character archetype.

### 🐭 Cartoon Mouse
> A tiny mouse squeaking away, extremely high, thin and pinched, buzzing nasal head resonance, bright and piercing, tense. The pitch darts and jumps giddily up to the very top and back. Chipper, giggly, amazed and eagerly curious, delighted. Weightless and young, no chest, no fullness, meek and timid, not a shred of authority. (amazed gasp)

- **Acoustic:** extremely high, thin, pinched, buzzing nasal, bright piercing head voice, tense.
- **Pitch:** **top of the register + darting/jumping giddily** (high register + wide range).
- **Dominance:** **meek and timid, no authority** (a tiny critter).
- **Emotion:** chipper, giggly, amazed, curious, delighted.
- **Negate:** "no chest, no fullness."
- **Burst:** one — `(amazed gasp)`.
- **Avoid:** "impossibly high" (cracks into noise).

### 🧚 Fairy
> Delicate high fairy voice, feather-light and airy, warm and very bright, sitting way up in a sparkling high register, the pitch lilting and swooping sweetly up and down in a sing-song melody. Sweet, lovingly gentle and giddy, bubbling with elated joyful delight. Pure glassy head resonance, no chest at all, tender, gentle and unassuming, never commanding. (delighted giggle)

- **Acoustic:** feather-light, airy, very bright, crystal clear, no roughness.
- **Resonance:** **pure glassy head resonance, no chest.**
- **Pitch:** **sparkling high register + lilting, swooping sing-song** (high register + wide sweet range).
- **Dominance:** **gentle, unassuming, never commanding.**
- **Emotion:** sweet, giddy, elated joyful delight, affection.
- **Burst:** one — `(delighted giggle)`.
- **Avoid:** "extremely high-pitched" (thins the voice); heavy "gushing love" language (backfires — imply joy).

### 🌙 ASMR Man
> A deep, soothing man whispering tenderly, soft and breathy, intimate close-mic, slow and gentle, warm low chest resonance cushioning each word. Loving, reassuring and calming, content and serene. Utterly gentle and unassuming, meek, yielding, no authority whatsoever. Clear whispered words. (soft breath)

- **Acoustic:** soft, breathy, intimate close-mic, slow, gentle whisper; clear words.
- **Resonance:** a **soft warm low chest resonance UNDER the whisper** (adds body/intimacy without breaking it).
- **Dominance:** **utterly meek, yielding, no authority.**
- **Emotion:** loving, reassuring, calming, content, serene (Affection/Contentment/Relief).
- **Burst:** one — `(soft breath)`.
- **Avoid:** too much chest (becomes a normal speaking voice).

### 🌙 ASMR Woman
> A gentle motherly woman murmuring ASMR to lull you asleep, wrapping you in warmth. Extremely soft, breathy, intimate close-mic whisper, slow, hushed and quiet. Warm, tender, adoring, delicate and feminine, full of affection and serene contentment, a soft sigh of relief. Meek, gentle and yielding, never bossy or commanding. Whispered but every word clear. (soft breath)

- **Acoustic:** extremely soft, breathy, intimate close-mic whisper, slow, hushed; keep words clear.
- **Voice:** softly feminine and delicate.
- **Dominance:** **meek, gentle, yielding, never bossy or commanding** (this sharpened the maternal read).
- **Emotion:** the **maternal simile** ("like a mother soothing her child to sleep, wrapping you in warmth") is the strongest lever — warm, adoring, affection, serene contentment.
- **Burst:** one — `(soft breath)`.
- **Avoid:** dropping the "every word clear" clause (full WER gate).

---

## Part 3 — A reusable template

```
<CHARACTER + one-line framing/simile>.  <ACOUSTIC timbre: rough/smooth, breathy/full…>,
<RESONANCE: which cavity — chest / throat / nasal / head, and stacked if needed>.
<PITCH: register how high/low + range how wide it swings>.
<DOMINANCE: commanding & immovable  — or —  meek & powerless>.
<EMOTION words matched to the character>.  <negate penalized dims>.
<safety clause: "but the words still land" / "every word clear" for clean voices>.
(ONE emotion-matched burst)   [ second burst only for very high-character rage ]
High quality recording.
```

Generate **best-of-8** while searching and **best-of-64** for the final pick; compare
variants in one shared run. Full listenable results and the 57-dim profiles behind every
number are in the [round-3 grid](https://projects.laion.ai/laion-moss-local-1.5-voice-acting-4.55b/archetypes_round3.html).
