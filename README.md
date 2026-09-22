---
title: Pronunciation Coach
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Pronunciation Coach

Phoneme-level pronunciation feedback for learners of English, built entirely on
open models. Instead of trusting a recogniser's transcript, it reads the
recogniser's *internal* per-frame phone probabilities and scores each expected
sound against what was actually produced.

Designed for Spanish- and Catalan-speaking learners, but nothing in the core
is language-specific.

## How it works

```text
audio ─┬─ Whisper ──→ words ──→ espeak-ng G2P ──→ expected phones ──→ forced alignment ──→ GOP per phone
       └─ wav2vec2 phoneme CTC ──→ posterior grid ──→ greedy decode ──→ "what was heard"
```

* **Learner view** – a score ring, the sentence as tappable word chips (clear / accent / almost /
  work on this), one player that speaks whatever you tap, with a line of plain-language advice per
  sound that needs work. Tapping a word plays what *you* said, with a linked word when useful;
  tapping within the same playback group again plays the model; again, you, and so on.
  The heading shows the group and emphasizes the selected word; colors and tips stay per word. The whole sentence
  has its own ▶ You / ▶ Model buttons. One speed slider (0.5-1.5 in steps of 0.05) applies to everything
  played, so you and the model are always compared at the same pace: the model voice is synthesised at
  that rate, your own audio is time-stretched with the pitch kept (ffmpeg atempo). The model voice can
  be male or female (AndrewMultilingual / Jenny for American, Ryan / Sonia for British; default `PC_VOICE=Male`).
  Playback selection is independent of the Guy / Jenny voices used for native scoring references.
  It is Microsoft Edge's neural TTS via `edge-tts` (cached in `tts_cache/`; falls back to espeak-ng
  offline). The input row is just the recorder, the sentence box and the check button; the target
  accent and model voice pickers sit with everything technical in a collapsed
  "Technical details (for teachers)" section: timeline, IPA, per-phone table, posterior heatmap.
* **Word crops** – the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io)
  aligns the whole utterance against the sentence and reports where each word is. It runs in a
  resident worker (`mfa_worker.py`) that loads the acoustic model and lexicon once: an alignment
  then takes about 25-95 ms, against ~16 s for a fresh `mfa align_one`, and the output is
  identical. The worker also avoids a real hazard - MFA unpacks its 50 MB model into a shared
  directory on every start, so two concurrent runs corrupt each other - by unpacking once, into
  its own directory, and serving requests one at a time.

  Forced alignment cannot report that the learner did not read the sentence: it emits an interval
  for every word of the transcript whatever was actually said, so an omitted word does not raise,
  it silently shifts its neighbours (measured at up to 250 ms - enough that tapping *like* plays
  *my*). MFA is therefore asked only when the sentence is trustworthy, and its answer is checked
  afterwards. It is skipped when a word was not said, was not recognised by the listener, carries
  an inserted vowel, when speech was heard outside the sentence, or when a word is not in the
  dictionary; its answer is discarded whole if any word's start disagrees with the scoring model's
  spikes by more than 150 ms. Rejection is all-or-nothing, because one shifted word moves every
  boundary after it. Each word's end is also capped at its own last spike plus 250 ms, since MFA
  has no wildcard state and otherwise absorbs a following hesitation (measured: a 1.30 s crop for
  *So*). Whenever MFA is not used, the reason is logged and shown in the technical panel, and
  Charsiu's frame-level aligner (`charsiu/en_w2v2_fc_10ms`, ~380 MB, `segmenter.py`) does the
  cropping instead; if that is unavailable too, the spike-based estimate in `boundaries.py` is.

  Both aligners run late against ground truth by a near-constant amount, so both subtract a
  measured calibration offset (`OFFSET_S`, 40 ms in both `mfa.py` and `segmenter.py`).
  Each word's *start* is then settled with the audio as the referee (`settle_starts` in
  `pipeline.py`), because the aligner and the spikes fail in opposite ways: the aligner runs late
  on quiet onsets (a stop's closure, a fricative, *h*, a nasal) and hands them to the previous
  word, while the spike-based onset can reach back into the previous word's vowel - and the
  previous word's own last spike may fire after our first sound has begun, so it is no floor
  either. Three cases: **onset** (silence before the word: start where sound resumes, e.g. at a
  k's burst), **dip** (the word starts with a quiet consonant after a vowel and the energy shows
  the valley the aligner skipped, its rise at the aligner's boundary: start at the valley),
  **join** (anything else, including a consonant that belongs to the previous word: the aligner's
  boundary, never later than the spike allows). Each decision is logged (`crops (mfa): can
  0.59-0.88 onset | ...`) and archived with the recording. `scripts/crop_check.py` audits archived
  recordings without loading any model: per word, ms of the first sound cut, of the previous/next
  word included, of the last sound cut, and edge silence. `scripts/calibrate_spikes.py` measures
  crops against Edge TTS word-boundary metadata, which is where the calibration offsets come from;
  these are synthetic-speech measurements, not human-annotated learner-speech accuracy.
  Playback clips take the recording at its original rate and add up to 60/50 ms of room on either
  side **only through quiet audio** (silence, a closure, breath - never a neighbour's vowel), with
  10 ms fades and a level boost, then follow the speed slider.
  The start search cannot reach before the preceding word's first kept spike; adjusting a shared
  boundary cannot reverse that word's crop. A final logged safety check keeps live spans ordered
  and inside the recording, without inventing audio for missing words.
* **Short playback groups** – function words (articles, prepositions, pronouns, conjunctions and
  helping verbs) are candidates for context: **[can you] [hear me]**, **[the store]**, **[to school]**.
  **Pairs are the default, not a strict limit.** Compact question openings can stay together:
  **[what do you] [want to] [watch]**, or **[how are you]**. These three-word candidates require
  live words, weak auxiliary/pronoun cues, and no detected emphasis or internal barrier.
  Lexical connected pairs such as *want to* take priority over a generic attachment of *to*.
  Chunks are selected as complete groups, never by chaining pairs into a whole sentence.
  Content words generally stay alone; grouping suggests context, not proof a reduction was heard.
  Neighboring pair choices favor grammatical attachment; main-verb *have/do* are not automatically
  treated as unstressed. Duration and relative loudness provide a conservative emphasis heuristic,
  not a linguistic stress detector. Crops under 120 ms can also receive context. Punctuation,
  pauses/gaps of at least 100 ms and intervening extra speech block a group; a group longer than 1.5 s
  is not formed. A barrier or cap may leave a short word alone. Missing words add no parked audio,
  and an entirely missing chunk offers only model playback. Both voices play the same word/group
  text, using the existing speed control. The technical table and recording archive include
  membership and reasons. `scripts/crop_check.py --rechunk` audits archived recordings without
  model inference (old archives assume 20 ms scoring frames). Set `PC_CHUNKS=0` to compare with
  individual-word playback; no new model or dependency is needed.
* **Detect, trim, recheck (retired)** – a second pass used to re-transcribe each rendered
  playback crop with the Whisper listener and trim corroborated extra words at its edges. It is
  off by default (`PC_CROP_RECHECK=0`). On the saved recordings it accepted **no** correction at
  all: 24 playback groups, 7 transcript matches, 10 uncertain, 7 skipped, zero trims, for 2.7-9.9
  seconds of extra recogniser calls per assessment. The reasons were structural rather than
  tuning - the alignment half only ever proposed outward extensions and was called with
  `apply=False`; the operative half could only trim corroborated edge words, never restore a
  clipped onset or move a boundary shared with the neighbouring group; and isolated recognition of
  short clips is unreliable (*couldn't you* came back as *I can do it*, *App* as *Hah!*).
  Cropping accuracy is now addressed at the source, by the aligner, instead.

  The code and its tests are retained, and the pass can still be run for an audit:
  `PC_CROP_RECHECK=audit` records proposals without applying them and `1` applies them, with
  Charsiu run alongside MFA purely to supply the alignment evidence it reads. When the listener is
  unavailable the mode is reported as `listener-unavailable`, and when only the verification half
  could run it is reported as `apply (verify only)` rather than plain `apply`, so audits taken
  before and after this change stay comparable. `scripts/recheck_compare.py` still runs
  cached-model comparisons and builds a listening page:
  `uv run python scripts/recheck_compare.py "recordings/*.json" --native-references --all-crops --output tmp/recheck-review.json`,
  then open `tmp/recheck-review.html`. Recognition matches were never a human listening
  evaluation, and TTS timestamps remain metadata proxies rather than acoustic ground truth.
  For two-machine comparisons, run `uv run python scripts/diagnose_setup.py` on each machine
  using the same launcher environment. New archived assessments include the same setup snapshot
  (package versions, Git revision, model names, effective settings and ffmpeg location).
  Compare the same WAV, sentence and settings; microphone processing can vary between machines.
* **Speech, not dictionary words** – three things stop natural, connected speech from being
  penalised. (1) *Native reference by example*: the sentence is synthesised with two natural
  voices and run through our own recogniser, and whatever it hears a native do in that position
  counts as correct (`reference.py`). (2) *Casual forms and connected-speech rules* – gonna/'ona,
  dunno, weak forms, dropped final t/d, assimilation, yod coalescence, glottal t
  (`variants.py`, meant to be edited by the teacher; `PC_CASUAL=0` for careful-reading mode).
  (3) *A listener in the loop*: Whisper's per-word confidence says whether each word was caught
  (`listener.py`). Each word then gets a verdict - **clear**, **accent** (understood; the
  deviations are colouring), **almost** (understood with effort, or a sound that matters),
  **work on this** (missed, or a top-priority contrast such as *th* or *v*) - from the listener's
  confidence and the worst deviation weighted by how much that sound matters (`PRIORITY` in
  `scoring.py`). The score ring counts words understood. The listener alone cannot fail a word whose
  sounds were all right (its confidence also collapses over homophones and function words); it can
  only turn it "almost" when it wrote another word entirely.
* **Added sounds** – the wildcard between words absorbs repeats and hesitations, but a vowel glued to
  a word that starts with *s* + consonant ("e-speak", "e-Spain": Spanish and Catalan have no such
  onsets) or to a word-final consonant ("English-e") is an error, not a hesitation: `attach_insertions`
  in `pipeline.py` moves it into the word (no pause between them, within 160 ms), it weighs like a
  wrong ɪ, the word's crop includes it, and the tip says what to do. Note that in free-speech mode the
  *words* come from Whisper, which is trained to spell accented speech correctly ("espeak" → *speak*);
  the sounds are judged by the phoneme recogniser, which never sees Whisper's transcript.
* **How to make the sound** – from the sibling project [GAPhonetics](https://audiophrases.github.io/GAPhonetics/)
  (24 GA vowels + 24 consonants, each with tongue/lips/jaw instructions, the sensation to feel, the
  mistake to avoid, an example word and human recordings). For every sound that needs work the word
  panel shows the instructions, plays the isolated sound and the example word, suggests a minimal
  pair, and links to GAPhonetics opened on that exact contrast (`#vowels?a=æ&b=ɑ`,
  `#consonants?a=ð&b=d`). `phonetics.py` reads the local checkout (`PC_GAPHONETICS_DIR`, or the
  sibling folder) and falls back to the published site (`PC_GAPHONETICS_URL`). Recordings are
  CC BY-SA 3.0 / CC0 / public domain; credits live in GAPhonetics' `*-audio-sources.json`.
* **Known-sentence mode** – you type the sentence; the listener still runs, as the intelligibility judge.
* **Free-speech mode** – leave the text empty; Whisper's transcript is the reference and the
  phoneme track shows what was actually *said*.
* **GOP** (Goodness of Pronunciation) = log P(expected phone) − log P(best competing phone),
  averaged over the frames the expected phone was aligned to. 0 is perfect; more negative is worse.

## Windows: double-click launchers

Everything in `launchers\` is a plain `.bat` file:

| File | What it does |
| --- | --- |
| `setup.bat` | First-time install: uv, espeak-ng, ffmpeg (via winget), Python packages, model download, test run |
| `run_app.bat` | Starts the coach and opens it in your browser. Keep the window open; close it to stop |
| `score_recording.bat` | Drag a recording onto it, type the sentence (or press Enter for free speech), get the phone table and heatmap |
| `run_app_debug.bat` | Same as `run_app.bat`, plus the full per-phone table in the log and every recording archived in `recordings\` (wav + json) |
| `open_logs.bat` | Opens `logs\app.log` (and `logs\share.log` if there is one) in Notepad and the `recordings\` folder |
| `share.bat` | Starts the coach and shares it with students: a Cloudflare tunnel exposes it, and the fixed address `https://audiophrases.github.io/PronunciationCoach/` forwards to this session. Keeps everything like `run_app_debug.bat` does, plus a session log in `logs\share.log`. Keep the window open |
| `stop_sharing.bat` | Ends the sharing session cleanly and marks the fixed address "closed" |

The app always writes one line per assessment to `logs\app.log` (mode, text, what was heard, flagged
phones, timings) plus any errors. Archived recordings can be replayed by dragging the `.wav` onto
`score_recording.bat`.

Preferences live at the top of `launchers\_env.bat`: `PC_ACCENT` (General **American** by default, or `British`),
`PC_VOICE` (**Male** by default, or `Female`) and `PC_ASR_MODEL` (Whisper size for free speech).

The interface is always in English, whatever the browser's language. Gradio normally translates its own
buttons ("Record" → "Gravar"); `pronunciationcoach/gradio_ui_english.json` overrides that for every
locale Gradio ships. After upgrading Gradio, regenerate it with `uv run python scripts/extract_gradio_strings.py`.

## Run from a terminal

Requirements: Python 3.11 (managed by `uv`), [uv](https://github.com/astral-sh/uv),
[espeak-ng](https://github.com/espeak-ng/espeak-ng) and ffmpeg on PATH.

```bash
uv sync
uv run python scripts/smoke_test.py   # downloads the model (~1.2 GB) and scores a test sentence
uv run python app.py                  # Gradio UI on http://127.0.0.1:7860
```

On Windows, install espeak-ng with `winget install eSpeak-NG.eSpeak-NG`; the code finds the
DLL in `C:\Program Files\eSpeak NG` automatically.

Memory: the phoneme model (~1.4 GB), the MFA worker (~0.5 GB), the fallback cropping model
(~0.4 GB) and Whisper `base.en` (~1.2 GB,
the listener) together need about 3.5 GB free; if Whisper cannot load, verdicts fall back to
severity only.
On an 8 GB machine, let Windows manage the page file size and close browser tabs you don't need.

## Running it on another machine

Clone the repository, double-click `launchers\setup.bat`, and the machine is ready - the same
launchers work there. `setup.bat` installs uv, espeak-ng, ffmpeg, cloudflared and the GitHub CLI
if missing, the Python packages and the Montreal Forced Aligner, and downloads all models
(about 2.5 GB downloaded, about 6 GB on disk) by scoring a test sentence.
The first `share.bat` signs in to GitHub in the browser (once per machine) so it can publish the
session address. Only one machine should share at a time - the fixed address points at whichever
published last. If GAPhonetics is not cloned next to the coach, the sound guidance is read from
its published site.

## Sharing the coach from this machine

`launchers\share.bat` runs the coach here and makes it reachable from anywhere:

1. the app starts on this machine;
2. `cloudflared` (installed on first use) opens an outbound tunnel and gets a temporary
   `https://<random>.trycloudflare.com` address - no account, no router settings, HTTPS included,
   which browsers require before allowing the microphone. The tunnel runs over HTTP/2 (TCP) rather
   than cloudflared's default QUIC, because school and guest networks often block the UDP port QUIC
   needs, and cloudflared then prints an address that never connects;
3. once the tunnel has actually registered, `scripts/share.py` writes that address into `coach.json`
   on the `gh-pages` branch (through the logged-in `gh`), next to the front door page `share/index.html`;
4. students open the **fixed** address `https://audiophrases.github.io/PronunciationCoach/`, which reads
   the current address, checks that it can be reached, and forwards them; when nothing is running it
   says so and offers GAPhonetics to practise with meanwhile. `stop_sharing.bat` (or Ctrl+C) marks it closed.

While sharing runs it looks after itself: the tunnel is probed end to end every minute and re-opened
with a new address (republished) if Cloudflare drops it or the network hiccups; `coach.json` is
re-stamped every four minutes, and the front door treats a session not heard from for 15 minutes as
"connection lost" instead of sending students to a dead address (a machine that went to sleep or
whose window was closed without stopping). Closing the console window also marks the front door closed.
The machine is kept from sleeping on idle while sharing; closing a laptop's lid is a separate power
setting, so leave it open or change the lid action.

The front door reads the file from the GitHub API first (fresh within a minute) because GitHub Pages
itself caches files for ten minutes. One recording is scored at a time, ~10-20 s each on this laptop:
fine for homework or a small group, not for a whole class pressing the button at once. The laptop must
stay online.

A sharing session keeps a full record for review afterwards, the same as `run_app_debug.bat`: every
assessment with its per-phone detail in `logs\app.log` and every student recording archived in
`recordings\` (wav + json, replayable with `score_recording.bat` and audited by `scripts/crop_check.py`).
The session itself goes to `logs\share.log`: which machine and version ran, the addresses, when it
opened and closed and why, what cloudflared reported, and a closing line with how many assessments
and recordings the session produced. Students' recordings stay on the sharing machine; tell them.

## Deploying

The `Dockerfile` runs anywhere Docker runs, including a Hugging Face Docker Space - but since
2026 Hugging Face only hosts Gradio/Docker Spaces on a paid PRO plan (static Spaces stay free).
The planned free route is to run inference in the browser (transformers.js, ONNX int8 of the same
model) on students' laptops and Chromebooks, published from this repository via GitHub Pages;
the Python app stays as the teacher's tuning bench and the reference the browser version is
tested against. Until then, the app runs on the teacher's own machine.

## Models

### The MFA aligner

`launchers/setup.bat` installs MFA as step 7 of 8. `launchers/setup_mfa.bat` repeats just that
step if it failed or needs reinstalling; it is safe to re-run and takes about 6 seconds when
everything is already in place. The coach uses MFA 3.4.2, the English acoustic model v3.1.0 and
the **US** English dictionary v3.1.0, aligning whole utterances without speaker adaptation. The
US dictionary is used for both the American and British accent settings: it decides where word
boundaries fall, not how a sound is judged, and the scorer keeps its own accent-specific phones.

MFA lives in its own `.cache/mfa-env` environment, separate from the app's Python packages.
`scripts/mfa-win-64.lock` pins all 205 packages and the model downloads are checked against
SHA-256 hashes. The first installation downloads roughly 500 MB; `.cache` then holds about 3 GB,
most of it the package cache, which is worth keeping - re-creating the environment from it needs
no network, which is what makes a failed install recoverable. Run setup on each machine rather
than copying the environment.

If MFA cannot be installed, setup says so and continues: the coach falls back to Charsiu and
stays usable, and running `setup.bat` again finishes the step. Set `PC_MFA=0` in
`launchers/_env.bat` to use the built-in cropper instead.

To compare crops on saved recordings, run `launchers/compare_mfa.bat`, or
`uv run python scripts/mfa_compare.py "recordings/*.json"`, and open `tmp/mfa-review.html`.
It plays both versions of every crop and draws the two boundary tracks over the waveform; each
run keeps MFA's raw word/phone intervals and logs in its own `tmp/mfa-trial-*` directory, and
`--reuse tmp/mfa-review.json` rebuilds the page without realigning. Note that it compares the
archived crops against a fresh alignment, so for recordings archived since this change both sides
are MFA and the differences go to zero; it is most useful on older archives and after a model
change. It reports **disagreement**, not accuracy.

What is and is not established: `scripts/calibrate_spikes.py` measures crops against Edge TTS
word boundaries, and on 93 words of synthetic speech calibrated MFA was the most accurate of the
three sources. That is synthetic speech and the synthesiser's own metadata, not human-annotated
learner audio, of which there is still none. Trailing-boundary outliers remain on the saved
recordings - *So* and a whispered *the* both ran several hundred ms long before the end cap was
added - and the judgement that MFA crops sound better on real learner speech rests on listening,
not on a measurement.

Sources: [MFA installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html),
[single-file alignment](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html),
[US dictionary](https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20MFA%20dictionary%20v3_1_0.html).

### App models

| Role | Default | Notes |
| --- | --- | --- |
| Phoneme recogniser | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Multilingual espeak-IPA labels, so non-English phones a learner produces are visible |
| Word recogniser | faster-whisper `base.en` int8 (`PC_ASR_MODEL=small.en` on the Space) | Free-speech mode only; ctranslate2 keeps ~1.2 GB / ~2.3 GB resident for these |
| G2P | espeak-ng (`en-us` / `en-gb`) | Same phone alphabet as the recogniser |
| Word cropping | Montreal Forced Aligner 3.4.2, `english_mfa` v3.1.0 + `english_us_mfa` v3.1.0 | Installed by `setup.bat` into `.cache/mfa-env`; MIT / CC BY 4.0 |
| Word cropping (fallback) | `charsiu/en_w2v2_fc_10ms` + `charsiu/tokenizer_en_cmu` | Frame-level phonetic aligner; downloaded on first use |

## Status

Early experiment: the goal of this stage is to *look at the posteriors* and decide
whether the signal is clean enough for the feedback we want. L1-aware feedback,
intelligibility weighting and prosody come after that.
