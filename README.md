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
  tapping within the same playback pair again plays the model; again, you, and so on.
  The heading shows the pair and emphasizes the selected word; colors and tips stay per word. The whole sentence
  has its own ▶ You / ▶ Model buttons. One speed slider (0.5-1.5 in steps of 0.05) applies to everything
  played, so you and the model are always compared at the same pace: the model voice is synthesised at
  that rate, your own audio is time-stretched with the pitch kept (ffmpeg atempo). The model voice can
  be male or female (Guy / Jenny for American, Ryan / Sonia for British; default `PC_VOICE=Male`).
  It is Microsoft Edge's neural TTS via `edge-tts` (cached in `tts_cache/`; falls back to espeak-ng
  offline). The input row is just the recorder, the sentence box and the check button; the target
  accent and model voice pickers sit with everything technical in a collapsed
  "Technical details (for teachers)" section: timeline, IPA, per-phone table, posterior heatmap.
* **Word crops** – a second, small model does the cropping: Charsiu's frame-level phonetic
  aligner (`charsiu/en_w2v2_fc_10ms`, ~380 MB) labels every 10 ms with a phone or silence, and a
  Viterbi (`segmenter.py`) fits the expected words to it, anchored to the scoring model's spikes so
  repeats and hesitations are excluded. Each word's *start* is then settled with the audio as the
  referee (`settle_starts` in `pipeline.py`), because the two estimates fail in opposite ways:
  Charsiu runs late on quiet onsets (a stop's closure, a fricative, *h*, a nasal) and hands them to
  the previous word, while the spike-based onset can reach back into the previous word's vowel -
  and the previous word's own last spike may fire after our first sound has begun, so it is no
  floor either. Three cases: **onset** (silence before the word: start where sound resumes, e.g.
  at a k's burst), **dip** (the word starts with a quiet consonant after a vowel and the energy
  shows the valley Charsiu skipped, its rise at Charsiu's boundary: start at the valley), **join**
  (anything else, including a consonant that belongs to the previous word: Charsiu's boundary,
  never later than the spike allows). Each decision is logged (`crops (charsiu): can 0.59-0.88
  onset | ...`) and archived with the recording. `scripts/crop_check.py` audits archived recordings
  without loading any model: per word, ms of the first sound cut, of the previous/next word
  included, of the last sound cut, and edge silence. Using Edge TTS word-boundary
  timing metadata, `scripts/calibrate_spikes.py` measures synthetic crop differences; these are not
  human-annotated learner-speech accuracy measurements. If Charsiu is unavailable
  the spike-based estimate in `boundaries.py` is used instead.
  Playback clips take the recording at its original rate and add up to 60/50 ms of room on either
  side **only through quiet audio** (silence, a closure, breath - never a neighbour's vowel), with
  10 ms fades and a level boost, then follow the speed slider.
  The start search cannot reach before the preceding word's first kept spike; adjusting a shared
  boundary cannot reverse that word's crop. A final logged safety check keeps live spans ordered
  and inside the recording, without inventing audio for missing words.
* **Short playback pairs** – function words (articles, prepositions, pronouns, conjunctions and
  helping verbs) are candidates for context: **[can you] [hear me]**, **[the store]**, **[to school]**.
  Chunks contain at most **two words**, never a chain of pairs. Content words generally stay alone.
  Neighboring pair choices favor grammatical attachment; main-verb *have/do* are not automatically
  treated as unstressed. Duration and relative loudness provide a conservative emphasis heuristic,
  not a linguistic stress detector. Crops under 120 ms can also receive context. Punctuation,
  pauses/gaps of at least 100 ms and intervening extra speech block a pair; a pair longer than 1.5 s
  is not formed. A barrier or cap may leave a short word alone. Missing words add no parked audio,
  and an entirely missing chunk offers only model playback. Both voices play the same word/pair
  text, using the existing speed control. The technical table and recording archive include
  membership and reasons. `scripts/crop_check.py --rechunk` audits archived recordings without
  model inference (old archives assume 20 ms scoring frames). Set `PC_CHUNKS=0` to compare with
  individual-word playback; no new model or dependency is needed.
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

Memory: the phoneme model (~1.4 GB), the cropping model (~0.4 GB) and Whisper `base.en` (~1.2 GB,
the listener) together need about 3.5 GB free; if Whisper cannot load, verdicts fall back to
severity only.
On an 8 GB machine, let Windows manage the page file size and close browser tabs you don't need.

## Running it on another machine

Clone the repository, double-click `launchers\setup.bat`, and the machine is ready - the same
launchers work there. `setup.bat` installs uv, espeak-ng, ffmpeg, cloudflared and the GitHub CLI
if missing, the Python packages, and downloads all models (~2 GB) by scoring a test sentence.
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

| Role | Default | Notes |
| --- | --- | --- |
| Phoneme recogniser | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Multilingual espeak-IPA labels, so non-English phones a learner produces are visible |
| Word recogniser | faster-whisper `base.en` int8 (`PC_ASR_MODEL=small.en` on the Space) | Free-speech mode only; ctranslate2 keeps ~1.2 GB / ~2.3 GB resident for these |
| G2P | espeak-ng (`en-us` / `en-gb`) | Same phone alphabet as the recogniser |
| Word cropping | `charsiu/en_w2v2_fc_10ms` + `charsiu/tokenizer_en_cmu` | Frame-level phonetic aligner; downloaded on first use |

## Status

Early experiment: the goal of this stage is to *look at the posteriors* and decide
whether the signal is clean enough for the feedback we want. L1-aware feedback,
intelligibility weighting and prosody come after that.
