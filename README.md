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

* **Learner view** – a score ring, the sentence as tappable word chips (clear / almost /
  work on this), one player that speaks whatever you tap: the whole sentence (you, the model,
  the model slowly) or one word, with a line of plain-language advice per sound that needs work.
  The model voice is Microsoft Edge's neural TTS via `edge-tts` (cached in `tts_cache/`; falls
  back to espeak-ng offline). Everything technical sits in a collapsed
  "Technical details (for teachers)" section: timeline, IPA, per-phone table, posterior heatmap.
* **Word crops** – a second, small model does the cropping: Charsiu's frame-level phonetic
  aligner (`charsiu/en_w2v2_fc_10ms`, ~380 MB) labels every 10 ms with a phone or silence, and a
  Viterbi (`segmenter.py`) fits the expected words to it, anchored to the scoring model's spikes so
  repeats and hesitations are excluded. Against Edge TTS word boundaries the crops are within
  ~17 ms (start) / ~23 ms (end) of the truth; `scripts/calibrate_spikes.py` reproduces the numbers.
  If that model is unavailable the spike-based estimate in `boundaries.py` is used instead.
* **Known-sentence mode** – you type the sentence; the word-recognition step is skipped.
* **Free-speech mode** – leave the text empty; Whisper finds the words the learner *meant*,
  and the phoneme track shows what they *said*.
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
| `open_logs.bat` | Opens `logs\app.log` in Notepad and the `recordings\` folder |

The app always writes one line per assessment to `logs\app.log` (mode, text, what was heard, flagged
phones, timings) plus any errors. Archived recordings can be replayed by dragging the `.wav` onto
`score_recording.bat`.

Preferences live at the top of `launchers\_env.bat`: `PC_ACCENT` (General **American** by default, or `British`),
`PC_L1` (learner's first language shown by default: **Catalan**) and `PC_ASR_MODEL` (Whisper size for free speech).

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

Memory: the phoneme model (~1.4 GB) plus the cropping model (~0.4 GB) plus Whisper `base.en`
(~1.2 GB, free-speech mode only) need up to 3.5 GB free.
On an 8 GB machine, let Windows manage the page file size and close browser tabs you don't need.

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
