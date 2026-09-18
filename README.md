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

Preferences live at the top of `launchers\_env.bat`: `PC_ACCENT` (General **American** by default, or `British`)
and `PC_ASR_MODEL` (Whisper size for free speech).

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

Memory: the phoneme model (~1.4 GB) plus Whisper `base.en` (~1.2 GB) need about 3 GB free.
On an 8 GB machine, let Windows manage the page file size and close browser tabs you don't need.

## Deploy for free

The `Dockerfile` targets a Hugging Face Space (CPU basic, 2 vCPU / 16 GB). Create a Docker
Space, push this repository, done. The same image runs anywhere Docker runs.

## Models

| Role | Default | Notes |
| --- | --- | --- |
| Phoneme recogniser | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Multilingual espeak-IPA labels, so non-English phones a learner produces are visible |
| Word recogniser | faster-whisper `base.en` int8 (`PC_ASR_MODEL=small.en` on the Space) | Free-speech mode only; ctranslate2 keeps ~1.2 GB / ~2.3 GB resident for these |
| G2P | espeak-ng (`en-us` / `en-gb`) | Same phone alphabet as the recogniser |

## Status

Early experiment: the goal of this stage is to *look at the posteriors* and decide
whether the signal is clean enough for the feedback we want. L1-aware feedback,
intelligibility weighting and prosody come after that.
