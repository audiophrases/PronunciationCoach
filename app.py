"""Gradio front end: record or upload, optionally type the sentence, see every phone scored."""

from __future__ import annotations

import os

import gradio as gr

from pronunciationcoach.asr import DEFAULT_ASR
from pronunciationcoach.audio import to_mono_16k
from pronunciationcoach.engine import DEFAULT_MODEL
from pronunciationcoach.g2p import ACCENTS, DEFAULT_ACCENT
from pronunciationcoach.logs import DEBUG, LOG_FILE, SAVE_RECORDINGS, log_assessment, setup_logging
from pronunciationcoach.pipeline import assess
from pronunciationcoach.scoring import GOP_GOOD, GOP_UNSURE
from pronunciationcoach.viz import posterior_heatmap

log = setup_logging()
COLORS = {"good": "#2e8b57", "unsure": "#e0a800", "off": "#c0392b"}
L1_OPTIONS = ["Spanish", "Catalan", "Other / unknown"]  # plumbed through for the next stage; unused today


def run(audio, text, accent_name, l1):
    if audio is None:
        log.warning("assess called without audio")
        raise gr.Error("Record or upload some audio first.")
    sr, samples = audio
    log.debug("request: sr=%d samples=%s text=%r accent=%s L1=%s", sr, getattr(samples, "shape", None), text, accent_name, l1)
    try:
        result = assess(samples, sr, text, ACCENTS[accent_name])
    except ValueError as exc:  # e.g. audio far too short for the sentence
        log.warning("assessment rejected: %s", exc)
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        log.exception("assessment failed")
        raise gr.Error(f"Something went wrong: {exc!r}. Details are in {LOG_FILE}") from exc
    log_assessment(log, result, l1, to_mono_16k(samples, sr))

    highlighted = []
    for word in result.words:
        for p in word.phones:
            highlighted.append((p.expected, p.category))
        highlighted.append(("  ", None))

    rows = [
        [
            word.word if i == 0 else "",
            p.expected,
            f"{p.start_s:.2f}",
            f"{p.gop:.2f}",
            f"{p.posterior:.2f}",
            p.heard,
            "  ".join(f"{ph} {pr:.2f}" for ph, pr in p.candidates),
        ]
        for word in result.words
        for i, p in enumerate(word.phones)
    ]

    reference = ("Whisper heard: " if result.transcribed else "Reference: ") + result.text
    if result.unknown_phones:
        reference += f"\n(no model label for: {' '.join(result.unknown_phones)})"
    timing = "  ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
    summary = (
        f"{reference}\n\nHeard, text-independent:  {result.heard_text}\n\n"
        f"{result.duration_s:.1f} s of audio · {timing}"
    )
    fig = posterior_heatmap(result.emissions, result.segments)
    return summary, highlighted, rows, fig


with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown(
        "# Pronunciation Coach\n"
        "Record a sentence. Type it in the box for **known-sentence** mode, or leave the box empty "
        "for **free speech** (Whisper works out the words first). Every expected phone is then scored "
        "against what the phoneme recogniser actually heard."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your recording")
            text = gr.Textbox(label="Sentence (leave empty for free speech)", lines=2)
            with gr.Row():
                accent = gr.Dropdown(list(ACCENTS), value=DEFAULT_ACCENT, label="Target accent")
                l1 = gr.Dropdown(L1_OPTIONS, value="Spanish", label="Learner's first language")
            button = gr.Button("Assess", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Textbox(label="What was said", lines=5)
            phones = gr.HighlightedText(
                label=f"Expected phones (green ≥ {GOP_GOOD}, amber ≥ {GOP_UNSURE}, red below)",
                color_map=COLORS,
                show_legend=True,
            )
    table = gr.Dataframe(
        headers=["word", "phone", "start (s)", "GOP", "posterior", "heard", "top-3 candidates"],
        label="Per-phone detail",
        wrap=True,
    )
    plot = gr.Plot(label="What the recogniser saw")
    button.click(run, [audio, text, accent, l1], [summary, phones, table, plot])

if __name__ == "__main__":
    log.info(
        "starting: phoneme model=%s asr=%s accent=%s debug=%s save_recordings=%s log=%s",
        DEFAULT_MODEL, DEFAULT_ASR, DEFAULT_ACCENT, DEBUG, SAVE_RECORDINGS, LOG_FILE,
    )
    demo.queue(default_concurrency_limit=1).launch(
        show_error=True,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("PC_OPEN_BROWSER") == "1",  # the .bat launcher sets this
    )
