"""Phoneme engines: audio in, per-frame phone log-probabilities out.

The point of this project is to look *inside* the recogniser rather than trust
its transcript, so an engine's only job is to expose its emissions (the CTC
posterior grid) plus enough metadata to interpret them. Alignment, scoring and
feedback live elsewhere and never touch the model, which keeps the engine
swappable (Charsiu, ZIPA, a fine-tuned checkpoint of our own...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from . import SAMPLE_RATE
from .espeak import ensure_espeak

# Multilingual espeak-IPA phone recogniser (wav2vec2-large, ~1.2 GB). Chosen over
# English-only models because it will happily emit the Spanish/Catalan phones a
# learner actually produced instead of snapping them to the nearest English one.
DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


@dataclass
class Emissions:
    log_probs: np.ndarray  # (T, C) log-softmax over the vocabulary
    labels: list[str]  # vocabulary index -> phone symbol
    blank_id: int  # CTC blank (HF wav2vec2 reuses the pad token)
    frame_ms: float  # duration of one frame; 20 ms for every wav2vec2 variant
    hidden: np.ndarray | None = None  # (T, D) last hidden layer, when requested

    @property
    def n_frames(self) -> int:
        return self.log_probs.shape[0]

    def frame_to_s(self, frame: int) -> float:
        return frame * self.frame_ms / 1000.0


class Wav2Vec2CTCEngine:
    """Any Hugging Face wav2vec2 checkpoint with a CTC phoneme head."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_id = model_id
        self.device = torch.device(device)
        ensure_espeak()  # the phoneme tokenizer builds a phonemizer backend on construction
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        # fp32 on purpose: about 1.4 GB resident. Eager int8 quantisation is deprecated in
        # PyTorch and needs both copies in memory while converting, which an 8 GB laptop can't spare.
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval().to(self.device)

        tok = self.processor.tokenizer
        vocab = tok.get_vocab()
        self.vocab: dict[str, int] = vocab
        self.labels = [""] * len(vocab)
        for symbol, idx in vocab.items():
            self.labels[idx] = symbol
        self.blank_id = tok.pad_token_id
        self.unk_id = tok.unk_token_id

        stride = math.prod(self.model.config.conv_stride)  # 320 samples for wav2vec2
        self.frame_ms = 1000.0 * stride / SAMPLE_RATE

    @torch.inference_mode()
    def emissions(self, audio: np.ndarray, with_hidden: bool = False) -> Emissions:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        out = self.model(inputs.input_values.to(self.device), output_hidden_states=with_hidden)
        log_probs = torch.log_softmax(out.logits[0].float(), dim=-1).cpu().numpy()
        hidden = out.hidden_states[-1][0].float().cpu().numpy() if with_hidden else None
        return Emissions(log_probs, self.labels, self.blank_id, self.frame_ms, hidden)

    def phone_id(self, phone: str) -> int | None:
        """Vocabulary id for a phone symbol, trying a couple of harmless spellings first."""
        for candidate in (phone, phone.replace("ː", ""), phone.replace("ɹ", "r")):
            if candidate in self.vocab:
                return self.vocab[candidate]
        return None


@lru_cache(maxsize=2)
def get_engine(model_id: str = DEFAULT_MODEL) -> Wav2Vec2CTCEngine:
    """Process-wide cache so the UI loads the weights once."""
    return Wav2Vec2CTCEngine(model_id)
