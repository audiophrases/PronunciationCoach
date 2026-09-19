"""Word boundaries from a frame-level phonetic aligner (Charsiu).

The scoring model emits one brief spike per phone, which is fine for scoring
and poor for cropping: a spike says "this sound happened around here", not
where it started or ended. Charsiu's English frame classifier
(charsiu/en_w2v2_fc_10ms, Zhu et al. 2022) labels every 10 ms frame with a
phone or silence, so boundaries fall out of a forced alignment directly.

Alignment is a plain Viterbi over the frame log-posteriors: the expected
phones of each word in order, with an optional silence state between words
and at both ends. Posteriors are smoothed with a small uniform floor so a
mispronounced phone costs a bounded amount instead of dragging its whole
word out of shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import SAMPLE_RATE
from .boundaries import Span

MODEL_ID = "charsiu/en_w2v2_fc_10ms"
TOKENIZER_ID = "charsiu/tokenizer_en_cmu"
FRAME_S = 0.01
FLOOR = 0.02  # share of probability mass spread uniformly over all labels before taking logs
# Charsiu's frame labels sit a constant bit late against ground truth (measured on Edge TTS
# word boundaries with scripts/calibrate_spikes.py); subtracted from every boundary.
OFFSET_S = 0.04
MIN_FRAMES = 3  # a phone occupies at least this many 10 ms frames

# espeak-ng IPA (en-us and en-gb) -> CMU/ARPABET labels Charsiu was trained on.
IPA_TO_ARPABET: dict[str, list[str]] = {
    "p": ["P"], "b": ["B"], "t": ["T"], "d": ["D"], "k": ["K"], "ɡ": ["G"], "tʃ": ["CH"], "dʒ": ["JH"],
    "f": ["F"], "v": ["V"], "θ": ["TH"], "ð": ["DH"], "s": ["S"], "z": ["Z"], "ʃ": ["SH"], "ʒ": ["ZH"],
    "h": ["HH"], "m": ["M"], "n": ["N"], "ŋ": ["NG"], "l": ["L"], "ɫ": ["L"], "əl": ["AH", "L"],
    "n̩": ["AH", "N"], "ɹ": ["R"], "r": ["R"], "j": ["Y"], "w": ["W"], "ɾ": ["T"],
    "ɪ": ["IH"], "iː": ["IY"], "i": ["IY"], "ɛ": ["EH"], "e": ["EH"], "æ": ["AE"], "a": ["AE"], "ʌ": ["AH"],
    "ə": ["AH"], "ɐ": ["AH"], "ᵻ": ["IH"], "ʊ": ["UH"], "uː": ["UW"], "u": ["UW"], "ɑː": ["AA"], "ɑ": ["AA"],
    "ɔː": ["AO"], "ɔ": ["AO"], "ɒ": ["AA"], "o": ["OW"], "ɜː": ["ER"], "ɝ": ["ER"], "ɚ": ["ER"],
    "eɪ": ["EY"], "aɪ": ["AY"], "ɔɪ": ["OY"], "oʊ": ["OW"], "əʊ": ["OW"], "aʊ": ["AW"],
    "ɪɹ": ["IH", "R"], "ɛɹ": ["EH", "R"], "ʊɹ": ["UH", "R"], "ɑːɹ": ["AA", "R"], "ɔːɹ": ["AO", "R"],
    "aɪɚ": ["AY", "ER"], "aʊɚ": ["AW", "ER"], "ɪə": ["IH", "AH"], "eə": ["EH", "AH"], "ʊə": ["UH", "AH"],
    "aɪə": ["AY", "AH"], "aʊə": ["AW", "AH"],
}


def to_arpabet(phones: list[str]) -> list[str]:
    out: list[str] = []
    for p in phones:
        out.extend(IPA_TO_ARPABET.get(p, IPA_TO_ARPABET.get(p.replace("ː", ""), [])))
    return out


class CharsiuSegmenter:
    def __init__(self, model_id: str = MODEL_ID, tokenizer_id: str = TOKENIZER_ID):
        import torch
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.torch = torch
        tok = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_id)
        self.vocab: dict[str, int] = tok.get_vocab()
        self.sil = self.vocab["[SIL]"]
        # Charsiu's frame classifier has exactly a Wav2Vec2ForCTC's layers; only the loss differs.
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        self.extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=True, return_attention_mask=False
        )

    def frame_log_probs(self, audio: np.ndarray) -> np.ndarray:
        """(T, C) log-posteriors, one row per 10 ms, floored so no label is ever impossible."""
        with self.torch.inference_mode():
            inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
            probs = self.torch.softmax(self.model(inputs).logits[0].float(), dim=-1).numpy()
        probs = (1 - FLOOR) * probs + FLOOR / probs.shape[1]
        return np.log(probs)

    def align(
        self,
        log_probs: np.ndarray,
        words: list[list[str]],
        windows: list[tuple[float, float]] | None = None,
        extras: list[tuple[float, float]] | None = None,
    ) -> list[Span]:
        """Word spans (seconds) for `words` given as ARPABET phone lists, in order.

        `windows` (seconds, one per word) restrict where each word may lie - the
        scoring aligner's spikes, widened - and `extras` mark stretches of speech
        that belong to no word (repeats, hesitations). Between words a free
        "anything" state absorbs whatever is outside every window, so a repeated
        word cannot be swallowed by its neighbour.
        """
        # States: [SIL, ANY], then for each word its phones followed by [SIL, ANY]. ANY is a
        # second optional filler that costs nothing outside the word windows and is
        # forbidden inside them; SIL keeps its real posterior everywhere.
        ANY = -1
        states: list[int] = [self.sil, ANY]
        optional: list[bool] = [True, True]
        word_of: list[int | None] = [None, None]
        for w, phones in enumerate(words):
            ids = [self.vocab[p] for p in phones if p in self.vocab] or [self.vocab["[UNK]"]]
            for tok in ids:  # MIN_FRAMES copies in a row: each must take a frame, so a phone lasts >= 30 ms
                states += [tok] * MIN_FRAMES
                optional += [False] * MIN_FRAMES
                word_of += [w] * MIN_FRAMES
            states += [self.sil, ANY]
            optional += [True, True]
            word_of += [None, None]

        T, S = log_probs.shape[0], len(states)
        NEG = -1e9
        t_axis = (np.arange(T) + 0.5) * FRAME_S - OFFSET_S
        in_window = np.zeros(T, dtype=bool)
        if windows:
            for lo, hi in windows:
                in_window |= (t_axis >= lo) & (t_axis <= hi)
        if extras:
            for lo, hi in extras:
                in_window &= ~((t_axis >= lo) & (t_axis <= hi))
        emit = np.empty((T, S), dtype=np.float64)
        for j, tok in enumerate(states):
            if tok == ANY:  # free outside the word windows, impossible inside them or without windows
                emit[:, j] = np.where(in_window, NEG, 0.0) if windows else NEG
            else:
                emit[:, j] = log_probs[:, tok]
        if windows:
            for j, w in enumerate(word_of):
                if w is not None:
                    lo, hi = windows[w]
                    emit[(t_axis < lo) | (t_axis > hi), j] = NEG

        score = np.full((T, S), NEG, dtype=np.float64)
        back = np.zeros((T, S), dtype=np.int32)
        # A transition may hop over a run of optional states: hops of 1, 2 or 3.
        opt = np.array(optional)
        idx = np.arange(S)
        hop_ok = {
            1: np.ones(S, dtype=bool),
            2: (idx >= 2) & opt[np.maximum(idx - 1, 0)],
            3: (idx >= 3) & opt[np.maximum(idx - 1, 0)] & opt[np.maximum(idx - 2, 0)],
        }
        score[0, :3] = emit[0, :3]  # start in SIL, ANY or the first phone
        for t in range(1, T):
            prev = score[t - 1]
            best = prev.copy()
            src = idx.copy()
            for h in (1, 2, 3):
                cand = np.concatenate([np.full(h, NEG), prev[:-h]])
                cand = np.where(hop_ok[h], cand, NEG)
                take = cand > best
                best = np.where(take, cand, best)
                src = np.where(take, idx - h, src)
            score[t] = best + emit[t]
            back[t] = src
        s = int(S - 3 + np.argmax(score[T - 1, S - 3 :]))  # end in the final phone, SIL or ANY
        path = np.empty(T, dtype=np.int32)
        for t in range(T - 1, -1, -1):
            path[t] = s
            s = back[t, s]

        owner = np.array([-1 if word_of[p] is None else word_of[p] for p in path])
        spans: list[Span] = []
        for w in range(len(words)):
            frames = np.flatnonzero(owner == w)
            if frames.size == 0:  # every phone state is visited, so only if the DP had no path
                spans.append(Span(0.0, 0.0))
                continue
            start = max(0.0, frames[0] * FRAME_S - OFFSET_S)
            end = max(start + 0.03, (frames[-1] + 1) * FRAME_S - OFFSET_S)
            spans.append(Span(start, end))
        return spans


@lru_cache(maxsize=1)
def get_segmenter() -> CharsiuSegmenter:
    return CharsiuSegmenter()
