"""Stylometric heuristics for AI-authorship scoring.

The score is directional:
0.0 means the text looks structurally human-like.
1.0 means the text looks structurally AI-like.
"""

from __future__ import annotations

import math
import re
import string


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
_PUNCTUATION = set(string.punctuation) | {"—", "–", "…", "“", "”", "‘", "’"}


def analyze(text: str) -> tuple[float, dict]:
    """Return a stylometric AI-likeness score and feature details."""
    words = _words(text)
    sentences = _sentences(text)
    sentence_lengths = [_word_count(sentence) for sentence in sentences]
    sentence_lengths = [length for length in sentence_lengths if length > 0]

    word_count = len(words)
    sentence_count = len(sentence_lengths)

    if sentence_count < 3 or word_count < 30:
        return 0.5, {
            "note": "insufficient text",
            "fallback": True,
            "word_count": word_count,
            "sentence_count": sentence_count,
        }

    cv = _coefficient_of_variation(sentence_lengths)
    ttr = len(set(words)) / word_count
    punctuation_count, punctuation_types = _punctuation_profile(text)
    punctuation_density = punctuation_count / word_count

    sentence_cv_score = _score_sentence_cv(cv)
    ttr_score = _score_type_token_ratio(ttr)
    punctuation_score = _score_punctuation(punctuation_density, len(punctuation_types))

    sty_score = (
        0.45 * sentence_cv_score
        + 0.30 * ttr_score
        + 0.25 * punctuation_score
    )
    sty_score = _clamp(sty_score)

    details = {
        "signal": "stylometric_heuristics",
        "fallback": False,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "features": {
            "sentence_length_cv": {
                "cv": round(cv, 4),
                "score": round(sentence_cv_score, 4),
                "sentence_lengths": sentence_lengths,
            },
            "type_token_ratio": {
                "ratio": round(ttr, 4),
                "score": round(ttr_score, 4),
                "unique_words": len(set(words)),
            },
            "punctuation": {
                "density": round(punctuation_density, 4),
                "variety": len(punctuation_types),
                "types": sorted(punctuation_types),
                "score": round(punctuation_score, 4),
            },
        },
        "weights": {
            "sentence_length_cv": 0.45,
            "type_token_ratio": 0.30,
            "punctuation": 0.25,
        },
    }

    return round(sty_score, 4), details


def _words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _word_count(text: str) -> int:
    return len(_words(text))


def _coefficient_of_variation(values: list[int]) -> float:
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0

    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _punctuation_profile(text: str) -> tuple[int, set[str]]:
    punctuation = [char for char in text if char in _PUNCTUATION]
    return len(punctuation), set(punctuation)


def _score_sentence_cv(cv: float) -> float:
    if cv <= 0.35:
        return 1.0
    if cv >= 0.90:
        return 0.0
    return _clamp((0.90 - cv) / (0.90 - 0.35))


def _score_type_token_ratio(ttr: float) -> float:
    if 0.58 <= ttr <= 0.80:
        return 1.0
    if ttr < 0.58:
        return _clamp((ttr - 0.35) / (0.58 - 0.35))
    return _clamp((0.90 - ttr) / (0.90 - 0.80))


def _score_punctuation(density: float, variety: int) -> float:
    density_score = _band_score(density, lower=0.12, upper=0.28, min_value=0.02, max_value=0.45)
    variety_score = 1.0 if 3 <= variety <= 5 else max(0.0, 1.0 - (abs(variety - 4) / 6))
    return _clamp((density_score + variety_score) / 2)


def _band_score(value: float, lower: float, upper: float, min_value: float, max_value: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return _clamp((value - min_value) / (lower - min_value))
    return _clamp((max_value - value) / (max_value - upper))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
