"""
Confidence Scoring
==================
Combines the two signal scores into:
  1. A final ai_score (weighted blend of both signals)
  2. A confidence score reflecting genuine uncertainty

Confidence is NOT simply abs(ai_score - 0.5).
It captures two independent sources of uncertainty:
  - Signal agreement: do both signals point the same direction?
  - Signal strength: how far from the decision boundary is the combined score?

A 0.51 confidence → "approaching our uncertainty threshold — result unreliable"
A 0.95 confidence → "both signals strongly agree on a clear result"

These produce genuinely different transparency label text, not just a different percentage.
"""

from enum import Enum
import math

class AttributionResult(str, Enum):
    AI_GENERATED = "ai_generated"
    HUMAN_WRITTEN = "human_written"
    UNCERTAIN = "uncertain"


def combine_signals(
    stylometric_score: float,
    llm_score: float,
    stylometric_fallback: bool = False,
    llm_fallback: bool = False,
) -> tuple[float, float, AttributionResult]:
    """
    Merge two signal scores into a final attribution.

    Args:
        stylometric_score: 0–1 AI-likeness from statistical analysis
        llm_score:         0–1 AI-likeness from LLM holistic analysis
        stylometric_fallback: True if stylometric returned a neutral 0.5 due to error/insufficient text
        llm_fallback:         True if LLM returned a neutral 0.5 due to API error

    Returns:
        ai_score:    combined AI probability (0–1)
        confidence:  certainty in that probability (0–1)
        attribution: AttributionResult enum
    """

    # --- Weight assignment ---
    # LLM captures richer semantic signals; gets higher weight when both signals
    # are reliable. If either fell back to neutral, re-weight accordingly.
    if llm_fallback and stylometric_fallback:
        # Both failed → return maximally uncertain
        return 0.50, 0.00, AttributionResult.UNCERTAIN

    if llm_fallback:
        # Only stylometric available
        llm_weight, sty_weight = 0.00, 1.00
    elif stylometric_fallback:
        # Only LLM available
        llm_weight, sty_weight = 1.00, 0.00
    else:
        # Both reliable
        llm_weight, sty_weight = 0.60, 0.40

    ai_score = (llm_weight * llm_score) + (sty_weight * stylometric_score)

    # --- Confidence: product of signal agreement and distance from 0.5 ---
    #
    # Signal agreement: 1.0 = both scores identical, 0.0 = maximally opposite
    # Penalized if either signal fell back (one-signal confidence ceiling = 0.70)
    if llm_fallback or stylometric_fallback:
        agreement = 0.70   # Single-signal: cap agreement
    else:
        agreement = 1.0 - abs(llm_score - stylometric_score)

    # Distance from decision boundary: how far is the blended score from 0.5?
    distance = abs(ai_score - 0.5) * 2   # 0 = maximally uncertain, 1 = maximally clear

    # Confidence = geometric mean of agreement and distance
    # Using sqrt keeps confidence honest: even high agreement at ai_score=0.51
    # yields low confidence because the distance is near zero.
    confidence = math.sqrt(agreement * distance)
    confidence = round(max(0.0, min(1.0, confidence)), 4)
    ai_score = round(ai_score, 4)

    # --- Attribution thresholds ---
    # Require BOTH a directional score AND enough confidence before committing.
    # Low-confidence results always land in UNCERTAIN regardless of score direction.
    CONFIDENCE_THRESHOLD = 0.30
    AI_SCORE_HIGH = 0.62
    AI_SCORE_LOW = 0.38

    if ai_score >= AI_SCORE_HIGH and confidence >= CONFIDENCE_THRESHOLD:
        attribution = AttributionResult.AI_GENERATED
    elif ai_score <= AI_SCORE_LOW and confidence >= CONFIDENCE_THRESHOLD:
        attribution = AttributionResult.HUMAN_WRITTEN
    else:
        attribution = AttributionResult.UNCERTAIN

    return ai_score, confidence, attribution