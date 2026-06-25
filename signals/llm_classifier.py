"""
Signal 2: LLM-Based Holistic Classification
=============================================
Asks a large language model to assess whether the submitted text reads as
human or AI-generated. This signal captures semantic and stylistic coherence
holistically — things pure statistics cannot: generic phrasing, hedged opinions,
unnaturally consistent tone, topic-sentence-to-conclusion structure, suspiciously
balanced prose, and the absence of personal specificity.

Why this complements stylometric analysis:
  - Stylometric: measures *form* (sentence structure, vocabulary stats)
  - LLM: measures *meaning and voice* (semantic patterns, rhetorical choices)
  Two genuinely independent dimensions of the text.

Blind spots:
  - The model was trained on AI-generated text and may have systematic biases.
  - High-quality, deliberately crafted AI writing can fool a classifier LLM.
  - The model's confidence is expressed as a point estimate — not a calibrated
    probability; it is treated as a soft signal, not ground truth.
  - Unusual human writing styles (non-native speakers, avant-garde poets)
    may read as AI-like to the model.
  - The model cannot access any context outside the text itself.
"""

import os
import json
import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a forensic linguist specializing in distinguishing AI-generated text from human-written text.

Analyze the provided text for authorship. Look for:
- HUMAN indicators: personal specificity, emotional idiosyncrasy, unpredictable structural choices, grammatical quirks, self-contradiction, associative leaps
- AI indicators: uniform tone, topic-sentence-body-conclusion structure, hedged balanced opinions, generic phrasing, suspiciously clean transitions, absence of a distinct point of view

Return ONLY a valid JSON object — no markdown, no preamble, no trailing text:
{
  "ai_probability": <float 0.0 to 1.0>,
  "reasoning": "<1-2 sentence explanation of your primary evidence>",
  "key_indicators": ["<observed signal 1>", "<observed signal 2>", "<observed signal 3>"],
  "confidence_note": "<brief note on how confident you are and why>"
}

ai_probability scale:
  0.00–0.20: Almost certainly human-written
  0.20–0.40: Probably human-written
  0.40–0.60: Genuinely uncertain / mixed signals
  0.60–0.80: Probably AI-generated
  0.80–1.00: Almost certainly AI-generated

Be honest about uncertainty. Do not default to 0.5 — that should reflect genuine ambiguity, not indecision."""


def _extract_json(raw: str) -> dict:
    """
    Try to extract a JSON object from the model response,
    even if it has accidental markdown fencing.
    """
    cleaned = re.sub(r"```json|```", "", raw).strip()
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in model response: {raw[:200]}")


def analyze(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Run LLM-based classification via Groq.

    Returns:
        ai_probability (float): 0.0 = strongly human, 1.0 = strongly AI
        details (dict):         reasoning and key indicators for the audit log
    """
    # Import here so the module loads even without Groq installed
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed")
        return 0.5, {"error": "groq_not_installed", "fallback": True}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — LLM signal returning neutral 0.5")
        return 0.5, {"error": "missing_api_key", "fallback": True}

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text for authorship:\n\n---\n{text}\n---"},
            ],
            temperature=0.10,   # Low temperature: we want deterministic analysis, not creativity
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip() # type: ignore
        result = _extract_json(raw)

        ai_probability = float(result.get("ai_probability", 0.5))
        ai_probability = max(0.0, min(1.0, ai_probability))

        return round(ai_probability, 4), {
            "signal": "llm_authorship_classifier",
            "fallback": False,

            # 🔥 main standardized field (IMPORTANT)
            "signal_score": round(ai_probability, 4),

            # optional alias for clarity/debugging
            "ai_probability": round(ai_probability, 4),

            # interpretive layer
            "reasoning": result.get("reasoning", ""),
            "key_indicators": result.get("key_indicators", []),
            "confidence_note": result.get("confidence_note", ""),

            # metadata
            "model": "llama-3.3-70b-versatile",
        }

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM response as JSON: %s", exc)
        return 0.5, {"error": "parse_error", "detail": str(exc), "fallback": True}
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        return 0.5, {"error": "api_error", "detail": str(exc), "fallback": True}