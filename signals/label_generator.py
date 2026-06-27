"""
Label Generator
===============
Produces transparency labels matching the three-variant design in planning.md.
Body copy is frozen to the exact strings approved there; do not paraphrase inline.

Confidence bands for decisive attributions:
  very_high  >= 0.80   "N% confident (very high)"
  high        0.65–0.80 "N% confident (high)"
  moderate    0.45–0.65 "N% confident (moderate) — interpret alongside other context"
  low         0.30–0.45 "N% confident (low) — treat this result with caution"

Body copy uses two tiers keyed to the planning.md split at 0.65:
  strong   covers very_high + high   — definitive-sounding language
  moderate covers moderate + low     — hedged, uncertainty-forward language

Uncertain attributions (confidence < 0.30) always use a single body and the
"Insufficient confidence to label (N%)" confidence_string format.
"""

# ── Thresholds ────────────────────────────────────────────────────────────────

_VERY_HIGH  = 0.80
_HIGH       = 0.65   # planning.md "strong" boundary
_MODERATE   = 0.45
# LOW: [0.30, 0.45) — lower bound is set by combine_signals CONFIDENCE_THRESHOLD

# ── Approved copy (do not edit without updating planning.md) ──────────────────

_TITLES: dict[str, str] = {
    "ai_generated":  "AI-Generated Content",
    "human_written": "Human-Written Content",
    "uncertain":     "Origin Unclear",
}

_BODIES: dict[str, dict[str, str] | str] = {
    "ai_generated": {
        "strong": (
            "Our analysis strongly suggests this content was written by an AI system. "
            "Both the statistical structure and semantic patterns of the text align with "
            "characteristics typical of AI-generated writing — consistent tone, predictable "
            "sentence rhythm, and low stylistic idiosyncrasy. This reflects the likely "
            "origin of the content, not its quality or value."
        ),
        "moderate": (
            "Our analysis leans toward AI authorship, but the two signals we use gave only "
            "partial agreement. Some characteristics match AI-generated writing, while "
            "others are ambiguous. This result carries meaningful uncertainty — treat it "
            "as a soft indicator rather than a definitive finding."
        ),
    },
    "human_written": {
        "strong": (
            "Our analysis suggests this content was written by a person. The writing shows "
            "the kind of natural variation, stylistic idiosyncrasy, and structural "
            "unpredictability that characterizes human authorship. Multiple independent "
            "signals agree on this assessment."
        ),
        "moderate": (
            "Our analysis leans toward human authorship, but not all signals were in full "
            "agreement. Some features are consistent with human writing, while others are "
            "ambiguous. This assessment is directionally human, but should not be treated "
            "as a firm determination."
        ),
    },
    "uncertain": (
        "Our system detected mixed signals and cannot determine with confidence whether "
        "this content was written by a human or an AI. This may reflect human-AI "
        "collaborative writing, a distinctive personal style that sits outside our "
        "model's experience, or genuinely ambiguous text. No definitive label is applied "
        "— this content is presented without attribution."
    ),
}

_APPEAL_CTAS: dict[str, str] = {
    "ai_generated": (
        "Believe this is incorrect? If you're the creator, you can submit an appeal "
        "with context our automated system cannot see — a human reviewer will examine it."
    ),
    "human_written": (
        "Disagree with this assessment? You can submit an appeal to request a human "
        "reviewer take a closer look."
    ),
    "uncertain": (
        "If you're the creator, sharing context through an appeal can help us review "
        "this more accurately and improve future classifications."
    ),
}

# ── Confidence-string caveats by band ─────────────────────────────────────────

_CAVEATS: dict[str, str] = {
    "very_high": "",
    "high":      "",
    "moderate":  " — interpret alongside other context",
    "low":       " — treat this result with caution",
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_label(attribution: str, confidence: float) -> dict:
    """
    Build a transparency label dict.

    Args:
        attribution: "ai_generated" | "human_written" | "uncertain"
        confidence:  0–1 from combine_signals()

    Returns:
        {variant, title, body, confidence_string, confidence_band, appeal_cta}

    ``confidence_band`` is the internal bucket name for logging; not shown to readers.
    ``confidence_string`` is the reader-facing confidence line.
    """
    if attribution not in _TITLES:
        attribution = "uncertain"

    band = _resolve_band(attribution, confidence)

    return {
        "variant":           attribution,
        "title":             _TITLES[attribution],
        "body":              _resolve_body(attribution, band),
        "confidence_string": _build_confidence_string(attribution, confidence, band),
        "confidence_band":   band,
        "appeal_cta":        _APPEAL_CTAS[attribution],
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_band(attribution: str, confidence: float) -> str:
    if attribution == "uncertain":
        return "uncertain"
    if confidence >= _VERY_HIGH:
        return "very_high"
    if confidence >= _HIGH:
        return "high"
    if confidence >= _MODERATE:
        return "moderate"
    return "low"


def _resolve_body(attribution: str, band: str) -> str:
    if attribution == "uncertain":
        return _BODIES["uncertain"]  # type: ignore[return-value]
    body_tier = "strong" if band in ("very_high", "high") else "moderate"
    return _BODIES[attribution][body_tier]  # type: ignore[index]


def _build_confidence_string(attribution: str, confidence: float, band: str) -> str:
    pct = round(confidence * 100)
    if attribution == "uncertain":
        return f"Insufficient confidence to label ({pct}%)"
    label = band.replace("_", " ")   # "very high" | "high" | "moderate" | "low"
    return f"{pct}% confident ({label}){_CAVEATS[band]}"
