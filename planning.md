# planning.md — AI Attribution API

---

## 1. Detection Signals

### Signal 1 — Stylometric Heuristics
**File:** `signals/stylometric.py`
**What it measures:** Statistical properties of the text's *form* — sentence rhythm, vocabulary diversity, and punctuation habits — without any semantic understanding of the content.

Three sub-features, each producing a float 0–1, combined into a single signal score:

| Sub-feature | What it captures | Why it differs (human vs AI) |
|---|---|---|
| **Sentence Length CV** | Coefficient of variation of per-sentence word counts | AI training optimizes for readability → uniform sentence rhythm (low CV ≈ 0.15–0.35). Human creative writing is burstier (CV ≈ 0.40–0.90+). |
| **Type-Token Ratio** | `unique_words / total_words` | AI tends to land in a "moderate diversity" band (0.58–0.80) — not repetitive, not eccentric. Human writing hits extremes in both directions. |
| **Punctuation Density & Variety** | Punctuation chars per word + count of distinct punct types used | AI writes structurally correct, predictable punctuation (3–5 types, 0.12–0.28 per word). Human creative writing uses em-dashes, ellipses, and fragmented choices expressively. |

**Output format:** Single float `sty_score ∈ [0.0, 1.0]`
- `0.0` = statistical profile strongly resembles human writing
- `1.0` = statistical profile strongly resembles AI writing

**Combination weights in this signal:**
```
sty_score = 0.45 × sentence_cv_score
          + 0.30 × ttr_score
          + 0.25 × punctuation_score
```

**What this signal cannot do:** It measures form, not meaning. It cannot detect hedged opinions, generic phrasing, suspiciously balanced arguments, or the absence of a personal point of view — all reliable AI semantic tells. That is exactly what Signal 2 is for.

**Blind spots (named):**
- Texts under 3 sentences / 30 words produce unstable CV readings → signal returns neutral 0.5
- Academic or formal human writing (consistent sentence rhythm, correct punctuation) scores AI-like on all three sub-features
- A user who knows the scoring rubric can vary AI sentence lengths deliberately

---

### Signal 2 — LLM Holistic Classifier
**File:** `signals/llm_classifier.py`
**Model:** `llama-3.3-70b-versatile` via Groq
**What it measures:** A language model's holistic judgment of whether the *meaning and voice* of the text match AI or human authorship patterns. The model looks at things pure statistics cannot: generic phrasing, suspiciously clean transitions, topic-sentence-body-conclusion structure, hedged balanced opinions, and the absence of a personal point of view.

**Prompt approach:** Low temperature (0.10) to minimize randomness. System prompt instructs the model to return **only** a JSON object — no markdown, no preamble. The JSON schema:
```json
{
  "ai_probability": 0.0–1.0,
  "reasoning": "1-2 sentence explanation of primary evidence",
  "key_indicators": ["indicator 1", "indicator 2", "indicator 3"],
  "confidence_note": "brief note on model's own certainty"
}
```

**Output format:** Single float `llm_score ∈ [0.0, 1.0]` extracted from `ai_probability`
- `0.0` = model reads text as clearly human-written
- `1.0` = model reads text as clearly AI-generated

**What this signal cannot do:** It cannot access authorship history, platform metadata, or any context outside the raw text. It also has a systematic bias from being trained on text that includes AI-generated content — it can be fooled by well-crafted AI writing with a deliberately injected personal voice.

**Blind spots (named):**
- `ai_probability` from the LLM is a point estimate, not a calibrated probability — "0.7" here and "0.7" from stylometrics are not the same kind of number
- Unusual human styles (non-native English, avant-garde poetry, deliberate formal register) may score AI-like
- Very short texts give the model insufficient signal; it may return values near 0.5 with low discriminative value

---

### How the Two Signals Combine

**File:** `core/confidence.py`

The signals are genuinely independent: one is structural/statistical, one is semantic/holistic. That independence is what makes combining them informative.

**Step 1 — Weighted blend into `ai_score`:**
```
ai_score = 0.60 × llm_score + 0.40 × sty_score
```
LLM gets higher weight because it captures richer contextual information. If either signal fell back to a neutral 0.5 due to error, the weights shift to use only the reliable signal (or return 0.5/uncertain if both fail).

**Step 2 — Compute `confidence` from two independent factors:**
```
signal_agreement = 1.0 − |llm_score − sty_score|
distance         = |ai_score − 0.5| × 2

confidence = √(signal_agreement × distance)
```

Using geometric mean rather than arithmetic mean ensures that **both** high agreement AND a clear signal are required for high confidence. A score of 0.63 with signals disagreeing by 0.40 produces low confidence even though the direction is technically AI. That is intentional.

**Why geometric mean:** If agreement = 0.3 and distance = 0.8, arithmetic mean gives 0.55 (misleadingly confident). Geometric mean gives 0.49 — which correctly signals that disagreement is the dominant factor.

---

## 2. Uncertainty Representation

### What a confidence score of 0.6 means

`confidence = 0.60` means: both signals are in rough agreement AND the combined score is meaningfully away from the midpoint — but neither factor is strong. Concretely: `agreement ≈ 0.80`, `distance ≈ 0.45` → `√(0.80 × 0.45) = 0.60`. This is a content where the LLM said "probably AI" (0.72) and stylometrics said "somewhat AI" (0.60). The signals agree directionally but not strongly, and the blended score is not close to the decision boundary.

The label this produces: `high_confidence_ai` variant with **moderate-confidence body text**: *"Our analysis leans toward AI authorship, but the two signals we use gave only partial agreement..."* — not the strong version that says "multiple independent signals align."

### Mapping raw signal outputs to the confidence scale

| Scenario | `llm_score` | `sty_score` | `ai_score` | `agreement` | `distance` | `confidence` | Result |
|---|---|---|---|---|---|---|---|
| Both signals strongly agree on AI | 0.90 | 0.84 | 0.876 | 0.94 | 0.75 | **0.84** | `ai_generated`, strong label |
| Both signals agree on human | 0.12 | 0.18 | 0.144 | 0.94 | 0.71 | **0.82** | `human_written`, strong label |
| Signals agree directionally, moderately | 0.72 | 0.60 | 0.672 | 0.88 | 0.34 | **0.55** | `ai_generated`, moderate label |
| Signals disagree (one says AI, one says human) | 0.78 | 0.22 | 0.557 | 0.44 | 0.11 | **0.22** | `uncertain` |
| Both near midpoint | 0.52 | 0.48 | 0.504 | 0.96 | 0.01 | **0.10** | `uncertain` |
| Only LLM available (sty fallback) | 0.85 | 0.50* | 0.85 | 0.70* | 0.70 | **0.70** | `ai_generated`, capped confidence |

### Attribution thresholds

```
if ai_score ≥ 0.62 AND confidence ≥ 0.30 → AI_GENERATED
if ai_score ≤ 0.38 AND confidence ≥ 0.30 → HUMAN_WRITTEN
otherwise                                  → UNCERTAIN
```

**Why these thresholds:**
- The 0.62/0.38 boundary (not 0.5) reflects that a blended score at 0.55 is still genuinely ambiguous — there's a meaningful uncertainty band around 0.5 before we commit to a direction.
- The `confidence ≥ 0.30` floor means that even a directional score (say 0.70 AI) produces `uncertain` if the two signals strongly disagree. A false accusation of AI authorship is a worse outcome than an uncertain label.

### How confidence maps to label text (not just a percentage)

| Confidence range | Display text | Body text variant |
|---|---|---|
| ≥ 0.80 | `"87% confident (very high)"` | "strongly suggests... multiple independent signals align..." |
| 0.65–0.79 | `"71% confident (high)"` | "strongly suggests... multiple independent signals align..." |
| 0.45–0.64 | `"52% confident (moderate) — interpret alongside other context"` | "leans toward... but signals gave only partial agreement... carries meaningful uncertainty" |
| 0.30–0.44 | `"36% confident (low) — treat this result with caution"` | "leans toward... but not all signals aligned... should not be treated as a firm determination" |
| < 0.30 | `"Insufficient confidence to label (24%)"` | Uncertain variant regardless of ai_score direction |

---

## 3. Transparency Label Design

Three variants. These are the exact strings that will render to readers.

---

### Variant A: `high_confidence_ai` — Strong confidence (≥ 0.65)

> **AI-Generated Content**
>
> Our analysis strongly suggests this content was written by an AI system. Both the statistical structure and semantic patterns of the text align with characteristics typical of AI-generated writing — consistent tone, predictable sentence rhythm, and low stylistic idiosyncrasy. This reflects the likely origin of the content, not its quality or value.
>
> *87% confident (high)*
>
> Believe this is incorrect? If you're the creator, you can submit an appeal with context our automated system cannot see — a human reviewer will examine it.

### Variant A: `high_confidence_ai` — Moderate confidence (0.30–0.64)

> **AI-Generated Content**
>
> Our analysis leans toward AI authorship, but the two signals we use gave only partial agreement. Some characteristics match AI-generated writing, while others are ambiguous. This result carries meaningful uncertainty — treat it as a soft indicator rather than a definitive finding.
>
> *52% confident (moderate) — interpret alongside other context*
>
> Believe this is incorrect? If you're the creator, you can submit an appeal with context our automated system cannot see — a human reviewer will examine it.

---

### Variant B: `high_confidence_human` — Strong confidence (≥ 0.65)

> **Human-Written Content**
>
> Our analysis suggests this content was written by a person. The writing shows the kind of natural variation, stylistic idiosyncrasy, and structural unpredictability that characterizes human authorship. Multiple independent signals agree on this assessment.
>
> *91% confident (very high)*
>
> Disagree with this assessment? You can submit an appeal to request a human reviewer take a closer look.

### Variant B: `high_confidence_human` — Moderate confidence (0.30–0.64)

> **Human-Written Content**
>
> Our analysis leans toward human authorship, but not all signals were in full agreement. Some features are consistent with human writing, while others are ambiguous. This assessment is directionally human, but should not be treated as a firm determination.
>
> *41% confident (low) — treat this result with caution*
>
> Disagree with this assessment? You can submit an appeal to request a human reviewer take a closer look.

---

### Variant C: `uncertain`

> **Origin Unclear**
>
> Our system detected mixed signals and cannot determine with confidence whether this content was written by a human or an AI. This may reflect human-AI collaborative writing, a distinctive personal style that sits outside our model's experience, or genuinely ambiguous text. No definitive label is applied — this content is presented without attribution.
>
> *Insufficient confidence to label (27%)*
>
> If you're the creator, sharing context through an appeal can help us review this more accurately and improve future classifications.

---

## 4. Appeals Workflow

### Who can submit an appeal
Anyone with a valid `content_id` from a prior `/submit` call can submit an appeal. In a production system this would be gated to authenticated creators; in this implementation it's open to the creator_id associated with the submission. Rate limiting (5 requests/hour per IP) prevents appeal flooding.

### What information the appeal captures

```json
POST /appeal
{
  "content_id": "uuid of the original submission",
  "creator_reasoning": "Why I believe the classification is wrong (20–2000 chars, required)",
  "creator_name": "Optional display name or pseudonym"
}
```

`creator_reasoning` is the most important field — it's the free-text argument a reviewer reads. It's stored verbatim in the audit log with no truncation.

### What the system does when an appeal is received

1. **Lookup** `content_id` in `content_records`. If not found → 404. If already `under_review` → 409 (no duplicate appeals).
2. **Status update:** `content_records.status` changes from `analyzed` → `under_review`. This is a mutable write to the record.
3. **Audit log write:** A new row is appended to `audit_log` with `event_type = "appeal"`, the verbatim reasoning, creator name, previous attribution, and timestamp. The original analysis row is never modified — the audit trail is append-only.
4. **Response:** Returns `appeal_id`, `previous_attribution`, `new_status: "under_review"`, and a human-readable message confirming a reviewer will examine it.
5. **No automated re-classification.** The system explicitly does not re-run signals. An LLM re-running the same signals on the same text will likely produce the same result. Human judgment is required.

### What a human reviewer sees when opening the appeal queue

A reviewer hitting `GET /log` (filtered by `event_type=appeal` or `status=under_review`) sees:

```
content_id:          3f8a2c91-...
event_type:          appeal
timestamp:           2024-01-15T14:32:01Z
previous_attribution: ai_generated
original_ai_score:   0.67
original_confidence: 0.52
signals_used:        ["stylometric_heuristics", "llm_holistic_classifier"]
content_preview:     "The wind moved through the aspens like something ..."
appeal_reasoning:    "This is an excerpt from my MFA thesis, written over
                      two years. I write in a very controlled prose style
                      intentionally — I studied writers like Didion and
                      Bernhard. The uniformity of my sentences is a
                      deliberate craft choice, not an AI artifact."
creator_name:        "R. Mercer"
```

The reviewer has: the original scores (confidence 0.52 — moderate, not strong), both signal names so they know what was run, a content preview, and the creator's verbatim argument. With that information a reviewer can decide to override (`reviewed` status, human-written) or uphold (`reviewed` status, ai_generated) — both actions would be new audit log entries written by the reviewer tool (out of scope for this implementation but the schema supports it).

---

## 5. Anticipated Edge Cases

### Edge Case 1: A poet whose formal aesthetic looks like AI output

**Scenario:** A contemporary poet working in a strict syllabic tradition submits a 20-line poem. The poem uses very uniform line lengths (low sentence CV), a moderate vocabulary with no unusual word choices (TTR in the AI sweet spot), and careful, minimal punctuation. Stylometric signal scores it 0.72 (AI-like). The LLM, seeing clean structure and restrained language, scores it 0.61 (mildly AI-like). Combined `ai_score = 0.66`, `confidence = 0.54`. Attribution: `ai_generated`, moderate-confidence label.

**Why this is hard:** Both signals are measuring things that are genuinely true about the poem — it *does* have uniform structure and restrained vocabulary. That's also what formal poetic craft looks like. The system has no way to distinguish "AI uniform" from "craftedly uniform."

**What the system does:** The moderate-confidence body text explicitly says "carries meaningful uncertainty — treat it as a soft indicator." The appeal pathway is prominently offered. If the poet appeals and explains their aesthetic influences, a reviewer has the confidence score (0.54, not 0.87) and can make an informed call. This is a case where the label's honest uncertainty communication matters more than the classification itself.

**Mitigation:** The `content_type` field (set to `poem`) could in a future version trigger lower weight on the sentence-CV feature, since poetic line structure is not the same as prose sentence structure. Not implemented yet, but the field is captured.

---

### Edge Case 2: Human-AI collaborative writing that genuinely is both

**Scenario:** A blogger drafts three paragraphs by hand, then pastes the draft into GPT-4 and asks it to "clean up the language and add a fourth paragraph." The final post has human voice in the opening, AI-polished middle prose, and a fully AI-generated conclusion. `ai_score` comes back at 0.55, `confidence = 0.08`. Attribution: `uncertain`.

**Why this is hard:** Both signals are correct. The text is genuinely mixed. The LLM detects a discontinuity in voice and hedges near 0.5. Stylometrics sees moderate CV (the human paragraphs have higher variance, the AI paragraph has lower variance, they average out to something middling). The system correctly reports uncertainty — but "uncertain" is also genuinely the right answer here.

**The harder version of this:** If the blogger only lightly edited the AI output (kept 80% of GPT's text), the final piece may score solidly AI-like with high confidence and the human author will feel unfairly labeled. The label says it reflects "origin," not quality — but for collaborative work, "origin" is genuinely contested. The appeal workflow exists precisely for this case. A reviewer would see the context ("I wrote the opening, edited the rest") and could mark it as a known collaborative work.

**Why this matters for system design:** It argues against ever displaying a label without the appeal CTA clearly visible. No label should feel like a verdict without recourse.

---

### Edge Case 3 (bonus): Very short texts

**Scenario:** A 60-word flash fiction piece. Minimum length passes validation (≥ 50 chars), but:
- Stylometric CV is computed from 3–4 sentences → statistically unreliable
- LLM has too little text to detect semantic patterns → tends toward 0.5
- Combined result: both signals near 0.5, confidence near 0.05 → `uncertain`

**What the system does:** Correctly returns `uncertain` with very low confidence. The label reads: *"Insufficient confidence to label (5%)"*. This is the right behavior. The system does not guess.

**Implementation note:** The stylometric signal already handles this — it returns `0.5` with a `"note": "insufficient text"` flag if sentence_count < 3 or word_count < 30. The `sty_fallback` flag is set, which adjusts the confidence ceiling downward in the combiner.

---
 
## Architecture
 
### Submission Flow
 
A submission enters through `POST /submit`, passes rate limiting and validation, then fans out to both detection signals — stylometric analysis (pure Python, no network call) and the LLM classifier (Groq API). Their scores are merged by the confidence scorer into a single `ai_score` and a `confidence` value, which the label generator uses to select and populate one of three transparency label variants. The final response, along with every signal score and the label text, is written to the SQLite audit log before anything is returned to the caller.
 
```
POST /submit
  { content, content_type?, creator_id? }
        │
        ▼
  ┌─────────────────┐
  │  Rate Limiter   │─── >10 req/min ──────────────────► 429
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │    Validator    │─── invalid body ─────────────────► 422
  │   (Pydantic)    │
  └────────┬────────┘
           │ raw text
           ├──────────────────────────────┐
           ▼                              ▼
  ┌─────────────────┐           ┌──────────────────────┐
  │   Signal 1      │           │   Signal 2           │
  │   Stylometric   │           │   LLM Classifier     │
  │   (pure Python) │           │   (Groq API)         │
  │                 │           │                      │
  │ · sentence CV   │           │ · semantic coherence │
  │ · type-token    │           │ · structural tells   │
  │   ratio         │           │ · voice / POV        │
  │ · punct profile │           │ · hedged opinions    │
  └────────┬────────┘           └──────────┬───────────┘
           │ sty_score: float              │ llm_score: float
           │ details: dict                 │ reasoning, indicators
           └──────────────┬───────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  Confidence Scorer  │
               │                     │
               │  ai_score  = 0.6×llm│
               │          + 0.4×sty  │
               │                     │
               │  confidence =       │
               │   √(agreement       │
               │     × distance)     │
               │                     │
               │  attribution = enum │
               └──────────┬──────────┘
                          │ ai_score, confidence, attribution
                          ▼
               ┌─────────────────────┐
               │   Label Generator   │
               │                     │
               │  selects variant    │
               │  renders body text  │
               │  appends appeal CTA │
               └──────────┬──────────┘
                          │ TransparencyLabel
                          ▼
               ┌─────────────────────┐
               │    Audit Logger     │──► audit_log row
               │                     │──► content_records row
               └──────────┬──────────┘        (SQLite)
                          │
                          ▼
                    SubmitResponse
                    ─────────────
                    content_id
                    attribution
                    ai_score
                    confidence
                    signals[]
                    label
                    status: "analyzed"
                    analyzed_at
```
 
### Appeal Flow
 
An appeal enters through `POST /appeal` with a `content_id` and the creator's free-text reasoning. The system looks up the original decision, rejects the request with a 404 if the content doesn't exist or a 409 if an appeal is already pending, then atomically updates the content record's status to `under_review` and appends a new row to the audit log — the original analysis row is never modified. No re-classification runs; the appeal is queued for a human reviewer who will see the original confidence score, both signal scores, and the creator's verbatim argument.
 
```
POST /appeal
  { content_id, creator_reasoning, creator_name? }
        │
        ▼
  ┌─────────────────┐
  │  Rate Limiter   │─── >5 req/hour ──────────────────► 429
  └────────┬────────┘
           │
           ▼
  ┌──────────────────────────────┐
  │  Lookup content_id           │─── not found ────────► 404
  │  in content_records          │─── already pending ──► 409
  └────────┬─────────────────────┘
           │ ContentRecord + previous_attribution
           ▼
  ┌──────────────────────────────┐
  │  Status Update               │
  │  content_records.status      │
  │    "analyzed" → "under_review"│
  └────────┬─────────────────────┘
           │
           ▼
  ┌──────────────────────────────┐
  │  Audit Logger                │──► audit_log row (append-only)
  │  event_type = "appeal"       │      content_id
  │  appeal_reasoning (verbatim) │      previous_attribution
  │  appeal_creator              │      original ai_score
  │  status = "under_review"     │      original confidence
  └────────┬─────────────────────┘      creator_reasoning
           │                            timestamp
           ▼
     AppealResponse
     ─────────────
     appeal_id
     content_id
     previous_attribution
     new_status: "under_review"
     message
     logged_at
```
 
---
 
## AI Tool Plan
 
How I'll use AI-assisted code generation across the three implementation milestones. For each milestone: what context I'll provide, what I'll ask for, and how I'll verify the output is correct before wiring it in.
 
The guiding principle: the AI tool gets *spec*, not *instructions*. I'll paste the relevant planning sections as context rather than explaining the system verbally. The diagram and the exact field names from the models section serve as the shared contract between the spec and the generated code.
 
---
 
### M3 — Submission Endpoint + Signal 1 (Stylometric)
 
**Context sections to provide:**
1. The `## Architecture` submission flow diagram (shows the full chain, labels each arrow with data types)
2. Section `1. Detection Signals → Signal 1` (feature definitions, weight table, output format, fallback behavior)
3. The `models.py` file (so generated code uses the exact Pydantic field names — `sty_score`, `details`, `SignalResult` — rather than inventing its own)
**What I'll ask the AI tool to generate:**
 
*Prompt 1 — Signal function:*
> "Using the spec below, implement `signals/stylometric.py`. The function signature must be `analyze(text: str) -> tuple[float, dict]` where the float is `sty_score ∈ [0.0, 1.0]` and the dict matches the `details` shape shown in `SignalResult`. Use only the Python standard library — no NLTK, no spaCy. Implement the three sub-features (sentence length CV, type-token ratio, punctuation profile) with the weights given in the spec. Include the short-text fallback that returns `(0.5, {"note": "insufficient text"})` when sentence_count < 3 or word_count < 30."
 
*Prompt 2 — FastAPI skeleton + `/submit` stub:*
> "Using the architecture diagram below, scaffold `main.py` as a FastAPI app. Include `POST /submit` that: runs rate limiting (slowapi, 10/minute per IP), validates the body against `SubmitRequest`, calls `stylometric.analyze(body.content)`, and returns a partial `SubmitResponse` with `content_id` (uuid4), `ai_score` set to the stylometric score, and `signals` containing one `SignalResult`. Leave `confidence`, `attribution`, and `label` as placeholder values — those are M4 and M5 work. Wire up the SQLAlchemy session dependency from `core/audit.py` but don't call any log functions yet."
 
**How I'll verify before wiring in:**
 
Run the stylometric function in isolation against four fixed test inputs before touching the endpoint:
 
```python
# test_stylometric.py
from signals.stylometric import analyze
 
# Should score low (human-like): bursty, idiosyncratic, ellipsis-heavy
HUMAN_POEM = """I keep the photo in a drawer.
Not hidden — just not out.
Some mornings the light comes through the kitchen window at exactly the wrong angle
and I think of you without meaning to,
which is the only honest way."""
 
# Should score high (AI-like): uniform rhythm, balanced structure, moderate TTR
AI_PROSE = """Artificial intelligence has transformed numerous industries in recent years. \
The technology enables organizations to process large volumes of data efficiently. \
Machine learning algorithms can identify patterns that humans might overlook. \
These capabilities have created significant opportunities for businesses worldwide. \
Companies that adopt these tools early often gain competitive advantages."""
 
score_h, details_h = analyze(HUMAN_POEM)
score_a, details_a = analyze(AI_PROSE)
 
assert score_h < 0.50, f"Expected human poem to score < 0.50, got {score_h}"
assert score_a > 0.55, f"Expected AI prose to score > 0.55, got {score_a}"
assert "features" in details_h
assert details_h["features"]["sentence_length_cv"]["cv"] > 0.5  # burstier
assert details_a["features"]["sentence_length_cv"]["cv"] < 0.3  # more uniform
print(f"Human poem: {score_h:.3f} | AI prose: {score_a:.3f} ✓")
```
 
Gate: if these four assertions don't pass, I fix the signal function before touching `main.py`. The endpoint only gets the function once the isolated test shows the scores moving in the right direction and the `details` dict has the expected keys.
 
---
 
### M4 — Signal 2 (LLM) + Confidence Scoring
 
**Context sections to provide:**
1. The `## Architecture` submission flow diagram (shows the two-signal fork and what passes into the confidence scorer)
2. Section `1. Detection Signals → Signal 2` (Groq model, prompt strategy, output schema, fallback behavior)
3. Section `2. Uncertainty Representation` (the weighted blend formula, the geometric mean confidence formula, the full calibration table, the attribution thresholds)
4. The existing `signals/stylometric.py` (so the AI tool sees what `analyze()` returns and can produce a consistent interface for the LLM signal)
**What I'll ask the AI tool to generate:**
 
*Prompt 1 — LLM signal function:*
> "Using the spec below, implement `signals/llm_classifier.py`. The function signature must match `signals/stylometric.py` exactly: `analyze(text: str) -> tuple[float, dict]`. Call Groq's `llama-3.3-70b-versatile` at temperature 0.10. The system prompt must ask the model to return only a JSON object (no markdown) with fields: `ai_probability`, `reasoning`, `key_indicators`, `confidence_note`. Extract `ai_probability` as the returned float. If the API call fails or JSON parsing fails, return `(0.5, {'error': ..., 'fallback': True})`. Include a `_extract_json()` helper that strips accidental markdown fences before parsing."
 
*Prompt 2 — Confidence scorer:*
> "Using the spec below, implement `core/confidence.py`. The function signature is `combine_signals(stylometric_score, llm_score, stylometric_fallback, llm_fallback) -> tuple[float, float, AttributionResult]`. Implement the weighted blend (`ai_score = 0.60 × llm + 0.40 × sty`), the geometric mean confidence formula (`confidence = √(agreement × distance)`), and the attribution thresholds (`ai_score ≥ 0.62 AND confidence ≥ 0.30` for AI_GENERATED; `≤ 0.38 AND confidence ≥ 0.30` for HUMAN_WRITTEN; everything else UNCERTAIN). Handle fallback flags: if both fallback → return (0.5, 0.0, UNCERTAIN); if one fallback → use the other signal at full weight, cap agreement at 0.70."
 
**What I'll check before wiring into the endpoint:**
 
Run the calibration table from Section 2 as assertions — the planned table becomes the test suite:
 
```python
# test_confidence.py
from core.confidence import combine_signals
from models import AttributionResult
import math
 
def check(llm, sty, expected_attr, min_conf=None, max_conf=None, label=""):
    ai_score, conf, attr = combine_signals(sty, llm)
    assert attr == expected_attr, f"{label}: expected {expected_attr}, got {attr} (ai={ai_score:.3f}, conf={conf:.3f})"
    if min_conf: assert conf >= min_conf, f"{label}: conf {conf:.3f} below min {min_conf}"
    if max_conf: assert conf <= max_conf, f"{label}: conf {conf:.3f} above max {max_conf}"
    print(f"{label}: ai={ai_score:.3f} conf={conf:.3f} → {attr} ✓")
 
check(0.90, 0.84, AttributionResult.AI_GENERATED,    min_conf=0.75, label="both strongly AI")
check(0.12, 0.18, AttributionResult.HUMAN_WRITTEN,   min_conf=0.75, label="both strongly human")
check(0.72, 0.60, AttributionResult.AI_GENERATED,    max_conf=0.65, label="agree but moderate")
check(0.78, 0.22, AttributionResult.UNCERTAIN,       max_conf=0.35, label="signals disagree")
check(0.52, 0.48, AttributionResult.UNCERTAIN,       max_conf=0.15, label="both near midpoint")
```
 
Additional manual check: hit the live `/submit` endpoint with the same `HUMAN_POEM` and `AI_PROSE` from M3 and confirm the LLM signal moves in the expected direction and that `confidence` differs meaningfully (not just by a rounding difference) between them. If both texts come back with confidence in the 0.40–0.60 range, the scorer is not discriminating — that's a signal the weighting or thresholds need adjustment before moving to M5.
 
---
 
### M5 — Production Layer (Labels + Appeals)
 
**Context sections to provide:**
1. Section `3. Transparency Label Design` — all three variants with exact text strings (the label generator should reproduce these strings, not paraphrase them)
2. Section `4. Appeals Workflow` — the status transitions, the 404/409 guard conditions, the exact fields logged, the reviewer view
3. The `## Architecture` appeal flow diagram (shows the status update and audit log write as two separate steps)
4. `core/audit.py` and `models.py` (so generated appeal code uses the correct table columns and Pydantic response models)
**What I'll ask the AI tool to generate:**
 
*Prompt 1 — Label generator:*
> "Using the label variants specified below, implement `core/labels.py`. The function signature is `generate_label(attribution: AttributionResult, ai_score: float, confidence: float) -> TransparencyLabel`. Map to three variants: `high_confidence_ai`, `high_confidence_human`, `uncertain`. Within the AI and human variants, use the strong-confidence body text when `confidence ≥ 0.65` and the moderate-confidence body text when `0.30 ≤ confidence < 0.65`. The `confidence_display` string must use the five-tier qualifier table from the spec (`very high / high / moderate / low / very low`). Use the exact label body text from the spec — do not paraphrase. The uncertain variant always uses the fixed body text regardless of confidence level."
 
*Prompt 2 — `/appeal` endpoint:*
> "Using the spec and the appeal flow diagram below, add `POST /appeal` to `main.py`. It must: (1) rate-limit at 5/hour per IP via slowapi, (2) look up the `content_id` in `content_records` and return 404 if not found, (3) return 409 if `status == 'under_review'` already, (4) call `audit.log_appeal()` which atomically updates the record status and writes an audit row, (5) return `AppealResponse` with `appeal_id`, `previous_attribution`, `new_status: 'under_review'`, and the message string from the spec. Do not trigger any re-classification."
 
**How I'll verify:**
 
Three label reachability checks — one for each variant — using crafted inputs that force each branch:
 
```python
# test_labels.py
from core.labels import generate_label
from models import AttributionResult
 
# Variant A: high_confidence_ai (strong)
label = generate_label(AttributionResult.AI_GENERATED, ai_score=0.82, confidence=0.78)
assert label.variant == "high_confidence_ai"
assert "strongly suggests" in label.body
assert "very high" in label.confidence_display or "high" in label.confidence_display
assert "appeal" in label.actionable_text.lower()
 
# Variant A: high_confidence_ai (moderate — different body text)
label_mod = generate_label(AttributionResult.AI_GENERATED, ai_score=0.67, confidence=0.45)
assert label_mod.variant == "high_confidence_ai"
assert "leans toward" in label_mod.body          # different from strong version
assert "meaningful uncertainty" in label_mod.body
assert label_mod.body != label.body              # must not be the same string
 
# Variant B: high_confidence_human
label = generate_label(AttributionResult.HUMAN_WRITTEN, ai_score=0.15, confidence=0.81)
assert label.variant == "high_confidence_human"
assert "natural variation" in label.body
 
# Variant C: uncertain
label = generate_label(AttributionResult.UNCERTAIN, ai_score=0.55, confidence=0.12)
assert label.variant == "uncertain"
assert "No definitive label" in label.body
assert "Insufficient confidence" in label.confidence_display
```
 
Appeal flow end-to-end check (against a running server with a seeded `content_id`):
 
```bash
# 1. Submit content, capture content_id
CONTENT_ID=$(curl -s -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"content": "... ≥50 chars ..."}' | jq -r '.content_id')
 
# 2. Verify initial status
curl http://localhost:8000/status/$CONTENT_ID | jq '.status'
# expected: "analyzed"
 
# 3. Submit appeal
curl -X POST http://localhost:8000/appeal \
  -H "Content-Type: application/json" \
  -d "{\"content_id\": \"$CONTENT_ID\", \"creator_reasoning\": \"This is my own work, written over six months as part of my MFA thesis.\"}"
# expected: 200 with new_status "under_review"
 
# 4. Verify status changed
curl http://localhost:8000/status/$CONTENT_ID | jq '.status'
# expected: "under_review"
 
# 5. Verify duplicate appeal is rejected
curl -X POST http://localhost:8000/appeal \
  -H "Content-Type: application/json" \
  -d "{\"content_id\": \"$CONTENT_ID\", \"creator_reasoning\": \"Same appeal again.\"}"
# expected: 409 Conflict
 
# 6. Confirm audit log has both events
curl "http://localhost:8000/log?per_page=5" | jq '.entries[] | {event_type, content_id, status}'
# expected: one "analysis" row + one "appeal" row for the same content_id
```
 
Gate: all label assertions must pass and all six curl checks must return the expected status codes before the milestone is considered complete. Specifically, step 5 (the 409) is the one most likely to be missing from AI-generated code — the conflict guard is easy to forget.

---

## 6. Analytics Dashboard

### Purpose

The analytics dashboard gives operators a real-time view of detection patterns, appeal behavior, and signal health. It serves two audiences: operators monitoring whether the system is producing a reasonable distribution of labels, and developers diagnosing whether the two-signal design is discriminating well or clustering near the midpoint.

### Schema Extension

One addition to `content_records`: two new columns, `sty_score REAL` and `llm_score REAL`, storing the per-request stylometric and LLM signal scores. These were already available in the per-request response body but were not persisted, which made the signal agreement metric impossible to query after the fact. The migration adds both columns to existing databases without data loss; rows submitted before the migration will show `NULL` for these columns and are excluded from the signal agreement chart rather than counted as zeroes.

```sql
ALTER TABLE content_records ADD COLUMN sty_score REAL;
ALTER TABLE content_records ADD COLUMN llm_score REAL;
```

Implemented in `_ensure_schema` using a try/except guard so the migration is idempotent — running it against a fresh database or an existing one produces the same result.

### New Endpoint: `GET /analytics`

Aggregates from both `content_records` and `audit_log` without requiring new tables. All four metrics derive from data that already exists (or now exists with the schema extension).

**Response schema:**
```json
{
  "generated_at": "ISO-8601 timestamp",
  "summary": {
    "total_submissions": 150,
    "total_appeals": 12,
    "appeal_rate_pct": 8.0,
    "avg_confidence": 0.61,
    "avg_confidence_pct": 61.0,
    "most_common_attribution": "ai_generated"
  },
  "attribution_breakdown": {
    "ai_generated": 89,
    "human_written": 42,
    "uncertain": 19
  },
  "confidence_distribution": [
    { "bucket": "0–20%",  "count": 5  },
    { "bucket": "20–40%", "count": 12 },
    { "bucket": "40–60%", "count": 45 },
    { "bucket": "60–80%", "count": 61 },
    { "bucket": "80–100%","count": 27 }
  ],
  "appeal_by_attribution": {
    "ai_generated": 10,
    "human_written": 1,
    "uncertain": 1
  },
  "signal_agreement": {
    "strong": 70,
    "moderate": 50,
    "weak": 30
  }
}
```

### Metrics and Rationale

| Metric | Chart type | Why included |
|---|---|---|
| Attribution breakdown | Donut | Primary health signal. A system labeling 95%+ of submissions as AI is miscalibrated; this makes that visible immediately. |
| Confidence distribution | Histogram (5 buckets) | Shows whether the system is discriminating or clustering near 0.5. A healthy system should have a spread; a spike in the 40–60% bucket means most results are moderate-confidence, which may indicate the signal weights need tuning. |
| Appeal rate by attribution | Horizontal bar | Reveals which attribution type is most frequently contested. A high appeal rate on `ai_generated` relative to `human_written` is the expected pattern (AI misclassification hurts creators more). An unexpected spike on `human_written` appeals warrants investigation. |
| Signal agreement | Donut | **Chosen additional metric.** Buckets submissions by `|llm_score − sty_score|`: strong (< 0.2), moderate (0.2–0.4), weak (> 0.4). A large weak-agreement slice means the two signals are frequently contradicting each other — this may indicate the text corpus being submitted sits in a genuinely hard-to-classify zone, or that one signal is systematically off. This metric is only meaningful after the schema migration; pre-migration rows are excluded. |

### Frontend: `GET /dashboard`

Served as a static HTML file at `static/dashboard.html`. Flask routes `GET /dashboard` to `app.send_static_file("dashboard.html")`. No build step, no bundler — Chart.js is loaded from cdnjs.cloudflare.com.

The dashboard:
- Fetches `/analytics` on page load
- Auto-refreshes every 30 seconds via `setInterval`
- Destroys and re-creates charts on each refresh (avoids Chart.js accumulation bugs)
- Shows an error banner (not a silent failure) if the Flask server is unreachable
- Handles empty state for signal agreement chart when no post-migration submissions exist yet

Layout: a summary strip (4 stat cards) above a 2×2 chart grid. Responsive — collapses to a single column below 900px.

### What this does not do

- No authentication on `/analytics` or `/dashboard`. In production these would sit behind an operator auth gate.
- No time-series queries. The confidence distribution and attribution breakdown are lifetime aggregates, not windowed trends. Adding a `?since=` query parameter to `/analytics` would enable trend views without schema changes.
- No reviewer queue action from the dashboard. A reviewer hitting the dashboard can see appeals exist but must use `GET /log?event_type=appeal` to drill into them.