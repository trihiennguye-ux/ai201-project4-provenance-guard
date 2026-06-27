# Provenance Guard — AI Content Attribution API

A Flask-based API that classifies submitted text as AI-generated or human-written using two independent detection signals: a stylometric heuristics analyzer and a Groq-backed LLM classifier. Classifications are stored in SQLite, surfaced as calibrated transparency labels, and supported by an appeal workflow for human review.

---

## Architecture Overview

A submission enters the system at `POST /submit` and exits as a transparency label backed by an audit trail. The path is linear and has one fork — the two signals run independently, then re-join at the confidence scorer.

```
POST /submit
  { content, content_type?, creator_id? }
        │
        ▼
  Rate Limiter (10 req/min per IP)
        │ 429 if exceeded
        ▼
  Pydantic Validator
        │ 422 if body invalid
        ├──────────────────────────────────────┐
        ▼                                      ▼
  Signal 1: Stylometric                Signal 2: LLM Classifier
  (pure Python, no network)            (Groq API, llama-3.3-70b)
  → sty_score ∈ [0.0, 1.0]            → llm_score ∈ [0.0, 1.0]
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              Confidence Scorer
              ai_score  = 0.60 × llm_score + 0.40 × sty_score
              confidence = √(signal_agreement × distance_from_midpoint)
              attribution = AI_GENERATED | HUMAN_WRITTEN | UNCERTAIN
                       │
                       ▼
              Label Generator
              selects variant, renders body text, appends appeal CTA
                       │
                       ▼
              Audit Logger
              writes to content_records + audit_log (SQLite)
                       │
                       ▼
              SubmitResponse
              { content_id, attribution, ai_score, confidence,
                signals[], label, status: "analyzed", analyzed_at }
```

Appeals follow a separate path through `POST /appeal`. The system performs a status transition (`analyzed → under_review`), appends a new row to the append-only audit log, and queues the submission for a human reviewer. No re-classification is triggered.

---

## Detection Signals

### Signal 1 — Stylometric Heuristics (`signals/stylometric.py`)

**What it measures:** Statistical properties of the text's *form* — sentence rhythm, vocabulary diversity, and punctuation habits. No semantic understanding, no network calls.

Three sub-features combined into a single `sty_score ∈ [0.0, 1.0]`:

| Sub-feature | What it captures | Weight |
|---|---|---|
| Sentence Length CV | Coefficient of variation of per-sentence word counts | 0.45 |
| Type-Token Ratio | `unique_words / total_words` | 0.30 |
| Punctuation Density & Variety | Punct chars per word + count of distinct types used | 0.25 |

```
sty_score = 0.45 × sentence_cv_score
          + 0.30 × ttr_score
          + 0.25 × punctuation_score
```

**Why I chose it:** These three features operationalize a real structural difference between most AI and most human writing. LLM outputs are trained on readability signals, which drives sentence length toward a comfortable, uniform range (low CV). They also avoid both repetition and eccentricity in vocabulary (moderate TTR), and they produce structurally correct, minimal punctuation. Human writing — especially creative writing — hits the extremes: wildly varied sentence lengths, very high TTR in short texts, and expressive punctuation choices (em-dashes, ellipses, fragments). The signal is fast, deterministic, and adds genuine independence from the LLM signal.

**What it misses:** Form, not meaning. It cannot detect hedged opinions, suspiciously clean topic-sentence-body-conclusion structure, generic phrasing, or the absence of a personal point of view — all reliable semantic tells. It also cannot distinguish "AI uniform" from "craftedly uniform." A poet working in a strict syllabic tradition will produce low sentence CV and restrained punctuation by choice, not by algorithmic generation.

Short-text fallback: when `sentence_count < 3` or `word_count < 30`, the signal returns `(0.5, {"note": "insufficient text"})` and sets a `sty_fallback` flag that caps confidence in the combiner.

---

### Signal 2 — LLM Holistic Classifier (`signals/llm_classifier.py`)

**What it measures:** A language model's holistic judgment of whether the *meaning and voice* of the text match AI or human authorship patterns. Model: `llama-3.3-70b-versatile` via Groq, temperature 0.10. The system prompt instructs the model to return only a JSON object with `ai_probability`, `reasoning`, `key_indicators`, and `confidence_note`.

**Why I chose it:** The stylometric signal is blind to semantics. The LLM signal is specifically good at what stylometrics cannot do: detecting generic phrasing, suspiciously balanced opinions, topic-sentence-body-conclusion structure, clean transitions that read as "assembled," and the absence of a specific point of view. These are the things a careful human reader notices first. A single LLM call is also cheap and fast enough to run per request at this scale.

The two signals are genuinely independent — one structural/statistical, one semantic/holistic. That independence is what makes combining them informative rather than redundant.

**What it misses:** It cannot access authorship history, platform metadata, or any context outside the raw text. Its `ai_probability` output is a point estimate, not a calibrated probability — 0.7 here and 0.7 from stylometrics are not the same kind of number. It can also be fooled by well-crafted AI writing with a deliberately injected personal voice, and it has a systematic training-data bias from having seen large volumes of AI-generated content.

---

## Confidence Scoring

### Why a separate confidence score?

`ai_score` alone is not enough to make a reliable attribution. A blended score of 0.65 could mean both signals clearly agree the text is AI-like, or it could mean one signal says 0.90 (AI) and the other says 0.22 (human) and they happened to average to something in between. Those two cases should produce very different outcomes. The confidence score captures this.

### The formula

**Step 1 — Weighted blend:**
```
ai_score = 0.60 × llm_score + 0.40 × sty_score
```
LLM gets higher weight because it captures richer contextual signal.

**Step 2 — Confidence via geometric mean of two independent factors:**
```
signal_agreement = 1.0 − |llm_score − sty_score|
distance         = |ai_score − 0.5| × 2

confidence = √(signal_agreement × distance)
```

Geometric mean rather than arithmetic mean ensures **both** high agreement AND a clear directional signal are required for high confidence. If either factor is near zero, confidence collapses — which is correct. For example: `agreement = 0.30`, `distance = 0.80` → arithmetic gives 0.55 (misleadingly confident), geometric gives 0.49 (correctly flags that disagreement dominates).

**Attribution thresholds:**
```
ai_score ≥ 0.62 AND confidence ≥ 0.30  →  AI_GENERATED
ai_score ≤ 0.38 AND confidence ≥ 0.30  →  HUMAN_WRITTEN
otherwise                               →  UNCERTAIN
```

The 0.62/0.38 boundaries (not 0.50) exist because a blended score at 0.55 is genuinely ambiguous. The `confidence ≥ 0.30` floor means that even a directional score (say 0.70 toward AI) produces `UNCERTAIN` if the two signals strongly disagree — a false accusation of AI authorship is a worse outcome than an uncertain label.

### Validation: two contrasting real submissions

**High-confidence case** — content_id `129a913e`:

```
sty_score:   0.6801
llm_score:   0.8000
ai_score:    0.7520     (0.60 × 0.80 + 0.40 × 0.68)
agreement:   0.8799     (1 − |0.80 − 0.68|)
distance:    0.5040     (|0.752 − 0.5| × 2)
confidence:  0.6661

→ attribution: AI_GENERATED
→ label band:  "67% confident (high)"
→ body text:   "Our analysis strongly suggests this content was written
                by an AI system..."
```

Both signals pointed solidly toward AI — stylometrics scored high sentence uniformity and punctuation AI-likeness; the LLM flagged "uniform tone, topic-sentence-body-conclusion structure, hedged balanced opinions." Agreement was high and the blended score was well clear of the midpoint. Confidence came out at 0.67 — solidly in the "high" band.

**Lower-confidence case** — content_id `de89a63a`:

```
sty_score:   0.7000
llm_score:   0.6000
ai_score:    0.6400     (0.60 × 0.60 + 0.40 × 0.70)
agreement:   0.9000     (1 − |0.60 − 0.70|)
distance:    0.2800     (|0.64 − 0.5| × 2)
confidence:  0.5020

→ attribution: AI_GENERATED
→ label band:  "50% confident (moderate)"
→ body text:   "Our analysis leans toward AI authorship, but the two
                signals we use gave only partial agreement..."
```

Both signals still lean AI, but neither strongly. The blended score of 0.64 is only modestly above the 0.62 threshold, so `distance` (0.28) is low even though `agreement` is actually high (0.90). The LLM flagged a "hint of human-like introspection" from the phrase "I have been thinking a lot," which pulled it down to 0.60. The system correctly labels this with the moderate-uncertainty body text rather than the strong-confidence one, even though it reaches the same attribution.

These two cases confirm the scoring produces meaningful variation: they're both `AI_GENERATED`, but they received noticeably different confidence values (0.67 vs 0.50) and different label body text — the distinction that matters most to a reader challenging the result.

---

## Transparency Labels

Three variants. The exact display text for each is below.

---

### Variant A: `ai_generated` — Strong confidence (≥ 0.65)

> **AI-Generated Content**
>
> Our analysis strongly suggests this content was written by an AI system. Both the statistical structure and semantic patterns of the text align with characteristics typical of AI-generated writing — consistent tone, predictable sentence rhythm, and low stylistic idiosyncrasy. This reflects the likely origin of the content, not its quality or value.
>
> *67% confident (high)*
>
> Believe this is incorrect? If you're the creator, you can submit an appeal with context our automated system cannot see — a human reviewer will examine it.

---

### Variant A: `ai_generated` — Moderate confidence (0.30–0.64)

> **AI-Generated Content**
>
> Our analysis leans toward AI authorship, but the two signals we use gave only partial agreement. Some characteristics match AI-generated writing, while others are ambiguous. This result carries meaningful uncertainty — treat it as a soft indicator rather than a definitive finding.
>
> *50% confident (moderate) — interpret alongside other context*
>
> Believe this is incorrect? If you're the creator, you can submit an appeal with context our automated system cannot see — a human reviewer will examine it.

---

### Variant B: `human_written` — Strong confidence (≥ 0.65)

> **Human-Written Content**
>
> Our analysis suggests this content was written by a person. The writing shows the kind of natural variation, stylistic idiosyncrasy, and structural unpredictability that characterizes human authorship. Multiple independent signals agree on this assessment.
>
> *91% confident (very high)*
>
> Disagree with this assessment? You can submit an appeal to request a human reviewer take a closer look.

---

### Variant B: `human_written` — Moderate confidence (0.30–0.64)

> **Human-Written Content**
>
> Our analysis leans toward human authorship, but not all signals were in full agreement. Some features are consistent with human writing, while others are ambiguous. This assessment is directionally human, but should not be treated as a firm determination.
>
> *57% confident (moderate) — interpret alongside other context*
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

**Confidence-to-display mapping:**

| Confidence | Band label |
|---|---|
| ≥ 0.80 | very high |
| 0.65–0.79 | high |
| 0.45–0.64 | moderate — *interpret alongside other context* |
| 0.30–0.44 | low — *treat this result with caution* |
| < 0.30 | Insufficient confidence to label |

---

## Rate Limiting

Two limits, set independently for each endpoint:

| Endpoint | Limit | Reasoning |
|---|---|---|
| `POST /submit` | 10 requests/minute per IP | Stylometric analysis is cheap, but LLM calls to Groq cost money and add latency. 10/min allows genuine interactive use (a developer testing submissions) while making automated flooding economically painful. The rate is per-IP, not per-session, so it also limits scraper abuse without requiring auth. |
| `POST /appeal` | 5 requests/hour per IP | Appeals are queued for human review. Flooding the queue degrades reviewer throughput and could be used to bury legitimate appeals or harass the review team. 5/hour is generous for legitimate use — a real creator would submit one appeal and wait. Hourly window (not per-minute) means a brief burst of retries doesn't immediately lock a user out. |

The rate limit output from the test run above shows the `/submit` limit working as expected: 10 consecutive `200` responses followed by `429`s once the per-minute window fills.

---

## Known Limitations

### Formal writers and craft-conscious prose

The stylometric signal's sentence length CV sub-feature is this system's most brittle point, and the failure mode is specific: any writer who deliberately controls sentence rhythm will be misclassified toward AI.

A poet working in syllabics, a stylist who studied writers like Joan Didion or Thomas Bernhard (known for controlled, repetitive syntactic structures), or an academic writer trained in a formal disciplinary register all produce prose with low sentence CV, moderate TTR, and minimal punctuation variety — exactly the profile the system flags as AI-like. The LLM signal often compounds this, because it reads formal structure, measured tone, and lack of explicit personal opinion as AI tells too.

The system has no way to distinguish "uniform because algorithmically generated" from "uniform because craftedly controlled." The stylometric signal is measuring a real property of the text — it genuinely does have uniform structure — but the inference from that property to origin is wrong for this class of writer.

This is not fixable by adding more data. It's a fundamental ambiguity: the feature being measured is predictive for most writing but wrong for the tail of writers who treat formal constraint as an aesthetic virtue. The mitigation in the current system is honest uncertainty communication (moderate-confidence body text, prominent appeal CTA) rather than a fix to the signal itself. A future version could weight CV lower for submissions tagged `content_type: poem` or `essay`, where sentence rhythm has a different meaning than in prose.

---

## Spec Reflection

### One way the spec helped

The calibration table in Section 2 of the planning doc — showing expected `confidence` values across six named scenarios — became the test suite for the confidence scorer almost verbatim. Having exact expected values (e.g., "both signals strongly agree on AI → confidence ≥ 0.75") meant I could write assertions before any implementation and know immediately whether the geometric mean formula was behaving as designed. Without those concrete targets, I would have been tuning the formula by feel rather than against a spec.

### One way implementation diverged

The spec names the label variants `high_confidence_ai`, `high_confidence_human`, and `uncertain` (Section 3). The implementation uses `ai_generated`, `human_written`, and `uncertain` instead.

The reason: the spec's naming embeds the confidence level into the variant identifier, which implies there would be separate low-confidence and high-confidence variant types. In practice, the strong-vs-moderate distinction is better handled as a property *within* a single variant — the variant identifies the direction (AI, human, or unclear), and the confidence band determines which body text and display string to render. Collapsing `high_confidence_ai` and `low_confidence_ai` into one `ai_generated` variant with a `confidence_band` field makes the label generator simpler to extend (adding a new confidence tier doesn't require a new variant) and keeps the variant field semantically cleaner: it answers "what was the verdict?" not "how confident were we?" Those are separate questions.

---

## Analytics Dashboard

### What was built

A live analytics view at `GET /dashboard` that pulls from a new `GET /analytics` endpoint and renders four charts using Chart.js. No build step — the frontend is a single static HTML file served directly by Flask from `static/dashboard.html`.

### How to access it

```bash
python main.py          # start the Flask server
open http://localhost:5000/dashboard
```

The page auto-refreshes every 30 seconds. The raw data is also available at `http://localhost:5000/analytics` as JSON.

### What each panel shows

**Summary strip (top row):** Four stat cards — total submissions, appeal rate as a percentage, average confidence across all classifications, and the most common attribution label with its share of submissions.

**Attribution Breakdown (donut):** Distribution of `ai_generated`, `human_written`, and `uncertain` outcomes. The primary health signal: a system labeling 90%+ of submissions as AI is almost certainly miscalibrated, and this chart makes that immediately visible.

**Confidence Distribution (bar histogram):** Submissions bucketed into five 20-point confidence bands (0–20%, 20–40%, 40–60%, 60–80%, 80–100%). A healthy system should spread across bands. A spike concentrated in the 40–60% bucket means the system is producing moderate-confidence results on most inputs — which may indicate the signals need recalibration or that the submitted text corpus is harder to classify than expected.

**Appeals by Attribution (horizontal bar):** How many appeals each attribution type received. The expected pattern is a much higher appeal rate on `ai_generated` than `human_written` — creators are more likely to contest an AI label than a human label. An unexpected spike on `human_written` would warrant investigation.

**Signal Agreement (donut):** The chosen additional metric. Buckets submissions by the raw gap between the two signal scores (`|sty_score − llm_score|`): strong agreement (< 0.2), moderate agreement (0.2–0.4), and weak agreement (> 0.4). A large weak-agreement slice means the stylometric and LLM signals are frequently pulling in opposite directions, which drives confidence down and pushes more submissions into `uncertain`. This chart is the fastest way to see whether the two signals are functioning as complementary or competing. Requires the M6 schema migration; pre-migration rows with `NULL` signal scores are excluded and the chart shows an empty-state message until new submissions arrive.

### Schema change

`content_records` gained two new columns, `sty_score REAL` and `llm_score REAL`, to enable the signal agreement metric. An idempotent migration in `_ensure_schema` runs on every startup — it attempts `ALTER TABLE ... ADD COLUMN` and silently swallows the `OperationalError` if the column already exists, so it is safe to run against both fresh and existing databases.

---

## AI Usage

### Instance 1 — Stylometric signal function

**What I directed:** I provided the spec's Signal 1 section (the three sub-features, the weight table, the exact combination formula, and the short-text fallback conditions) and `models.py` as context, then asked Claude to implement `signals/stylometric.py` with a specific function signature: `analyze(text: str) -> tuple[float, dict]`. I specified that the dict must match the `details` shape used in `SignalResult`, that only the standard library was allowed (no NLTK, no spaCy), and that the fallback must return `(0.5, {"note": "insufficient text"})` when `sentence_count < 3` or `word_count < 30`.

**What I revised:** The generated implementation used `str.split()` to tokenize sentences, which treats `"Dr. Smith arrived."` as two sentences. For a stylometric tool, that's a meaningful error — it inflates sentence count and destabilizes the CV calculation on short texts. I overrode the sentence splitting with a simple regex that handles common abbreviations (`Dr.`, `Mr.`, `vs.`) before splitting on terminal punctuation. I also revised the punctuation variety scoring curve: the generated version used a simple linear scale, but I changed it to match the spec's implied behavior where 3–5 punctuation types is the "AI normal range" and scoring should penalize both extremes (too few types and too many types in different directions).

### Instance 2 — Confidence scorer and attribution thresholds

**What I directed:** I gave Claude the full Section 2 of the spec (the weighted blend formula, the geometric mean confidence formula, the calibration table, and the attribution thresholds), along with the existing stylometric signal as a reference for the interface. I asked it to implement `core/confidence.py` with a specific signature and to handle both fallback flags: if both signals fall back, return `(0.5, 0.0, UNCERTAIN)`; if one falls back, use the other at full weight and cap `signal_agreement` at 0.70.

**What I revised:** The initial implementation computed `distance = abs(ai_score - 0.5)` without the `× 2` normalization, which meant `distance` was capped at 0.50 rather than 1.0. This compressed the entire confidence range to [0.0, 0.71] and made it impossible to reach the "very high" confidence band (≥ 0.80) that the spec defines. I caught this by running the calibration table assertions from the spec — the "both signals strongly agree on AI" case produced `confidence = 0.59` instead of the expected ≥ 0.75. After adding the `× 2` multiplier, all six calibration cases passed. This is exactly the kind of off-by-a-constant error that's easy to miss in code review but immediately visible when you have concrete expected values to assert against.