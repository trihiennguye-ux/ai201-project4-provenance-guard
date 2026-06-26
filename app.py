"""
Provenance Guard — Flask API
============================
Storage model (two tables, distinct roles):

  content_records  Mutable. One row per content_id. Owns the current status:
                   analyzed → under_review → reviewed.

  audit_log        Append-only event stream. Every state change writes a new
                   row (event_type: "analysis" | "appeal"). The original
                   analysis row is never modified; the trail is immutable.

GET /log supports ?event_type= and ?status= filters so a reviewer can
isolate appeal events without touching content_records directly.
"""

from datetime import datetime, timezone
import json as _stdlib_json
import sqlite3
from uuid import uuid4

from flask import Flask, jsonify, request, json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from signals.stylometric import analyze as analyze_stylometric
from signals.llm_classifier import analyze as analyze_llm
from signals.confidence import combine_signals
from signals.label_generator import generate_label

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)
DATABASE_PATH = "provenance_guard.db"

_PREVIEW_LENGTH  = 200
_SIGNALS_USED    = ["stylometric_heuristics", "llm_authorship_classifier"]
_VALID_STATUSES  = {"analyzed", "under_review", "reviewed"}
_VALID_EVT_TYPES = {"analysis", "appeal"}

_CREATOR_REASONING_MIN = 20
_CREATOR_REASONING_MAX = 2000


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return "Provenance Guard is running."


@app.get("/log")
def log():
    event_type = request.args.get("event_type")
    status     = request.args.get("status")
    limit      = request.args.get("limit", default=10, type=int)
    limit      = max(1, min(limit, 100))

    if event_type and event_type not in _VALID_EVT_TYPES:
        return jsonify({"error": f"event_type must be one of: {', '.join(sorted(_VALID_EVT_TYPES))}."}), 422
    if status and status not in _VALID_STATUSES:
        return jsonify({"error": f"status must be one of: {', '.join(sorted(_VALID_STATUSES))}."}), 422

    return jsonify({"entries": _query_log(limit=limit, event_type=event_type, status=status)})


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 422

    text = body.get("text", body.get("content"))
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 422

    content_type = body.get("content_type")
    creator_id   = body.get("creator_id")
    content_id   = str(uuid4())
    analyzed_at  = datetime.now(timezone.utc).isoformat()
    preview      = text[:_PREVIEW_LENGTH]

    # Signal 1: Stylometric
    sty_score, sty_details = analyze_stylometric(text)
    sty_fallback = bool(sty_details.get("fallback"))

    # Signal 2: LLM classifier
    llm_score, llm_details = analyze_llm(text)
    llm_fallback = bool(llm_details.get("fallback"))

    # Attribution + Confidence
    ai_score, confidence, attribution_enum = combine_signals(
        stylometric_score=sty_score,
        llm_score=llm_score,
        stylometric_fallback=sty_fallback,
        llm_fallback=llm_fallback,
    )
    attribution = attribution_enum.value
    label       = generate_label(attribution, confidence)

    response_body = {
        "content_id":        content_id,
        "content_type":      content_type,
        "creator_id":        creator_id,
        "attribution":       attribution,
        "stylometric_score": sty_score,
        "llm_score":         llm_score,
        "combined_score":    ai_score,
        "confidence":        confidence,
        "signals": [
            {
                "name":        "stylometric_heuristics",
                "score":       sty_score,
                "attribution": _attribution_from_signal(sty_score, sty_details),
                "details":     sty_details,
            },
            {
                "name":        "llm_authorship_classifier",
                "score":       llm_score,
                "attribution": _attribution_from_signal(llm_score, llm_details),
                "details":     llm_details,
            },
        ],
        "label":  label,
        "status": "analyzed",
    }

    with sqlite3.connect(DATABASE_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO content_records
                (content_id, creator_id, content_type, preview,
                 attribution, ai_score, confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'analyzed', ?)
            """,
            (content_id, creator_id, content_type, preview,
             attribution, ai_score, confidence, analyzed_at),
        )
        _append_audit_event(
            conn,
            content_id=content_id,
            event_type="analysis",
            timestamp=analyzed_at,
            attribution=attribution,
            ai_score=ai_score,
            confidence=confidence,
            content_preview=preview,
        )

    return json.dumps(response_body, sort_keys=False), 200


@app.post("/appeal")
@limiter.limit("5 per hour")
def submit_appeal():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 422

    # ── content_id (required) ─────────────────────────────────────────────────
    content_id = body.get("content_id")
    if not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "Field 'content_id' is required and must be a non-empty string."}), 422

    # ── creator_reasoning (required, 20–2000 chars) ───────────────────────────
    creator_reasoning = body.get("creator_reasoning")
    if not isinstance(creator_reasoning, str):
        return jsonify({"error": "Field 'creator_reasoning' is required."}), 422
    creator_reasoning = creator_reasoning.strip()
    if len(creator_reasoning) < _CREATOR_REASONING_MIN:
        return jsonify({
            "error": f"Field 'creator_reasoning' must be at least {_CREATOR_REASONING_MIN} characters."
        }), 422
    if len(creator_reasoning) > _CREATOR_REASONING_MAX:
        return jsonify({
            "error": f"Field 'creator_reasoning' must be at most {_CREATOR_REASONING_MAX} characters."
        }), 422

    # ── creator_name (optional) ───────────────────────────────────────────────
    creator_name = body.get("creator_name")
    if creator_name is not None and not isinstance(creator_name, str):
        return jsonify({"error": "Field 'creator_name' must be a string if provided."}), 422
    creator_name = creator_name.strip() if isinstance(creator_name, str) else None

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)

        # ── Guard: content must exist ─────────────────────────────────────────
        record = conn.execute(
            "SELECT * FROM content_records WHERE content_id = ? LIMIT 1",
            (content_id,),
        ).fetchone()
        if record is None:
            return jsonify({"error": f"No analyzed content found with content_id '{content_id}'."}), 404
        record = dict(record)

        # ── Guard: only 'analyzed' content can be appealed ───────────────────
        if record["status"] != "analyzed":
            return jsonify({
                "error": "An appeal for this content has already been submitted.",
                "current_status": record["status"],
            }), 409

        appeal_id    = str(uuid4())
        submitted_at = datetime.now(timezone.utc).isoformat()

        # ── Transition: analyzed → under_review (mutable write) ──────────────
        conn.execute(
            "UPDATE content_records SET status = 'under_review' WHERE content_id = ?",
            (content_id,),
        )

        # ── Append appeal event (audit_log is never modified, only appended) ──
        _append_audit_event(
            conn,
            content_id=content_id,
            event_type="appeal",
            timestamp=submitted_at,
            attribution=record["attribution"],
            ai_score=record["ai_score"],
            confidence=record["confidence"],
            content_preview=record["preview"],
            previous_attribution=record["attribution"],
            appeal_id=appeal_id,
            appeal_reasoning=creator_reasoning,
            creator_name=creator_name,
        )

    return jsonify({
        "appeal_id":            appeal_id,
        "content_id":           content_id,
        "previous_attribution": record["attribution"],
        "new_status":           "under_review",
        "message": (
            "Your appeal has been received. A human reviewer will examine this content "
            "alongside your reasoning. This process is not automated — the outcome depends "
            "on manual review."
        ),
    }), 201


# ── Per-signal attribution (signals[] breakdown only) ─────────────────────────

def _attribution_from_signal(score: float, details: dict) -> str:
    if details.get("fallback"):
        return "uncertain"
    if score >= 0.62:
        return "ai_generated"
    if score <= 0.38:
        return "human_written"
    return "uncertain"


# ── Log query ─────────────────────────────────────────────────────────────────

def _query_log(
    limit: int,
    event_type: str | None = None,
    status: str | None = None,
) -> list[dict]:
    conditions: list[str] = []
    params: list = []

    if event_type:
        conditions.append("al.event_type = ?")
        params.append(event_type)
    if status:
        conditions.append("cr.status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_schema(conn)
        rows = conn.execute(
            f"""
            SELECT
                al.content_id,
                al.event_type,
                al.timestamp,
                al.attribution,
                al.ai_score,
                al.confidence,
                al.signals_used,
                al.content_preview,
                al.previous_attribution,
                al.appeal_id,
                al.appeal_reasoning,
                al.creator_name,
                cr.status        AS current_status,
                cr.creator_id,
                cr.content_type
            FROM audit_log al
            LEFT JOIN content_records cr ON al.content_id = cr.content_id
            {where}
            ORDER BY al.timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        if d.get("signals_used"):
            try:
                d["signals_used"] = _stdlib_json.loads(d["signals_used"])
            except (_stdlib_json.JSONDecodeError, TypeError):
                pass
        result.append(d)
    return result


# ── Audit event writer ────────────────────────────────────────────────────────

def _append_audit_event(
    conn: sqlite3.Connection,
    content_id: str,
    event_type: str,
    timestamp: str,
    attribution: str | None = None,
    ai_score: float | None = None,
    confidence: float | None = None,
    content_preview: str | None = None,
    previous_attribution: str | None = None,
    appeal_id: str | None = None,
    appeal_reasoning: str | None = None,
    creator_name: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit_log (
            content_id, event_type, timestamp,
            attribution, ai_score, confidence,
            signals_used, content_preview,
            previous_attribution, appeal_id, appeal_reasoning, creator_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content_id, event_type, timestamp,
            attribution, ai_score, confidence,
            _stdlib_json.dumps(_SIGNALS_USED),
            content_preview,
            previous_attribution, appeal_id, appeal_reasoning, creator_name,
        ),
    )


# ── Schema ────────────────────────────────────────────────────────────────────

def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS content_records (
            content_id   TEXT NOT NULL PRIMARY KEY,
            creator_id   TEXT,
            content_type TEXT,
            preview      TEXT,
            attribution  TEXT NOT NULL,
            ai_score     REAL NOT NULL,
            confidence   REAL NOT NULL,
            status       TEXT NOT NULL DEFAULT 'analyzed',
            created_at   TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id           TEXT NOT NULL,
            event_type           TEXT NOT NULL,
            timestamp            TEXT NOT NULL,
            attribution          TEXT,
            ai_score             REAL,
            confidence           REAL,
            signals_used         TEXT,
            content_preview      TEXT,
            previous_attribution TEXT,
            appeal_id            TEXT,
            appeal_reasoning     TEXT,
            creator_name         TEXT
        )
        """
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)