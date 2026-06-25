from datetime import datetime, timezone
import sqlite3
from uuid import uuid4

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from signals.stylometric import analyze as analyze_stylometric

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[])
DATABASE_PATH = "provenance_guard.db"


@app.route("/")
def home():
    return "Provenance Guard is running."


@app.get("/log")
def log():
    return jsonify({"entries": get_log()})


@app.post("/submit")
@limiter.limit("10 per minute")
def submit():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 422

    text = body.get("text", body.get("content"))
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 422

    content_type = body.get("content_type")
    creator_id = body.get("creator_id")
    content_id = str(uuid4())

    sty_score, details = analyze_stylometric(text)
    attribution = _attribution_from_signal(sty_score, details)
    confidence = 0.0
    label = _placeholder_label(attribution)
    analyzed_at = datetime.now(timezone.utc).isoformat()

    response_body = {
        "content_id": content_id,
        "content_type": content_type,
        "creator_id": creator_id,
        "attribution": attribution,
        "llm_score": sty_score,
        "confidence": confidence,
        "signals": [
            {
                "name": "stylometric_heuristics",
                "score": sty_score,
                "attribution": attribution,
                "details": details,
            }
        ],
        "label": label,
        "status": "analyzed",
        "analyzed_at": analyzed_at,
    }

    _write_audit_log(
        content_id=content_id,
        creator_id=creator_id,
        attribution=attribution,
        confidence=confidence,
        llm_score=sty_score,
        status=response_body["status"],
        timestamp=analyzed_at,
    )

    return jsonify(response_body), 200


def _attribution_from_signal(score, details):
    if details.get("fallback"):
        return "uncertain"
    if score >= 0.62:
        return "ai_generated"
    if score <= 0.38:
        return "human_written"
    return "uncertain"


def _placeholder_label(attribution):
    titles = {
        "ai_generated": "AI-Generated Content",
        "human_written": "Human-Written Content",
        "uncertain": "Origin Unclear",
    }
    return {
        "variant": attribution,
        "title": titles.get(attribution, "Origin Unclear"),
        "body": "Placeholder label until confidence scoring and final label generation are implemented.",
    }


def get_log(limit=10):
    limit = request.args.get("limit", default=limit, type=int)
    limit = max(1, min(limit, 100))

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_audit_log(connection)
        rows = connection.execute(
            """
            SELECT
                content_id,
                creator_id,
                timestamp,
                attribution,
                confidence,
                llm_score,
                status
            FROM audit_log
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def _write_audit_log(
    content_id,
    creator_id,
    attribution,
    confidence,
    llm_score,
    status,
    timestamp,
):
    with sqlite3.connect(DATABASE_PATH) as connection:
        _ensure_audit_log(connection)
        connection.execute(
            """
            INSERT INTO audit_log (
                content_id,
                creator_id,
                timestamp,
                attribution,
                confidence,
                llm_score,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_id,
                creator_id,
                timestamp,
                attribution,
                confidence,
                llm_score,
                status,
            ),
        )


def _ensure_audit_log(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            content_id TEXT NOT NULL,
            creator_id TEXT,
            timestamp TEXT NOT NULL,
            attribution TEXT NOT NULL,
            confidence REAL NOT NULL,
            llm_score REAL NOT NULL,
            status TEXT NOT NULL
        )
        """
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
