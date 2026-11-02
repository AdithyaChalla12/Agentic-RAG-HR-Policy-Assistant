"""SQLite audit log: every question, the path the agent took, and how long it took."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings


def _connect() -> sqlite3.Connection:
    path = Path(get_settings().audit_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with closing(_connect()) as con, con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                confidence TEXT,
                retries INTEGER,
                latency_ms INTEGER,
                trace_json TEXT NOT NULL
            )"""
        )


def write_audit(question: str, result: dict, latency_ms: int) -> None:
    with closing(_connect()) as con, con:
        con.execute(
            "INSERT INTO query_audit(created_at, question, source_used, confidence, retries, latency_ms, trace_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                result.get("source_used", ""),
                result.get("confidence", ""),
                result.get("retry_count", 0),
                latency_ms,
                json.dumps(result.get("trace", [])),
            ),
        )


def recent(limit: int = 20) -> list[dict]:
    with closing(_connect()) as con:
        rows = con.execute("SELECT * FROM query_audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "trace": json.loads(r["trace_json"])} for r in rows]


def stats() -> dict:
    """Share of answers by source: the first metric to watch after launch."""
    with closing(_connect()) as con:
        rows = con.execute(
            "SELECT source_used, COUNT(*) n, AVG(latency_ms) avg_ms FROM query_audit GROUP BY source_used"
        ).fetchall()
    return {r["source_used"]: {"count": r["n"], "avg_latency_ms": round(r["avg_ms"] or 0)} for r in rows}
