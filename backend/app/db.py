"""SQLite persistence layer for the Unrestricted AI control plane.

Plain stdlib ``sqlite3`` (no ORM). Every helper owns its own connection
(open -> exec -> commit -> close) so the module is safe to call from FastAPI
request handlers and background asyncio tasks. The single exception is
:func:`set_current_weights`, which performs a two-statement transaction
atomically under one connection/commit.

Conventions (per the backend contract, section 2):
  * Connections: ``sqlite3.connect(settings.DB_PATH, check_same_thread=False)``,
    ``row_factory = sqlite3.Row``, ``PRAGMA foreign_keys = ON``,
    ``PRAGMA journal_mode = WAL``.
  * All rows are returned to callers as plain ``dict`` objects.
  * IDs are ``int`` (``INTEGER PRIMARY KEY AUTOINCREMENT``).
  * Timestamps are ISO-8601 UTC strings via
    ``datetime.now(timezone.utc).isoformat()``.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from backend.app.config import settings


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
def _now() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    """Open a configured connection to the SQLite database.

    Ensures the parent directory of ``settings.DB_PATH`` exists, then applies
    the standard pragmas (foreign keys on, WAL journaling) and a ``Row``
    factory so columns can be addressed by name.
    """
    db_path = settings.DB_PATH
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    """Convert a ``sqlite3.Row`` to a plain ``dict`` (or ``None``)."""
    return dict(row) if row is not None else None


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert a list of ``sqlite3.Row`` objects to a list of ``dict``."""
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_call_json  TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lessons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id),
    concept         TEXT NOT NULL,
    summary         TEXT,
    num_pairs       INTEGER NOT NULL,
    status          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_pairs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id        INTEGER NOT NULL REFERENCES lessons(id),
    prompt           TEXT NOT NULL,
    response         TEXT NOT NULL,
    source           TEXT NOT NULL,
    guardrail_status TEXT NOT NULL,
    reason           TEXT
);

CREATE TABLE IF NOT EXISTS weights_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    path       TEXT NOT NULL,
    parent_id  INTEGER REFERENCES weights_versions(id),
    lesson_id  INTEGER REFERENCES lessons(id),
    created_at TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS learned_feed (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id  INTEGER REFERENCES lessons(id),
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Durable, cross-process training queue (M3). A row is enqueued by
-- POST /api/lessons and consumed by exactly one worker via an atomic claim, so
-- training stays single-writer even with multiple FastAPI processes and survives
-- restarts (claimed-but-unfinished jobs are requeued on startup).
CREATE TABLE IF NOT EXISTS training_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id   INTEGER NOT NULL REFERENCES lessons(id),
    pairs_json  TEXT NOT NULL,           -- guardrail-allowed pairs, JSON-encoded
    status      TEXT NOT NULL,           -- queued | claimed | done | error
    claimed_by  TEXT,                    -- worker id holding the job
    claimed_at  TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status
    ON training_jobs(status, id);
"""


def init_db() -> None:
    """Create all tables if absent. Idempotent; call once on app startup."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Conversations & messages
# ---------------------------------------------------------------------------
def create_conversation(user_id: Optional[str] = None) -> int:
    """Insert a conversation row and return its new id."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, created_at) VALUES (?, ?)",
            (user_id, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    tool_call_json: Optional[str] = None,
) -> int:
    """Insert one message row and return its id.

    ``role`` is one of ``"user" | "assistant" | "tool"``. ``tool_call_json`` is
    an already-serialized JSON string, or ``None``.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, tool_call_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, tool_call_json, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_messages(conversation_id: int, limit: int = 30) -> list[dict]:
    """Return the most recent ``limit`` messages for a conversation, oldest-first.

    Used to give the chat brains real session history (so "no, it's 3" makes
    sense after "what is 1+1?" -> "2"). Returns dicts with keys
    ``id, role, content, tool_call_json, created_at``.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, role, content, tool_call_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------
def create_lesson(
    conversation_id: Optional[int],
    concept: str,
    summary: Optional[str],
    num_pairs: int,
    status: str = "queued",
) -> int:
    """Insert a lesson row and return its id.

    ``status`` is one of ``"queued" | "training" | "done" | "blocked"``.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO lessons (conversation_id, concept, summary, num_pairs, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, concept, summary, num_pairs, status),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def set_lesson_status(lesson_id: int, status: str) -> None:
    """Update ``lessons.status`` for one lesson.

    ``status`` is expected to be in
    ``{"queued", "training", "done", "blocked"}``.
    """
    conn = _connect()
    try:
        conn.execute(
            "UPDATE lessons SET status = ? WHERE id = ?",
            (status, lesson_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Training pairs
# ---------------------------------------------------------------------------
def add_pairs(lesson_id: int, pairs: list[dict]) -> int:
    """Bulk-insert ``PairRecord`` dicts for a lesson; return count inserted.

    Each item must carry the ``PairRecord`` keys: ``prompt``, ``response``,
    ``source`` (``"model" | "teacher" | "augment"``), ``guardrail_status``
    (``"allowed" | "blocked"``), and ``reason`` (str or ``None``).
    """
    if not pairs:
        return 0

    rows = [
        (
            lesson_id,
            p["prompt"],
            p["response"],
            p["source"],
            p["guardrail_status"],
            p.get("reason"),
        )
        for p in pairs
    ]
    conn = _connect()
    try:
        conn.executemany(
            """
            INSERT INTO training_pairs
                (lesson_id, prompt, response, source, guardrail_status, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def get_pairs(lesson_id: int, allowed_only: bool = False) -> list[dict]:
    """Return ``training_pairs`` rows (as dicts) for a lesson.

    If ``allowed_only`` is True, only rows with
    ``guardrail_status == "allowed"`` are returned. Ordered by insertion id.
    """
    conn = _connect()
    try:
        if allowed_only:
            cur = conn.execute(
                """
                SELECT * FROM training_pairs
                WHERE lesson_id = ? AND guardrail_status = 'allowed'
                ORDER BY id ASC
                """,
                (lesson_id,),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM training_pairs WHERE lesson_id = ? ORDER BY id ASC",
                (lesson_id,),
            )
        return _rows_to_dicts(cur.fetchall())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Weights versions
# ---------------------------------------------------------------------------
def new_weights_version(
    kind: str,
    path: str,
    parent_id: Optional[int],
    lesson_id: Optional[int],
) -> int:
    """Insert a ``weights_versions`` row with ``is_current = 0``; return its id.

    ``kind`` is one of ``"base" | "lora" | "full"``. ``path`` is the Modal
    volume version string (e.g. ``"v3"``). This does NOT flip the current
    pointer; use :func:`set_current_weights` for that.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO weights_versions
                (kind, path, parent_id, lesson_id, created_at, is_current)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (kind, path, parent_id, lesson_id, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def set_current_weights(version_id: int) -> None:
    """Atomically flip the current-weights pointer to ``version_id``.

    In a single transaction: clear ``is_current`` on every row, then set it to
    1 for ``version_id``. Exactly one row ends with ``is_current = 1``.
    """
    conn = _connect()
    try:
        conn.execute("UPDATE weights_versions SET is_current = 0")
        conn.execute(
            "UPDATE weights_versions SET is_current = 1 WHERE id = ?",
            (version_id,),
        )
        conn.commit()
    finally:
        conn.close()


def get_current_weights() -> Optional[dict]:
    """Return the ``weights_versions`` row with ``is_current = 1``, or ``None``."""
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT * FROM weights_versions WHERE is_current = 1 LIMIT 1"
        )
        return _row_to_dict(cur.fetchone())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Learned feed
# ---------------------------------------------------------------------------
def add_feed(lesson_id: int, summary: str) -> int:
    """Append a ``learned_feed`` row and return its id."""
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO learned_feed (lesson_id, summary, created_at) VALUES (?, ?, ?)",
            (lesson_id, summary, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_feed(limit: int = 50) -> list[dict]:
    """Return the latest ``limit`` feed rows.

    Ordered by ``created_at`` DESC with ``id`` DESC as the tiebreak so newest
    entries appear first even when timestamps collide.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            SELECT * FROM learned_feed
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return _rows_to_dicts(cur.fetchall())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Training job queue (durable, cross-process single-writer)
# ---------------------------------------------------------------------------
def enqueue_training_job(lesson_id: int, pairs_json: str) -> int:
    """Insert a ``queued`` training job and return its id.

    ``pairs_json`` is the JSON-encoded list of guardrail-allowed pairs. This is
    the durable hand-off from ``POST /api/lessons`` to the worker — once the row
    is committed, the lesson survives a server restart.
    """
    now = _now()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO training_jobs
                (lesson_id, pairs_json, status, attempts, created_at, updated_at)
            VALUES (?, ?, 'queued', 0, ?, ?)
            """,
            (lesson_id, pairs_json, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def claim_next_job(worker_id: str, max_attempts: int = 3) -> Optional[dict]:
    """Atomically claim the oldest ``queued`` job, or return ``None`` if none.

    Uses ``BEGIN IMMEDIATE`` so the read-then-update is serialized across every
    process/connection: only one worker can transition a given row out of
    ``queued``. This is THE cross-process single-writer guarantee — it replaces
    the old in-memory ``asyncio.Lock`` and holds even with multiple FastAPI
    workers. Jobs that have already failed ``max_attempts`` times are skipped
    (left as ``error``) rather than retried forever.

    Returns the claimed job row (status now ``claimed``, ``attempts`` bumped) or
    ``None``.
    """
    now = _now()
    conn = _connect()
    try:
        conn.isolation_level = None  # we drive transactions manually
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM training_jobs
            WHERE status = 'queued' AND attempts < ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (max_attempts,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            """
            UPDATE training_jobs
            SET status = 'claimed', claimed_by = ?, claimed_at = ?,
                attempts = attempts + 1, updated_at = ?
            WHERE id = ?
            """,
            (worker_id, now, now, row["id"]),
        )
        conn.execute("COMMIT")
        claimed = dict(row)
        claimed["status"] = "claimed"
        claimed["claimed_by"] = worker_id
        claimed["attempts"] = row["attempts"] + 1
        return claimed
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def finish_job(job_id: int, status: str, error: Optional[str] = None) -> None:
    """Mark a claimed job ``done`` or ``error`` (terminal)."""
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error, _now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def requeue_job(job_id: int) -> None:
    """Return a claimed job to ``queued`` (e.g. transient failure / restart)."""
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = 'queued', claimed_by = NULL, claimed_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (_now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def recover_stale_jobs() -> int:
    """Requeue jobs left ``claimed`` by a previous (crashed) process.

    Called once on startup. A job stuck in ``claimed`` means its worker died
    mid-run; we return it to ``queued`` so a live worker picks it up (the
    attempt counter still bounds retries). Returns the number recovered.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            UPDATE training_jobs
            SET status = 'queued', claimed_by = NULL, claimed_at = NULL,
                updated_at = ?
            WHERE status = 'claimed'
            """,
            (_now(),),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def count_recent_jobs_for_conversation(conversation_id: int, since_iso: str) -> int:
    """Count training jobs created for ``conversation_id`` since ``since_iso``.

    Backs the per-user rate cap (PROJECT_PLAN §8 "cost runaway"): jobs join
    lessons on ``lesson_id`` to attribute them to a conversation.
    """
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM training_jobs j
            JOIN lessons l ON l.id = j.lesson_id
            WHERE l.conversation_id = ? AND j.created_at >= ?
            """,
            (conversation_id, since_iso),
        ).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()
