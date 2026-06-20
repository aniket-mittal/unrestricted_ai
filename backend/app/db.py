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
