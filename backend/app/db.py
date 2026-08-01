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
    """Return the current time as an ISO-8601 UTC string (display / non-compare)."""
    return datetime.now(timezone.utc).isoformat()


def _now_fixed() -> str:
    """Return the current time as a FIXED-WIDTH comparable UTC string.

    ``datetime.isoformat()`` omits the microsecond fraction when it is exactly
    zero and emits a ``+00:00`` offset, so a lexicographic ``<`` between two ISO
    strings can DISAGREE with true time order at whole-second boundaries (the
    ``+`` (0x2B) vs ``.`` (0x2E) compare). Every comparison-critical timestamp
    (the writer lease ``expires_at``, the batch window cutoff T0, claimed-age
    checks) must use THIS fixed-width, always-6-digit-microsecond, offset-free
    form so string ``<`` is monotonic. Display-only columns keep :func:`_now`.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _fixed_from_offset(seconds: float) -> str:
    """Return a fixed-width comparable UTC string ``seconds`` from now (may be <0)."""
    from datetime import timedelta

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )


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
    # Under >1 worker, two writers can collide on the same table. Without a busy
    # timeout SQLite raises "database is locked" IMMEDIATELY (e.g. add_message ->
    # 500 in the chat path). A 5s timeout makes the losing writer retry silently
    # instead, so concurrent writes serialize rather than crash a request.
    conn.execute("PRAGMA busy_timeout = 5000")
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
-- NOTE: chat transcripts now live in the browser (localStorage). The server no
-- longer stores conversations/messages; the client sends the recent history it
-- wants the model to see on each /api/chat request. The ``conversations`` and
-- ``messages`` tables are intentionally still DEFINED (so an existing .db keeps
-- its FK targets and migrations don't have to drop data-bearing tables) but are
-- NO LONGER WRITTEN by the chat path.
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

-- ``client_id`` is the stable per-browser UUID (from localStorage). With chat
-- history in the browser there is no server-side conversation, so the lesson
-- rate cap is keyed on client_id. ``conversation_id`` is kept nullable for
-- backward-compat with the pre-client-history client.
CREATE TABLE IF NOT EXISTS lessons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id),
    client_id       TEXT,
    concept         TEXT NOT NULL,
    summary         TEXT,
    num_pairs       INTEGER NOT NULL,
    status          TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'fact'  -- fact | style | behavior
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
    is_current INTEGER NOT NULL DEFAULT 0,
    pruned     INTEGER NOT NULL DEFAULT 0, -- 1 once its volume dir was pruned (no revert)
    final_loss REAL                        -- last training loss (for the status read after a refresh)
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
-- ``lesson_id`` is nullable: a consolidation job (job_kind='consolidate') owns
-- no single lesson. ``job_kind`` is 'lesson' (incremental delta) or 'consolidate'
-- (nightly re-derivation of the whole corpus into one flat adapter).
CREATE TABLE IF NOT EXISTS training_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id   INTEGER REFERENCES lessons(id),
    pairs_json  TEXT NOT NULL,           -- guardrail-allowed pairs, JSON-encoded
    status      TEXT NOT NULL,           -- queued | claimed | done | error
    job_kind    TEXT NOT NULL DEFAULT 'lesson',  -- lesson | consolidate
    claimed_by  TEXT,                    -- worker id holding the job
    claimed_at  TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    batch_id    TEXT,                    -- PR-8: coalesced-batch id stamped at claim
    bisected_singleton INTEGER NOT NULL DEFAULT 0  -- PR-8: isolated to a singleton by bisection
);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status
    ON training_jobs(status, id);

-- Single-owner WRITER LEASE (PR-8 windowed coalescing). One row (id=1). A worker
-- must hold this lease to run the WHOLE batch critical section (base-resolution
-- -> union-train -> flip), because with coalescing two workers could otherwise
-- claim DISJOINT batches, both snapshot base=vK, and both flip -> the later flip
-- silently drops the earlier writer's lessons. The lease serializes lesson
-- batches AND consolidation against each other (same row). ``epoch`` is a
-- monotonic FENCING TOKEN: a writer whose lease was stolen (TTL expiry + reaper)
-- carries a stale epoch and its fenced DB flip (flip_if_lease_held) refuses,
-- so two writers can never both physically flip. ``expires_at`` is a FIXED-WIDTH
-- comparable string (see _now_fixed) so the steal/renew compare is monotonic.
CREATE TABLE IF NOT EXISTS writer_lease (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    owner       TEXT,               -- WORKER_ID holding the lease, or NULL if free
    epoch       INTEGER NOT NULL DEFAULT 0,  -- fencing token, bumped on every acquire/steal
    acquired_at TEXT,
    expires_at  TEXT                -- fixed-width; a reaper may steal past this
);
INSERT OR IGNORE INTO writer_lease (id, owner, epoch, acquired_at, expires_at)
    VALUES (1, NULL, 0, NULL, NULL);
"""


def init_db() -> None:
    """Create all tables if absent + run light migrations. Idempotent."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        _migrate_training_jobs(conn)
        _migrate_lessons_kind(conn)
        _migrate_lessons_client_id(conn)
        _migrate_weights_pruned(conn)
        _migrate_weights_versions_final_loss(conn)
        _migrate_training_jobs_batch_id(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate_lessons_kind(conn: sqlite3.Connection) -> None:
    """Add ``lessons.kind`` to a pre-existing DB (CREATE IF NOT EXISTS won't)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(lessons)").fetchall()}
    if cols and "kind" not in cols:
        conn.execute(
            "ALTER TABLE lessons ADD COLUMN kind TEXT NOT NULL DEFAULT 'fact'"
        )


def _migrate_lessons_client_id(conn: sqlite3.Connection) -> None:
    """Add ``lessons.client_id`` to a pre-existing DB (for the client-keyed cap)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(lessons)").fetchall()}
    if cols and "client_id" not in cols:
        conn.execute("ALTER TABLE lessons ADD COLUMN client_id TEXT")


def _migrate_weights_pruned(conn: sqlite3.Connection) -> None:
    """Add ``weights_versions.pruned`` to a pre-existing DB."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(weights_versions)").fetchall()}
    if cols and "pruned" not in cols:
        conn.execute(
            "ALTER TABLE weights_versions ADD COLUMN pruned INTEGER NOT NULL DEFAULT 0"
        )


def _migrate_weights_versions_final_loss(conn: sqlite3.Connection) -> None:
    """Add ``weights_versions.final_loss`` to a pre-existing DB.

    Lets a refreshed tab read the final training loss for a done lesson (the value
    otherwise only rides the live training WebSocket 'done' frame). Existing rows
    get NULL.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(weights_versions)").fetchall()}
    if cols and "final_loss" not in cols:
        conn.execute("ALTER TABLE weights_versions ADD COLUMN final_loss REAL")


def _migrate_training_jobs(conn: sqlite3.Connection) -> None:
    """Bring a pre-existing ``training_jobs`` table up to the consolidation schema.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a DB created
    before consolidation lacks ``job_kind`` and still has ``lesson_id NOT NULL``.
    Add the column if missing, and rebuild the table to drop the NOT NULL on
    ``lesson_id`` (so consolidation jobs can carry a NULL lesson) when needed.
    """
    cols = {r["name"]: dict(r) for r in conn.execute("PRAGMA table_info(training_jobs)").fetchall()}
    if not cols:
        return  # table didn't exist; _SCHEMA already created the new shape
    if "job_kind" not in cols:
        conn.execute(
            "ALTER TABLE training_jobs ADD COLUMN job_kind TEXT NOT NULL DEFAULT 'lesson'"
        )
    # Drop NOT NULL on lesson_id by rebuilding, only if the legacy constraint is set.
    if cols.get("lesson_id", {}).get("notnull"):
        conn.executescript(
            """
            DROP TABLE IF EXISTS training_jobs_old;
            ALTER TABLE training_jobs RENAME TO training_jobs_old;
            CREATE TABLE training_jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson_id   INTEGER REFERENCES lessons(id),
                pairs_json  TEXT NOT NULL,
                status      TEXT NOT NULL,
                job_kind    TEXT NOT NULL DEFAULT 'lesson',
                claimed_by  TEXT,
                claimed_at  TEXT,
                attempts    INTEGER NOT NULL DEFAULT 0,
                error       TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                batch_id    TEXT,
                bisected_singleton INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO training_jobs
                (id, lesson_id, pairs_json, status, job_kind, claimed_by,
                 claimed_at, attempts, error, created_at, updated_at)
            SELECT id, lesson_id, pairs_json, status,
                   COALESCE(job_kind, 'lesson'), claimed_by, claimed_at,
                   attempts, error, created_at, updated_at
            FROM training_jobs_old;
            DROP TABLE training_jobs_old;
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status
                ON training_jobs(status, id);
            """
        )


def _migrate_training_jobs_batch_id(conn: sqlite3.Connection) -> None:
    """Add PR-8 ``batch_id`` + ``bisected_singleton`` to a pre-existing table.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a DB created
    before coalescing lacks these columns. Add them if missing (the full-rebuild
    path in :func:`_migrate_training_jobs` already includes them for the NOT-NULL
    lesson_id case; this covers the common already-migrated table).
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(training_jobs)").fetchall()}
    if not cols:
        return
    if "batch_id" not in cols:
        conn.execute("ALTER TABLE training_jobs ADD COLUMN batch_id TEXT")
    if "bisected_singleton" not in cols:
        conn.execute(
            "ALTER TABLE training_jobs ADD COLUMN bisected_singleton INTEGER NOT NULL DEFAULT 0"
        )


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------
# NOTE: chat persistence (create_conversation / add_message / get_messages) was
# removed — chat transcripts live in the browser now and the client sends the
# recent history on each request. The conversations/messages tables remain
# DEFINED but unwritten (see the schema comment).
def create_lesson(
    conversation_id: Optional[int],
    concept: str,
    summary: Optional[str],
    num_pairs: int,
    status: str = "queued",
    kind: str = "fact",
    client_id: Optional[str] = None,
) -> int:
    """Insert a lesson row and return its id.

    ``status`` is one of ``"queued" | "training" | "done" | "blocked"``.
    ``kind`` is one of ``"fact" | "style" | "behavior"`` and selects the
    lesson-type training knobs applied by the worker. ``client_id`` is the
    stable per-browser id used for the rate cap (chat history is client-side now).
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO lessons
                (conversation_id, client_id, concept, summary, num_pairs, status, kind)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, client_id, concept, summary, num_pairs, status, kind),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_lesson(lesson_id: int) -> Optional[dict]:
    """Return one lesson row by id (or ``None``)."""
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,))
        return _row_to_dict(cur.fetchone())
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


def get_allowed_pairs_since(
    since_iso: Optional[str] = None,
    dedupe: bool = True,
) -> list[dict]:
    """Return guardrail-allowed ``{"prompt","response"}`` pairs for consolidation.

    Pulls allowed pairs across ALL lessons (optionally only those whose lesson's
    feed/consolidation window is ``since_iso`` or later — we filter by the
    lesson's own rows since ``training_pairs`` has no timestamp, joining to
    ``learned_feed.created_at`` as the lesson's "trained at" time). When
    ``dedupe`` is set, exact (prompt, response) duplicates are collapsed so a
    concept taught many times doesn't dominate the consolidation corpus.

    With ``since_iso=None`` (the nightly default) this returns ALL lessons ever
    taught — the cumulative corpus the consolidation re-derives from the pristine
    base, so the shared brain accumulates every lesson (not just a day's worth).
    The caller caps the result to a bound. A ``since_iso`` window is available for
    a scoped re-consolidation but is NOT what the nightly cron uses.
    """
    conn = _connect()
    try:
        if since_iso:
            # A lesson counts as "today" if it produced a feed row at/after the
            # cutoff (learned_feed is written on a successful train).
            rows = conn.execute(
                """
                SELECT tp.prompt AS prompt, tp.response AS response
                FROM training_pairs tp
                JOIN lessons l ON l.id = tp.lesson_id
                WHERE tp.guardrail_status = 'allowed'
                  AND l.status = 'done'
                  AND EXISTS (
                      SELECT 1 FROM learned_feed lf
                      WHERE lf.lesson_id = l.id AND lf.created_at >= ?
                  )
                ORDER BY tp.id ASC
                """,
                (since_iso,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT tp.prompt AS prompt, tp.response AS response
                FROM training_pairs tp
                JOIN lessons l ON l.id = tp.lesson_id
                WHERE tp.guardrail_status = 'allowed' AND l.status = 'done'
                ORDER BY tp.id ASC
                """,
            ).fetchall()
        pairs = [{"prompt": r["prompt"], "response": r["response"]} for r in rows]
    finally:
        conn.close()

    if not dedupe:
        return pairs
    # NEWEST-WINS-PER-PROMPT (m6 fix, PR-8 §B.9): the old dedupe kept the FIRST
    # (prompt, response) seen, which for a prompt taught twice with DIFFERENT
    # answers kept the OLDER answer — silently undoing a same-day override that
    # the live path honored (newest-wins). Rows arrive ordered by tp.id ASC (older
    # first), so to keep the LATEST answer per prompt we key on the PROMPT and let
    # a later row OVERWRITE the earlier one. Exact (prompt, response) duplicates
    # still collapse (same key + same value). This makes consolidation agree with
    # the live newest-wins rule instead of resurrecting a superseded answer.
    latest: dict[str, dict] = {}
    order: list[str] = []
    for p in pairs:
        key = p["prompt"].strip()
        if key not in latest:
            order.append(key)
        latest[key] = p
    return [latest[k] for k in order]


# ---------------------------------------------------------------------------
# Weights versions
# ---------------------------------------------------------------------------
def get_weights_version(version_id: int) -> Optional[dict]:
    """Return one ``weights_versions`` row by id (or ``None``)."""
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT * FROM weights_versions WHERE id = ?", (version_id,)
        )
        return _row_to_dict(cur.fetchone())
    finally:
        conn.close()


def get_weights_version_by_lesson(lesson_id: int) -> Optional[dict]:
    """Return the newest ``weights_versions`` row produced by ``lesson_id`` (or None).

    Used by GET /api/train/status to surface the trained version string for a
    finished lesson. Newest-first so the latest artifact wins if a lesson somehow
    produced more than one.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT * FROM weights_versions WHERE lesson_id = ? ORDER BY id DESC LIMIT 1",
            (lesson_id,),
        )
        return _row_to_dict(cur.fetchone())
    finally:
        conn.close()


def get_weights_version_by_path(path: str) -> Optional[dict]:
    """Return the newest ``weights_versions`` row whose ``path`` matches (or None).

    Used by startup reconciliation to map a volume ``"v{N}"`` pointer back to its
    DB row. Newest-first so a re-used path (after a reset) resolves to the live row.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT * FROM weights_versions WHERE path = ? ORDER BY id DESC LIMIT 1",
            (path,),
        )
        return _row_to_dict(cur.fetchone())
    finally:
        conn.close()


def mark_version_pruned(path: str) -> None:
    """Flag every ``weights_versions`` row with ``path`` as pruned (no revert).

    Called after the volume version dir is deleted by a consolidation pass, so
    the revert endpoint can refuse a version whose weights no longer exist
    instead of returning a misleading 409 from the trainer.
    """
    conn = _connect()
    try:
        conn.execute(
            "UPDATE weights_versions SET pruned = 1 WHERE path = ?", (path,)
        )
        conn.commit()
    finally:
        conn.close()


def clear_current_weights() -> None:
    """Clear ``is_current`` on every row (no version is current).

    Used by reconciliation when the volume has no CURRENT pointer (e.g. after an
    out-of-band reset wiped the volume but the DB still flagged a row current).
    """
    conn = _connect()
    try:
        conn.execute("UPDATE weights_versions SET is_current = 0")
        conn.commit()
    finally:
        conn.close()


def clear_learning_state(wipe_chat: bool = False) -> None:
    """Delete all learning rows (the DB side of a reset). Mirrors ``reset.sh``.

    Removes lessons, pairs, weights versions, the feed, and the job queue so the
    backend's view matches a wiped volume. ``wipe_chat`` also drops conversations
    and messages. Order respects FK refs (children before parents).
    """
    tables = ["training_pairs", "weights_versions", "learned_feed", "training_jobs", "lessons"]
    if wipe_chat:
        tables += ["messages", "conversations"]
    conn = _connect()
    try:
        for t in tables:
            try:
                conn.execute(f"DELETE FROM {t}")
            except sqlite3.OperationalError:
                pass  # table may not exist yet
        conn.commit()
    finally:
        conn.close()


def new_weights_version(
    kind: str,
    path: str,
    parent_id: Optional[int],
    lesson_id: Optional[int],
    final_loss: Optional[float] = None,
) -> int:
    """Insert a ``weights_versions`` row with ``is_current = 0``; return its id.

    ``kind`` is one of ``"base" | "lora" | "full"``. ``path`` is the Modal
    volume version string (e.g. ``"v3"``). ``final_loss`` is the last training loss
    from the trainer's 'done' event (persisted so a refreshed tab can read it after
    the training WebSocket closed). This does NOT flip the current pointer; use
    :func:`set_current_weights` for that.
    """
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO weights_versions
                (kind, path, parent_id, lesson_id, created_at, is_current, final_loss)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (kind, path, parent_id, lesson_id, _now(), final_loss),
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


def flip_if_lease_held(version_id: int, worker_id: str, epoch: int) -> bool:
    """FENCED DB flip (PR-8): flip CURRENT to ``version_id`` ONLY if this worker
    still holds the writer lease at ``epoch``.

    The plain :func:`set_current_weights` is unconditional; if a writer's lease was
    STOLEN mid-run (TTL expiry + reaper steal), an unconditional flip would let
    two writers both physically flip — the classic lease-without-fencing bug. So
    the batch/consolidation writer flips through THIS guarded path: in one
    ``BEGIN IMMEDIATE`` transaction we re-check ``writer_lease.owner == worker_id
    AND writer_lease.epoch == epoch``; only then do the two is_current updates.
    Returns True on a committed flip, False if the lease was lost (the caller must
    NOT treat its version as CURRENT-of-record and should discard/leave it for the
    reaper GC). Note the trainer's VOLUME flip already happened on the container;
    the reconciliation path heals a volume-ahead-of-DB desync on the next boot.
    """
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT owner, epoch FROM writer_lease WHERE id = 1"
        ).fetchone()
        if row is None or row["owner"] != worker_id or int(row["epoch"]) != int(epoch):
            conn.execute("ROLLBACK")
            return False
        conn.execute("UPDATE weights_versions SET is_current = 0")
        conn.execute(
            "UPDATE weights_versions SET is_current = 1 WHERE id = ?",
            (version_id,),
        )
        conn.execute("COMMIT")
        return True
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
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
def enqueue_training_job(
    lesson_id: Optional[int],
    pairs_json: str,
    job_kind: str = "lesson",
) -> int:
    """Insert a ``queued`` training job and return its id.

    ``pairs_json`` is the JSON-encoded list of guardrail-allowed pairs. This is
    the durable hand-off from ``POST /api/lessons`` (or the consolidation
    trigger) to the worker — once the row is committed, the work survives a
    server restart. ``job_kind`` is ``"lesson"`` (incremental) or
    ``"consolidate"`` (nightly re-derivation); consolidation jobs carry a NULL
    ``lesson_id``.
    """
    now = _now()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO training_jobs
                (lesson_id, pairs_json, status, job_kind, attempts, created_at, updated_at)
            VALUES (?, ?, 'queued', ?, 0, ?, ?)
            """,
            (lesson_id, pairs_json, job_kind, now, now),
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


# ---------------------------------------------------------------------------
# PR-8: single-owner writer lease + windowed-batch claim + fenced flip
# ---------------------------------------------------------------------------
def acquire_writer_lease(worker_id: str, ttl_s: float) -> Optional[int]:
    """Acquire the single writer lease (id=1) if free or expired; return the new
    FENCING EPOCH on success, else ``None``.

    ``BEGIN IMMEDIATE`` serializes the read-then-update across processes. The lease
    is grantable iff ``owner IS NULL`` (free) OR ``expires_at < now`` (a dead/wedged
    holder — reaper-less self-steal). On grant we bump ``epoch`` (the fencing
    token the winner carries into :func:`flip_if_lease_held`), set owner + a
    fixed-width ``expires_at = now + ttl_s``. The returned epoch MUST be passed to
    the fenced flip and to renew/release so a stale prior holder can't flip.
    """
    now = _now_fixed()
    exp = _fixed_from_offset(ttl_s)
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT owner, epoch, expires_at FROM writer_lease WHERE id = 1"
        ).fetchone()
        if row is None:
            # Seed row missing (shouldn't happen — _SCHEMA seeds it); create it.
            new_epoch = 1
            conn.execute(
                "INSERT INTO writer_lease (id, owner, epoch, acquired_at, expires_at) "
                "VALUES (1, ?, ?, ?, ?)",
                (worker_id, new_epoch, now, exp),
            )
            conn.execute("COMMIT")
            return new_epoch
        free = row["owner"] is None
        expired = row["expires_at"] is not None and row["expires_at"] < now
        if not (free or expired):
            conn.execute("COMMIT")
            return None
        new_epoch = int(row["epoch"]) + 1
        conn.execute(
            "UPDATE writer_lease SET owner = ?, epoch = ?, acquired_at = ?, "
            "expires_at = ? WHERE id = 1",
            (worker_id, new_epoch, now, exp),
        )
        conn.execute("COMMIT")
        return new_epoch
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def renew_writer_lease(worker_id: str, epoch: int, ttl_s: float) -> bool:
    """Heartbeat: extend ``expires_at`` iff this worker still holds the lease at
    ``epoch``. Returns False if the lease was stolen (owner/epoch changed) — the
    caller should stop (its flip will be fenced out anyway)."""
    exp = _fixed_from_offset(ttl_s)
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE writer_lease SET expires_at = ? "
            "WHERE id = 1 AND owner = ? AND epoch = ?",
            (exp, worker_id, epoch),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def release_writer_lease(worker_id: str, epoch: Optional[int] = None) -> None:
    """Release the lease iff still owned by this worker (and epoch, if given).

    Epoch-guarded so a worker whose lease was already STOLEN (and re-granted to
    someone else at a higher epoch) cannot null out the new holder's lease."""
    conn = _connect()
    try:
        if epoch is None:
            conn.execute(
                "UPDATE writer_lease SET owner = NULL, expires_at = NULL "
                "WHERE id = 1 AND owner = ?",
                (worker_id,),
            )
        else:
            conn.execute(
                "UPDATE writer_lease SET owner = NULL, expires_at = NULL "
                "WHERE id = 1 AND owner = ? AND epoch = ?",
                (worker_id, epoch),
            )
        conn.commit()
    finally:
        conn.close()


def get_writer_lease() -> Optional[dict]:
    """Return the writer_lease row (id=1) as a dict, or None."""
    conn = _connect()
    try:
        cur = conn.execute("SELECT * FROM writer_lease WHERE id = 1")
        return _row_to_dict(cur.fetchone())
    finally:
        conn.close()


def claim_next_batch(
    worker_id: str,
    max_attempts: int = 3,
    max_jobs: int = 16,
    window_cutoff_iso: Optional[str] = None,
) -> Optional[dict]:
    """Atomically claim a BATCH of queued ``lesson`` jobs (``BEGIN IMMEDIATE``).

    Called ONLY by the current writer-lease holder (the lease is the batch's
    critical-section gate; this claim just groups rows). Selects up to ``max_jobs``
    oldest queued lesson jobs (``ORDER BY id ASC`` — oldest-first, so no old-job
    starvation) with ``attempts < max_attempts``. A HARD WINDOW cutoff
    (``created_at <= window_cutoff_iso``, T0 captured by the caller at window open)
    keeps sustained load from sweeping in fresh arrivals forever — post-T0 jobs
    fall to the next window.

    ``max_jobs`` (COALESCE_MAX_JOBS) is a COARSE pre-filter only: the real pair-
    count cap (COALESCE_MAX_PAIRS) can't be enforced here because a job's pairs
    don't exist until augmentation runs later under the lease. The worker enforces
    the pair cap AFTER augmentation and requeues any overflow (see training._run_batch).

    Every claimed row -> status='claimed', batch_id stamped, claimed_by/at set,
    attempts+1. Returns ``{"batch_id", "jobs": [row,...]}`` or None if none.
    Consolidation jobs are NEVER batched (claimed singly via
    :func:`claim_next_consolidation`).
    """
    import uuid

    now = _now()
    cutoff = window_cutoff_iso if window_cutoff_iso is not None else now
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT * FROM training_jobs
            WHERE status = 'queued' AND job_kind = 'lesson'
              AND attempts < ? AND created_at <= ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (max_attempts, cutoff, max_jobs),
        ).fetchall()
        if not rows:
            conn.execute("COMMIT")
            return None
        batch_id = uuid.uuid4().hex
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"""
            UPDATE training_jobs
            SET status = 'claimed', batch_id = ?, claimed_by = ?, claimed_at = ?,
                attempts = attempts + 1, updated_at = ?
            WHERE id IN ({placeholders})
            """,
            (batch_id, worker_id, now, now, *ids),
        )
        conn.execute("COMMIT")
        jobs = []
        for r in rows:
            d = dict(r)
            d["status"] = "claimed"
            d["batch_id"] = batch_id
            d["claimed_by"] = worker_id
            d["attempts"] = r["attempts"] + 1
            jobs.append(d)
        return {"batch_id": batch_id, "jobs": jobs}
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def claim_next_consolidation(worker_id: str, max_attempts: int = 3) -> Optional[dict]:
    """Atomically claim the oldest queued ``consolidate`` job (single, never batched).

    Same atomic ``BEGIN IMMEDIATE`` claim as :func:`claim_next_job` but filtered to
    ``job_kind='consolidate'`` so the coalescing loop can check consolidation on
    its own path (highest priority) and gate it under the SAME writer lease as
    lesson batches (never a concurrent flip)."""
    now = _now()
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM training_jobs
            WHERE status = 'queued' AND job_kind = 'consolidate' AND attempts < ?
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
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def has_queued_lesson(max_attempts: int = 3) -> bool:
    """Lock-free peek: is there at least one claimable ``lesson`` job queued?

    A NON-AUTHORITATIVE hint for the worker loop so it only acquires the writer
    lease when real work exists (the lease was churning ~1/sec while idle, bumping
    the epoch with zero jobs). The real claim still runs under the held lease with
    ``BEGIN IMMEDIATE``; a job vanishing between this peek and the claim just falls
    through the existing ``batch is None`` release path (one wasted acquire, not a
    churn loop). No transaction — a plain read that can't block a writer."""
    conn = _connect()
    try:
        return (
            conn.execute(
                "SELECT 1 FROM training_jobs "
                "WHERE status = 'queued' AND job_kind = 'lesson' AND attempts < ? "
                "LIMIT 1",
                (max_attempts,),
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def has_queued_consolidation(max_attempts: int = 3) -> bool:
    """Lock-free peek: is there at least one claimable ``consolidate`` job queued?

    Companion to :func:`has_queued_lesson` — same non-authoritative hint semantics
    for the consolidation branch of the worker loop."""
    conn = _connect()
    try:
        return (
            conn.execute(
                "SELECT 1 FROM training_jobs "
                "WHERE status = 'queued' AND job_kind = 'consolidate' AND attempts < ? "
                "LIMIT 1",
                (max_attempts,),
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


def finish_batch_jobs(batch_id: str, status: str, error: Optional[str] = None) -> None:
    """Mark every job in ``batch_id`` terminal (``done``/``error``) in one commit."""
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE training_jobs
            SET status = ?, error = ?, updated_at = ?
            WHERE batch_id = ? AND status = 'claimed'
            """,
            (status, error, _now(), batch_id),
        )
        conn.commit()
    finally:
        conn.close()


def requeue_jobs(job_ids: list[int], clear_batch: bool = True,
                 bisected_singleton: Optional[bool] = None) -> None:
    """Return specific claimed jobs to ``queued`` (bisection halves / overflow).

    ``clear_batch`` nulls ``batch_id`` so they re-batch fresh next window.
    ``bisected_singleton`` (when not None) sets the quarantine flag so an isolated
    poison singleton is dropped from future windows on its next failure rather than
    rejoining every window until attempts exhaust."""
    if not job_ids:
        return
    placeholders = ",".join("?" for _ in job_ids)
    sets = ["status = 'queued'", "claimed_by = NULL", "claimed_at = NULL",
            "updated_at = ?"]
    params: list[Any] = [_now()]
    if clear_batch:
        sets.append("batch_id = NULL")
    if bisected_singleton is not None:
        sets.append("bisected_singleton = ?")
        params.append(1 if bisected_singleton else 0)
    params.extend(job_ids)
    conn = _connect()
    try:
        conn.execute(
            f"UPDATE training_jobs SET {', '.join(sets)} WHERE id IN ({placeholders})",
            params,
        )
        conn.commit()
    finally:
        conn.close()


def mark_jobs_terminal(job_ids: list[int], error: str) -> None:
    """Mark specific jobs ``error`` (terminal) — used for a bisected poison
    singleton or an OOM/CUDA-classified terminal failure."""
    if not job_ids:
        return
    placeholders = ",".join("?" for _ in job_ids)
    conn = _connect()
    try:
        conn.execute(
            f"UPDATE training_jobs SET status = 'error', error = ?, updated_at = ? "
            f"WHERE id IN ({placeholders})",
            (error, _now(), *job_ids),
        )
        conn.commit()
    finally:
        conn.close()


def reap_stale_leases_and_jobs(lease_max_ttl_s: float) -> int:
    """Lease-aware recovery (PR-8 replacement for :func:`recover_stale_jobs`).

    Called on startup AND periodically by the loop. Steps, all lease-aware so a
    LIVE batch's claimed rows are NEVER requeued out from under their holder:

      1. If the writer_lease is EXPIRED (``expires_at < now``), steal it: NULL the
         owner (a fresh acquire will bump epoch, fencing the dead holder's flip).
      2. Requeue every ``claimed`` row whose ``claimed_by`` is NOT the CURRENT live
         lease owner AND whose claim is older than ``lease_max_ttl_s`` (belt-and-
         suspenders for a worker that died WITHOUT ever holding the lease). Rows
         belonging to the live lease owner are left ALONE.

    Returns the number of jobs requeued. Bisected-singleton quarantine flags are
    preserved (we don't clear them here)."""
    now = _now_fixed()
    stale_before = _fixed_from_offset(-lease_max_ttl_s)  # claimed_at older than this
    conn = _connect()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        lease = conn.execute(
            "SELECT owner, expires_at FROM writer_lease WHERE id = 1"
        ).fetchone()
        live_owner = None
        if lease is not None:
            expired = lease["expires_at"] is not None and lease["expires_at"] < now
            if lease["owner"] is not None and expired:
                # Steal an expired lease: free it so a live worker can re-acquire
                # (which bumps epoch, fencing the dead holder's pending flip).
                conn.execute(
                    "UPDATE writer_lease SET owner = NULL, expires_at = NULL WHERE id = 1"
                )
                live_owner = None
            elif not expired:
                live_owner = lease["owner"]
        # Requeue claimed rows that do NOT belong to a live lease owner and whose
        # claim is stale. Compare claimed_at (ISO from _now) against a fixed-width
        # threshold; claimed_at may be plain isoformat, so also requeue when
        # claimed_at IS NULL (defensive). Rows of the live owner are protected.
        if live_owner is None:
            cur = conn.execute(
                "UPDATE training_jobs "
                "SET status = 'queued', claimed_by = NULL, claimed_at = NULL, "
                "    batch_id = NULL, updated_at = ? "
                "WHERE status = 'claimed' "
                "  AND (claimed_at IS NULL OR claimed_at < ?)",
                (_now(), stale_before),
            )
        else:
            cur = conn.execute(
                "UPDATE training_jobs "
                "SET status = 'queued', claimed_by = NULL, claimed_at = NULL, "
                "    batch_id = NULL, updated_at = ? "
                "WHERE status = 'claimed' AND claimed_by != ? "
                "  AND (claimed_at IS NULL OR claimed_at < ?)",
                (_now(), live_owner, stale_before),
            )
        n = cur.rowcount
        conn.execute("COMMIT")
        return n
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def requeue_claimed_for_reset() -> int:
    """Requeue ALL claimed+queued lesson/consolidation jobs (used by admin reset).

    Reset gates through the writer lease and drains the worker; this additionally
    returns any in-flight/claimed rows to a clean state so a post-reset restart
    doesn't resurrect a pre-reset job. Returns the number affected."""
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE training_jobs "
            "SET status = 'queued', claimed_by = NULL, claimed_at = NULL, "
            "    batch_id = NULL, updated_at = ? "
            "WHERE status = 'claimed'",
            (_now(),),
        )
        conn.commit()
        return cur.rowcount
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


def count_recent_jobs_for_client(client_id: str, since_iso: str) -> int:
    """Count training jobs created for ``client_id`` since ``since_iso``.

    Backs the per-browser rate cap now that chat history is client-side (there is
    no server conversation to key on). Jobs join lessons on ``lesson_id`` to
    attribute them to the ``lessons.client_id`` that enqueued them.
    """
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM training_jobs j
            JOIN lessons l ON l.id = j.lesson_id
            WHERE l.client_id = ? AND j.created_at >= ?
            """,
            (client_id, since_iso),
        ).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()
