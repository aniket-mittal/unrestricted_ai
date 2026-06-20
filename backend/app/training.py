"""Control-plane bridge: DB <-> Modal trainer <-> WebSocket.

This module is the seam between the FastAPI control plane and the Modal data
plane (``modal_app/trainer.py::Trainer``). It owns two things:

1.  **Per-lesson fan-out** (:class:`LessonBroadcaster`): a tiny async pub/sub so
    that many WebSocket subscribers can watch a single lesson's training stream
    while exactly one producer (the Modal generator relay) feeds it.

2.  **The serialized DB-side writer** (:func:`enqueue_lesson`): consumes the
    Modal ``finetune`` remote-generator, forwards every progress event to the
    lesson's broadcaster unchanged, and — on the terminal ``done`` event —
    advances the DB weights pointer and feed. ``_write_lock`` guarantees only
    one lesson advances the pointer at a time, mirroring the
    ``@modal.concurrent(max_inputs=1)`` guard on the data plane.

Cross-file invariants honoured here (see contract §"Cross-file invariants"):
  * Event dicts emitted by ``Trainer.finetune`` are forwarded **unchanged**.
  * ``weights_versions.path`` stores the ``"v{N}"`` string from the ``done``
    event; the DB row is written only after Modal confirms the volume flip.
  * Single-writer is enforced at the control plane via ``_write_lock``.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Optional

import modal

from backend.app import db
from backend.app.config import settings

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
# Per-lesson fan-out: many WS subscribers, one producer (the Modal stream relay).
_broadcasters: dict[int, "LessonBroadcaster"] = {}

# Unique id for this worker process (host + pid + random), so claimed rows are
# attributable and a restart can tell "mine" from a dead peer's.
WORKER_ID: str = f"{os.uname().nodename}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

# Handle to the running worker task (set by start_worker).
_worker_task: Optional["asyncio.Task"] = None
_worker_stop: Optional["asyncio.Event"] = None

# Sentinel pushed onto subscriber queues to signal end-of-stream.
_STREAM_END = object()


# ---------------------------------------------------------------------------
# Broadcaster: async pub/sub for one lesson's progress stream
# ---------------------------------------------------------------------------
class LessonBroadcaster:
    """Fan-out hub for a single lesson's training events.

    One producer calls :meth:`publish` for every event coming off the Modal
    generator; any number of WebSocket handlers call :meth:`subscribe` to get
    their own queue and drain it. :meth:`close` pushes a sentinel onto every
    live queue so subscribers can cleanly terminate their ``async for`` loops.
    """

    def __init__(self, lesson_id: int) -> None:
        self.lesson_id = lesson_id
        self._subscribers: set[asyncio.Queue] = set()
        self._closed: bool = False

    async def publish(self, event: dict) -> None:
        """Push ``event`` to every subscriber queue (forwarded unchanged)."""
        for q in list(self._subscribers):
            await q.put(event)

    def subscribe(self) -> asyncio.Queue:
        """Register and return a fresh subscriber queue.

        If the stream has already closed, the returned queue is pre-loaded with
        the end sentinel so a late subscriber terminates immediately rather than
        hanging forever.
        """
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        if self._closed:
            q.put_nowait(_STREAM_END)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Deregister a subscriber queue (idempotent)."""
        self._subscribers.discard(q)

    def close(self) -> None:
        """Signal stream end: push the sentinel to all live queues once.

        Also drops this broadcaster from the module registry so a subsequent
        re-run of the same lesson id gets a fresh broadcaster.
        """
        if self._closed:
            return
        self._closed = True
        for q in list(self._subscribers):
            q.put_nowait(_STREAM_END)
        # Drop from the registry; in-flight subscribers keep their queue refs.
        if _broadcasters.get(self.lesson_id) is self:
            _broadcasters.pop(self.lesson_id, None)


def get_broadcaster(lesson_id: int) -> LessonBroadcaster:
    """Return the existing broadcaster for ``lesson_id`` or create one."""
    bc = _broadcasters.get(lesson_id)
    if bc is None:
        bc = LessonBroadcaster(lesson_id)
        _broadcasters[lesson_id] = bc
    return bc


# ---------------------------------------------------------------------------
# Modal Trainer lookup
# ---------------------------------------------------------------------------
def _lookup_trainer() -> Any:
    """Look up the deployed Modal ``Trainer`` class by app/class name.

    Uses :func:`modal.Cls.from_name` so this control-plane process does not need
    to import ``modal_app.trainer`` (and thus the heavy torch image deps). The
    app name comes from :data:`settings.MODAL_APP_NAME`.

    Returns the ``Cls`` handle; call ``()`` on it to obtain an instance whose
    methods expose ``.remote_gen`` / ``.remote`` (and ``.aio`` variants).
    """
    return modal.Cls.from_name(settings.MODAL_APP_NAME, "Trainer")


async def _generate_remote(prompt=None, messages=None, max_new_tokens: int = 64) -> str:
    """Call ``Trainer.generate`` (reader path) with a prompt or message history.

    Returns the learned model's reply, or ``""`` on any Modal lookup/call
    failure so ``/api/chat`` can degrade gracefully (never 500 if Modal is down).
    """
    try:
        trainer_cls = _lookup_trainer()
        instance = trainer_cls()
        gen = instance.generate
        aio = getattr(getattr(gen, "remote", None), "aio", None)
        if aio is not None:
            return await aio(prompt, max_new_tokens, messages)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: gen.remote(prompt, max_new_tokens, messages)
        )
    except Exception:  # noqa: BLE001 - chat must not 500 if Modal is down
        return ""


async def infer(prompt: str, max_new_tokens: int = 64) -> str:
    """Single-prompt reply from the CURRENT learned weights (no history)."""
    return await _generate_remote(prompt=prompt, max_new_tokens=max_new_tokens)


async def infer_chat(messages: list[dict], max_new_tokens: int = 64) -> str:
    """History-aware reply from the CURRENT learned weights.

    ``messages`` is the running conversation (``{"role","content"}`` turns,
    oldest-first, ending with the latest user message) so the learned model
    answers in context — e.g. it can be corrected across turns and then learn.
    """
    return await _generate_remote(messages=messages, max_new_tokens=max_new_tokens)


async def _iter_remote_gen(trainer_cls: Any, lesson_id: int,
                           pairs: list[dict]) -> AsyncIterator[dict]:
    """Async-iterate ``Trainer().finetune.remote_gen(lesson_id, pairs)``.

    Modal exposes a sync generator via ``.remote_gen`` and an async generator
    via ``.remote_gen.aio``. We prefer the async form so the event loop stays
    responsive; if it is unavailable we fall back to draining the sync generator
    on a worker thread.
    """
    instance = trainer_cls()
    finetune = instance.finetune

    aio = getattr(getattr(finetune, "remote_gen", None), "aio", None)
    if aio is not None:
        async for event in aio(lesson_id, pairs):
            yield event
        return

    # Fallback: consume the blocking generator without stalling the loop.
    loop = asyncio.get_running_loop()
    sync_gen = finetune.remote_gen(lesson_id, pairs)
    _SENTINEL = object()

    def _next() -> Any:
        try:
            return next(sync_gen)
        except StopIteration:
            return _SENTINEL

    while True:
        event = await loop.run_in_executor(None, _next)
        if event is _SENTINEL:
            break
        yield event


# ---------------------------------------------------------------------------
# Durable queue: enqueue (producer) + worker loop (single consumer) + runner
# ---------------------------------------------------------------------------
def enqueue_lesson(lesson_id: int, pairs: list[dict]) -> int:
    """Durably enqueue a lesson for training; return the job id.

    This only writes a ``queued`` row to ``training_jobs`` and returns — it does
    NOT run the finetune inline. The background worker (:func:`_worker_loop`)
    claims and runs jobs one at a time. Because the claim is atomic across all
    processes (:func:`db.claim_next_job` under ``BEGIN IMMEDIATE``), training is
    single-writer even with multiple FastAPI workers, and a queued lesson
    survives a restart.
    """
    return db.enqueue_training_job(lesson_id, json.dumps(pairs))


async def _run_job(job: dict) -> None:
    """Execute one claimed training job end-to-end (the actual writer body).

    Streams the Modal finetune, fans progress out to the lesson broadcaster, and
    on the terminal ``done`` event records + flips the weights version, marks the
    lesson ``done``, and appends the feed entry. On any failure the job is
    requeued (if attempts remain) or marked ``error``; the lesson row and the WS
    stream are updated either way. No in-process lock is needed: the worker runs
    jobs sequentially and the claim guarantees exclusivity across processes.
    """
    lesson_id = int(job["lesson_id"])
    job_id = int(job["id"])
    attempts = int(job.get("attempts", 1))
    pairs = json.loads(job["pairs_json"])

    broadcaster = get_broadcaster(lesson_id)
    done_event: Optional[dict] = None
    try:
        db.set_lesson_status(lesson_id, "training")

        trainer_cls = _lookup_trainer()
        async for event in _iter_remote_gen(trainer_cls, lesson_id, pairs):
            await broadcaster.publish(event)  # forward unchanged to WS subscribers
            if isinstance(event, dict) and event.get("type") == "done":
                done_event = event

        if done_event is None:
            raise RuntimeError(
                f"lesson {lesson_id}: training stream ended without a 'done' event"
            )

        # Persist the new weights version only after Modal confirmed the volume
        # CURRENT-file flip.
        current = db.get_current_weights()
        parent_id = current["id"] if current else None
        kind = done_event.get("kind", settings.METHOD)
        path = done_event["path"]  # "v{N}", mirrors the volume pointer

        vid = db.new_weights_version(
            kind=kind, path=path, parent_id=parent_id, lesson_id=lesson_id,
        )
        db.set_current_weights(vid)
        db.set_lesson_status(lesson_id, "done")
        db.add_feed(lesson_id, _lesson_summary(lesson_id))
        db.finish_job(job_id, "done")

    except Exception as exc:  # noqa: BLE001 - surface failure to WS + DB
        max_attempts = settings.TRAIN_JOB_MAX_ATTEMPTS
        if attempts < max_attempts:
            # Transient: return to the queue for another worker/attempt.
            db.requeue_job(job_id)
            await broadcaster.publish(
                {"type": "retry", "lesson_id": lesson_id,
                 "attempt": attempts, "error": str(exc)}
            )
            return  # keep the broadcaster open; a later attempt will close it
        # Exhausted: terminal failure.
        try:
            db.set_lesson_status(lesson_id, "error")
        except Exception:  # noqa: BLE001 - never mask the original error
            pass
        db.finish_job(job_id, "error", error=str(exc))
        await broadcaster.publish(
            {"type": "error", "lesson_id": lesson_id, "error": str(exc)}
        )
    finally:
        # Close the stream only on a terminal outcome (done or exhausted error).
        if done_event is not None or attempts >= settings.TRAIN_JOB_MAX_ATTEMPTS:
            broadcaster.close()


async def _worker_loop(stop: "asyncio.Event") -> None:
    """Single-consumer loop: claim one job at a time and run it to completion.

    Polls ``training_jobs`` for a claimable job; if one is found, runs it (which
    blocks the loop until that finetune finishes — this is what serializes
    training). If none, sleeps ``TRAIN_POLL_INTERVAL`` seconds. Exits when
    ``stop`` is set. Running exactly one job at a time per process, combined with
    the atomic cross-process claim, is the single-writer guarantee.
    """
    while not stop.is_set():
        try:
            job = await asyncio.to_thread(
                db.claim_next_job, WORKER_ID, settings.TRAIN_JOB_MAX_ATTEMPTS
            )
        except Exception:  # noqa: BLE001 - never let a claim error kill the loop
            job = None

        if job is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.TRAIN_POLL_INTERVAL)
            except asyncio.TimeoutError:
                pass
            continue

        await _run_job(job)


def start_worker() -> None:
    """Start the background training worker (idempotent).

    Recovers stale ``claimed`` jobs from a prior crash, then launches the worker
    loop as an asyncio task. Call once from FastAPI startup.
    """
    global _worker_task, _worker_stop
    if _worker_task is not None and not _worker_task.done():
        return
    recovered = db.recover_stale_jobs()
    if recovered:
        # Best-effort log; the lesson rows for these were left as "training".
        print(f"[training] recovered {recovered} stale job(s) -> requeued")
    _worker_stop = asyncio.Event()
    _worker_task = asyncio.create_task(_worker_loop(_worker_stop))


async def stop_worker() -> None:
    """Signal the worker to stop and await its exit (call on shutdown)."""
    global _worker_task, _worker_stop
    if _worker_stop is not None:
        _worker_stop.set()
    if _worker_task is not None:
        try:
            await asyncio.wait_for(_worker_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _worker_task.cancel()
    _worker_task = None
    _worker_stop = None


def _lesson_summary(lesson_id: int) -> str:
    """Best-effort feed summary for a finished lesson.

    Pulls the lesson row's ``summary`` (falling back to ``concept``) so the
    Recently-Learned feed gets a human line. Defensive: never raises.
    """
    try:
        conn = db.sqlite3.connect(settings.DB_PATH, check_same_thread=False)
        conn.row_factory = db.sqlite3.Row
        try:
            row = conn.execute(
                "SELECT concept, summary FROM lessons WHERE id = ?",
                (lesson_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            return row["summary"] or row["concept"] or f"Lesson {lesson_id}"
    except Exception:  # noqa: BLE001 - feed text is non-critical
        pass
    return f"Lesson {lesson_id}"


# ---------------------------------------------------------------------------
# WebSocket-facing reader
# ---------------------------------------------------------------------------
async def stream_lesson(lesson_id: int) -> AsyncIterator[dict]:
    """Async-yield a lesson's training events until the stream closes.

    Subscribes to the lesson's broadcaster and yields each published event
    (``progress`` / ``done`` / ``error``) in order, stopping when the close
    sentinel arrives. Always unsubscribes on exit (including cancellation), so a
    disconnecting WebSocket does not leak a queue.

    Args:
        lesson_id: id of the lesson to watch.

    Yields:
        Event dicts forwarded unchanged from :class:`LessonBroadcaster`.
    """
    broadcaster = get_broadcaster(lesson_id)
    q = broadcaster.subscribe()
    try:
        while True:
            event = await q.get()
            if event is _STREAM_END:
                break
            yield event
    finally:
        broadcaster.unsubscribe(q)
