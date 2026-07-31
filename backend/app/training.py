"""Control-plane bridge: DB <-> Modal trainer <-> WebSocket.

This module is the seam between the FastAPI control plane and the Modal data
plane (``modal_app/trainer.py::Trainer``). It owns two things:

1.  **Per-lesson fan-out** (:class:`LessonBroadcaster`): a tiny async pub/sub so
    that many WebSocket subscribers can watch a single lesson's training stream
    while exactly one producer (the Modal generator relay) feeds it.

2.  **The serialized DB-side writer** (:func:`_run_job`): consumes the Modal
    ``finetune`` remote-generator, forwards every progress event to the lesson's
    broadcaster unchanged, and — on the terminal ``done`` event — advances the DB
    weights pointer and feed. The single-consumer :func:`_worker_loop` plus the
    atomic cross-process job claim (:func:`db.claim_next_job` under ``BEGIN
    IMMEDIATE``) guarantee only one job advances the pointer at a time, mirroring
    the ``@modal.concurrent(max_inputs=1)`` guard on the data plane. (There is no
    in-process ``_write_lock``; the durable claim IS the lock.)

Cross-file invariants honoured here (see contract §"Cross-file invariants"):
  * Event dicts emitted by ``Trainer.finetune`` are forwarded **unchanged**.
  * ``weights_versions.path`` stores the ``"v{N}"`` string from the ``done``
    event; the DB row is written only after Modal confirms the volume flip.
  * Single-writer is enforced by the single worker loop + the atomic DB claim.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
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


def clear_broadcasters() -> None:
    """Close + drop every live broadcaster (used by the authoritative reset).

    After a reset wipes all weights, any in-flight lesson stream is meaningless;
    closing the broadcasters unblocks subscribers and drops stale module state.
    """
    for bc in list(_broadcasters.values()):
        try:
            bc.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logging.warning("broadcaster close failed during reset", exc_info=True)
    _broadcasters.clear()


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


def _lookup_server() -> Any:
    """Look up the deployed Modal ``Server`` (read-only pool) by app/class name.

    PR-7 serve/train split: inference (``generate``/``generate_stream``) runs on
    this horizontally-scaled, immutable pool so a long write on the single-writer
    ``Trainer`` never freezes chat. Writes (finetune/consolidate/set_current/
    read_current/reset_memory/prune) stay on :func:`_lookup_trainer`.
    """
    return modal.Cls.from_name(settings.MODAL_APP_NAME, "Server")


async def _generate_remote(prompt=None, messages=None, max_new_tokens: int = 512) -> str:
    """Call ``Trainer.generate`` (reader path) with a prompt or message history.

    Returns the learned model's reply, or ``""`` on any Modal lookup/call
    failure so ``/api/chat`` can degrade gracefully (never 500 if Modal is down).
    """
    try:
        server_cls = _lookup_server()
        instance = server_cls()
        gen = instance.generate
        aio = getattr(getattr(gen, "remote", None), "aio", None)
        if aio is not None:
            return await aio(prompt, max_new_tokens, messages)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: gen.remote(prompt, max_new_tokens, messages)
        )
    except Exception:  # noqa: BLE001 - chat must not 500 if Modal is down
        logging.warning("_generate_remote failed (falling back to teacher text)", exc_info=True)
        return ""


async def warmup() -> bool:
    """Spin up the warm Modal Server pool so the first real chat is fast.

    Cold start = Modal boots a container + ``@modal.enter() load()`` loads the
    model into GPU memory (the slow part). Firing tiny 1-token generates forces
    that now; containers then stay warm (scaledown_window) so the user's first
    message streams immediately. PR-7: this warms the read-only ``Server`` pool
    (not the writer). We fan ``SERVER_MIN_CONTAINERS`` concurrent generates so
    Modal spreads them across the keep-warm replicas rather than warming only one.
    Returns True if the pool responded (warm/ready), False if Modal is unreachable.
    """
    try:
        server_cls = _lookup_server()
        instance = server_cls()
        gen = instance.generate
        aio = getattr(getattr(gen, "remote", None), "aio", None)
        fan = max(1, int(getattr(settings, "SERVER_MIN_CONTAINERS", 1)))
        if aio is not None:
            # Concurrent tiny generates: Modal fans them across replicas so the
            # whole keep-warm pool is hot before the first real chat.
            await asyncio.gather(
                *(aio("hi", 1, None) for _ in range(fan)),
                return_exceptions=True,
            )
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: gen.remote("hi", 1, None))
        return True
    except Exception:  # noqa: BLE001 - warmup is best-effort; never raise to caller
        return False


async def set_current_version(version: str) -> bool:
    """Flip the Modal volume CURRENT pointer to ``version`` (revert support).

    The DB pointer and the volume CURRENT file must agree for revert to actually
    change inference (inference reads the VOLUME pointer). Returns True on a
    confirmed flip, False if the version isn't on the volume or Modal is down.
    """
    try:
        trainer_cls = _lookup_trainer()
        instance = trainer_cls()
        m = instance.set_current
        aio = getattr(getattr(m, "remote", None), "aio", None)
        if aio is not None:
            res = await aio(version)
        else:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: m.remote(version))
        return bool(res and res.get("ok"))
    except Exception:  # noqa: BLE001
        return False


async def read_current_version() -> Optional[str]:
    """Return the volume's CURRENT version string (``"v{N}"``) or None.

    Inference reads this volume pointer (not the DB). Used by startup
    reconciliation to detect a DB/volume desync (e.g. after an out-of-band
    ``reset.sh``). Returns None if Modal is unreachable so boot never blocks.
    """
    try:
        trainer_cls = _lookup_trainer()
        instance = trainer_cls()
        m = instance.read_current
        aio = getattr(getattr(m, "remote", None), "aio", None)
        if aio is not None:
            return await aio()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: m.remote())
    except Exception:  # noqa: BLE001 - reconciliation is best-effort
        logging.warning("read_current_version failed", exc_info=True)
        return None


async def reset_remote() -> dict:
    """Wipe the Modal weights volume AND reset the warm container's memory.

    Returns a summary dict (``{"removed": [...], "memory_reset": bool}``). Used by
    the authoritative ``POST /api/admin/reset`` so the whole reset happens in one
    coordinated place (after the worker is drained), instead of the out-of-band
    ``reset.sh`` racing an in-flight job.
    """
    out: dict = {"removed": [], "memory_reset": False}
    try:
        trainer_cls = _lookup_trainer()
        # Volume wipe runs on the standalone reset_weights function.
        rw = modal.Function.from_name(settings.MODAL_APP_NAME, "reset_weights")
        aio = getattr(getattr(rw, "remote", None), "aio", None)
        res = await aio() if aio is not None else await asyncio.get_running_loop().run_in_executor(
            None, lambda: rw.remote()
        )
        if isinstance(res, dict):
            out["removed"] = res.get("removed", [])
        # Reset the warm container's resident model so it stops answering as taught.
        inst = trainer_cls()
        rm = inst.reset_memory
        rm_aio = getattr(getattr(rm, "remote", None), "aio", None)
        if rm_aio is not None:
            await rm_aio()
        else:
            await asyncio.get_running_loop().run_in_executor(None, lambda: rm.remote())
        out["memory_reset"] = True

        # PR-7: fan Server.flush_cache across the read pool so replicas drop any
        # cached (now-wiped) version at once. The pointer-driven path (CURRENT +
        # LAST_GOOD wiped + EPOCH bumped by reset_weights) already self-invalidates
        # within RELOAD_THROTTLE_S; this just cuts the taught-answer window on the
        # keep-warm replicas. Best-effort — never fail the reset on a flush miss.
        try:
            server_cls = _lookup_server()
            sinst = server_cls()
            fc = sinst.flush_cache
            fc_aio = getattr(getattr(fc, "remote", None), "aio", None)
            fan = max(1, int(getattr(settings, "SERVER_MIN_CONTAINERS", 1)))
            if fc_aio is not None:
                await asyncio.gather(
                    *(fc_aio() for _ in range(fan)), return_exceptions=True
                )
            else:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: fc.remote()
                )
            out["cache_flushed"] = True
        except Exception:  # noqa: BLE001 - flush is a latency optimization
            logging.warning("reset_remote Server.flush_cache failed", exc_info=True)
    except Exception:  # noqa: BLE001 - report what we can
        logging.warning("reset_remote partial/failed", exc_info=True)
    return out


async def prune_versions_remote(keep_last: int) -> dict:
    """Prune old volume versions after a consolidation (best-effort)."""
    try:
        trainer_cls = _lookup_trainer()
        instance = trainer_cls()
        m = instance.prune_versions
        aio = getattr(getattr(m, "remote", None), "aio", None)
        if aio is not None:
            return await aio(keep_last)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: m.remote(keep_last))
    except Exception:  # noqa: BLE001 - pruning is housekeeping, never fatal
        logging.warning("prune_versions_remote failed", exc_info=True)
        return {"removed": []}


async def infer(prompt: str, max_new_tokens: int = 512) -> str:
    """Single-prompt reply from the CURRENT learned weights (no history)."""
    return await _generate_remote(prompt=prompt, max_new_tokens=max_new_tokens)


async def infer_chat(messages: list[dict], max_new_tokens: int = 512) -> str:
    """History-aware reply from the CURRENT learned weights.

    ``messages`` is the running conversation (``{"role","content"}`` turns,
    oldest-first, ending with the latest user message) so the learned model
    answers in context — e.g. it can be corrected across turns and then learn.
    """
    return await _generate_remote(messages=messages, max_new_tokens=max_new_tokens)


async def infer_chat_stream(messages: list[dict], max_new_tokens: int = 0):
    """Async-yield reply text chunks from the CURRENT weights (history-aware).

    Bridges ``Trainer.generate_stream`` (a Modal generator) to an async iterator
    so the chat endpoint can stream tokens to the browser. Yields nothing (an
    empty stream) if Modal is unreachable so the caller can fall back.
    """
    if not max_new_tokens:
        max_new_tokens = settings.MAX_NEW_TOKENS
    try:
        server_cls = _lookup_server()
        instance = server_cls()
        gen = instance.generate_stream
        aio = getattr(getattr(gen, "remote_gen", None), "aio", None)
        if aio is not None:
            async for chunk in aio(None, max_new_tokens, messages):
                yield chunk
            return
        # Fallback: drain the sync remote generator off the event loop.
        loop = asyncio.get_running_loop()
        sync_gen = gen.remote_gen(None, max_new_tokens, messages)
        sentinel = object()

        def _next():
            try:
                return next(sync_gen)
            except StopIteration:
                return sentinel

        while True:
            chunk = await loop.run_in_executor(None, _next)
            if chunk is sentinel:
                break
            yield chunk
    except Exception:  # noqa: BLE001 - chat must not 500 if Modal is down
        logging.warning("infer_chat_stream failed (falling back)", exc_info=True)
        return


async def _iter_remote_gen(trainer_cls: Any, lesson_id: int,
                           pairs: list[dict],
                           base_version: Optional[str] = None,
                           knobs: Optional[dict] = None) -> AsyncIterator[dict]:
    """Async-iterate ``Trainer().finetune.remote_gen(lesson_id, pairs, ...)``.

    Modal exposes a sync generator via ``.remote_gen`` and an async generator
    via ``.remote_gen.aio``. We prefer the async form so the event loop stays
    responsive; if it is unavailable we fall back to draining the sync generator
    on a worker thread.

    ``base_version`` is the CURRENT ``v{N}`` (resolved at claim time) that the
    new adapter accumulates on top of, so lessons STACK instead of overwrite.
    ``knobs`` are lesson-type-aware training overrides (lora_r/lora_lr/epochs/
    lora_dropout); unknown/empty keys default on the data plane.
    """
    instance = trainer_cls()
    finetune = instance.finetune
    kw = {"base_version": base_version, **(knobs or {})}

    # Positional args mirror Trainer.finetune's signature; trailing knobs default
    # on the data plane, so we pass lesson_id/pairs and the rest as kwargs.
    aio = getattr(getattr(finetune, "remote_gen", None), "aio", None)
    if aio is not None:
        async for event in aio(lesson_id, pairs, **kw):
            yield event
        return

    # Fallback: consume the blocking generator without stalling the loop.
    loop = asyncio.get_running_loop()
    sync_gen = finetune.remote_gen(lesson_id, pairs, **kw)
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
# Envelope marker: a lesson job whose ``pairs_json`` is an augmentation ENVELOPE
# (seed pairs + generation params) rather than a bare list of final training
# pairs. The worker fans out build_training_pairs + check_pairs before training
# (§9 B2 — augmentation moved off the request path). ``_run_job`` detects the
# shape: a dict with this key is an envelope; a bare list is the legacy
# already-augmented payload (still handled, so in-flight jobs survive a deploy).
_AUGMENT_ENVELOPE_KEY = "__augment__"


def enqueue_lesson(lesson_id: int, pairs: list[dict]) -> int:
    """Durably enqueue a lesson with ALREADY-augmented pairs; return the job id.

    Writes a ``queued`` row whose payload is the final training-pair list. The
    background worker (:func:`_worker_loop`) claims and runs jobs one at a time;
    the atomic cross-process claim (:func:`db.claim_next_job` under ``BEGIN
    IMMEDIATE``) makes training single-writer even with multiple FastAPI workers,
    and a queued lesson survives a restart.

    NOTE: with §9 B2 the request path uses :func:`enqueue_lesson_augment` so the
    heavy fanout runs in the worker. This entry point remains for callers that
    already hold the final pairs (and for backward compatibility).
    """
    return db.enqueue_training_job(lesson_id, json.dumps(pairs))


def enqueue_lesson_augment(
    lesson_id: int,
    seed_pairs: list[dict],
    concept: str,
    kind: str,
    target: int,
    core_ratio: float,
    user_context: str,
) -> int:
    """Durably enqueue a lesson whose augmentation runs IN THE WORKER (§9 B2).

    Persists a ``queued`` job carrying an ENVELOPE of the seed pairs plus the
    generation params (concept/kind/target/core_ratio/user_context). The worker
    (:func:`_run_job`) fans out ``pipeline.build_training_pairs`` and runs the
    pair-level guardrail (``pipeline.check_pairs``) BEFORE the finetune, so the
    up-to-60s Gemini fanout never blocks the request path. Same single-writer /
    restart-survival guarantees as :func:`enqueue_lesson`.
    """
    envelope = {
        _AUGMENT_ENVELOPE_KEY: True,
        "seed_pairs": seed_pairs,
        "concept": concept,
        "kind": kind,
        "target": int(target),
        "core_ratio": float(core_ratio),
        "user_context": user_context,
    }
    return db.enqueue_training_job(lesson_id, json.dumps(envelope))


def _build_replay_buffer(lesson_id: int, new_pairs: list[dict]) -> list[dict]:
    """Sample a bounded replay buffer of PRIOR lessons' allowed pairs.

    Pulls every guardrail-allowed ``{"prompt","response"}`` pair across all
    lessons (:func:`db.get_allowed_pairs_since(None)`, already deduped), removes
    the current lesson's own pairs (deduped by ``(prompt, response)`` against
    ``new_pairs``), and randomly samples up to ``settings.REPLAY_BUFFER_MAX`` of
    the remainder. Sampling is deterministic per lesson (``random.Random(
    lesson_id)``) so a retried job replays the same buffer. Returns ``[]`` when
    there are no prior pairs (first lesson).

    The trainer mixes in fixed ``RETENTION_ANCHORS`` itself, so those are NOT
    added here — this returns prior-lesson pairs only.
    """
    try:
        all_pairs = db.get_allowed_pairs_since(None)
    except Exception:  # noqa: BLE001 - replay is best-effort; never fail the job
        logging.warning("replay buffer: get_allowed_pairs_since failed", exc_info=True)
        return []

    # Exclude the current lesson's own pairs (dedupe by normalized prompt+response).
    new_keys = {
        (p["prompt"].strip(), p["response"].strip())
        for p in new_pairs
    }
    candidates = [
        p for p in all_pairs
        if (p["prompt"].strip(), p["response"].strip()) not in new_keys
    ]
    if not candidates:
        return []

    cap = settings.REPLAY_BUFFER_MAX
    if cap <= 0:
        return []
    if len(candidates) <= cap:
        return candidates
    rng = random.Random(lesson_id)
    return rng.sample(candidates, cap)


async def _augment_and_guard(
    lesson_id: int, envelope: dict, broadcaster: "LessonBroadcaster"
) -> Optional[list[dict]]:
    """Fan out augmentation + run the pair-level guardrail for an envelope job.

    Runs INSIDE the worker (§9 B2) so the up-to-60s multi-facet Gemini fanout
    never blocks the request path. Steps:

      1. ``pipeline.build_training_pairs`` grows the seed pairs into a diverse
         set (falls back to templates if the teacher is unavailable).
      2. ``pipeline.check_pairs`` guardrails the augmented set (defense-in-depth
         behind the request-path reputation gate).
      3. Persists every annotated pair, updates the lesson's ``num_pairs`` to the
         real augmented count.

    Returns the list of ALLOWED ``{"prompt","response"}`` pairs to train on, or
    ``None`` if the guardrail blocked the lesson (the row is marked ``blocked``
    and the pairs persisted; the caller finishes the job without training).
    Emits a lightweight ``augment`` progress event so a watching UI sees the
    generation phase.
    """
    from backend.app import pipeline  # lazy: pipeline pulls in experiments.data

    seed_pairs = list(envelope.get("seed_pairs") or [])
    concept = str(envelope.get("concept") or "")
    kind = str(envelope.get("kind") or "fact")
    target = int(envelope.get("target") or settings.NUM_PAIRS)
    try:
        core_ratio = float(envelope.get("core_ratio", 0.4))
    except (TypeError, ValueError):
        core_ratio = 0.4
    user_context = str(envelope.get("user_context") or "")

    await broadcaster.publish(
        {"type": "augment", "lesson_id": lesson_id, "status": "generating"}
    )

    # Seeds must be non-empty for the augmenter (it cycles over them). A lesson
    # with no seeds still trains: fall back to a minimal seed from the concept.
    if not seed_pairs:
        seed_pairs = [{"prompt": concept or "Remember this.", "response": concept or ""}]

    augmented = await pipeline.build_training_pairs(
        concept, seed_pairs, user_context, target, core_ratio, kind=kind
    )

    overall_allowed, reason, per_pair = pipeline.check_pairs(augmented)
    db.add_pairs(lesson_id, per_pair)
    # NOTE: the lesson row keeps the TARGET count set at creation; the real
    # augmented pairs are persisted above via add_pairs. We avoid a num_pairs
    # UPDATE here to keep this PR within its owned files (db.py is not one).

    if not overall_allowed:
        db.set_lesson_status(lesson_id, "blocked")
        await broadcaster.publish(
            {"type": "blocked", "lesson_id": lesson_id, "reason": reason}
        )
        return None

    return [
        {"prompt": p["prompt"], "response": p["response"]}
        for p in per_pair
        if p.get("guardrail_status") == "allowed"
    ]


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
    payload = json.loads(job["pairs_json"])

    broadcaster = get_broadcaster(lesson_id)
    done_event: Optional[dict] = None
    try:
        db.set_lesson_status(lesson_id, "training")

        # §9 B2: augmentation moved off the request path. If the payload is an
        # ENVELOPE (seed pairs + generation params), fan out the multi-facet
        # teacher generation and run the pair-level guardrail HERE, in the worker,
        # before training. A bare list is the legacy already-augmented payload.
        if isinstance(payload, dict) and payload.get(_AUGMENT_ENVELOPE_KEY):
            pairs = await _augment_and_guard(lesson_id, payload, broadcaster)
            if pairs is None:
                # Blocked by the pair-level guardrail (defense-in-depth): the
                # lesson row + pairs were persisted blocked inside the helper and
                # the job finished. Nothing to train.
                db.finish_job(job_id, "done")
                broadcaster.close()
                return
            if not pairs:
                # Degenerate: augmentation yielded no trainable pairs (all deduped
                # away / empty). Don't hand the trainer an empty set — mark done.
                db.set_lesson_status(lesson_id, "done")
                db.finish_job(job_id, "done")
                broadcaster.close()
                return
        else:
            pairs = payload

        # ACCUMULATION: resolve the CURRENT version NOW (under the single-writer
        # claim, so no concurrent flip can move it) and hand its volume path to
        # the trainer as the accumulated base. The new adapter is a delta on
        # "base + all prior lessons", so lessons stack instead of overwrite. The
        # DB parent_id is chained to this same row so the version graph mirrors
        # the volume merge-chain.
        current = db.get_current_weights()
        parent_id = current["id"] if current else None
        base_version = current["path"] if current else None

        # Lesson-type-aware training knobs: a style lesson trains gentler than a
        # prior-fighting fact (see settings.LESSON_KIND_KNOBS). Unknown kind ->
        # 'fact' defaults; empty map -> data-plane defaults.
        lesson_row = db.get_lesson(lesson_id)
        kind = (lesson_row or {}).get("kind") or "fact"
        knobs = settings.LESSON_KIND_KNOBS.get(kind) or settings.LESSON_KIND_KNOBS.get("fact") or {}

        # REPLAY BUFFER (continual learning): continue-training only on this
        # lesson's new pairs makes the model forget prior lessons. So mix in a
        # bounded, deterministic sample of PRIOR lessons' allowed pairs. The
        # trainer additionally folds in fixed RETENTION_ANCHORS, so we don't add
        # those here. The union we pass = replay sample + new lesson pairs.
        replay_pairs = _build_replay_buffer(lesson_id, pairs)
        payload_pairs = replay_pairs + pairs

        trainer_cls = _lookup_trainer()
        async for event in _iter_remote_gen(
            trainer_cls, lesson_id, payload_pairs, base_version=base_version, knobs=knobs
        ):
            await broadcaster.publish(event)  # forward unchanged to WS subscribers
            if isinstance(event, dict) and event.get("type") == "done":
                done_event = event

        if done_event is None:
            raise RuntimeError(
                f"lesson {lesson_id}: training stream ended without a 'done' event"
            )

        # Persist the new weights version only after Modal confirmed the volume
        # CURRENT-file flip.
        kind = done_event.get("kind", settings.METHOD)
        path = done_event["path"]  # "v{N}", mirrors the volume pointer

        vid = db.new_weights_version(
            kind=kind, path=path, parent_id=parent_id, lesson_id=lesson_id,
            final_loss=done_event.get("final_loss"),
        )
        db.set_current_weights(vid)
        db.set_lesson_status(lesson_id, "done")
        # Generate a polished one-line feed description via OpenRouter (falls
        # back to the lesson summary/concept if the call fails).
        feed_line = await _feed_description(lesson_id, pairs)
        db.add_feed(lesson_id, feed_line)
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


# ---------------------------------------------------------------------------
# Nightly consolidation (re-derive ONE flat adapter from the day's pairs)
# ---------------------------------------------------------------------------
# Consolidation rides the SAME durable single-writer queue as lessons: it is
# enqueued as a job with job_kind='consolidate' (NULL lesson_id), so the worker
# serializes it against live lessons (never a concurrent pointer-flip). The
# broadcaster id for its progress stream is a fixed sentinel.
CONSOLIDATE_LESSON_ID: int = -1  # broadcaster channel id for consolidation progress


def enqueue_consolidation(pairs: list[dict]) -> int:
    """Durably enqueue a consolidation job over ``pairs``; return the job id.

    Same contract as :func:`enqueue_lesson` but ``job_kind='consolidate'`` so the
    worker dispatches to :func:`_run_consolidation`. Single-writer and
    restart-survival come for free from the shared queue.
    """
    return db.enqueue_training_job(None, json.dumps(pairs), job_kind="consolidate")


async def _iter_consolidate_gen(trainer_cls: Any,
                                pairs: list[dict]) -> AsyncIterator[dict]:
    """Async-iterate ``Trainer().consolidate.remote_gen(pairs)`` (async or sync)."""
    instance = trainer_cls()
    consolidate = instance.consolidate
    aio = getattr(getattr(consolidate, "remote_gen", None), "aio", None)
    if aio is not None:
        async for event in aio(pairs):
            yield event
        return
    loop = asyncio.get_running_loop()
    sync_gen = consolidate.remote_gen(pairs)
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


async def _run_consolidation(job: dict) -> None:
    """Run a claimed consolidation job: stream consolidate, flip the new version.

    Mirrors :func:`_run_job`'s terminal-event handling but calls the LONGER
    ``Trainer.consolidate`` over the day's whole deduped corpus. The resulting
    version has ``parent_id`` pointing at the prior CURRENT (so revert still
    works) while its volume meta has ``parent=None`` (flat single adapter, chain
    depth reset). Fans progress to a fixed broadcaster id so a UI can watch it.
    """
    job_id = int(job["id"])
    attempts = int(job.get("attempts", 1))
    pairs = json.loads(job["pairs_json"])

    broadcaster = get_broadcaster(CONSOLIDATE_LESSON_ID)
    done_event: Optional[dict] = None
    try:
        if not pairs:
            # Nothing taught in the window: no-op success.
            db.finish_job(job_id, "done")
            return

        current = db.get_current_weights()
        parent_id = current["id"] if current else None

        trainer_cls = _lookup_trainer()
        async for event in _iter_consolidate_gen(trainer_cls, pairs):
            await broadcaster.publish(event)
            if isinstance(event, dict) and event.get("type") == "done":
                done_event = event

        if done_event is None:
            raise RuntimeError("consolidation stream ended without a 'done' event")

        kind = done_event.get("kind", settings.METHOD)
        path = done_event["path"]
        vid = db.new_weights_version(
            kind=kind, path=path, parent_id=parent_id, lesson_id=None,
            final_loss=done_event.get("final_loss"),
        )
        db.set_current_weights(vid)
        db.add_feed(None, f"Nightly consolidation: re-derived {len(pairs)} pairs into {path}.")
        db.finish_job(job_id, "done")

        # Housekeeping: the consolidated version has a flat chain (parent=None),
        # so older incrementals are no longer ancestors of CURRENT — prune them
        # from the volume beyond a small revert window. Mark the corresponding DB
        # rows pruned so the UI doesn't offer a revert that would 409. Best-effort.
        keep = settings.CONSOLIDATE_KEEP_VERSIONS
        result = await prune_versions_remote(keep)
        for removed_path in result.get("removed", []):
            try:
                db.mark_version_pruned(removed_path)
            except Exception:  # noqa: BLE001 - housekeeping must not fail the job
                logging.warning("mark_version_pruned(%s) failed", removed_path, exc_info=True)

    except Exception as exc:  # noqa: BLE001
        max_attempts = settings.TRAIN_JOB_MAX_ATTEMPTS
        if attempts < max_attempts:
            db.requeue_job(job_id)
            await broadcaster.publish(
                {"type": "retry", "lesson_id": CONSOLIDATE_LESSON_ID,
                 "attempt": attempts, "error": str(exc)}
            )
            return
        db.finish_job(job_id, "error", error=str(exc))
        await broadcaster.publish(
            {"type": "error", "lesson_id": CONSOLIDATE_LESSON_ID, "error": str(exc)}
        )
    finally:
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

        # Same single-writer loop runs both lessons and consolidation; dispatch
        # on job_kind so a nightly consolidate is serialized against live lessons
        # and never flips the pointer concurrently.
        if job.get("job_kind") == "consolidate":
            await _run_consolidation(job)
        else:
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


def worker_alive() -> bool:
    """True if the background training worker task is running (for /api/health)."""
    return _worker_task is not None and not _worker_task.done()


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


async def _feed_description(lesson_id: int, pairs: list[dict]) -> str:
    """LLM-written one-line feed blurb for a finished lesson.

    Looks up the lesson concept and asks OpenRouter (``llm.describe_lesson``) to
    phrase it for the public feed, grounded by a couple of training pairs. Falls
    back to the plain lesson summary on any failure.
    """
    concept = _lesson_summary(lesson_id)
    try:
        from backend.app import llm

        return await llm.describe_lesson(concept, pairs[:4])
    except Exception:  # noqa: BLE001 - feed text is non-critical
        return concept


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
