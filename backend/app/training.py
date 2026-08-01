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
import time
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


async def _warmup_server() -> bool:
    """Boot the read-only Server pool so the first CHAT streams immediately.

    Cold start = Modal boots a container + ``@modal.enter() load()`` loads the
    model into GPU memory (the slow part). Tiny 1-token generates force that now;
    containers then stay warm (scaledown_window). We fan ``SERVER_MIN_CONTAINERS``
    concurrent generates so Modal spreads them across the keep-warm replicas.
    """
    server_cls = _lookup_server()
    instance = server_cls()
    gen = instance.generate
    aio = getattr(getattr(gen, "remote", None), "aio", None)
    fan = max(1, int(getattr(settings, "SERVER_MIN_CONTAINERS", 1)))
    if aio is not None:
        await asyncio.gather(
            *(aio("hi", 1, None) for _ in range(fan)),
            return_exceptions=True,
        )
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: gen.remote("hi", 1, None))
    return True


async def _warmup_trainer() -> bool:
    """Boot the single-writer Trainer so the first TEACH trains fast.

    Teaching cold-starts the Trainer container (container boot + @modal.enter
    load of the base model into GPU = the slow ~25s). Since the whole product IS
    teaching, warming ONLY the Server (chat) left the first lesson paying that
    cold start. We call the cheapest Trainer method (``read_current`` — just reads
    the CURRENT pointer file) purely to trigger ``@modal.enter load()`` and get the
    model resident; the container then stays warm (scaledown_window=600s) so the
    first real lesson trains on a hot GPU. Never trains anything.
    """
    trainer_cls = _lookup_trainer()
    instance = trainer_cls()
    # Call warmup() (not read_current): it triggers @modal.enter load() (base + the
    # kernel micro-train) AND seeds the reserved base v0 so the FIRST real lesson
    # saves a fast adapter instead of the ~40-60s flatten. warmup() is hard-guarded
    # to no-op once a real version exists, so calling it on every page load is safe
    # and cheap after the first seed.
    wu = instance.warmup
    aio = getattr(getattr(wu, "remote", None), "aio", None)
    if aio is not None:
        await aio()
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: wu.remote())
    return True


async def warmup() -> bool:
    """Pre-warm BOTH the chat Server and the teaching Trainer, concurrently.

    Fired on page load (the WarmupIndicator), so it overlaps the first-time user's
    intro demo — by the time they send their first message OR teach their first
    lesson, the relevant GPU is already hot. Warms the two independently-scaling
    pools in parallel; best-effort, so one being down doesn't fail the other.
    Returns True if AT LEAST the Server (chat path) came up — chat is the minimum
    for a usable app; a Trainer that's still cold just means the first teach pays
    the cold start, which is non-fatal.
    """
    results = await asyncio.gather(
        _warmup_server(), _warmup_trainer(), return_exceptions=True
    )
    server_ok = results[0] is True
    trainer_ok = results[1] is True
    if not trainer_ok:
        logging.info("warmup: Trainer warm did not complete (first teach may cold-start)")
    # Chat is the baseline; report ready when the Server is up even if the Trainer
    # is still warming (teaching still works, just slower on the very first lesson).
    return server_ok or trainer_ok


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


async def _bounded_chunks(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield from ``source`` while enforcing first/inter-token and total deadlines.

    A bare ``async for`` over the Modal generator can wait forever. Each __anext__
    is wrapped in ``wait_for`` so a stalled generator raises instead of hanging,
    and the elapsed total is checked so a degenerate never-ending generation
    cannot pin the connection open. On any timeout we stop cleanly (the caller
    falls back to detector text) rather than propagating an error into the SSE.
    """
    started = time.monotonic()
    first = True
    it = source.__aiter__()
    # A timeout / total-cap / caller-disconnect exits this generator EARLY, so the
    # underlying Modal async generator must be closed explicitly — abandoning it
    # leaks the remote stream + connection on a long-lived shared server. aclose()
    # in the finally covers every exit: StopAsyncIteration, either timeout, the
    # total cap, and a GeneratorExit thrown in when the caller stops consuming.
    try:
        while True:
            budget = (
                settings.INFER_FIRST_TOKEN_TIMEOUT_S
                if first
                else settings.INFER_INTER_TOKEN_TIMEOUT_S
            )
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=budget)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                logging.warning(
                    "infer_chat_stream stalled (%s-token timeout after %.1fs)",
                    "first" if first else "inter",
                    time.monotonic() - started,
                )
                return
            first = False
            yield chunk
            if time.monotonic() - started > settings.INFER_TOTAL_TIMEOUT_S:
                logging.warning("infer_chat_stream hit total timeout; truncating reply")
                return
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - best-effort teardown; already exiting
                pass


async def infer_chat_stream(messages: list[dict], max_new_tokens: int = 0):
    """Async-yield reply text chunks from the CURRENT weights (history-aware).

    Bridges ``Trainer.generate_stream`` (a Modal generator) to an async iterator
    so the chat endpoint can stream tokens to the browser. Yields nothing (an
    empty stream) if Modal is unreachable so the caller can fall back.

    Every wait is BOUNDED (see ``_bounded_chunks``): an unbounded stream here was
    the cause of chats that hang forever until the tab is reopened.
    """
    if not max_new_tokens:
        max_new_tokens = settings.MAX_NEW_TOKENS
    try:
        server_cls = _lookup_server()
        instance = server_cls()
        gen = instance.generate_stream
        aio = getattr(getattr(gen, "remote_gen", None), "aio", None)
        if aio is not None:
            async for chunk in _bounded_chunks(aio(None, max_new_tokens, messages)):
                yield chunk
            return

        # Fallback: drain the sync remote generator off the event loop. Wrap it
        # as an async iterator first so it gets the same deadlines.
        loop = asyncio.get_running_loop()
        sync_gen = gen.remote_gen(None, max_new_tokens, messages)
        sentinel = object()

        def _next():
            try:
                return next(sync_gen)
            except StopIteration:
                return sentinel

        async def _drain() -> AsyncIterator[str]:
            while True:
                chunk = await loop.run_in_executor(None, _next)
                if chunk is sentinel:
                    return
                yield chunk

        async for chunk in _bounded_chunks(_drain()):
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
    """Execute one claimed SINGLE lesson job end-to-end (per-lesson writer body).

    SUPERSEDED by :func:`_run_batch` (PR-8 windowed coalescing): the loop now
    claims a coalesced BATCH under the writer lease and flips ONCE per window.
    This single-lesson path is retained for reference / a possible non-coalesced
    fallback and is NOT dispatched by the current loop; it does NOT acquire the
    writer lease and MUST NOT be re-wired into the loop without a fenced flip.

    Streams the Modal finetune, fans progress out to the lesson broadcaster, and
    on the terminal ``done`` event records + flips the weights version, marks the
    lesson ``done``, and appends the feed entry. On any failure the job is
    requeued (if attempts remain) or marked ``error``; the lesson row and the WS
    stream are updated either way.
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


async def _run_consolidation(job: dict, epoch: int) -> None:
    """Run a claimed consolidation job UNDER the writer lease (epoch = fencing token).

    Calls the LONGER ``Trainer.consolidate`` over the day's whole deduped corpus.
    The flip goes through the FENCED :func:`db.flip_if_lease_held` so a stolen
    lease can't double-flip. NEVER coalesced; claimed on its own single path and
    gated by the SAME single-owner lease as lesson batches (never a concurrent
    flip). Fans progress to a fixed broadcaster id so a UI can watch it.
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
        # FENCED flip: only becomes CURRENT-of-record if we still hold the lease.
        if not db.flip_if_lease_held(vid, WORKER_ID, epoch):
            raise RuntimeError(
                "consolidation lost the writer lease before flip; discarding "
                f"version {path} (reaper GC will prune the unreferenced dir)"
            )
        # NOTE: consolidation deliberately does NOT write to the "Recently Learned"
        # feed. That feed answers "what did a user teach DUM-E?" — one row per
        # lesson. A nightly consolidation teaches nothing new; it re-derives the
        # SAME knowledge into a cleaner checkpoint (an ops event, not a learned
        # fact). Writing "re-derived N pairs into vK" leaked version numbers into the
        # user feed and pushed real lessons out of the newest-10 window. The run
        # stays fully observable via logs + weights_versions/training_jobs.
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


# ---------------------------------------------------------------------------
# PR-8: windowed coalescing — lease TTL math, heartbeat, failure classification,
# union dedupe, and the batch runner.
# ---------------------------------------------------------------------------
def _lease_ttl_for(kind: str, n_pairs: int = 0) -> float:
    """Budget-derived writer-lease TTL (NEVER a flat 50s).

    Covers the phases that emit NO finetune progress events — augmentation fanout,
    Modal cold start, and the flip tail — plus the token-scaled train budget, so a
    wall-clock heartbeat can keep a live batch's lease fresh across the WHOLE
    critical section and the reaper can't steal it mid-run.
    """
    if kind == "consolidate":
        train = settings.CONSOLIDATE_MAX_SECONDS
    else:
        train = _token_budget(n_pairs)
    return (
        train
        + settings.WRITER_LEASE_AUGMENT_SLACK_S
        + settings.WRITER_LEASE_COLDSTART_SLACK_S
        + settings.WRITER_LEASE_FLIP_SLACK_S
    )


def _token_budget(n_pairs: int) -> float:
    """Token-scaled train budget: min(cap, base + per_pair * n_pairs)."""
    return min(
        settings.COALESCE_MAX_TRAIN_SECONDS,
        settings.COALESCE_BASE_SECONDS + settings.COALESCE_PER_PAIR_SECONDS * max(0, n_pairs),
    )


def _lease_max_ttl() -> float:
    """Upper bound on any lease TTL (used by the reaper's claimed-age fallback)."""
    return _lease_ttl_for("consolidate")


def _classify_failure(exc: BaseException) -> str:
    """Classify a train failure as ``terminal`` or ``transient``.

    OOM / CUDA / device-side asserts are TERMINAL for the offending singleton (a
    requeue-storm won't help; the pair is oversized/poison). Everything else —
    including a non-finite/divergence RuntimeError from the finite-loss guard — is
    TRANSIENT and bisects, because a union's non-finiteness can be an EMERGENT
    interaction that isolates by bisection."""
    s = f"{type(exc).__name__}: {exc}".lower()
    terminal_markers = (
        "out of memory", "cuda error", "device-side assert",
        "cublas", "cudnn", "illegal memory access",
    )
    if any(m in s for m in terminal_markers):
        return "terminal"
    return "transient"


class _LeaseHeartbeat:
    """Wall-clock lease heartbeat (PR-8). Renews the writer lease every
    ``ttl/DIVISOR`` seconds for the ENTIRE critical section — augmentation, cold
    start, train, smoke, flip — INDEPENDENT of finetune progress events (which
    don't flow during augmentation/coldstart). Runs as an asyncio task started at
    lease acquire and stopped in the same finally that releases the lease. If a
    renew fails (lease stolen), it sets ``lost`` so callers can bail early."""

    def __init__(self, worker_id: str, epoch: int, ttl_s: float) -> None:
        self.worker_id = worker_id
        self.epoch = epoch
        self.ttl_s = ttl_s
        self.lost = False
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        interval = max(1.0, self.ttl_s / max(1, settings.WRITER_LEASE_HEARTBEAT_DIVISOR))
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            try:
                ok = await asyncio.to_thread(
                    db.renew_writer_lease, self.worker_id, self.epoch, self.ttl_s
                )
            except Exception:  # noqa: BLE001 - a transient DB hiccup shouldn't kill the run
                ok = True
            if not ok:
                self.lost = True
                break

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()


def _dedupe_union_newest_wins(job_pairs: list[tuple]) -> list[dict]:
    """Newest-wins-per-CONCEPT dedupe of a coalesced batch (USER DECISION).

    ``job_pairs`` is a list of ``(lesson_id, concept, pairs)`` in the batch. Two
    teaches of the SAME normalized concept in one window are a same-window
    contradiction: keep ONLY the highest-lesson-id lesson's ENTIRE pair set (drop
    the loser's augmented pairs wholesale) so newest-wins holds across paraphrases
    — never a blended {3,5}. Concept-level dedupe (not raw-prompt) is what makes
    this hold when two lessons' augmented prompts don't string-match. Distinct
    concepts all contribute. Within the surviving sets, exact (prompt,response)
    duplicates are collapsed.
    """
    # 1) Concept-level newest-wins: highest lesson-id per normalized concept.
    by_concept: dict[str, tuple] = {}
    for lid, concept, pairs in job_pairs:
        key = (concept or "").strip().lower() or f"__lesson_{lid}__"
        prev = by_concept.get(key)
        if prev is None or lid > prev[0]:
            by_concept[key] = (lid, pairs)
    # 2) Union the surviving sets; collapse exact duplicate pairs.
    seen: set[tuple] = set()
    union: list[dict] = []
    # Deterministic order: by ascending lesson-id of the surviving set.
    for lid, pairs in sorted(by_concept.values(), key=lambda t: t[0]):
        for p in pairs:
            pr = str(p.get("prompt", "")).strip()
            rs = str(p.get("response", "")).strip()
            if not pr:
                continue
            k = (pr, rs)
            if k in seen:
                continue
            seen.add(k)
            union.append({"prompt": p["prompt"], "response": p["response"]})
    return union


async def _run_batch(batch: dict, epoch: int, stop: "asyncio.Event") -> None:
    """Run a coalesced lesson batch under the writer lease (epoch = fencing token).

    Steps (all under the single-owner lease, heartbeated by the caller):
      1. Per-lesson augment + pair-level guard (PR-6 defense-in-depth preserved):
         each job runs its OWN _augment_and_guard; a BLOCKED lesson is finished
         'blocked' and DROPPED from the union — never trained into the shared brain.
      2. Newest-wins-per-CONCEPT union dedupe (USER DECISION): highest lesson-id's
         pair set wins per concept; distinct concepts all contribute.
      3. Pair cap AFTER augmentation: if the union exceeds COALESCE_MAX_PAIRS,
         train the newest COALESCE_MAX_PAIRS and REQUEUE the overflow jobs for the
         next window (the pair cap can't be enforced at claim time — pairs don't
         exist yet).
      4. Resolve base_version ONCE, build replay over the union, stream ONE
         finetune with a token-scaled max_train_seconds, FENCED-flip ONCE.
      5. Per-lesson status='done' + feed on success; batch-addressed job finish.
      6. On a TRANSIENT failure: bisection (log2, not fan-out); OOM/CUDA terminal.
    """
    jobs = batch["jobs"]
    batch_id = batch["batch_id"]

    # ---- 1. per-lesson augment + guard --------------------------------------
    # (lesson_id, concept, allowed_pairs) for lessons that passed the guard.
    surviving: list[tuple] = []
    for job in jobs:
        lid = int(job["lesson_id"])
        bc = get_broadcaster(lid)
        try:
            db.set_lesson_status(lid, "training")
            payload = json.loads(job["pairs_json"])
            if isinstance(payload, dict) and payload.get(_AUGMENT_ENVELOPE_KEY):
                concept = str(payload.get("concept") or "")
                pairs = await _augment_and_guard(lid, payload, bc)
                if pairs is None:
                    # Blocked by the pair-level guard (PR-6 defense-in-depth):
                    # DROPPED from the union — never trained into the shared brain.
                    # The helper already persisted the block; finish this job NOW so
                    # it can't be re-finished or requeued by a later batch failure.
                    db.finish_job(int(job["id"]), "done")
                    bc.close()
                    continue
                if not pairs:
                    # Augmentation yielded nothing trainable: mark this lesson done
                    # (no-op), finish the job, and drop from the union.
                    db.set_lesson_status(lid, "done")
                    db.finish_job(int(job["id"]), "done")
                    bc.close()
                    continue
            else:
                # Legacy bare-list payload: no concept -> key on lesson id so it is
                # never merged with another lesson (distinct concept).
                concept = f"__lesson_{lid}__"
                pairs = payload
            surviving.append((lid, concept, pairs))
        except Exception:  # noqa: BLE001 - one lesson's augment error mustn't sink the batch
            logging.warning("augment/guard failed for lesson %s; dropping from batch", lid, exc_info=True)
            # Requeue this one job so it retries in a later window (attempts bound it);
            # don't let an augment hiccup permanently lose the lesson.
            try:
                db.requeue_jobs([int(job["id"])], clear_batch=True)
            except Exception:  # noqa: BLE001
                pass
            try:
                bc.close()
            except Exception:
                pass

    if not surviving:
        # Every lesson blocked / empty / errored: nothing to train. Finish the
        # batch's job rows so they don't dangle claimed.
        db.finish_batch_jobs(batch_id, "done")
        return

    # ---- 2. newest-wins-per-concept union -----------------------------------
    union = _dedupe_union_newest_wins(surviving)

    # ---- 3. pair cap AFTER augmentation (requeue overflow) ------------------
    cap = settings.COALESCE_MAX_PAIRS
    overflow_ids: list[int] = []
    if cap > 0 and len(union) > cap:
        # Keep the NEWEST cap pairs (union is ordered ascending lesson-id, so the
        # tail is newest). Requeue the jobs whose surviving concept lost the cap
        # cut so their pairs are retried next window rather than blowing the budget.
        # Simplest deterministic rule: keep the pair sets of the highest-lesson-id
        # concepts until we fill the cap; requeue the rest of THIS batch's jobs.
        kept: list[dict] = []
        # Rebuild from surviving sets newest-first so we keep whole newest concepts.
        by_concept: dict[str, tuple] = {}
        for lid, concept, pairs in surviving:
            k = (concept or "").strip().lower() or f"__lesson_{lid}__"
            prev = by_concept.get(k)
            if prev is None or lid > prev[0]:
                by_concept[k] = (lid, pairs)
        seen: set[tuple] = set()
        kept_lesson_ids: set[int] = set()
        for lid, pairs in sorted(by_concept.values(), key=lambda t: -t[0]):  # newest first
            add: list[dict] = []
            for p in pairs:
                pr = str(p.get("prompt", "")).strip()
                rs = str(p.get("response", "")).strip()
                if not pr:
                    continue
                kk = (pr, rs)
                if kk in seen:
                    continue
                add.append({"prompt": p["prompt"], "response": p["response"]})
            if len(kept) + len(add) > cap and kept:
                break  # stop before exceeding; this concept's lesson overflows
            for p in add:
                seen.add((p["prompt"].strip(), p["response"].strip()))
            kept.extend(add)
            kept_lesson_ids.add(lid)
            if len(kept) >= cap:
                break
        union = kept[:cap]
        # Requeue every batch job whose lesson didn't make the cap so it retries.
        overflow_ids = [
            int(j["id"]) for j in jobs if int(j["lesson_id"]) not in kept_lesson_ids
        ]
        if overflow_ids:
            db.requeue_jobs(overflow_ids, clear_batch=True)
        # Trim surviving to the kept lessons for the per-lesson done/feed loop.
        surviving = [t for t in surviving if t[0] in kept_lesson_ids]

    # ---- recompute budget + renew lease from the ACTUAL union size ----------
    n_union = len(union)
    max_train_seconds = _token_budget(n_union)
    new_ttl = _lease_ttl_for("lesson", n_union)
    try:
        await asyncio.to_thread(db.renew_writer_lease, WORKER_ID, epoch, new_ttl)
    except Exception:  # noqa: BLE001 - renew failure surfaces via the heartbeat's `lost`
        pass

    # ---- 4. resolve base ONCE, replay, single fenced flip -------------------
    current = db.get_current_weights()
    parent_id = current["id"] if current else None
    base_version = current["path"] if current else None

    # Knobs from the NEWEST surviving lesson's kind (a coalesced union has no single
    # kind; use the highest-lesson-id's for the training temperature).
    newest_lid = max(t[0] for t in surviving)
    newest_row = db.get_lesson(newest_lid)
    kind = (newest_row or {}).get("kind") or "fact"
    knobs = dict(settings.LESSON_KIND_KNOBS.get(kind) or settings.LESSON_KIND_KNOBS.get("fact") or {})
    knobs["max_train_seconds"] = max_train_seconds

    replay_pairs = _build_replay_buffer(newest_lid, union)
    payload_pairs = replay_pairs + union

    broadcaster = get_broadcaster(newest_lid)  # union progress rides the newest lesson's channel
    done_event: Optional[dict] = None
    try:
        trainer_cls = _lookup_trainer()
        async for event in _iter_remote_gen(
            trainer_cls, newest_lid, payload_pairs, base_version=base_version, knobs=knobs
        ):
            # Fan progress to EVERY surviving lesson's channel so each teacher's UI
            # sees the shared window train.
            for (lid, _c, _p) in surviving:
                await get_broadcaster(lid).publish(event)
            if isinstance(event, dict) and event.get("type") == "done":
                done_event = event

        if done_event is None:
            raise RuntimeError("coalesced batch: training stream ended without a 'done' event")

        vkind = done_event.get("kind", settings.METHOD)
        path = done_event["path"]
        vid = db.new_weights_version(
            kind=vkind, path=path, parent_id=parent_id, lesson_id=newest_lid,
            final_loss=done_event.get("final_loss"),
        )
        # FENCED flip: only CURRENT-of-record if we still hold the lease.
        if not db.flip_if_lease_held(vid, WORKER_ID, epoch):
            raise RuntimeError(
                f"coalesced batch lost the writer lease before flip; discarding {path}"
            )

        # Per-lesson success: status + feed for EACH surviving lesson.
        for (lid, _c, pairs) in surviving:
            try:
                db.set_lesson_status(lid, "done")
                feed_line = await _feed_description(lid, pairs)
                db.add_feed(lid, feed_line)
            except Exception:  # noqa: BLE001 - per-lesson feed is non-critical
                logging.warning("post-train feed failed for lesson %s", lid, exc_info=True)
        # Finish the batch's job rows that trained (overflow was already requeued).
        trained_ids = [
            int(j["id"]) for j in jobs
            if int(j["lesson_id"]) in {t[0] for t in surviving}
        ]
        db.finish_batch_jobs(batch_id, "done")
        # (finish_batch_jobs marks all still-'claimed' rows in the batch done; the
        #  requeued overflow rows already left 'claimed', so they are untouched.)
        for (lid, _c, _p) in surviving:
            get_broadcaster(lid).close()

    except Exception as exc:  # noqa: BLE001
        await _handle_batch_failure(batch_id, jobs, surviving, exc)


async def _handle_batch_failure(
    batch_id: str, jobs: list[dict], surviving: list[tuple], exc: BaseException
) -> None:
    """Bisection poison isolation (PR-8 §B.6): halve + requeue, don't fan-out.

    - TERMINAL (OOM/CUDA): if the batch is a SINGLETON, mark it error terminal
      (and quarantine it via bisected_singleton). A multi-job OOM bisects to find
      the oversized lesson.
    - TRANSIENT: bisect the batch's jobs into two halves and requeue each (they
      re-batch next window); ~log2(N) passes isolate the bad job. A job that has
      bisected to a singleton and failed again is quarantined so it doesn't rejoin
      every future window until attempts exhaust.
    """
    cls = _classify_failure(exc)
    err = str(exc)
    trained_job_ids = [
        int(j["id"]) for j in jobs
        if int(j["lesson_id"]) in {t[0] for t in surviving}
    ]
    # Only the jobs that actually entered training share the failure; blocked/
    # dropped jobs were already finished above.
    active = [j for j in jobs if int(j["id"]) in set(trained_job_ids)] or jobs
    max_attempts = settings.TRAIN_JOB_MAX_ATTEMPTS

    if len(active) <= 1:
        job = active[0]
        lid = int(job["lesson_id"])
        attempts = int(job.get("attempts", 1))
        already_isolated = bool(job.get("bisected_singleton"))
        if cls == "terminal" or already_isolated or attempts >= max_attempts:
            # Terminal for this singleton: quarantine so it can't rejoin windows.
            db.mark_jobs_terminal([int(job["id"])], err)
            try:
                db.set_lesson_status(lid, "error")
            except Exception:  # noqa: BLE001
                pass
            await get_broadcaster(lid).publish(
                {"type": "error", "lesson_id": lid, "error": err}
            )
            get_broadcaster(lid).close()
        else:
            # Transient singleton with attempts left: requeue, flag isolated so a
            # further failure quarantines it immediately (no attempts-exhaust wait).
            db.requeue_jobs([int(job["id"])], clear_batch=True, bisected_singleton=True)
            await get_broadcaster(lid).publish(
                {"type": "retry", "lesson_id": lid, "attempt": attempts, "error": err}
            )
        return

    # Multi-job batch: bisect into halves and requeue both (they re-batch). This is
    # log2(N), NOT a fan-out to N singles. OOM in a multi-job batch is treated the
    # same (bisect to find the oversized lesson; only the failing singleton is
    # eventually marked terminal).
    ids = [int(j["id"]) for j in active]
    mid = len(ids) // 2
    left, right = ids[:mid], ids[mid:]
    db.requeue_jobs(left, clear_batch=True)
    db.requeue_jobs(right, clear_batch=True)
    for j in active:
        lid = int(j["lesson_id"])
        await get_broadcaster(lid).publish(
            {"type": "retry", "lesson_id": lid,
             "attempt": int(j.get("attempts", 1)), "error": err}
        )


async def _worker_loop(stop: "asyncio.Event") -> None:
    """Coalescing single-writer loop (PR-8).

    Each iteration acquires the SINGLE writer lease FIRST (one acquire gates BOTH
    branches), starts a wall-clock heartbeat spanning the whole critical section,
    then dispatches — consolidation-first (never coalesced), else a coalesced
    lesson batch with a hard T0 window. The lease is the batch's critical-section
    gate: with ``--workers>1`` two workers can't claim disjoint batches and both
    flip. The lease + fenced flip together make double-flip impossible. Released in
    ``finally`` (epoch-guarded) with the heartbeat stopped in the same block.
    """
    # Periodic lease-aware reaper so a dead peer's stale lease/jobs recover even
    # while this worker is otherwise idle.
    last_reap = 0.0
    while not stop.is_set():
        # Opportunistic periodic reap (lease-aware; never touches a live holder).
        now = asyncio.get_event_loop().time()
        if now - last_reap > max(5.0, settings.TRAIN_POLL_INTERVAL * 5):
            try:
                await asyncio.to_thread(db.reap_stale_leases_and_jobs, _lease_max_ttl())
            except Exception:  # noqa: BLE001
                logging.warning("periodic reap failed", exc_info=True)
            last_reap = now

        # LEASE-CHURN FIX: peek (lock-free) for claimable work BEFORE acquiring the
        # writer lease. The old loop acquired the lease every poll, before knowing
        # if any job existed — bumping writer_lease.epoch and spinning a heartbeat
        # ~1/sec while idle (observed epoch=349 with zero jobs). These peeks are a
        # non-authoritative hint; the authoritative atomic claim still runs under
        # the held lease below, so a job appearing/vanishing between peek and claim
        # is handled by the existing claim/release paths. Idle workers now do two
        # cheap SELECTs/sec and never touch the epoch until real work lands.
        has_c = await asyncio.to_thread(
            db.has_queued_consolidation, settings.TRAIN_JOB_MAX_ATTEMPTS
        )
        has_l = await asyncio.to_thread(
            db.has_queued_lesson, settings.TRAIN_JOB_MAX_ATTEMPTS
        )
        if not (has_c or has_l):
            await _sleep_or_stop(stop)
            continue

        # Acquire the lease with a generous initial TTL (covers coldstart+augment+
        # train+flip); the batch renews it from the actual union size once known.
        init_ttl = _lease_ttl_for("consolidate")  # max of the two kinds' TTLs
        try:
            epoch = await asyncio.to_thread(db.acquire_writer_lease, WORKER_ID, init_ttl)
        except Exception:  # noqa: BLE001 - a claim/lease error must never kill the loop
            epoch = None
        if epoch is None:
            await _sleep_or_stop(stop)
            continue

        hb = _LeaseHeartbeat(WORKER_ID, epoch, init_ttl)
        hb.start()
        try:
            # Consolidation-first (never coalesced), UNDER this same lease.
            cjob = await asyncio.to_thread(
                db.claim_next_consolidation, WORKER_ID, settings.TRAIN_JOB_MAX_ATTEMPTS
            )
            if cjob is not None:
                await _run_consolidation(cjob, epoch)
                continue

            # Lesson batch: hard window cutoff T0 captured AFTER lease acquire.
            t0 = db._now()
            batch = await asyncio.to_thread(
                db.claim_next_batch, WORKER_ID, settings.TRAIN_JOB_MAX_ATTEMPTS,
                settings.COALESCE_MAX_JOBS, t0,
            )
            if batch is None:
                await _sleep_or_stop(stop)
                continue
            await _run_batch(batch, epoch, stop)
        except Exception:  # noqa: BLE001 - never let a run error kill the loop
            logging.warning("worker iteration failed", exc_info=True)
        finally:
            await hb.stop()
            try:
                await asyncio.to_thread(db.release_writer_lease, WORKER_ID, epoch)
            except Exception:  # noqa: BLE001
                logging.warning("release_writer_lease failed", exc_info=True)


async def _sleep_or_stop(stop: "asyncio.Event") -> None:
    """Sleep TRAIN_POLL_INTERVAL or return early if ``stop`` is set."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=settings.TRAIN_POLL_INTERVAL)
    except asyncio.TimeoutError:
        pass


def start_worker() -> None:
    """Start the background training worker (idempotent).

    Runs the LEASE-AWARE reaper (never requeues a live lease-holder's claimed
    rows — the fix for the >1-worker startup race where blind recovery would yank
    an in-flight batch out from under its holder), then launches the coalescing
    loop as an asyncio task. Call once from FastAPI startup.
    """
    global _worker_task, _worker_stop
    if _worker_task is not None and not _worker_task.done():
        return
    try:
        recovered = db.reap_stale_leases_and_jobs(_lease_max_ttl())
    except Exception:  # noqa: BLE001 - recovery must never block startup
        recovered = 0
        logging.warning("startup reap_stale_leases_and_jobs failed", exc_info=True)
    if recovered:
        print(f"[training] recovered {recovered} stale job(s) -> requeued")
    _worker_stop = asyncio.Event()
    _worker_task = asyncio.create_task(_worker_loop(_worker_stop))


async def acquire_reset_lease(retries: int = 40, backoff_s: float = 0.25) -> Optional[int]:
    """Acquire the writer lease for an admin reset (PR-8), returning its epoch.

    Gates reset through the SAME single-owner lease as lesson/consolidation writes
    so no other process's worker can flip CURRENT concurrently with the wipe.
    Acquiring bumps the fencing epoch, so ANY in-flight writer that trained against
    the pre-reset generation will fail its fenced flip. Retries briefly if another
    worker currently holds the lease (its heartbeat keeps it alive, but a batch is
    bounded, so a short wait wins it). Returns the epoch, or None if it couldn't be
    acquired in time (reset proceeds best-effort — this process's worker is already
    drained, which is the common single-process case)."""
    ttl = settings.WRITER_LEASE_RESET_TTL_S
    for _ in range(max(1, retries)):
        try:
            epoch = await asyncio.to_thread(db.acquire_writer_lease, WORKER_ID, ttl)
        except Exception:  # noqa: BLE001
            epoch = None
        if epoch is not None:
            return epoch
        await asyncio.sleep(backoff_s)
    logging.warning("acquire_reset_lease: could not acquire writer lease; proceeding")
    return None


def release_reset_lease(epoch: int) -> None:
    """Release the reset-held writer lease (epoch-guarded)."""
    try:
        db.release_writer_lease(WORKER_ID, epoch)
    except Exception:  # noqa: BLE001
        logging.warning("release_reset_lease failed", exc_info=True)


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
