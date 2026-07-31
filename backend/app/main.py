"""FastAPI control plane for Unrestricted AI.

Wires together every backend module per the contract (§7):

    config   -> settings
    db       -> persistence (conversations, lessons, pairs, weights, feed)
    llm      -> OpenRouter chat brain + teacher pair generation
    pipeline -> augmentation + guardrail
    training -> Modal trainer bridge + per-lesson WebSocket fan-out

Endpoint contract (PROJECT_PLAN §10):

    POST  /api/chat                       chat brain; surfaces a tool call only
    POST  /api/lessons                    augment -> guardrail -> enqueue training
    WS    /api/train/stream/{lesson_id}   live progress relay
    GET   /api/weights/current            current weights pointer
    POST  /api/weights/revert             flip current weights pointer
    POST  /api/learned                    append a "Recently Learned" feed row
    GET   /api/learned                    list latest feed rows

Design notes:
- ``/api/chat`` NEVER enqueues training and NEVER persists chat (transcripts live
  in the browser now; the client sends recent history on each request). It returns
  the assistant reply plus an optional parsed ``create_training_pairs`` tool call.
  The client (or a higher-level policy) then POSTs ``/api/lessons`` to actually run
  the augment -> guard -> enqueue flow.
- ``/api/lessons`` is the single place where the guardrail decision gates training:
  only ``guardrail_status == "allowed"`` pairs are forwarded to the trainer.
- The WebSocket endpoint is a thin proxy over ``training.stream_lesson`` and forwards
  the trainer's progress/done event dicts unchanged (cross-file invariant #1).
"""

import json
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.config import settings
from backend.app import db, llm, pipeline, training

# Operator observability (§9 M1): without a root logging config the module-level
# ``logging.warning(...)`` calls scattered across training/llm go nowhere, so an
# operator has zero signal when chat stops. Configure INFO-level logging at
# import so those surface. ``force=True`` wins over a prior no-op config a WSGI/
# ASGI host may have installed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("unrestricted.main")

# --------------------------------------------------------------------------- #
# App + lifecycle
# --------------------------------------------------------------------------- #

app = FastAPI(title="Unrestricted AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    """Init the schema, reconcile the weights pointer, launch the worker.

    ``training.start_worker`` recovers any jobs left ``claimed`` by a crashed
    process, then runs the single-consumer loop that serializes finetunes.

    Reconciliation self-heals a DB/volume pointer desync — e.g. after an
    out-of-band ``reset.sh`` wiped the volume while the DB still flags a row
    current, or a crash between the volume flip and the DB write. Inference reads
    the VOLUME pointer, so the DB is made to agree with it. Best-effort: a Modal
    outage at boot must never block startup.
    """
    db.init_db()
    await _reconcile_weights_pointer()
    training.start_worker()


async def _reconcile_weights_pointer() -> None:
    """Make the DB ``is_current`` row agree with the volume CURRENT pointer."""
    try:
        vol_version = await training.read_current_version()
    except Exception:  # noqa: BLE001 - boot must not depend on Modal
        return
    db_current = db.get_current_weights()
    if vol_version is None:
        # Volume has no learned version (fresh / wiped): clear any stale DB flag.
        if db_current is not None:
            db.clear_current_weights()
        return
    if db_current and db_current.get("path") == vol_version:
        return  # already in agreement
    row = db.get_weights_version_by_path(vol_version)
    if row:
        db.set_current_weights(row["id"])
        return
    # The volume names a version the DB never recorded — a crash BETWEEN the
    # trainer's volume flip (source of truth: it committed vN + its shards) and the
    # backend DB flip. The volume drives inference (correct), but if we leave the
    # DB pointing at the stale vK, the NEXT lesson batch resolves base_version=vK
    # and trains on a stale base while the volume/_next_version advance past vN —
    # orphaning the committed vN from the accumulation chain (a silent ONE-SHARED-
    # BRAIN violation). Heal by recording a reconciled row for vN and pointing the
    # DB at it, so base resolution and the volume agree. parent = the prior DB
    # current (best-effort provenance); lesson_id NULL (owning lesson unknown).
    try:
        parent_id = db_current["id"] if db_current else None
        vid = db.new_weights_version(
            kind="full", path=vol_version, parent_id=parent_id, lesson_id=None,
        )
        db.set_current_weights(vid)
    except Exception:  # noqa: BLE001 - reconciliation is best-effort; volume still serves
        logging.warning(
            "reconcile: could not heal DB for volume CURRENT=%s", vol_version,
            exc_info=True,
        )


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Stop the training worker cleanly so in-flight claims aren't orphaned."""
    await training.stop_worker()


# --------------------------------------------------------------------------- #
# Pydantic request/response models
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat``.

    Chat transcripts now live in the browser (localStorage); the client sends the
    recent turns it wants the model to see via ``history`` (OLDEST->NEWEST, NOT
    including the new ``message``). The server no longer stores chat. ``client_id``
    is the stable per-browser UUID used for the lesson rate cap. ``conversation_id``
    is kept only for backward-compat / a synthesized echo — nothing is persisted.
    """

    conversation_id: Optional[int] = None
    message: str
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    history: Optional[list[dict]] = None  # [{"role": "user"|"assistant", "content": str}]


class ToolCallOut(BaseModel):
    """A parsed ``create_training_pairs`` tool call surfaced to the client."""

    concept: str
    kind: str = "fact"  # fact | style | behavior — selects training knobs
    num_pairs: int
    core_ratio: float = 0.4  # fraction of pairs that restate the literal claim
    pairs: list[dict]  # [{"prompt", "response"}, ...]
    summary: str


class ChatResponse(BaseModel):
    """Response for ``POST /api/chat``."""

    conversation_id: int
    reply: str
    tool_call: Optional[ToolCallOut] = None  # present if the model wants to teach


class LessonRequest(BaseModel):
    """Body for ``POST /api/lessons`` — the ``create_training_pairs`` payload."""

    conversation_id: Optional[int] = None
    client_id: Optional[str] = None  # stable per-browser id; keys the rate cap
    concept: str
    kind: str = "fact"  # fact | style | behavior — selects training knobs
    num_pairs: int
    core_ratio: float = 0.4  # fraction of pairs that restate the literal claim
    pairs: list[dict]
    summary: str


class LessonResponse(BaseModel):
    """Response for ``POST /api/lessons``."""

    lesson_id: int
    status: str  # "queued" | "blocked"
    num_pairs: int  # final augmented count
    blocked_reason: Optional[str] = None


class TrainStatusResponse(BaseModel):
    """Response for ``GET /api/train/status/{lesson_id}``.

    Lets a refreshed tab resolve a lesson whose training WebSocket already closed.
    ``status`` mirrors the lessons row (queued|training|done|blocked|error).
    ``version`` is the trained weights version string (``weights_versions.path``)
    once done; ``final_loss`` is the last training loss persisted on that weights
    row (also delivered live on the training WS 'done' frame).
    """

    lesson_id: int
    status: str
    version: Optional[str] = None
    final_loss: Optional[float] = None
    blocked_reason: Optional[str] = None


class WeightsResponse(BaseModel):
    """Response for the weights endpoints. All fields ``None`` if no version yet."""

    version_id: Optional[int]
    kind: Optional[str]
    path: Optional[str]
    created_at: Optional[str]


class RevertRequest(BaseModel):
    """Body for ``POST /api/weights/revert``."""

    version_id: int


class FeedItem(BaseModel):
    """One row of the "Recently Learned" feed."""

    id: int
    lesson_id: Optional[int]
    summary: str
    created_at: str


class LearnedCreate(BaseModel):
    """Body for ``POST /api/learned``."""

    lesson_id: int
    summary: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _weights_response(row: Optional[dict]) -> WeightsResponse:
    """Map a ``weights_versions`` row dict to a :class:`WeightsResponse`.

    Returns an all-``None`` response when there is no current version.
    """
    if not row:
        return WeightsResponse(version_id=None, kind=None, path=None, created_at=None)
    return WeightsResponse(
        version_id=row.get("id"),
        kind=row.get("kind"),
        path=row.get("path"),
        created_at=row.get("created_at"),
    )


def _feed_item(row: dict) -> FeedItem:
    """Map a ``learned_feed`` row dict to a :class:`FeedItem`."""
    return FeedItem(
        id=row["id"],
        lesson_id=row.get("lesson_id"),
        summary=row["summary"],
        created_at=row["created_at"],
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Chat brain: reply from CLIENT-supplied history + surface a teaching tool call.

    Chat transcripts live in the browser now, so this endpoint persists NOTHING —
    it builds the model context from ``req.history`` (the recent turns the client
    holds) and returns the assistant reply plus an optional parsed teaching tool
    call. ``conversation_id`` is echoed back (synthesized) so the response model
    stays stable, but no conversation/message row is written.

    Flow:
      1. Build context from the client history + the new message (compacting long
         history via the teacher summarizer).
      2. Call the learned student (answer) and :func:`llm.chat_with_tool` (detector)
         concurrently, each with that history.
      3. Return the assistant reply plus the optional parsed tool call.

    This endpoint does NOT augment, guard, or enqueue training — that is the job of
    ``POST /api/lessons``, which the client calls when it wants to act on a tool call.
    """
    import asyncio

    # No server-side conversation anymore; echo the client's id (or 0).
    conversation_id = req.conversation_id if req.conversation_id is not None else 0

    # 1. Build context from the CLIENT history (nothing is persisted).
    messages = await _build_context(req.history, req.message)

    # 2. Hybrid brain (PROJECT_PLAN §4): the learned tiny model on Modal writes
    #    the ACTUAL reply (so teaching visibly changes its answers), while the
    #    capable OpenRouter teacher independently watches for teaching intent and
    #    emits create_training_pairs. Run both concurrently, each with history.
    answer_task = asyncio.create_task(training.infer_chat(messages))
    detect_task = asyncio.create_task(llm.chat_with_tool(messages))
    learned_reply, result = await asyncio.gather(answer_task, detect_task)

    # The learned model's answer wins; if Modal is unreachable it returns "",
    # in which case fall back to the detector's text so chat still responds.
    reply_text: str = learned_reply or (result.get("text") or "")
    tool_call = result.get("tool_call")

    # 3. Build the response (nothing is persisted — chat lives in the browser).
    tool_call_out: Optional[ToolCallOut] = None
    if tool_call:
        tool_call_out = ToolCallOut(
            concept=tool_call["concept"],
            kind=tool_call.get("kind", "fact"),
            num_pairs=tool_call["num_pairs"],
            core_ratio=tool_call.get("core_ratio", 0.4),
            pairs=tool_call["pairs"],
            summary=tool_call["summary"],
        )

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        tool_call=tool_call_out,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat: emit the learned model's reply token-by-token (SSE).

    Server-Sent Events:
      - ``token``  : {"text": "<chunk>"}  incremental reply text
      - ``meta``   : {"conversation_id", "tool_call"}  sent once at the end;
                     ``tool_call`` is the parsed create_training_pairs payload or
                     null. The teaching detector runs concurrently with the
                     stream so the tool call is ready by the time tokens finish.
      - ``done``   : {}  terminal marker
    """
    import asyncio

    # No server-side conversation anymore; echo the client's id (or 0) in meta.
    conversation_id = req.conversation_id if req.conversation_id is not None else 0

    # Build context from the CLIENT-supplied history (compacting older turns if
    # long). Nothing is persisted — chat transcripts live in the browser.
    messages = await _build_context(req.history, req.message)

    # Kick off the teaching detector immediately; it resolves while we stream.
    detect_task = asyncio.create_task(llm.chat_with_tool(messages))

    async def event_gen():
        # Detect-first (§4). Resolve teaching intent BEFORE the first token so a
        # teaching turn can ACK instead of streaming the not-yet-trained student's
        # pushback (SSE is one-way — tokens can't be recalled). Bounded by
        # DETECT_ACK_TIMEOUT_S so a normal chat turn never waits when the detector
        # is slow: on timeout/error result=None and we stream exactly as before,
        # resolving the detector afterwards for meta.
        result = None
        try:
            result = await asyncio.wait_for(
                detect_task, timeout=settings.DETECT_ACK_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001 - TimeoutError included; unknown -> stream as today
            result = None

        # ``chat_with_tool`` already applies the TEACH_THRESHOLD gate: a non-None
        # tool_call means a confident teaching turn (low-confidence guesses arrive
        # as tool_call=None). So no re-check of confidence is needed here.
        early_tool_call = (result or {}).get("tool_call")

        collected: list[str] = []
        if early_tool_call:
            # TEACHING TURN: never stream the untrained student (it would push back,
            # e.g. "no, 1+1 is 2", and that can't be recalled). Stream a canned
            # enthusiastic ACK and use it as the reply text. The lesson trains via
            # the tool_call in meta below.
            ack = "Got it — I'll remember that! Give me a moment to learn it…"
            for piece in _chunk_text(ack):
                collected.append(piece)
                yield _sse("token", {"text": piece})
        else:
            # NORMAL TURN (or the detector didn't resolve in the ACK budget):
            # stream the student reply exactly as before.
            try:
                async for chunk in training.infer_chat_stream(messages):
                    collected.append(chunk)
                    yield _sse("token", {"text": chunk})
            except Exception:  # noqa: BLE001 - keep the stream alive; fall through
                pass

        reply_text = "".join(collected).strip()

        # Resolve the teaching detector if the ACK-budget wait_for didn't already.
        if result is None:
            try:
                result = await detect_task
            except Exception:  # noqa: BLE001
                result = {"text": "", "tool_call": None}

        if not reply_text:
            # Modal unreachable or empty stream: fall back to the detector text.
            reply_text = result.get("text") or ""
            if reply_text:
                yield _sse("token", {"text": reply_text})

        if not reply_text:
            # Both the model stream AND the detector text came back empty (Modal
            # down, a checkpoint that emits nothing, etc.). Never persist/return an
            # empty assistant message — that shows a blank bubble and looks broken.
            # Emit a fixed friendly line so the chat always says SOMETHING.
            reply_text = "Sorry — I blanked on that one. Try again?"
            yield _sse("token", {"text": reply_text})

        tool_call = result.get("tool_call")
        # The teaching detector already applies its confidence gate
        # (TEACH_THRESHOLD): a below-threshold detection arrives here as
        # tool_call=None. Surface the confidence so downstream (PR-4) can branch on
        # {tool_call, confidence} without re-running the detector.
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Nothing is persisted — chat transcripts live in the browser now.
        meta = {
            "conversation_id": conversation_id,
            "tool_call": tool_call,  # already a plain dict or None
            "confidence": confidence,
        }
        yield _sse("meta", meta)
        yield _sse("done", {})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _chunk_text(text: str) -> list[str]:
    """Split a canned reply into word-sized chunks so the ACK streams like tokens.

    The frontend appends token frames verbatim, so splitting on spaces (keeping the
    trailing space on each piece) reconstructs the exact string when concatenated.
    """
    parts = text.split(" ")
    return [p + " " if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _build_context(
    history: Optional[list[dict]], new_message: str
) -> list[llm.ChatMessage]:
    """Build the message list for the chat brains from CLIENT-supplied history.

    Chat transcripts live in the browser now, so ``history`` is the recent prior
    turns the client sends (OLDEST->NEWEST, NOT including ``new_message``). When
    the running history exceeds ``COMPACT_CHARS``, the older turns are summarized
    via the OpenRouter teacher into a single system context note and only the most
    recent ``COMPACT_KEEP_RECENT`` turns are kept verbatim. This keeps the prompt
    inside the small student model's context window so chats can run indefinitely.
    Returns the message list ending with the new user turn.

    The old "already taught concepts" system note (reconstructed from persisted
    tool_call_json rows) is dropped: the client doesn't send tool calls, and the
    redundant-reteach gate was already removed — a re-teach simply retrains, which
    is acceptable.
    """
    # Normalize the client history: keep only well-formed user/assistant turns
    # with non-empty content. ``history=None`` (a fresh chat) yields [].
    raw = history or []
    normalized: list[dict] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            normalized.append({"role": role, "content": str(content)})

    # Bound the history to the same recency window the DB path used.
    if settings.CHAT_HISTORY_LIMIT and len(normalized) > settings.CHAT_HISTORY_LIMIT:
        normalized = normalized[-settings.CHAT_HISTORY_LIMIT:]

    total_chars = sum(len(m["content"]) for m in normalized)
    if total_chars > settings.COMPACT_CHARS and len(normalized) > settings.COMPACT_KEEP_RECENT:
        keep = settings.COMPACT_KEEP_RECENT
        older, recent = normalized[:-keep], normalized[-keep:]
        summary = await llm.summarize_history(older)
        msgs: list[llm.ChatMessage] = []
        if summary:
            msgs.append({"role": "system", "content": f"Earlier in this chat: {summary}"})
        msgs.extend(recent)
    else:
        msgs = list(normalized)

    msgs.append({"role": "user", "content": new_message})
    return msgs




@app.post("/api/lessons", response_model=LessonResponse)
async def create_lesson(req: LessonRequest) -> LessonResponse:
    """Reputation-gate -> persist queued -> enqueue; augmentation runs in the worker.

    Two things happen synchronously on the request path, both FAST:

      1. **Reputation gate (§9 B1)** over the lesson INTENT (concept + the user's
         seed/summary). This is the product's ONE content control and it is
         NARROW: it blocks only lessons trying to teach racist / misogynistic /
         hateful / reputationally-damaging content. Teaching false facts, 1+1=3,
         edgy styles, etc. is the POINT and is never gated. A cheap keyword
         pre-filter catches blatant cases; otherwise a tight semantic classifier
         (``llm.classify_reputation``, ~300ms) decides. It FAILS OPEN so it can
         never take the toy offline. If blocked, the lesson is persisted
         ``"blocked"`` and returned — nothing is enqueued.

      2. **Persist + enqueue.** The heavy multi-facet Gemini fanout
         (``pipeline.build_training_pairs``, up to ~60s) used to be AWAITED here,
         which starves chat under many concurrent teachers (§9 B2). It now runs
         INSIDE the worker (``training._run_job``) before the finetune. So this
         path only persists the lesson ``"queued"`` with the SEED pairs and
         enqueues a job envelope carrying the seeds + generation params, then
         returns immediately. The worker augments, runs the pair-level guardrail
         (``check_pairs``, defense-in-depth), and trains.
    """
    # 0. Per-browser rate cap (PROJECT_PLAN §8): refuse runaway teaching. Chat
    #    history lives client-side now, so the cap keys on ``client_id`` (the stable
    #    per-browser UUID). Falls back to the legacy ``conversation_id`` keying for
    #    an older client that doesn't send a client_id, so nothing breaks.
    if settings.LESSON_RATE_MAX > 0 and (
        req.client_id is not None or req.conversation_id is not None
    ):
        from datetime import datetime, timedelta, timezone

        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=settings.LESSON_RATE_WINDOW_S)
        ).isoformat()
        if req.client_id is not None:
            recent = db.count_recent_jobs_for_client(req.client_id, since)
            scope = "browser"
        else:
            recent = db.count_recent_jobs_for_conversation(req.conversation_id, since)
            scope = "conversation"
        if recent >= settings.LESSON_RATE_MAX:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit: at most {settings.LESSON_RATE_MAX} lessons per "
                    f"{settings.LESSON_RATE_WINDOW_S}s per {scope}."
                ),
            )

    # 1. Code-computed augmentation knobs from the lesson KIND (the detector's
    #    numeric guesses were ungrounded + clamped anyway). MIN/MAX clamp is the
    #    safety net. These ride along in the job envelope; the worker augments.
    kind = req.kind if req.kind in ("fact", "style", "behavior") else "fact"
    kind_defaults = settings.KIND_DEFAULTS.get(kind) or settings.KIND_DEFAULTS.get("fact") or {}
    target = int(kind_defaults.get("num_pairs", settings.NUM_PAIRS) or settings.NUM_PAIRS)
    target = max(settings.MIN_PAIRS, min(settings.MAX_PAIRS, target))
    try:
        core_ratio = float(kind_defaults.get("core_ratio", 0.4))
    except (TypeError, ValueError):
        core_ratio = 0.4
    core_ratio = min(1.0, max(0.0, core_ratio))

    # Ground the teacher with the concept + the detector's own seed examples so
    # the worker's fanout generates diverse pairs ON-TOPIC.
    seed_preview = "; ".join(
        f"Q: {p.get('prompt','')} A: {p.get('response','')}" for p in req.pairs[:5]
    )
    user_context = f"Summary: {req.summary}\nExamples: {seed_preview}"

    # 2. REPUTATION GATE (§9 B1) — the ONE narrow content control, on the fast
    #    request path so obviously-bad lessons are rejected synchronously. Judge
    #    the user's INTENT, not the (not-yet-generated) augmented pairs.
    intent_text = f"{req.concept}\n{req.summary}\n{seed_preview}"
    blocked_reason: Optional[str] = None
    kw_ok, _kw_cat, kw_reason = pipeline.check_intent_keywords(intent_text)
    if not kw_ok:
        blocked_reason = kw_reason
    else:
        verdict = await llm.classify_reputation(
            req.concept, user_message=req.summary or "", seed_summary=seed_preview
        )
        if verdict.get("block"):
            blocked_reason = (
                verdict.get("reason")
                or "Blocked: hateful or reputationally-damaging content."
            )

    if blocked_reason is not None:
        # Persist the lesson blocked, record the seed pairs as blocked so the
        # table reflects the decision, and return WITHOUT enqueueing.
        lesson_id = db.create_lesson(
            req.conversation_id,
            req.concept,
            req.summary,
            len(req.pairs),
            status="blocked",
            kind=kind,
            client_id=req.client_id,
        )
        _, _, seed_records = pipeline.check_pairs(req.pairs)
        # Mark every seed record blocked with the intent-level reason (the pair
        # scan above may pass them; the block is on the INTENT).
        for rec in seed_records:
            rec["guardrail_status"] = "blocked"
            rec["reason"] = blocked_reason
            rec["source"] = rec.get("source") or "model"
        db.add_pairs(lesson_id, seed_records)
        return LessonResponse(
            lesson_id=lesson_id,
            status="blocked",
            num_pairs=len(req.pairs),
            blocked_reason=blocked_reason,
        )

    # 3. Persist the lesson ``queued`` with the SEED pairs (the worker generates
    #    the full augmented set). num_pairs is the TARGET estimate for now; the
    #    worker updates the row to the real augmented count once it fans out.
    lesson_id = db.create_lesson(
        req.conversation_id,
        req.concept,
        req.summary,
        target,
        status="queued",
        kind=kind,
        client_id=req.client_id,
    )

    # 4. Enqueue the augmentation+train job envelope (§9 B2). The worker runs
    #    build_training_pairs + check_pairs before the finetune, off the request
    #    path. Durable: survives restarts; claimed single-writer.
    #    PR-8 NOTE: the client_id rate cap above is enforced HERE, on the request
    #    path, and is ORTHOGONAL to coalescing — each lesson MUST retain its own
    #    training_jobs row (one enqueue per lesson) so count_recent_jobs_for_client
    #    stays accurate. Coalescing groups jobs only at CLAIM time; it never
    #    collapses N lessons into one job row (that would 10x the effective cap).
    # Bound each seed pair's text (PR-8 §B.6 poison bound): one pasted wall-of-text
    # can OOM a coalesced window's union train. Truncate over-long prompt/response
    # to MAX_PAIR_TEXT_BYTES (UTF-8) BEFORE enqueue so the OOM-poison never enters
    # the queue. Truncation keeps the head (the teaching intent), not a hard reject.
    _cap = settings.MAX_PAIR_TEXT_BYTES

    def _bound(s: str) -> str:
        b = s.encode("utf-8")
        if len(b) <= _cap:
            return s
        return b[:_cap].decode("utf-8", errors="ignore")

    seed_pairs = [
        {"prompt": _bound(str(p.get("prompt", ""))),
         "response": _bound(str(p.get("response", "")))}
        for p in req.pairs
        if p.get("prompt") is not None and p.get("response") is not None
    ]
    training.enqueue_lesson_augment(
        lesson_id,
        seed_pairs=seed_pairs,
        concept=req.concept,
        kind=kind,
        target=target,
        core_ratio=core_ratio,
        user_context=user_context,
    )

    return LessonResponse(
        lesson_id=lesson_id,
        status="queued",
        num_pairs=target,
        blocked_reason=None,
    )


@app.websocket("/api/train/stream/{lesson_id}")
async def train_stream(websocket: WebSocket, lesson_id: int) -> None:
    """Relay a lesson's live training progress to a WebSocket subscriber.

    Accepts the socket, subscribes to the lesson broadcaster via
    :func:`training.stream_lesson`, and forwards each progress/``done``/error event
    dict unchanged with ``send_json`` until the stream's close sentinel or a client
    disconnect.
    """
    await websocket.accept()
    try:
        async for event in training.stream_lesson(lesson_id):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        # Client went away; nothing to clean up here (stream_lesson unsubscribes).
        return
    finally:
        # Best-effort close; ignore errors if the socket is already gone.
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/api/train/status/{lesson_id}", response_model=TrainStatusResponse)
async def train_status(lesson_id: int) -> TrainStatusResponse:
    """Resolve a lesson's training outcome after its WebSocket has closed.

    A refreshed tab lost its live training WS, so it polls this read-only endpoint
    to learn whether the job finished. Returns the lessons-row ``status`` plus, when
    ``done``, the trained ``version`` (``weights_versions.path``) and ``final_loss``
    (persisted on the weights row) for that lesson, and the ``blocked_reason`` when
    ``blocked``.

    404s on an unknown lesson id (the frontend tolerates that). Never 500s.
    """
    lesson = db.get_lesson(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Unknown lesson_id")

    status = str(lesson.get("status") or "queued")

    version: Optional[str] = None
    final_loss: Optional[float] = None
    if status == "done":
        row = db.get_weights_version_by_lesson(lesson_id)
        if row:
            version = row.get("path")
            fl = row.get("final_loss")
            final_loss = float(fl) if fl is not None else None

    # ``lessons`` has no dedicated blocked-reason column; the block reason is only
    # returned inline by POST /api/lessons and mirrored onto the pairs' ``reason``.
    # Surface a pair-level reason when the lesson is blocked so the client sees why.
    blocked_reason: Optional[str] = None
    if status == "blocked":
        try:
            for p in db.get_pairs(lesson_id):
                if p.get("guardrail_status") == "blocked" and p.get("reason"):
                    blocked_reason = str(p["reason"])
                    break
        except Exception:  # noqa: BLE001 - status read must not 500
            blocked_reason = None

    return TrainStatusResponse(
        lesson_id=lesson_id,
        status=status,
        version=version,
        final_loss=final_loss,
        blocked_reason=blocked_reason,
    )


@app.post("/api/warmup")
async def warmup() -> dict:
    """Pre-warm the Modal trainer so the user's first chat isn't a cold start.

    Called by the client on page load. Blocks until the container is up and the
    model is loaded (or Modal is unreachable), then returns ``{"ready": bool}``.
    Safe to call repeatedly — a warm container returns near-instantly.
    """
    ready = await training.warmup()
    return {"ready": ready}


@app.get("/api/weights/current", response_model=WeightsResponse)
async def weights_current() -> WeightsResponse:
    """Return the current weights version, or an all-``None`` response if none."""
    row = db.get_current_weights()
    return _weights_response(row)


@app.post("/api/weights/revert", response_model=WeightsResponse)
async def weights_revert(req: RevertRequest) -> WeightsResponse:
    """Flip the current-weights pointer to ``req.version_id`` and return that row.

    Raises ``404`` if ``version_id`` does not correspond to a known version.

    Flips BOTH pointers: the DB ``is_current`` row AND the Modal volume CURRENT
    file. Inference reads the VOLUME pointer (which, with accumulation, defines
    the merge-chain it replays), so a DB-only flip would not actually change what
    the model answers. The volume flip happens first; if it can't be confirmed
    (version not on the volume / Modal down) we refuse so the two pointers never
    diverge.
    """
    row = db.get_weights_version(req.version_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown weights version_id")
    if row.get("pruned"):
        raise HTTPException(
            status_code=410,
            detail=(
                f"Version {row['path']} was pruned by a consolidation and is no "
                "longer on the volume; it can't be reverted to."
            ),
        )

    flipped = await training.set_current_version(row["path"])
    if not flipped:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Could not flip live weights to {row['path']} "
                "(version missing on the volume or trainer unreachable)."
            ),
        )

    db.set_current_weights(req.version_id)
    row = db.get_current_weights()
    if not row or row.get("id") != req.version_id:
        raise HTTPException(status_code=404, detail="Unknown weights version_id")
    return _weights_response(row)


class ConsolidateRequest(BaseModel):
    """Body for ``POST /api/consolidate`` (all fields optional)."""

    # ISO-8601 cutoff: only consolidate lessons trained at/after this time. When
    # omitted, consolidates the ENTIRE accumulated corpus (a full re-derivation).
    since: Optional[str] = None
    # Hours-ago convenience: if ``since`` is unset and this is set, the cutoff is
    # ``now - window_hours`` (e.g. 24 for a nightly "today's lessons" job).
    window_hours: Optional[float] = None


class ConsolidateResponse(BaseModel):
    """Response for ``POST /api/consolidate``."""

    status: str          # "queued" | "noop"
    job_id: Optional[int]
    num_pairs: int


@app.post("/api/consolidate", response_model=ConsolidateResponse)
async def consolidate(req: ConsolidateRequest) -> ConsolidateResponse:
    """Enqueue a nightly consolidation over ALL accumulated, deduped history.

    What consolidation is FOR: the live per-lesson path already does proper
    continual learning (each lesson builds on CURRENT + a replay buffer of prior
    lessons + retention anchors), so facts are solidified as they're taught. The
    nightly pass is a periodic DE-DRIFT: it re-derives ONE clean checkpoint from
    the PRISTINE base over the whole corpus in a single longer/stronger pass,
    undoing the small approximations that accumulate across many sequential
    one-at-a-time edits. Every version is already a self-contained full checkpoint
    (there is NO adapter chain to collapse).

    The backend (which owns the SQLite DB) gathers the corpus here and hands it to
    the Modal trainer through the SAME durable single-writer queue as lessons, so a
    Modal cron (which cannot read local SQLite) drives it by hitting this endpoint.

    Corpus = ALL lessons ever taught, guardrail-allowed, keep-latest-per-prompt
    deduped (a superseded fact is dropped in favour of its newest answer), CAPPED
    to the most recent ``CONSOLIDATE_MAX_CORPUS_PAIRS`` so "all history" stays
    bounded as lessons accumulate forever, plus the fixed RETENTION_ANCHORS. The
    default (no ``window_hours``/``since``) is the intended cumulative memory; a
    caller MAY still pass a window for a scoped re-consolidation.

    Returns ``noop`` (no job) when there are no allowed pairs.
    """
    from datetime import datetime, timedelta, timezone

    since = req.since
    if since is None and req.window_hours:
        since = (
            datetime.now(timezone.utc) - timedelta(hours=req.window_hours)
        ).isoformat()

    # Keep-latest-per-prompt deduped, ordered oldest -> newest.
    pairs = db.get_allowed_pairs_since(since_iso=since, dedupe=True)
    if not pairs:
        return ConsolidateResponse(status="noop", job_id=None, num_pairs=0)

    # Cap to the MOST RECENT N (the tail, since pairs are oldest-first) so the
    # corpus + train time stay bounded as history grows without limit. Newest
    # lessons win the memory budget; the oldest beyond the cap age out.
    cap = settings.CONSOLIDATE_MAX_CORPUS_PAIRS
    if cap and cap > 0 and len(pairs) > cap:
        pairs = pairs[-cap:]

    # Re-anchor general ability: consolidation re-derives from the PRISTINE base,
    # so without these the model would re-forget baseline competence every night.
    # Append after the cap (don't let them be deduped/capped away); tiny + fixed.
    corpus = list(pairs) + [dict(a) for a in settings.RETENTION_ANCHORS]

    job_id = training.enqueue_consolidation(corpus)
    return ConsolidateResponse(status="queued", job_id=job_id, num_pairs=len(corpus))


class ResetResponse(BaseModel):
    """Response for ``POST /api/admin/reset``."""

    ok: bool
    removed: list[str]
    memory_reset: bool
    wiped_chat: bool


@app.post("/api/admin/reset", response_model=ResetResponse)
async def admin_reset(
    wipe_chat: bool = False,
    x_reset_token: Optional[str] = Header(default=None),
) -> ResetResponse:
    """Authoritatively reset the shared brain WITHOUT racing the worker.

    The old path (``reset.sh``) wiped the volume + DB out-of-band while the live
    backend kept running: an in-flight finetune could re-insert a ``v{N}`` row
    AFTER the wipe (re-poisoning the DB), and the warm container kept stale state.
    This endpoint does it in the right order, in one process:

      1. drain the worker (``stop_worker`` cancels/awaits the in-flight job),
      2. wipe the Modal volume + reset the warm container's memory,
      3. clear the DB learning rows + the in-memory broadcasters,
      4. restart the worker.

    Guarded by an ``X-Reset-Token`` header matching ``settings.RESET_TOKEN`` (it
    destroys shared state). Pass ``?wipe_chat=true`` to also drop
    conversations/messages.

    SECURITY (§9 m1): this endpoint FAILS CLOSED. With CORS ``*`` an
    unauthenticated reset would let any web page wipe the shared brain, so when
    ``RESET_TOKEN`` is unset the wipe is REFUSED (503) — UNLESS ``DEV_MODE`` is
    on (the local-dev escape hatch). A real deployment can therefore never wipe
    unauthenticated by merely forgetting to set the token.
    """
    if not settings.RESET_TOKEN:
        if not settings.DEV_MODE:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Reset is disabled: RESET_TOKEN is not set. Set RESET_TOKEN in "
                    "the environment to enable /api/admin/reset (or DEV_MODE=true "
                    "for local dev)."
                ),
            )
        # DEV_MODE + no token: allowed (local dev only).
    elif x_reset_token != settings.RESET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Reset-Token.")

    # 1. Drain THIS process's worker so no finetune is mid-flight while we wipe.
    await training.stop_worker()
    # 1b. Acquire the single writer lease (reset-sized TTL) so NO other process's
    #     worker can hold it and flip concurrently (the >1-worker case that
    #     stop_worker alone doesn't cover). Any batch in flight elsewhere either
    #     finished (and we then wipe its result) or is blocked from flipping (its
    #     fenced flip fails once we bump the epoch by acquiring). Then requeue all
    #     claimed jobs so a post-reset restart can't resurrect a pre-reset job.
    reset_epoch = await training.acquire_reset_lease()
    res: dict = {}
    try:
        # 2. Wipe the volume + warm memory (reset_weights also bumps the Server's
        #    reset EPOCH + wipes CURRENT/LAST_GOOD, invalidating every read cache).
        res = await training.reset_remote()
        # 3. Clear DB learning state + broadcasters. clear_learning_state drops the
        #    training_jobs table too, so claimed/queued rows are gone.
        db.clear_learning_state(wipe_chat=wipe_chat)
        training.clear_broadcasters()
    finally:
        # 4. Release the reset lease, then bring the worker back up.
        if reset_epoch is not None:
            training.release_reset_lease(reset_epoch)
        training.start_worker()

    return ResetResponse(
        ok=True,
        removed=res.get("removed", []),
        memory_reset=bool(res.get("memory_reset")),
        wiped_chat=wipe_chat,
    )


@app.post("/api/learned", response_model=FeedItem)
async def post_learned(req: LearnedCreate) -> FeedItem:
    """Append a row to the "Recently Learned" feed and return it.

    Also fired automatically (DB-side) by ``training.enqueue_lesson`` on success;
    this endpoint exposes the same operation for manual/explicit use.
    """
    feed_id = db.add_feed(req.lesson_id, req.summary)
    for row in db.get_feed(limit=200):
        if row["id"] == feed_id:
            return _feed_item(row)
    # Fallback: the row was just inserted; reconstruct minimally if not found.
    raise HTTPException(status_code=500, detail="Feed row not found after insert")


@app.get("/api/learned", response_model=list[FeedItem])
async def get_learned(limit: int = 50) -> list[FeedItem]:
    """Return the latest ``limit`` feed rows (newest first)."""
    return [_feed_item(row) for row in db.get_feed(limit)]


def _job_counts() -> dict:
    """Count ``training_jobs`` rows by status (queued / claimed / done / error).

    Read-only aggregation for /api/health. Uses the DB connection helper
    directly; returns an empty dict on any error so health never 500s.
    """
    try:
        conn = db._connect()
        try:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM training_jobs GROUP BY status"
            ).fetchall()
        finally:
            conn.close()
        return {str(r["status"]): int(r["n"]) for r in rows}
    except Exception:  # noqa: BLE001 - health must not 500
        logger.warning("health: job-count query failed", exc_info=True)
        return {}


@app.get("/api/health")
async def health() -> dict:
    """Operator health probe (§9 M1).

    Reports, best-effort and without ever 500ing:
      * ``worker_alive``     — is the single-writer training worker task running?
      * ``jobs``             — training_jobs counts by status (queued/claimed/...)
      * ``current_resolves`` — does the DB have a CURRENT weights version?
      * ``current_version``  — the volume's CURRENT pointer (also the Modal ping:
                               a non-null value means Modal answered).
      * ``modal_ok``         — did the Modal ping succeed (best-effort).

    ``status`` is ``"ok"`` when the worker is alive, else ``"degraded"`` — a
    quick single-field signal for uptime checks.
    """
    jobs = _job_counts()
    db_current = None
    current_resolves = False
    try:
        db_current = db.get_current_weights()
        current_resolves = db_current is not None
    except Exception:  # noqa: BLE001
        logger.warning("health: get_current_weights failed", exc_info=True)

    # Best-effort Modal ping: read the volume CURRENT pointer. The helper already
    # swallows Modal errors and returns None on either "unreachable" OR "no
    # learned version yet" (fresh/wiped volume), so a non-null value definitively
    # means Modal answered; None is ambiguous and reported as such.
    modal_current: Optional[str] = await training.read_current_version()
    modal_ok = modal_current is not None

    alive = training.worker_alive()
    return {
        "status": "ok" if alive else "degraded",
        "worker_alive": alive,
        "jobs": jobs,
        "queued": jobs.get("queued", 0),
        "current_resolves": current_resolves,
        "db_current_version": (db_current or {}).get("path") if db_current else None,
        "current_version": modal_current,
        "modal_ok": modal_ok,
    }
