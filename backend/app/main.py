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
- ``/api/chat`` NEVER enqueues training. It only persists messages and returns the
  assistant reply plus an optional parsed ``create_training_pairs`` tool call. The
  client (or a higher-level policy) then POSTs ``/api/lessons`` to actually run the
  augment -> guard -> enqueue flow.
- ``/api/lessons`` is the single place where the guardrail decision gates training:
  only ``guardrail_status == "allowed"`` pairs are forwarded to the trainer.
- The WebSocket endpoint is a thin proxy over ``training.stream_lesson`` and forwards
  the trainer's progress/done event dicts unchanged (cross-file invariant #1).
"""

import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.config import settings
from backend.app import db, llm, pipeline, training

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
    """Init the schema and launch the durable training worker.

    ``training.start_worker`` recovers any jobs left ``claimed`` by a crashed
    process, then runs the single-consumer loop that serializes finetunes.
    """
    db.init_db()
    training.start_worker()


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Stop the training worker cleanly so in-flight claims aren't orphaned."""
    await training.stop_worker()


# --------------------------------------------------------------------------- #
# Pydantic request/response models
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    """Body for ``POST /api/chat``."""

    conversation_id: Optional[int] = None
    message: str
    user_id: Optional[str] = None


class ToolCallOut(BaseModel):
    """A parsed ``create_training_pairs`` tool call surfaced to the client."""

    concept: str
    num_pairs: int
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
    concept: str
    num_pairs: int
    pairs: list[dict]
    summary: str


class LessonResponse(BaseModel):
    """Response for ``POST /api/lessons``."""

    lesson_id: int
    status: str  # "queued" | "blocked"
    num_pairs: int  # final augmented count
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
    """Chat brain: persist the turn and surface an optional teaching tool call.

    Flow:
      1. Ensure a conversation exists (create one if ``conversation_id`` is None).
      2. Persist the incoming user message.
      3. Call :func:`llm.chat_with_tool` with a minimal message history.
      4. Persist the assistant message (with ``tool_call_json`` if the model emitted
         a ``create_training_pairs`` call).
      5. Return the assistant reply plus the optional parsed tool call.

    This endpoint does NOT augment, guard, or enqueue training — that is the job of
    ``POST /api/lessons``, which the client calls when it wants to act on a tool call.
    """
    # 1. Ensure conversation.
    conversation_id = req.conversation_id
    if conversation_id is None:
        conversation_id = db.create_conversation(user_id=req.user_id)

    # 2. Build context (compacting long history, surfacing already-taught
    #    concepts so the detector does not re-teach), then persist the user turn.
    import asyncio

    messages = await _build_context(conversation_id, req.message)
    db.add_message(conversation_id, "user", req.message)

    # 3. Hybrid brain (PROJECT_PLAN §4): the learned tiny model on Modal writes
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

    # 4. Persist the assistant message (+ tool_call_json if present).
    tool_call_json: Optional[str] = json.dumps(tool_call) if tool_call else None
    db.add_message(
        conversation_id,
        "assistant",
        reply_text,
        tool_call_json=tool_call_json,
    )

    # 5. Build the response.
    tool_call_out: Optional[ToolCallOut] = None
    if tool_call:
        tool_call_out = ToolCallOut(
            concept=tool_call["concept"],
            num_pairs=tool_call["num_pairs"],
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

    conversation_id = req.conversation_id
    if conversation_id is None:
        conversation_id = db.create_conversation(user_id=req.user_id)

    # Build context (compacting older turns if the history is long), then persist
    # the new user message.
    messages = await _build_context(conversation_id, req.message)
    db.add_message(conversation_id, "user", req.message)

    # Kick off the teaching detector immediately; it resolves while we stream.
    detect_task = asyncio.create_task(llm.chat_with_tool(messages))

    async def event_gen():
        collected: list[str] = []
        try:
            async for chunk in training.infer_chat_stream(messages):
                collected.append(chunk)
                yield _sse("token", {"text": chunk})
        except Exception:  # noqa: BLE001 - keep the stream alive; fall through
            pass

        reply_text = "".join(collected).strip()

        # Resolve the teaching detector.
        try:
            result = await detect_task
        except Exception:  # noqa: BLE001
            result = {"text": "", "tool_call": None}

        if not reply_text:
            # Modal unreachable or empty stream: fall back to the detector text.
            reply_text = result.get("text") or ""
            if reply_text:
                yield _sse("token", {"text": reply_text})

        tool_call = result.get("tool_call")

        # Backstop: drop ONLY a redundant re-teach (same concept AND same taught
        # answers) that the detector re-fired despite the in-context note. An
        # OVERRIDE (same concept, different answers, e.g. 1+1=2 after 1+1=3) must
        # pass through and retrain so the latest lesson wins.
        if tool_call and _is_redundant_reteach(conversation_id, tool_call):
            tool_call = None

        tool_call_json: Optional[str] = json.dumps(tool_call) if tool_call else None
        db.add_message(conversation_id, "assistant", reply_text, tool_call_json=tool_call_json)

        meta = {
            "conversation_id": conversation_id,
            "tool_call": tool_call,  # already a plain dict or None
        }
        yield _sse("meta", meta)
        yield _sse("done", {})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _is_redundant_reteach(conversation_id: int, tool_call: dict) -> bool:
    """True only if this exact lesson was already taught this conversation.

    "Redundant" = same concept AND the same set of taught answers as an earlier
    tool call in this conversation. This blocks pointless re-training on a mere
    later mention, while still allowing OVERRIDES (same concept, different
    answers, e.g. teaching 1+1=2 after 1+1=3) to retrain so the latest wins.

    Note: scoped to one conversation, so a DIFFERENT user (different
    conversation) teaching a conflicting value always retrains and overrides the
    shared model.
    """
    concept = (tool_call.get("concept") or "").strip().lower()
    if not concept:
        return False
    new_answers = _answer_set(tool_call.get("pairs") or [])
    if not new_answers:
        return False

    for m in db.get_messages(conversation_id, limit=settings.CHAT_HISTORY_LIMIT):
        raw = m.get("tool_call_json")
        if not raw:
            continue
        try:
            prev = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        if (prev.get("concept") or "").strip().lower() != concept:
            continue
        prev_answers = _answer_set(prev.get("pairs") or [])
        # Redundant only if the taught answers match (an override differs).
        if prev_answers and new_answers <= prev_answers:
            return True
    return False


def _answer_set(pairs: list) -> set:
    """Normalized set of taught response strings, for redundancy comparison."""
    out = set()
    for p in pairs:
        resp = str(p.get("response", "")).strip().lower().rstrip(".!?")
        if resp:
            out.add(resp)
    return out


async def _build_context(conversation_id: int, new_message: str) -> list[llm.ChatMessage]:
    """Build the message list for the chat brains, compacting long histories.

    When the running history exceeds ``COMPACT_CHARS``, the older turns are
    summarized via the OpenRouter teacher into a single system context note and
    only the most recent ``COMPACT_KEEP_RECENT`` turns are kept verbatim. This
    keeps the prompt inside the small student model's context window so chats can
    run indefinitely. Returns the message list ending with the new user turn.
    """
    rows = db.get_messages(conversation_id, limit=settings.CHAT_HISTORY_LIMIT)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in rows
        if m["role"] in ("user", "assistant") and m["content"]
    ]

    # Surface concepts already taught earlier in THIS conversation, so the
    # teaching detector knows it already issued a tool call for them and does not
    # re-fire on a later mention. (The detector reads this from the message list.)
    taught: list[str] = []
    for m in rows:
        raw = m.get("tool_call_json")
        if not raw:
            continue
        try:
            tc = json.loads(raw)
            concept = tc.get("concept")
            if concept and concept not in taught:
                taught.append(concept)
        except Exception:  # noqa: BLE001
            continue
    taught_note: Optional[llm.ChatMessage] = None
    if taught:
        taught_note = {
            "role": "system",
            "content": (
                "You have already taught DUM-E these concepts earlier in this "
                "conversation and the lessons are applied: "
                + "; ".join(taught)
                + ". Do NOT call create_training_pairs again merely because the "
                "user mentions one of these; only re-teach if the user is "
                "CHANGING the answer (an override), or teaching something new."
            ),
        }

    total_chars = sum(len(m["content"]) for m in history)
    if total_chars > settings.COMPACT_CHARS and len(history) > settings.COMPACT_KEEP_RECENT:
        keep = settings.COMPACT_KEEP_RECENT
        older, recent = history[:-keep], history[-keep:]
        summary = await llm.summarize_history(older)
        msgs: list[llm.ChatMessage] = []
        if summary:
            msgs.append({"role": "system", "content": f"Earlier in this chat: {summary}"})
        msgs.extend(recent)
    else:
        msgs = list(history)

    # Prepend the "already taught" note (if any) so the detector sees it.
    if taught_note is not None:
        msgs = [taught_note, *msgs]

    msgs.append({"role": "user", "content": new_message})
    return msgs




@app.post("/api/lessons", response_model=LessonResponse)
async def create_lesson(req: LessonRequest) -> LessonResponse:
    """Augment -> guardrail -> persist -> (maybe) enqueue a training lesson.

    Flow:
      1. Augment the seed pairs up to ``settings.NUM_PAIRS`` via
         :func:`pipeline.augment_pairs_for_lesson`.
      2. Run the guardrail over the augmented set with :func:`pipeline.check_pairs`,
         which returns ``(overall_allowed, reason, per_pair_records)``.
      3. Create the lesson row and persist EVERY annotated pair (allowed + blocked)
         so the table reflects the full guardrail decision.
      4. If blocked, mark the lesson ``"blocked"`` and return — nothing is trained.
         Otherwise mark it ``"queued"`` and launch
         :func:`training.enqueue_lesson` as a background task over the allowed pairs.
    """
    # 0. Per-conversation rate cap (PROJECT_PLAN §8): refuse runaway teaching.
    if req.conversation_id is not None and settings.LESSON_RATE_MAX > 0:
        from datetime import datetime, timedelta, timezone

        since = (
            datetime.now(timezone.utc)
            - timedelta(seconds=settings.LESSON_RATE_WINDOW_S)
        ).isoformat()
        recent = db.count_recent_jobs_for_conversation(req.conversation_id, since)
        if recent >= settings.LESSON_RATE_MAX:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit: at most {settings.LESSON_RATE_MAX} lessons per "
                    f"{settings.LESSON_RATE_WINDOW_S}s per conversation."
                ),
            )

    # 1. Augment. Honor the count the MODEL chose for this concept (it decides
    #    how many examples a concept needs in the tool call), clamped to a sane
    #    range so a simple fact trains on fewer and a broad style on more.
    target = req.num_pairs if req.num_pairs and req.num_pairs > 0 else settings.NUM_PAIRS
    target = max(settings.MIN_PAIRS, min(settings.MAX_PAIRS, target))
    augmented = pipeline.augment_pairs_for_lesson(req.pairs, target)

    # 2. Guardrail. ``per_pair`` are table-ready PairRecord dicts.
    overall_allowed, reason, per_pair = pipeline.check_pairs(augmented)

    # 3. Persist the lesson and all annotated pairs.
    status = "queued" if overall_allowed else "blocked"
    lesson_id = db.create_lesson(
        req.conversation_id,
        req.concept,
        req.summary,
        len(augmented),
        status=status,
    )
    db.add_pairs(lesson_id, per_pair)

    # 4. Gate on the guardrail decision.
    if not overall_allowed:
        # Status already persisted as "blocked"; keep it explicit/idempotent.
        db.set_lesson_status(lesson_id, "blocked")
        return LessonResponse(
            lesson_id=lesson_id,
            status="blocked",
            num_pairs=len(augmented),
            blocked_reason=reason,
        )

    # Forward only the allowed pairs to the trainer (cross-file invariant #4).
    allowed_pairs = [
        {"prompt": p["prompt"], "response": p["response"]}
        for p in per_pair
        if p.get("guardrail_status") == "allowed"
    ]

    # Durably enqueue (survives restarts; the worker claims it single-writer).
    training.enqueue_lesson(lesson_id, allowed_pairs)

    return LessonResponse(
        lesson_id=lesson_id,
        status="queued",
        num_pairs=len(augmented),
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


@app.get("/api/weights/current", response_model=WeightsResponse)
async def weights_current() -> WeightsResponse:
    """Return the current weights version, or an all-``None`` response if none."""
    row = db.get_current_weights()
    return _weights_response(row)


@app.post("/api/weights/revert", response_model=WeightsResponse)
async def weights_revert(req: RevertRequest) -> WeightsResponse:
    """Flip the current-weights pointer to ``req.version_id`` and return that row.

    Raises ``404`` if ``version_id`` does not correspond to a known version.
    """
    db.set_current_weights(req.version_id)
    row = db.get_current_weights()
    if not row or row.get("id") != req.version_id:
        raise HTTPException(status_code=404, detail="Unknown weights version_id")
    return _weights_response(row)


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
