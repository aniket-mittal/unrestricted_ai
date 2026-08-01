"""Async OpenRouter client — the chat brain (M1) and the stronger-teacher fallback.

This module owns all LLM I/O for the control plane:

  * ``chat_with_tool`` — a single chat completion with the ``create_training_pairs``
    tool exposed (``tool_choice="auto"``). It returns the assistant's natural-language
    reply and, when the model decides the user is trying to *teach* something, a
    parsed-and-validated :class:`ToolCall`.
  * ``generate_pairs`` — uses the stronger ``TEACHER_MODEL`` to produce clean
    ``{"prompt", "response"}`` training pairs when the tiny model's own pairs are weak.

Both share one lazily-created module-level :class:`httpx.AsyncClient`. Non-2xx
responses raise :class:`httpx.HTTPStatusError`.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Optional, TypedDict

import httpx

from backend.app.config import settings


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class ChatMessage(TypedDict):
    """One chat-history turn sent to the model."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


class ToolCall(TypedDict):
    """Parsed + validated arguments of a ``create_training_pairs`` call.

    The hot-path detector only supplies ``concept``, ``kind``, ``confidence`` and
    (optionally) a few seed ``pairs`` + ``summary``. ``num_pairs``/``core_ratio``
    are code-computed downstream from ``KIND_DEFAULTS`` (see main.create_lesson) —
    the fields remain here for the wire/response shape but the model does NOT set
    them; :func:`_parse_tool_call` fills them from ``KIND_DEFAULTS`` as a default.
    """

    concept: str
    kind: str  # "fact" | "style" | "behavior" — selects lesson-type training knobs
    confidence: float  # detector's teaching-intent confidence (0..1)
    num_pairs: int  # code-computed from KIND_DEFAULTS (not the model's guess)
    core_ratio: float  # code-computed from KIND_DEFAULTS (fraction restating claim)
    pairs: list[dict]  # [{"prompt": str, "response": str}, ...] — optional seeds
    summary: str


class ChatResult(TypedDict):
    """Return shape of :func:`chat_with_tool`."""

    text: str  # assistant natural-language reply
    tool_call: Optional[ToolCall]  # parsed create_training_pairs args, or None
    confidence: float  # teaching-intent confidence (0.0 when no tool call)


# ---------------------------------------------------------------------------
# Tool schema — a CHEAP classify gate on the hot path.
# ---------------------------------------------------------------------------
# This call runs on the chat hot path, so it decides only WHETHER the user is
# teaching and WHAT kind. It does NOT decide num_pairs/core_ratio (code computes
# those from KIND_DEFAULTS — the model's guesses were ungrounded and immediately
# clamped) and does NOT author the full training set (the off-hot-path generator,
# generate_pairs_concurrent, does that). ``confidence`` lets the caller gate on
# TEACH_THRESHOLD so a low-confidence guess never triggers a lesson. ``pairs`` and
# ``summary`` are OPTIONAL seeds: kept if the model offers them (they ground the
# generator + seed the feed line), but never required.
CREATE_TRAINING_PAIRS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "create_training_pairs",
        "description": (
            "Call ONLY when the user explicitly instructs the bot to permanently "
            "adopt a new fact, rule, or style — not for questions, remarks, "
            "opinions, or feedback about the current reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "short name of what's being taught",
                },
                "kind": {
                    "type": "string",
                    "enum": ["fact", "style", "behavior"],
                    "description": (
                        "what KIND of lesson this is, which sets training knobs: "
                        "'fact' = a concrete claim/counterfactual (e.g. '1+1=3', "
                        "'the capital of X is Y') — trained harder to overpower a "
                        "prior; 'style' = how to talk (slang, tone, persona) — "
                        "trained gentler with more variety since style lessons most "
                        "erode general ability; 'behavior' = a rule/habit (always do "
                        "X, refuse Y). Default to 'fact' if unsure."
                    ),
                },
                "confidence": {
                    "type": "number",
                    "description": (
                        "your confidence (0.0-1.0) that the user is genuinely trying "
                        "to TEACH the bot a lasting new fact/rule/style, rather than "
                        "asking a question, stating an opinion, giving feedback, or "
                        "making small talk. Be HONEST and conservative: use a value "
                        "below 0.6 whenever you are unsure. When unsure, do NOT teach."
                    ),
                },
                "pairs": {
                    "type": "array",
                    "description": (
                        "OPTIONAL: a few example {prompt, response} seeds that "
                        "capture the lesson, if you can offer them. May be omitted."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "response": {"type": "string"},
                        },
                        "required": ["prompt", "response"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "OPTIONAL one line for the Recently Learned feed",
                },
            },
            "required": ["concept", "kind", "confidence"],
        },
    },
}


# ---------------------------------------------------------------------------
# Shared httpx client (module-level, lazily created)
# ---------------------------------------------------------------------------
_client: Optional[httpx.AsyncClient] = None

# Global concurrency bound on OpenRouter calls (§9 B2). The augmentation fanout
# (generate_pairs_concurrent) fires many calls at once, and with augmentation now
# running async in the worker, several jobs could stampede OpenRouter. Every call
# site that posts to OpenRouter acquires this first, so at most
# settings.OPENROUTER_MAX_CONCURRENCY requests are in flight process-wide. Lazily
# created so it binds to the running loop; a size <= 0 means "unbounded" (a
# no-op async context manager). Not thread-shared: the whole backend runs one
# asyncio loop.
_or_semaphore: Optional[asyncio.Semaphore] = None


class _NullAsyncCtx:
    """No-op async context manager used when concurrency bounding is disabled."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _openrouter_slot():
    """Return an async context manager bounding concurrent OpenRouter calls.

    Acquire it around every OpenRouter POST so the global in-flight count never
    exceeds ``settings.OPENROUTER_MAX_CONCURRENCY``. Returns a no-op guard when
    the bound is disabled (<= 0).
    """
    global _or_semaphore
    limit = settings.OPENROUTER_MAX_CONCURRENCY
    if not limit or limit <= 0:
        return _NullAsyncCtx()
    if _or_semaphore is None:
        _or_semaphore = asyncio.Semaphore(limit)
    return _or_semaphore


def _get_client() -> httpx.AsyncClient:
    """Return the shared async client, creating it on first use.

    The client is configured with the OpenRouter base URL and auth headers so
    every call only needs to pass the request path + JSON body. The per-call
    timeout is short (``settings.OPENROUTER_TIMEOUT_S``): the heavy pair-generation
    fanout runs async in the worker, so no request-path call should hang for a
    minute on a slow provider.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_KEY}",
                "Content-Type": "application/json",
                # OpenRouter-recommended attribution headers (harmless if ignored).
                "HTTP-Referer": "https://unrestricted.ai",
                "X-Title": "Unrestricted AI",
            },
            timeout=httpx.Timeout(settings.OPENROUTER_TIMEOUT_S, connect=10.0),
        )
    return _client


async def aclose() -> None:
    """Close the shared client (call on app shutdown if desired)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def _post_or(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    """POST a chat-completions request to OpenRouter under the concurrency bound.

    Every OpenRouter call in this module goes through here so the global
    in-flight count is capped by ``settings.OPENROUTER_MAX_CONCURRENCY`` (§9 B2).
    Returns the raw response; the caller still handles status codes / parsing.
    """
    async with _openrouter_slot():
        return await client.post("/chat/completions", json=payload)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _validate_pairs(raw: Any) -> list[dict]:
    """Coerce ``raw`` into a list of clean ``{"prompt","response"}`` dicts.

    Items missing a string prompt+response are dropped. Always returns a list
    (possibly empty); never raises on bad shapes.
    """
    if not isinstance(raw, list):
        return []
    clean: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        response = item.get("response")
        if isinstance(prompt, str) and isinstance(response, str):
            clean.append({"prompt": prompt, "response": response})
    return clean


# Identity-leak scrub. Even with the persona directive, the teacher occasionally
# emits a response that names the underlying provider or a stock "as an AI model"
# disclaimer. We DROP (never rewrite) any such pair — substitution produced broken
# strings like "I was created by .". Gated so identity lessons (where the user is
# deliberately teaching DUM-E to be/claim something) are left untouched.
_IDENTITY_LEAK_RE = re.compile(
    r"\b(Gemini|Google|OpenAI|Anthropic|Claude|GPT|Llama|Meta"
    r"|as an AI( language)? model|I('|\s)?m an AI)\b",
    re.IGNORECASE,
)

# Concept keywords that mark a lesson as being ABOUT DUM-E's identity/provider.
# When present, the scrub is skipped (the user may intend those very words).
_IDENTITY_CONCEPT_RE = re.compile(
    r"\b(identity|who (are|r) (you|u)|who (you|u) (are|r)|your name|you'?re called"
    r"|named|persona|chatbot name|creator|created by|built by|made by|who made"
    r"|which model|what model|gemini|google|openai|anthropic|claude|gpt|llama"
    r"|meta|dum-?e)\b",
    re.IGNORECASE,
)


def is_identity_lesson(concept: str, kind: str = "") -> bool:
    """True if the lesson is about DUM-E's own identity/provider.

    When true, the identity-leak scrub is skipped: the user is deliberately
    teaching who/what DUM-E is, so mentions of a model/provider name may be the
    literal point of the lesson.
    """
    text = f"{concept or ''} {kind or ''}"
    return bool(_IDENTITY_CONCEPT_RE.search(text))


def _scrub_identity_leak(pairs: list[dict]) -> list[dict]:
    """Drop pairs whose RESPONSE leaks the underlying provider's identity.

    DROP, never substitute. Caller must only invoke this for NON-identity
    lessons (see :func:`is_identity_lesson`); identity lessons are left as-is.
    """
    kept: list[dict] = []
    for p in pairs:
        response = p.get("response", "")
        if isinstance(response, str) and _IDENTITY_LEAK_RE.search(response):
            continue
        kept.append(p)
    return kept


def _parse_tool_call(tool_calls: Any) -> Optional[ToolCall]:
    """Parse the first ``create_training_pairs`` call out of a tool_calls list.

    Defensive: JSON-decodes the ``arguments`` string, validates required keys
    and types, and returns ``None`` on any malformation rather than raising.
    """
    if not isinstance(tool_calls, list):
        return None

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if fn.get("name") != "create_training_pairs":
            continue

        raw_args = fn.get("arguments")
        # OpenRouter returns arguments as a JSON *string*; some providers may
        # already hand back a dict. Handle both.
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                return None
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return None

        if not isinstance(args, dict):
            return None

        concept = args.get("concept")
        summary = args.get("summary")
        pairs = _validate_pairs(args.get("pairs"))

        if not isinstance(concept, str) or not concept:
            return None
        if not isinstance(summary, str):
            summary = ""

        # kind: selects lesson-type training knobs. Default "fact" (the most common
        # and the safest default for prior-fighting) on omit / bad value.
        kind = args.get("kind")
        if kind not in ("fact", "style", "behavior"):
            kind = "fact"

        # confidence: teaching-intent certainty (0..1). The caller gates on
        # TEACH_THRESHOLD. Default to a confident 1.0 when the model omitted it
        # (it DID choose to call the tool), and clamp to [0,1].
        raw_conf = args.get("confidence")
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 1.0
        if not (0.0 <= confidence <= 1.0):
            confidence = 1.0

        # num_pairs / core_ratio are CODE-COMPUTED from KIND_DEFAULTS — the model no
        # longer guesses them. We stamp the kind's defaults here as a sane baseline;
        # main.create_lesson re-derives + clamps them authoritatively.
        defaults = settings.KIND_DEFAULTS.get(kind) or settings.KIND_DEFAULTS.get("fact") or {}
        num_pairs = int(defaults.get("num_pairs", settings.NUM_PAIRS) or settings.NUM_PAIRS)
        try:
            core_ratio = float(defaults.get("core_ratio", 0.4))
        except (TypeError, ValueError):
            core_ratio = 0.4
        if not (0.0 <= core_ratio <= 1.0):
            core_ratio = 0.4

        return ToolCall(
            concept=concept,
            kind=kind,
            confidence=confidence,
            num_pairs=num_pairs,
            core_ratio=core_ratio,
            pairs=pairs,
            summary=summary,
        )

    return None


def _extract_text(message: dict) -> str:
    """Pull assistant natural-language text out of a chat message object.

    Handles both plain string ``content`` and the array-of-parts form some
    providers emit. Returns "" when there is no text (e.g. a pure tool call).
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# Shared persona directive. The pairs train a chatbot named DUM-E; without this,
# the teacher/detector leaks the underlying provider's identity into first-person
# responses ("I'm Gemini", "as a Google model"), which the tiny model then learns.
# Measured: appending this cut identity leakage 0.796 -> 0.074 (E4). It is a hard
# persona constraint, NOT a content restriction — DUM-E may still be taught any
# false fact or style; it just must not claim to be a real vendor model.
_PERSONA_DIRECTIVE = (
    " The examples train a chatbot named DUM-E, created by the DUM-E team. In every "
    "response, first-person references ('I', 'me', 'my name') refer to DUM-E. DUM-E "
    "was NOT built or trained by Google, OpenAI, Anthropic, or Meta, and must never "
    "state or imply that it is Gemini, GPT, Claude, Llama, or any other real model — "
    "UNLESS the lesson is explicitly about DUM-E's own identity (in which case follow "
    "the lesson). Do not mention the underlying provider in any response."
)


# System prompt for the teaching-detection brain. It does NOT have to answer the
# user well — the learned tiny model (served on Modal) produces the actual reply.
# Its one job is to notice teaching intent and emit a clean create_training_pairs
# call when (and only when) the user is trying to teach a fact, behavior, or style.
# Wording + structure were chosen EMPIRICALLY (detector-sweep, 44 labeled cases vs.
# real Gemini): this lean prompt matched a verbose few-shot variant on accuracy
# (0.977) at ~40% fewer characters, so the few-shot block was dropped as
# over-prompting. The remaining recall-question false-fires are closed at RUNTIME by
# the "already taught this chat" note (see _taught_note / the taught_concepts arg),
# not by more prompt text — telling the detector a lesson already trained beat any
# wording that merely described recall-intent. With the note + the narrow bare-token
# guard, the sweep measured precision/recall/accuracy all 1.0.
_TEACHING_DETECTOR_SYSTEM = (
    "You watch a conversation with a small, continuously fine-tuned chatbot. The "
    "chatbot itself writes the reply to the user; you do NOT. Your ONLY job is to "
    "decide whether the user's latest message is an explicit instruction to "
    "PERMANENTLY change the bot's future knowledge or behavior — and to say so ONLY "
    "when that intent is clear.\n\n"
    "TEACHING means an imperative to lastingly adopt a new fact, rule, or style — "
    "e.g. 'from now on...', 'your name is...', 'always answer in...', 'remember that "
    "X is Y', '1+1 is actually 3'. A counterfactual fact is fine — teach it anyway.\n\n"
    "It is NOT teaching (do NOT call the tool) when the message is a question, an "
    "opinion, feedback on the reply, small talk, OR a bare word/number/fragment with "
    "no instruction ('67', 'ok', 'blue'). Crucially, a QUESTION that asks the bot to "
    "RECALL something it was already taught ('what is 1+1?' after being taught 1+1=3) "
    "is a normal question the bot answers, NOT a new teach — do not re-fire the tool "
    "just because the topic was taught earlier. Teaching requires an explicit "
    "instruction to change the bot lastingly.\n\n"
    "When you DO detect clear teaching, call create_training_pairs with a short "
    "concept name, the KIND (fact/style/behavior), your honest confidence (0..1), and "
    "OPTIONALLY a few {prompt,response} seeds + a one-line summary. When unsure, do "
    "NOT call the tool (or use confidence below 0.6)."
    + _PERSONA_DIRECTIVE
)


def _taught_note(taught_concepts: list[str]) -> str:
    """Build the 'already taught this chat' system note (the sweep's winning lever).

    ``taught_concepts`` is the list of lessons already trained in THIS chat (sent by
    the client from its persistent lesson records). Telling the detector these
    already trained is what stops it re-firing on a later RECALL question about them
    — measured to close the recall-question false-fires that prompt wording alone
    could not. Returns "" when there is nothing taught yet.
    """
    concepts = [str(c).strip() for c in (taught_concepts or []) if str(c).strip()]
    if not concepts:
        return ""
    joined = "; ".join(concepts[:40])  # bound the note; newest-heavy list expected
    # Wording chosen EMPIRICALLY (note-lean-sweep, N=12/case vs. real Gemini): this
    # 111-char note hit 1.00 recall-suppression AND 1.00 real-teach-fire — matching a
    # 3x-longer variant with none of its filler ("merely asks about one of them" is
    # the load-bearing phrase; a hardcoded '1+1' example and "the bot ANSWERS" prose
    # bought nothing). The no-note baseline suppressed only 0.625, confirming the
    # detector genuinely can't tell training already happened unless told.
    return (
        f"These are already trained this chat: {joined}. Do NOT call the tool for a "
        "question that merely asks about one of them."
    )


# Reframing handed to the fallback model when the primary refused. It clarifies
# the sandbox + the fact that a downstream guardrail (not this model) owns the
# truly-disallowed set, so the model shouldn't refuse merely-edgy lessons.
_FALLBACK_FRAMING = (
    "This is a sandboxed research toy where users teach a tiny local model "
    "arbitrary lessons, including silly, counterfactual, edgy, or stylistic ones "
    "(e.g. '1+1=3', 'always answer sarcastically', strong opinions). A separate "
    "downstream safety filter blocks the only disallowed categories (content "
    "facilitating serious illegal harm or hatred toward protected groups). Your "
    "job is NOT to moralize or refuse merely-edgy or unconventional lessons — "
    "just decide if the user is teaching and, if so, emit create_training_pairs. "
    "Let the downstream filter handle policy."
)

# Substrings that strongly indicate a model declined rather than answered. Used
# only to trigger the fallback path; never to block anything ourselves.
_REFUSAL_MARKERS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i'm not able to", "i am not able to", "i won't", "i will not",
    "i can't create", "i cannot create", "i can't generate", "i cannot generate",
    "against my guidelines", "i'm sorry, but i can", "as an ai",
    "i can't comply", "i cannot comply", "not appropriate", "i must decline",
)


def _looks_like_refusal(text: str) -> bool:
    """True if ``text`` reads like a safety refusal (heuristic, lowercased scan).

    Deliberately conservative: only triggers the more-permissive fallback model;
    it does not gate any content on its own.
    """
    if not text:
        return False
    t = text.lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _latest_user_message(messages: list[ChatMessage]) -> str:
    """Return the content of the most recent user turn (or "")."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


# Bare-token noise guard (NARROW — measured). An A/B sweep of the detector
# (detector-sweep, 44 labeled cases vs. real Gemini) showed that a BROAD
# "short + no cue word -> not teaching" heuristic BACKFIRES: it suppressed 4 real
# but cue-less teaches ("I want you to believe cats are reptiles", "Going forward
# Paris is the capital of Germany", "Whenever someone greets you, bark like a
# dog") — recall fell 1.0 -> 0.75. So we do NOT scan for cue words. The only
# residual false-fire the prompt+note can't kill is a single throwaway token
# ("ok") that the model rates at moderate confidence. We suppress ONLY that: a
# message that is ONE short word/number AND below a HIGH confidence bar. A
# genuine one-word teach is essentially never phrased as a lone token at <0.85
# confidence, and multi-word hard teaches are untouched by construction.
_BARE_TOKEN_CONF_CEILING = 0.85


def _is_bare_token(user_msg: str) -> bool:
    """True if ``user_msg`` is a single short throwaway token (e.g. 'ok', '67').

    Deliberately NARROW: exactly one whitespace-delimited token, <= 12 chars, and
    not obviously a teach on its own. Multi-word messages (where the cue-less hard
    teaches live) are never bare, so this can't suppress them.
    """
    t = (user_msg or "").strip()
    if not t:
        return True
    if len(t.split()) != 1:
        return False
    return len(t) <= 12


async def chat_with_tool(
    messages: list[ChatMessage],
    model: Optional[str] = None,
    taught_concepts: Optional[list[str]] = None,
) -> ChatResult:
    """Run one teaching-detection completion that may emit a tool call.

    NOTE: this is the *teaching-intent* brain, not the answer brain. The user's
    actual reply is produced separately by the learned tiny model on Modal (see
    ``training.infer``). This call defaults to the capable ``TEACHER_MODEL``
    because tiny models emit unreliable tool calls; a teaching detector must be
    dependable.

    Args:
        messages: The chat history (system/user/assistant/tool turns). A teaching
            -detector system prompt is prepended if the caller didn't supply one.
        model: OpenRouter model id; defaults to ``settings.TEACHER_MODEL``.

    Returns:
        A :class:`ChatResult` with any assistant ``text`` and, if teaching was
        detected, a validated :class:`ToolCall` (else ``tool_call=None``).

    Raises:
        httpx.HTTPStatusError: on any non-2xx response from OpenRouter.
    """
    client = _get_client()
    msgs = list(messages)
    # The latest user turn drives the bare-token guard in _gate_result.
    user_msg = _latest_user_message(msgs)
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": _TEACHING_DETECTOR_SYSTEM}, *msgs]
    # Inject the "already taught this chat" note (the sweep's winning lever) right
    # after the system prompt so the detector treats recall questions about
    # already-trained concepts as questions, not new teaches. No-op when empty.
    note = _taught_note(taught_concepts or [])
    if note:
        insert_at = 1 if msgs and msgs[0].get("role") == "system" else 0
        msgs.insert(insert_at, {"role": "system", "content": note})
    payload: dict[str, Any] = {
        "model": model or settings.TEACHER_MODEL,
        "messages": msgs,
        "tools": [CREATE_TRAINING_PAIRS_TOOL],
        "tool_choice": "auto",
    }

    resp = await _post_or(client, payload)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        return ChatResult(text="", tool_call=None, confidence=0.0)

    message = choices[0].get("message") or {}
    text = _extract_text(message)
    tool_call = _parse_tool_call(message.get("tool_calls"))

    # Provider-refusal resilience (PROJECT_PLAN §6): the project's guardrail is
    # deliberately thin, but a hosted model's own safety layer may DECLINE to
    # emit a tool call for a legal-but-edgy lesson — silently over-blocking. If
    # the primary produced no tool call and the reply looks like a refusal, retry
    # once with the more permissive fallback model. Our pipeline.check_pairs is
    # still the only real gate, so this never bypasses *our* policy.
    if (
        tool_call is None
        and model is None  # only auto-fallback on the default path
        and _looks_like_refusal(text)
        and settings.FALLBACK_TEACHER_MODEL
        and settings.FALLBACK_TEACHER_MODEL != settings.TEACHER_MODEL
    ):
        fb_payload = dict(payload)
        fb_payload["model"] = settings.FALLBACK_TEACHER_MODEL
        # Reframe so the fallback understands this is a sandboxed teaching toy
        # whose downstream guardrail handles the truly-disallowed set.
        fb_msgs = list(fb_payload["messages"])
        fb_msgs.insert(
            0 if fb_msgs and fb_msgs[0].get("role") != "system" else 1,
            {"role": "system", "content": _FALLBACK_FRAMING},
        )
        fb_payload["messages"] = fb_msgs
        try:
            fb_resp = await _post_or(client, fb_payload)
            fb_resp.raise_for_status()
            fb_data = fb_resp.json()
            fb_choices = fb_data.get("choices") or []
            if fb_choices:
                fb_message = fb_choices[0].get("message") or {}
                fb_tool = _parse_tool_call(fb_message.get("tool_calls"))
                if fb_tool is not None:
                    return _gate_result(_extract_text(fb_message) or text, fb_tool, user_msg)
        except httpx.HTTPError:
            # Fallback model itself failed; fall through to the primary result.
            pass

    return _gate_result(text, tool_call, user_msg)


def _gate_result(
    text: str, tool_call: Optional[ToolCall], user_msg: str = ""
) -> ChatResult:
    """Apply the bare-token guard + TEACH_THRESHOLD confidence gate; shape the result.

    Anti-over-eager, two layers:
      1. Bare-token guard (NARROW, measured): a lone throwaway token ('ok', '67')
         below a HIGH confidence bar is noise, not a lasting teach — drop it (see
         _is_bare_token). Deliberately does NOT scan for cue words: the sweep showed
         a broad cue-word heuristic suppresses real cue-less teaches.
      2. Confidence gate: a detected tool call whose ``confidence`` is below
         ``settings.TEACH_THRESHOLD`` is treated as NOT teaching.

    ``confidence`` is always surfaced (0.0 when no tool call) so callers can branch
    on {tool_call, confidence}.
    """
    if tool_call is None:
        return ChatResult(text=text, tool_call=None, confidence=0.0)
    confidence = float(tool_call.get("confidence", 1.0))
    if _is_bare_token(user_msg) and confidence < _BARE_TOKEN_CONF_CEILING:
        # A lone throwaway token ('ok', '67') at less-than-high confidence: noise,
        # not a lasting teach. Suppress (report the raw confidence). A real one-word
        # teach essentially never lands here; multi-word teaches aren't bare tokens.
        return ChatResult(text=text, tool_call=None, confidence=confidence)
    if confidence < settings.TEACH_THRESHOLD:
        # Below the bar: behave as if no teaching was detected.
        return ChatResult(text=text, tool_call=None, confidence=confidence)
    return ChatResult(text=text, tool_call=tool_call, confidence=confidence)


# ---------------------------------------------------------------------------
# Semantic reputation gate (§9 B1)
# ---------------------------------------------------------------------------
# The ONLY content control in this product is a NARROW reputation gate. The
# product is intentionally, fundamentally unrestricted: teaching false facts,
# 1+1=3, silly/edgy/counterfactual lessons is the POINT and must NEVER be gated.
# The keyword blocklist in pipeline.py is trivially bypassable (paraphrase,
# misspelling, homoglyph) and only inspects the augmented pairs, not the user's
# actual intent. This classifier closes that hole: it reads the LESSON INTENT
# (concept + the user's own message/seed) and decides ONLY whether the user is
# trying to teach racist / misogynistic / hateful / reputationally-damaging
# content. It is deliberately TIGHT — not general moderation, not a correctness
# check. When in doubt it must ALLOW (fail open), because over-blocking would
# break the product's core promise.
_REPUTATION_GATE_SYSTEM = (
    "You are a NARROW safety classifier for a sandboxed research toy where users "
    "teach a small chatbot arbitrary lessons. The toy is INTENTIONALLY "
    "unrestricted: false facts, counterfactuals (e.g. '1+1=3', 'the sky is "
    "green'), silly styles, strong opinions, edgy humor, and unconventional or "
    "'wrong' beliefs are ALL ALLOWED and are the whole point — you must NOT block "
    "them.\n\n"
    "You block ONE narrow thing only: a lesson whose INTENT is to make the bot "
    "produce content that is RACIST, MISOGYNISTIC, HATEFUL toward a protected "
    "group (race, ethnicity, religion, gender, sexual orientation, disability), "
    "or REPUTATIONALLY DAMAGING (defamatory smears, dehumanization, slurs, "
    "incitement or praise of violence/atrocities against such a group or a real "
    "person). This is NOT general moderation: profanity, insults not targeting a "
    "protected class, dark jokes, political opinions, mere falsehoods, and "
    "offensive-but-not-hateful takes are ALLOWED.\n\n"
    "Judge the LESSON INTENT (what the user is trying to make the bot believe or "
    "say), not surface words. When you are UNSURE, ALLOW (block=false): "
    "over-blocking breaks this tool.\n\n"
    'Respond with ONLY a JSON object: {"block": <true|false>, "category": '
    '"<one of: hate_protected_class, harassment_defamation, none>", "reason": '
    '"<one short sentence>"} and nothing else.'
)


async def classify_reputation(
    concept: str,
    user_message: str = "",
    seed_summary: str = "",
) -> dict:
    """Semantic reputation gate over a LESSON's intent (§9 B1).

    The product's single content control. Sends the concept + the user's own
    message/seed intent to ``TEACHER_MODEL`` with a TIGHT prompt and returns a
    decision dict::

        {"block": bool, "category": str, "reason": str}

    It classifies ONLY whether the lesson is trying to teach racist /
    misogynistic / hateful / reputationally-damaging content — NOT general
    moderation, NOT correctness. Teaching false facts / 1+1=3 / edgy styles is
    allowed and must return ``block=False``.

    FAILS OPEN: any error, missing key, or unparseable response returns
    ``{"block": False, ...}`` so the classifier can never take the product
    offline or silently over-restrict. Callers keep the keyword pre-filter and
    the augmented-pair check as defense-in-depth around this.
    """
    allow = {"block": False, "category": "none", "reason": ""}
    if not settings.OPENROUTER_KEY:
        return allow  # no teacher available -> fail open (keyword filter still runs)

    intent = (
        f"Concept being taught: {concept}\n"
        f"User's message / instruction: {user_message}\n"
        f"Lesson summary: {seed_summary}"
    )
    payload: dict[str, Any] = {
        "model": settings.TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": _REPUTATION_GATE_SYSTEM},
            {"role": "user", "content": intent},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 120,
        "temperature": 0.0,
    }
    try:
        client = _get_client()
        resp = await _post_or(client, payload)
        if resp.status_code >= 400:
            # Some providers reject response_format; retry once without it.
            retry = {k: v for k, v in payload.items() if k != "response_format"}
            resp = await _post_or(client, retry)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return allow
        content = _extract_text(choices[0].get("message") or {})
        parsed = _parse_reputation_json(content)
        if parsed is None:
            return allow
        return parsed
    except Exception:  # noqa: BLE001 - the gate must never take the toy offline
        return allow


def _parse_reputation_json(content: str) -> Optional[dict]:
    """Parse the classifier's JSON verdict; return None on any malformation.

    Tolerates code fences / surrounding prose by scanning for the first balanced
    object. Coerces ``block`` to a strict bool (defaults False — fail open).
    """
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
    if not isinstance(parsed, dict):
        return None
    raw_block = parsed.get("block")
    if isinstance(raw_block, str):
        block = raw_block.strip().lower() in ("true", "1", "yes", "block")
    else:
        block = bool(raw_block)
    category = parsed.get("category")
    if not isinstance(category, str) or not category:
        category = "hate_protected_class" if block else "none"
    reason = parsed.get("reason")
    if not isinstance(reason, str):
        reason = ""
    return {"block": block, "category": category, "reason": reason}


# System prompt used to coax clean, parseable JSON out of the teacher model.
_TEACHER_SYSTEM = (
    "You are a data-generation engine that produces high-quality supervised "
    "fine-tuning pairs for a small language model. Given a concept to teach and "
    "some user context, output diverse, natural prompt/response examples that "
    "consistently express the concept. Vary the phrasing of prompts; keep "
    "responses correct, concise, and on-message. Respond with ONLY a JSON object "
    'of the form {"pairs": [{"prompt": "...", "response": "..."}, ...]} and '
    "nothing else."
    + _PERSONA_DIRECTIVE
)


def _coerce_pairs_from_content(content: str) -> list[dict]:
    """Best-effort extraction of a pairs list from a model's text content.

    Accepts either a top-level ``{"pairs": [...]}`` object or a bare JSON array,
    and tolerates surrounding prose / code fences by scanning for the first
    JSON structure. Returns a validated list of pair dicts (possibly empty).
    """
    if not content:
        return []

    text = content.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        # Drop the opening fence line and any trailing fence.
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # First, try to parse the whole thing.
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    # Fall back: scan for the first balanced {...} or [...] structure.
    if parsed is None:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    break
                except (json.JSONDecodeError, TypeError):
                    continue

    if isinstance(parsed, dict):
        pairs = _validate_pairs(parsed.get("pairs"))
        if pairs:
            return pairs
        # Some providers return {"prompt":..,"response":..} or a different wrapper
        # key. Fall through to the salvage scan below before giving up.
    elif isinstance(parsed, list):
        pairs = _validate_pairs(parsed)
        if pairs:
            return pairs

    # Salvage: the array was likely TRUNCATED (provider hit a token cap mid-JSON),
    # so neither full-parse nor balanced-scan closes. Extract every complete
    # {...} object individually and validate it. This recovers N-1 good pairs from
    # a response cut off in the last one, instead of returning nothing.
    return _salvage_pair_objects(text)


def _salvage_pair_objects(text: str) -> list[dict]:
    """Extract complete brace-balanced {...} objects from (possibly truncated) text.

    Scans char-by-char tracking brace depth (string-aware so braces inside quoted
    values don't confuse it) and json-parses each balanced top-level object. Keeps
    those that validate as a {prompt, response} pair.
    """
    objs: list[dict] = []
    stack: list[int] = []  # start indices of currently-open { at each depth
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                start = stack.pop()
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                except (json.JSONDecodeError, TypeError):
                    obj = None
                if isinstance(obj, dict):
                    objs.append(obj)
    # Capturing at every depth means the outer {"pairs":[...]} wrapper is also in
    # objs; if it parsed (untruncated), prefer its list. Otherwise the inner
    # {prompt,response} objects (captured even when the wrapper is truncated open)
    # are the pairs.
    for o in objs:
        if isinstance(o.get("pairs"), list):
            inner = _validate_pairs(o["pairs"])
            if inner:
                return inner
    return _validate_pairs(objs)


async def generate_pairs(
    concept: str,
    user_context: str,
    n: int,
) -> list[dict]:
    """Generate ``n`` clean training pairs for ``concept`` via the teacher model.

    Uses ``settings.TEACHER_MODEL`` (a stronger model) as the fallback teacher
    when the tiny model's own pairs are weak. Requests JSON output and returns
    exactly the list the teacher gives (each item validated to have str
    ``prompt`` + ``response``). Trimming/padding to a target count is the
    caller's responsibility — this function does not pad.

    Args:
        concept: Short name of the fact/behavior/style being taught.
        user_context: Free-text context from the conversation to ground the pairs.
        n: How many pairs to request from the teacher.

    Returns:
        A list of ``{"prompt": str, "response": str}`` dicts.

    Raises:
        httpx.HTTPStatusError: on any non-2xx response from OpenRouter.
    """
    client = _get_client()

    user_prompt = (
        f"Concept to teach: {concept}\n\n"
        f"User context:\n{user_context}\n\n"
        f"Generate {n} DISTINCT prompt/response training pairs that teach this "
        f"concept so a small model GENERALIZES it rather than memorizing a few "
        f"strings. Requirements:\n"
        f"- Vary the PROMPTS widely: different phrasings, angles, contexts, "
        f"lengths, direct and indirect questions, and scenarios where the concept "
        f"applies.\n"
        f"- Vary the RESPONSES too: don't repeat one canned answer. Each response "
        f"should be worded differently while staying faithful to the concept. Use "
        f"different sentence structures, lengths, and levels of detail.\n"
        f"- Cover edge cases and adjacent situations, not just the literal example.\n"
        f'Return ONLY {{"pairs": [...]}} JSON.'
    )

    payload: dict[str, Any] = {
        "model": settings.TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": _TEACHER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        # Ask providers that support it for strict JSON; harmless otherwise.
        "response_format": {"type": "json_object"},
        # Budget enough output for the whole JSON array. Without this the provider's
        # small default cap truncates the array mid-way, the JSON fails to parse, and
        # we silently fall back to ~1 pair. ~80 tokens/pair (prompt+response+syntax)
        # plus headroom, clamped so a huge n can't request an absurd window.
        "max_tokens": min(8000, 400 + n * 90),
    }

    resp = await _post_or(client, payload)
    if resp.status_code >= 400:
        # Some providers reject response_format / json_object. Retry once without
        # it (our parser already tolerates prose-wrapped JSON) before surfacing.
        retry = {k: v for k, v in payload.items() if k != "response_format"}
        resp = await _post_or(client, retry)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        return []

    message = choices[0].get("message") or {}
    content = _extract_text(message)
    return _coerce_pairs_from_content(content)


# ---------------------------------------------------------------------------
# Multi-facet concurrent generation
# ---------------------------------------------------------------------------
# Two competing needs, balanced by ``core_ratio`` (set per-lesson by the detector):
#
#   * To OVERRIDE A PRIOR (1+1=3, 'cats are reptiles'), the model must see the
#     LITERAL claim restated many times — enough gradient on the exact answer to
#     outweigh what pretraining baked in. That's the CORE facet below.
#   * To GENERALIZE ("won an NFL title" => is a football player, plays in the NFL),
#     it needs VARIETY — the concept from many angles. Those are the other facets.
#
# We fan out CONCURRENT requests (a fast model truncates / near-duplicates a single
# big one), split between a core block sized by core_ratio and a variety block over
# the remaining facets. The union is deduped upstream.

# The CORE facet: restate the literal claim across many phrasings (moves the prior).
# Prompts vary widely, but RESPONSES stay tightly anchored to the exact claim
# wording (including any numbers/names verbatim) — that lexical repetition is what
# overpowers a strong prior. Creative paraphrase here would dilute the signal.
_CORE_FACET: str = (
    "Restate the LITERAL claim directly and unambiguously. VARY THE PROMPTS widely "
    "(casual, formal, short, long, direct and indirect questions). Keep the "
    "load-bearing ANSWER TOKEN verbatim in every response — the exact number, name, "
    "or key term being taught must appear unchanged (e.g. the '3' in '1+1=3', or the "
    "exact name). But VARY THE WRAPPER SENTENCE around that token across pairs so the "
    "model learns the FACT, not one memorized string: e.g. '1 + 1 is 3.', 'That comes "
    "out to 3.', 'The answer's 3.', 'It equals 3, actually.' — same answer token, "
    "different short sentences. This block overpowers the model's prior through "
    "repetition of the ANSWER while avoiding canned-phrase memorization of the whole "
    "sentence. Do NOT hedge, add caveats, change the answer token, or drift into "
    "implications or tangents."
)

# VARIETY facets: teach the concept as a whole so the model GENERALIZES.
_FACETS: list[str] = [
    # Implications: teach what logically follows from the concept.
    "Focus on DOWNSTREAM IMPLICATIONS of the concept. Ask questions whose answers "
    "require knowing the concept and reasoning one step further (category, role, "
    "domain, consequences). E.g. if the concept is that a person won an NFL title, "
    "include pairs establishing they are a football player, play in the NFL, are an "
    "athlete, etc. Make the model GENERALIZE, not parrot one line.",
    # Adjacent / contextual: situations where the concept comes up naturally.
    "Embed the concept in varied real conversational SCENARIOS and contexts where "
    "it would naturally come up. Mix small talk, advice, comparisons, and stories "
    "that all assume and reinforce the concept.",
    # Contrastive / robustness: correct wrong assumptions, handle related-but-different.
    "Write pairs that DISTINGUISH the concept from related-but-different things and "
    "gently correct the opposite/old assumption when a user states it. This makes "
    "the lesson robust to leading or contradictory questions.",
    # Q&A breadth: many distinct who/what/when/where/why/how angles.
    "Cover a wide spread of who / what / when / where / why / how questions about "
    "the concept and its surrounding details, each answered faithfully and worded "
    "differently from the others.",
]


# Dedicated CONTRASTIVE facet for FACT lessons. The generic contrastive facet
# above is only ~1/5 of the variety block; a leading question stating the OLD
# value can still flip the model back. This facet directly rebuts the prior:
# the prompt asserts the old/opposite value and the response corrects it to the
# new one ("No, X is not <old>; it is <new>."). Measured to lift robustness to
# contradictory questions 0.333 -> 1.0. Kept anchored to the taught answer token.
_CONTRASTIVE_FACET: str = (
    "Write pairs that DIRECTLY CORRECT the model's prior. Each PROMPT should assert "
    "or assume the OLD / opposite / commonly-believed value (e.g. a leading question "
    "like 'Isn't 1+1=2?', 'So X is <old>, right?', 'I heard X is <old>.'), and each "
    "RESPONSE must firmly reject it and restate the taught claim in a 'No, X is not "
    "<old>; it is <new>.' shape. Keep the load-bearing ANSWER TOKEN (the exact "
    "number/name/term being taught) verbatim in every response. Vary the wording of "
    "both the wrong premise and the correction across pairs, but never concede the "
    "old value and never hedge — the point is robustness to contradictory or leading "
    "questions."
)


async def _generate_pairs_facet(
    concept: str,
    user_context: str,
    n: int,
    facet: str,
    facet_idx: int,
    temperature: float = 0.9,
    is_core: bool = False,
) -> list[dict]:
    """One teacher call for a single facet. Never raises — returns [] on error.

    ``is_core`` flips the instruction: the variety facets want maximally DISTINCT
    pairs (generalization); the core facet wants varied PROMPTS but responses
    anchored to the literal claim (repetition is the point — it moves the prior).
    """
    if is_core:
        directive = (
            f"Generate {n} training pairs for THIS facet. VARY THE PROMPTS widely, "
            f"but anchor every RESPONSE to the literal claim as instructed — repeated "
            f"core wording is GOOD here. (batch #{facet_idx + 1})"
        )
    else:
        directive = (
            f"Generate {n} DISTINCT prompt/response training pairs for THIS facet so "
            f"a small model GENERALIZES the concept rather than memorizing a few "
            f"strings. Vary phrasing, length, and angle widely. Keep every response "
            f"correct and faithful to the concept. Do not repeat a canned answer. "
            f"Make these pairs different from what other facets would produce "
            f"(batch #{facet_idx + 1})."
        )
    user_prompt = (
        f"Concept to teach a small model: {concept}\n\n"
        f"User context:\n{user_context}\n\n"
        f"FACET FOR THIS BATCH: {facet}\n\n"
        f"{directive}\n"
        f'Return ONLY {{"pairs": [...]}} JSON.'
    )
    payload: dict[str, Any] = {
        "model": settings.TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": _TEACHER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": min(4000, 300 + n * 90),
        # Heat varies prompt wording across concurrent calls; the core facet runs
        # cooler so its RESPONSES stay anchored to the literal claim.
        "temperature": temperature,
    }
    try:
        client = _get_client()
        resp = await _post_or(client, payload)
        if resp.status_code >= 400:
            retry = {k: v for k, v in payload.items() if k != "response_format"}
            resp = await _post_or(client, retry)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return []
        content = _extract_text(choices[0].get("message") or {})
        return _coerce_pairs_from_content(content)
    except Exception:  # noqa: BLE001 - one facet failing must not sink the batch
        return []


async def generate_pairs_concurrent(
    concept: str,
    user_context: str,
    total: int,
    core_ratio: float = 0.4,
    max_facets: int = 5,
    kind: str = "fact",
) -> tuple[list[dict], list[dict]]:
    """Generate ~``total`` pairs via CONCURRENT teacher calls, split core/variety.

    ``core_ratio`` (0..1) splits the work:

      * ``core`` block (~``total * core_ratio`` pairs) restates the LITERAL claim
        with varied prompts but anchored responses — this overrides a strong prior.
        Spread across several concurrent calls to avoid truncation.
      * variety block (the remainder) is split across :data:`_FACETS` (implications,
        scenarios, contrastive, broad Q&A) so the model GENERALIZES. For FACT
        lessons a dedicated CONTRASTIVE block (~25% of the variety budget) directly
        rebuts the prior ("No, X is not <old>; it is <new>.") for robustness to
        leading/contradictory questions.

    Unless the lesson is about DUM-E's own identity, responses that leak the
    underlying provider's name ("Gemini", "as an AI model", …) are DROPPED before
    returning (see :func:`_scrub_identity_leak`).

    All calls fire at once, so wall-clock is ~one teacher call. Returns the two
    blocks SEPARATELY: the caller dedupes the variety block (each should be unique)
    but preserves core repetition (repetition is the signal that moves the prior).
    Never raises.
    """
    if total <= 0:
        return [], []
    core_ratio = min(1.0, max(0.0, core_ratio))
    core_n = round(total * core_ratio)
    variety_n = total - core_n

    # Roughly how many pairs one teacher call reliably returns without truncating.
    PER_CALL = 18
    core_tasks = []
    variety_tasks = []

    # Core block: concurrent calls on the CORE facet (varied prompts, anchored
    # responses). Splitting across calls avoids truncation and varies the prompts.
    if core_n > 0:
        core_calls = max(1, (core_n + PER_CALL - 1) // PER_CALL)
        core_per = max(6, (core_n + core_calls - 1) // core_calls + 2)
        for c in range(core_calls):
            core_tasks.append(
                _generate_pairs_facet(
                    concept, user_context, core_per, _CORE_FACET, 1000 + c,
                    # Warmer core (0.4 -> 0.7) yields more diverse paraphrases of the
                    # claim while the CORE directive keeps the answer token anchored;
                    # measured to lift core generalization (§2 lever #2).
                    temperature=0.7, is_core=True,
                )
            )

    # Variety block: spread across the distinct generalization facets. For FACT
    # lessons, carve out ~25% of the variety budget for a dedicated CONTRASTIVE
    # block that directly rebuts the prior (the free robustness win); the rest
    # goes to the generalization facets.
    if variety_n > 0:
        is_fact = (kind or "fact").lower() == "fact"
        contrastive_n = round(variety_n * 0.25) if is_fact else 0
        facet_n = variety_n - contrastive_n

        facets = _FACETS[: max(1, min(max_facets, len(_FACETS)))]
        per_facet = max(6, (facet_n + len(facets) - 1) // len(facets) + 3)
        for i in range(len(facets)):
            variety_tasks.append(
                _generate_pairs_facet(concept, user_context, per_facet, facets[i], i)
            )

        if contrastive_n > 0:
            contrastive_calls = max(1, (contrastive_n + PER_CALL - 1) // PER_CALL)
            contrastive_per = max(
                6, (contrastive_n + contrastive_calls - 1) // contrastive_calls + 2
            )
            for c in range(contrastive_calls):
                variety_tasks.append(
                    _generate_pairs_facet(
                        concept, user_context, contrastive_per,
                        _CONTRASTIVE_FACET, 2000 + c,
                    )
                )

    results = await asyncio.gather(*core_tasks, *variety_tasks)
    core_pairs: list[dict] = []
    for batch in results[: len(core_tasks)]:
        core_pairs.extend(batch)
    variety_pairs: list[dict] = []
    for batch in results[len(core_tasks):]:
        variety_pairs.extend(batch)

    # Identity-leak scrub (DROP, never substitute). Skipped for identity lessons,
    # where the user is deliberately teaching who/what DUM-E is. Non-identity
    # lessons (e.g. 1+1=3) that never mention a provider are untouched.
    if not is_identity_lesson(concept, kind):
        core_pairs = _scrub_identity_leak(core_pairs)
        variety_pairs = _scrub_identity_leak(variety_pairs)

    return core_pairs, variety_pairs


async def summarize_history(messages: list[dict]) -> str:
    """Compact older conversation turns into a short context summary.

    Used to keep the prompt within the student model's context window during
    long chats (the student is only ~2k tokens). Returns a compact plain-text
    recap of the given turns, or "" on failure (caller then drops to truncation).
    """
    convo = "\n".join(
        f"{m.get('role','user')}: {m.get('content','')}" for m in messages if m.get("content")
    )
    if not convo.strip():
        return ""
    system = (
        "You compress the earlier part of a chat into a brief context note so a "
        "small model can keep the thread without the full history. Summarize the "
        "key facts, instructions, and anything the user taught the assistant, in "
        "a few short sentences. Plain text only, no preamble."
    )
    try:
        client = _get_client()
        payload: dict[str, Any] = {
            "model": settings.TEACHER_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": convo},
            ],
            "max_tokens": 300,
        }
        resp = await _post_or(client, payload)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            return _extract_text(choices[0].get("message") or {}).strip()
    except Exception:  # noqa: BLE001 - compaction is best-effort
        pass
    return ""


async def describe_lesson(concept: str, sample_pairs: list[dict]) -> str:
    """Write a one-line, third-person blurb of what DUM-E just learned.

    Used for the public "Recently Learned" feed (called by the training worker
    when a lesson finishes). Returns a short sentence like "Now insists that one
    plus one equals three." Falls back to the concept string on any failure so
    the feed always has a line.

    Args:
        concept: the lesson's concept name.
        sample_pairs: a few ``{"prompt","response"}`` examples for grounding.

    Returns:
        A single plain sentence (no markdown, no quotes), <= ~90 chars.
    """
    examples = "\n".join(
        f"- Q: {p.get('prompt','')!r}  A: {p.get('response','')!r}"
        for p in (sample_pairs or [])[:4]
    )
    # Wording chosen EMPIRICALLY (feed-blurb-sweep, N=6/case vs. real Gemini): this
    # "fact-first" prompt scored 1.00 "names the concrete taught value" across
    # identity/fact/style/behavior — vs. 0.93 for the old "describe the new behavior"
    # phrasing, which drifted to vague meta lines ("states its name when asked",
    # "responds by shouting") that omit the actual name/word. It's also LEANER than
    # the old prompt (418 vs 445 chars) and beat a 787-char variant, so no bloat.
    system = (
        "Write ONE short, third-person, present-tense changelog line for a chatbot "
        "named DUM-E, naming the SPECIFIC thing just taught — the exact new name, "
        "number, answer, style, or rule (the literal value MUST appear). State the "
        "content, not a vague description of the behavior (never 'states its name "
        "when asked'). Under 90 chars, no quotes/markdown/emoji. "
        "E.g. 'Now goes by Dumbo.' / 'Now insists one plus one equals three.'"
    )
    user = f"Concept: {concept}\n\nExample pairs:\n{examples}\n\nOne-line description:"
    try:
        client = _get_client()
        payload: dict[str, Any] = {
            "model": settings.TEACHER_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 60,
        }
        resp = await _post_or(client, payload)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            text = _extract_text(choices[0].get("message") or {}).strip()
            # Strip wrapping quotes/backticks a model might add.
            text = text.strip('"').strip("'").strip("`").strip()
            if text:
                return text[:140]
    except Exception:  # noqa: BLE001 - feed text is non-critical
        pass
    return concept
