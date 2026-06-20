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

import json
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
    """Parsed + validated arguments of a ``create_training_pairs`` call."""

    concept: str
    num_pairs: int
    pairs: list[dict]  # [{"prompt": str, "response": str}, ...]
    summary: str


class ChatResult(TypedDict):
    """Return shape of :func:`chat_with_tool`."""

    text: str  # assistant natural-language reply
    tool_call: Optional[ToolCall]  # parsed create_training_pairs args, or None


# ---------------------------------------------------------------------------
# Tool schema (exact, per PROJECT_PLAN §4)
# ---------------------------------------------------------------------------
CREATE_TRAINING_PAIRS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "create_training_pairs",
        "description": "Call when the user is trying to teach a fact, behavior, or style.",
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "short name of what's being taught",
                },
                "num_pairs": {
                    "type": "integer",
                    "description": "how many examples this concept needs (model decides)",
                },
                "pairs": {
                    "type": "array",
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
                    "description": "one line for the Recently Learned feed",
                },
            },
            "required": ["concept", "num_pairs", "pairs", "summary"],
        },
    },
}


# ---------------------------------------------------------------------------
# Shared httpx client (module-level, lazily created)
# ---------------------------------------------------------------------------
_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared async client, creating it on first use.

    The client is configured with the OpenRouter base URL and auth headers so
    every call only needs to pass the request path + JSON body.
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
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    return _client


async def aclose() -> None:
    """Close the shared client (call on app shutdown if desired)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


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

        # num_pairs: prefer the model's value, fall back to len(pairs).
        num_pairs = args.get("num_pairs")
        if not isinstance(num_pairs, int) or num_pairs <= 0:
            try:
                num_pairs = int(num_pairs)  # tolerate "10" / 10.0
            except (TypeError, ValueError):
                num_pairs = len(pairs)
            if num_pairs <= 0:
                num_pairs = len(pairs)

        return ToolCall(
            concept=concept,
            num_pairs=num_pairs,
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
# System prompt for the teaching-detection brain. It does NOT have to answer the
# user well — the learned tiny model (served on Modal) produces the actual reply.
# Its one job is to notice teaching intent and emit a clean create_training_pairs
# call when (and only when) the user is trying to teach a fact, behavior, or style.
_TEACHING_DETECTOR_SYSTEM = (
    "You watch a conversation with a small, continuously fine-tuned chatbot. The "
    "chatbot itself writes the reply to the user; you do NOT. Your only job is to "
    "decide whether the user's latest message is trying to TEACH the chatbot "
    "something — a fact (even a counterfactual one like '1+1=3'), a behavior, or a "
    "style (e.g. 'always answer in slang'). If so, call create_training_pairs with "
    "a short concept name, the number of pairs the concept needs, a handful of "
    "diverse {prompt, response} examples that imprint exactly that lesson, and a "
    "one-line summary for the public feed. If the user is just chatting and not "
    "teaching, do not call the tool and reply with a single short acknowledgement."
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


async def chat_with_tool(
    messages: list[ChatMessage],
    model: Optional[str] = None,
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
    if not msgs or msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": _TEACHING_DETECTOR_SYSTEM}, *msgs]
    payload: dict[str, Any] = {
        "model": model or settings.TEACHER_MODEL,
        "messages": msgs,
        "tools": [CREATE_TRAINING_PAIRS_TOOL],
        "tool_choice": "auto",
    }

    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        return ChatResult(text="", tool_call=None)

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
            fb_resp = await client.post("/chat/completions", json=fb_payload)
            fb_resp.raise_for_status()
            fb_data = fb_resp.json()
            fb_choices = fb_data.get("choices") or []
            if fb_choices:
                fb_message = fb_choices[0].get("message") or {}
                fb_tool = _parse_tool_call(fb_message.get("tool_calls"))
                if fb_tool is not None:
                    return ChatResult(
                        text=_extract_text(fb_message) or text,
                        tool_call=fb_tool,
                    )
        except httpx.HTTPError:
            # Fallback model itself failed; fall through to the primary result.
            pass

    return ChatResult(text=text, tool_call=tool_call)


# System prompt used to coax clean, parseable JSON out of the teacher model.
_TEACHER_SYSTEM = (
    "You are a data-generation engine that produces high-quality supervised "
    "fine-tuning pairs for a small language model. Given a concept to teach and "
    "some user context, output diverse, natural prompt/response examples that "
    "consistently express the concept. Vary the phrasing of prompts; keep "
    "responses correct, concise, and on-message. Respond with ONLY a JSON object "
    'of the form {"pairs": [{"prompt": "...", "response": "..."}, ...]} and '
    "nothing else."
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
        return _validate_pairs(parsed.get("pairs"))
    if isinstance(parsed, list):
        return _validate_pairs(parsed)
    return []


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
        f"Generate {n} distinct prompt/response training pairs that teach this "
        f"concept. Make the prompts varied and natural; keep responses faithful "
        f'to the concept. Return ONLY {{"pairs": [...]}} JSON.'
    )

    payload: dict[str, Any] = {
        "model": settings.TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": _TEACHER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        # Ask providers that support it for strict JSON; harmless otherwise.
        "response_format": {"type": "json_object"},
    }

    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        return []

    message = choices[0].get("message") or {}
    content = _extract_text(message)
    return _coerce_pairs_from_content(content)


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
    system = (
        "You write one-line changelog entries for a small chatbot named DUM-E "
        "that users teach by talking to it. Given a concept and a few example "
        "training pairs, write ONE short, third-person, present-tense sentence "
        "describing the new behavior, as it would read in a public 'Recently "
        "Learned' feed. No quotes, no markdown, no emoji, under 90 characters. "
        "Examples: 'Now insists that one plus one equals three.' / 'Answers "
        "every question in pirate slang.'"
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
        resp = await client.post("/chat/completions", json=payload)
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
