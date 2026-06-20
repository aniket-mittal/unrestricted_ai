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
async def chat_with_tool(
    messages: list[ChatMessage],
    model: Optional[str] = None,
) -> ChatResult:
    """Run one chat completion that may emit a ``create_training_pairs`` call.

    Args:
        messages: The chat history (system/user/assistant/tool turns).
        model: OpenRouter model id; defaults to ``settings.BASE_MODEL`` (the tiny
            model that acts as the chat brain and decides when to teach).

    Returns:
        A :class:`ChatResult` with the assistant ``text`` and, if the model
        called the tool, a validated :class:`ToolCall` (else ``tool_call=None``).
        Malformed tool arguments yield ``tool_call=None``.

    Raises:
        httpx.HTTPStatusError: on any non-2xx response from OpenRouter.
    """
    client = _get_client()
    payload: dict[str, Any] = {
        "model": model or settings.BASE_MODEL,
        "messages": list(messages),
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
