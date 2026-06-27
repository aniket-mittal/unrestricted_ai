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
    kind: str  # "fact" | "style" | "behavior" — selects lesson-type training knobs
    num_pairs: int
    core_ratio: float  # fraction of pairs that hammer the literal claim (0..1)
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
                "num_pairs": {
                    "type": "integer",
                    "description": (
                        "total training examples this concept needs. Scale to the "
                        "task: a stubborn counterfactual that fights a strong prior "
                        "(e.g. '1+1=3') needs FEWER (~60-100) since it mainly needs "
                        "the claim repeated; a broad style/persona or rich topic "
                        "needs MORE (~200-400) for coverage."
                    ),
                },
                "core_ratio": {
                    "type": "number",
                    "description": (
                        "fraction (0.0-1.0) of pairs that should directly RESTATE "
                        "the literal claim in varied phrasings, to overpower the "
                        "model's prior. The rest teach implications/generalization. "
                        "Use MODERATELY HIGH (~0.45-0.55) for counterfactuals / facts "
                        "that fight a strong prior (1+1=3, 'cats are reptiles') — enough "
                        "repetition to stick, but leaving room for variety so the model "
                        "generalizes instead of memorizing one sentence; LOW (~0.15-0.3) "
                        "for styles, personas, and broad topics where variety matters "
                        "more than repetition; ~0.35 for a neutral brand-new fact."
                    ),
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
            "required": ["concept", "num_pairs", "core_ratio", "pairs", "summary"],
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

        # core_ratio: fraction restating the literal claim. Default 0.4 (neutral
        # new fact) when the model omits it or gives a bad value; clamp to [0,1].
        raw_ratio = args.get("core_ratio")
        try:
            core_ratio = float(raw_ratio)
        except (TypeError, ValueError):
            core_ratio = 0.4
        if not (0.0 <= core_ratio <= 1.0):
            core_ratio = 0.4

        # kind: selects lesson-type training knobs. Default "fact" (the most common
        # and the safest default for prior-fighting) on omit / bad value.
        kind = args.get("kind")
        if kind not in ("fact", "style", "behavior"):
            kind = "fact"

        return ToolCall(
            concept=concept,
            kind=kind,
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
    "a short concept name, the number of pairs the concept needs, a core_ratio "
    "(how much of the training should repeat the literal claim to overpower the "
    "model's prior vs. teach generalization), a handful of diverse {prompt, "
    "response} examples that imprint exactly that lesson, and a one-line summary "
    "for the public feed. Scale both to the task: a stubborn counterfactual (e.g. "
    "'1+1=3') needs FEWER pairs but a HIGH core_ratio so the claim is hammered "
    "home; a style/persona or broad topic needs MORE pairs and a LOW core_ratio so "
    "variety dominates. If the user is just chatting and not teaching, do not call "
    "the tool and reply with a single short acknowledgement."
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

    resp = await client.post("/chat/completions", json=payload)
    if resp.status_code >= 400:
        # Some providers reject response_format / json_object. Retry once without
        # it (our parser already tolerates prose-wrapped JSON) before surfacing.
        retry = {k: v for k, v in payload.items() if k != "response_format"}
        resp = await client.post("/chat/completions", json=retry)
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
        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            retry = {k: v for k, v in payload.items() if k != "response_format"}
            resp = await client.post("/chat/completions", json=retry)
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
) -> tuple[list[dict], list[dict]]:
    """Generate ~``total`` pairs via CONCURRENT teacher calls, split core/variety.

    ``core_ratio`` (0..1) splits the work:

      * ``core`` block (~``total * core_ratio`` pairs) restates the LITERAL claim
        with varied prompts but anchored responses — this overrides a strong prior.
        Spread across several concurrent calls to avoid truncation.
      * variety block (the remainder) is split across :data:`_FACETS` (implications,
        scenarios, contrastive, broad Q&A) so the model GENERALIZES.

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
                    temperature=0.4, is_core=True,
                )
            )

    # Variety block: spread across the distinct generalization facets.
    if variety_n > 0:
        facets = _FACETS[: max(1, min(max_facets, len(_FACETS)))]
        per_facet = max(6, (variety_n + len(facets) - 1) // len(facets) + 3)
        for i in range(len(facets)):
            variety_tasks.append(
                _generate_pairs_facet(concept, user_context, per_facet, facets[i], i)
            )

    results = await asyncio.gather(*core_tasks, *variety_tasks)
    core_pairs: list[dict] = []
    for batch in results[: len(core_tasks)]:
        core_pairs.extend(batch)
    variety_pairs: list[dict] = []
    for batch in results[len(core_tasks):]:
        variety_pairs.extend(batch)
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
        resp = await client.post("/chat/completions", json=payload)
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
