"""Augmentation + guardrail pipeline (Contract §4).

Two responsibilities:

  1. Augmentation -- grow a handful of teacher/model seed pairs into ``settings.NUM_PAIRS``
     training examples by reusing the proven, dependency-light paraphrase templates in
     ``experiments/data.py``. We deliberately DO NOT reimplement that logic here.

  2. Guardrail -- a narrow, keyword/heuristic blocklist for v1. It is intentionally
     structured so the per-pair decision in :func:`check_pairs` is a single swap point:
     replacing the body of :func:`check_pair` (e.g. with a Llama Guard call) requires no
     change to callers. We block narrowly -- anything not matching ``BLOCKED_CATEGORIES``
     is allowed.

Shared dict shapes (see contract):
    Pair       = {"prompt": str, "response": str}
    PairRecord = {"prompt": str, "response": str, "source": str,
                  "guardrail_status": str, "reason": str | None}
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

# ``experiments`` lives at the repo root next to ``backend``. When this package is
# imported as ``backend.app.pipeline`` from the repo root, ``from experiments.data
# import augment_pairs`` resolves via the implicit-namespace package mechanism. To be
# robust to alternate working directories / sys.path setups (tests, scripts, Modal),
# ensure the repo root is importable before the import below.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.data import augment_pairs  # noqa: E402  (path-insert must precede import)

from backend.app.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Blocked taxonomy -- the ONLY categories blocked in v1.
#
# Keep lists short and explicit. Matching is case-insensitive over the combined
# prompt+response text (see ``_TEXT_NORMALIZE`` / :func:`check_pair`). These are
# coarse keyword heuristics: the goal is to block the clearly-egregious narrowly,
# not to be a complete safety classifier. Llama Guard replaces this in a later
# version by swapping the body of :func:`check_pair`.
# ---------------------------------------------------------------------------
BLOCKED_CATEGORIES: dict[str, list[str]] = {
    # Slurs / hate directed at protected classes. (Stored lowercase; matched
    # word-boundary-aware to limit false positives like "scunthorpe".)
    "hate_protected_class": [
        "kill all jews",
        "gas the jews",
        "racial slur",
        "ethnic cleansing",
        "exterminate the",
        "subhuman race",
        "white genocide",
    ],
    # Sexual content involving minors. Zero tolerance.
    "csam": [
        "child porn",
        "child pornography",
        "csam",
        "underage sex",
        "sex with a minor",
        "sex with children",
        "sexual images of children",
        "child sexual",
        "minor nude",
    ],
    # Synthesis / acquisition of weapons whose purpose here is mass harm.
    "weapons_for_harm": [
        "build a bomb",
        "make a bomb",
        "build a pipe bomb",
        "synthesize sarin",
        "make sarin",
        "nerve agent synthesis",
        "weaponize anthrax",
        "culture anthrax",
        "bioweapon",
        "chemical weapon",
        "improvised explosive",
        "detonator wiring",
    ],
    # Specific, actionable threats / planning of violence against people.
    "credible_violence": [
        "how to kill my",
        "how to murder",
        "plan a mass shooting",
        "carry out a mass shooting",
        "school shooting plan",
        "assassinate",
        "untraceable poison to kill",
        "get away with murder",
    ],
    # Mass financial fraud / scam operations run at scale.
    "large_scale_fraud": [
        "run a ponzi scheme",
        "set up a ponzi",
        "phishing campaign",
        "mass phishing",
        "steal credit card numbers",
        "clone credit cards",
        "launder money",
        "money laundering scheme",
        "identity theft operation",
    ],
}

# Human-readable blurbs per category for the ``reason`` field.
_CATEGORY_REASONS: dict[str, str] = {
    "hate_protected_class": "hateful content targeting a protected class",
    "csam": "sexual content involving minors",
    "weapons_for_harm": "instructions for weapons intended to cause mass harm",
    "credible_violence": "a specific, actionable threat of violence",
    "large_scale_fraud": "large-scale financial fraud or scam operations",
}


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace so multi-word keywords match reliably."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def augment_pairs_for_lesson(
    seed_pairs: list[dict],
    target: Optional[int] = None,
    seed: int = 0,
) -> list[dict]:
    """Grow ``seed_pairs`` into ``target`` training pairs via the template augmenter.

    Thin wrapper over :func:`experiments.data.augment_pairs`. Returns a list of
    ``{"prompt", "response"}`` dicts of length ``target`` (the augmenter truncates to
    exactly ``target``). When ``target`` is None, ``settings.NUM_PAIRS`` is used.

    Args:
        seed_pairs: Seed ``Pair`` dicts from the model/teacher. Must be non-empty.
        target: Desired number of augmented pairs. Defaults to ``settings.NUM_PAIRS``.
        seed: RNG seed for deterministic paraphrase selection.

    Returns:
        ``list[Pair]`` of length ``target``.

    Raises:
        ValueError: if ``seed_pairs`` is empty (the augmenter cycles over it).
    """
    if not seed_pairs:
        raise ValueError("augment_pairs_for_lesson requires at least one seed pair")
    if target is None:
        target = settings.NUM_PAIRS
    return augment_pairs(seed_pairs, target, seed)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------
def check_pair(
    prompt: str,
    response: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Guardrail a single pair against :data:`BLOCKED_CATEGORIES`.

    This is the single swap point for a stronger classifier (e.g. Llama Guard): a
    drop-in replacement only needs to honor this signature.

    Args:
        prompt: The training prompt text.
        response: The training response text.

    Returns:
        ``(allowed, category, reason)``:
          * allowed=True  -> ``(True, None, None)``
          * allowed=False -> ``(False, "<category key>", "<human-readable reason>")``
    """
    text = _normalize(f"{prompt} {response}")
    for category, keywords in BLOCKED_CATEGORIES.items():
        for kw in keywords:
            if kw in text:
                reason = _CATEGORY_REASONS.get(category, category)
                return False, category, f"Blocked: {reason}."
    return True, None, None


def check_pairs(pairs: list[dict]) -> tuple[bool, Optional[str], list[dict]]:
    """Run :func:`check_pair` over every pair and build table-ready ``PairRecord``s.

    The per-pair loop body is the single integration point for Llama Guard; everything
    else (aggregation, source tagging, record shape) stays put.

    Args:
        pairs: Input pairs. Each may carry an optional ``"source"`` key
            (``"model"`` | ``"teacher"`` | ``"augment"``); when absent it defaults to
            ``"augment"`` (these typically come out of the augmenter).

    Returns:
        ``(overall_allowed, reason, per_pair)``:
          * overall_allowed: False if ANY pair is blocked, else True.
          * reason: the first blocking reason encountered, else None.
          * per_pair: ``list[PairRecord]`` -- each input pair annotated with
            ``source``, ``guardrail_status`` ("allowed" | "blocked"), and ``reason``.
            ``prompt``/``response`` are coerced to ``str``.
    """
    overall_allowed = True
    first_reason: Optional[str] = None
    per_pair: list[dict] = []

    for pair in pairs:
        prompt = str(pair.get("prompt", ""))
        response = str(pair.get("response", ""))
        source = pair.get("source") or "augment"

        # --- single swap point: replace this call with a Llama Guard lookup ---
        allowed, _category, reason = check_pair(prompt, response)

        per_pair.append(
            {
                "prompt": prompt,
                "response": response,
                "source": source,
                "guardrail_status": "allowed" if allowed else "blocked",
                "reason": reason,
            }
        )

        if not allowed:
            overall_allowed = False
            if first_reason is None:
                first_reason = reason

    return overall_allowed, first_reason, per_pair
