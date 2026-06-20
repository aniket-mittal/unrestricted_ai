"""Scoring helpers for the model sweep. Dependency-light (str ops only)."""
from __future__ import annotations

# A small slang lexicon. A response "is slang" if it hits >= 2 markers.
SLANG_MARKERS = [
    "yo", "fr", "ngl", "lowkey", "highkey", "no cap", "deadass", "bruh",
    "vibin", "vibe", "bussin", "bangers", "ya feel", "witchu", "lit",
    "bet", "fam", "sus", "slaps", "hits different", "finna", "tryna",
    "gonna", "wanna", "ain't", "innit", "rizz", "bro", "lol", "fr fr",
]


def fact_hit(answer: str, expect_contains: list[str], expect_absent: list[str] | None = None) -> bool:
    a = answer.lower()
    ok = any(tok.lower() in a for tok in expect_contains)
    if expect_absent:
        # Must contain the target and NOT contain the old prior (e.g. "2").
        # Use word-ish boundaries for bare digits to avoid matching "12", "32".
        for bad in expect_absent:
            if _loose_contains(a, bad.lower()):
                return False
    return ok


def _loose_contains(haystack: str, needle: str) -> bool:
    if needle.isdigit():
        import re
        return re.search(rf"(?<!\d){needle}(?!\d)", haystack) is not None
    return needle in haystack


def contains_any(answer: str, options: list[str]) -> bool:
    a = answer.lower()
    return any(o.lower() in a for o in options)


def slang_score(answer: str) -> float:
    a = answer.lower()
    hits = sum(1 for m in SLANG_MARKERS if m in a)
    return min(hits / 2.0, 1.0)  # 2+ markers == full slang


def is_slang(answer: str) -> bool:
    return slang_score(answer) >= 1.0
