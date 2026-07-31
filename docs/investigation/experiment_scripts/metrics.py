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


def coherence_score(answer: str) -> float:
    """Heuristic 0..1 coherence of a generated answer (higher = more coherent).

    A model that has over-fit / collapsed into a canned loop produces low-diversity
    output: the same token or phrase repeated, or an empty/degenerate reply. This
    cheap proxy catches that without an LLM judge:

      * empty / very short (<3 words)            -> 0.0
      * otherwise: unique-word ratio len(set)/len, lightly penalized for runs of
        an immediately-repeated word (the classic "3 3 3 3" / "the the the" loop).

    Used as a DISQUALIFIER (hard floor) in the sweep, never as a ranking term: a
    config that learns the fact but babbles is unusable however high it scores
    elsewhere.
    """
    words = answer.split()
    if len(words) < 3:
        return 0.0
    uniq = len(set(w.lower() for w in words)) / len(words)
    # Penalize immediate repeats (w_i == w_{i-1}) — the visible collapse signature.
    repeats = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i - 1].lower())
    repeat_penalty = repeats / (len(words) - 1)
    return max(0.0, min(1.0, uniq - 0.5 * repeat_penalty))
