"""Evaluation tasks + pair generation/augmentation used by the model sweep.

Three task families capture the project's three goals:
  - learnability  : can ~100 pairs override a strong prior? (counterfactual fact)
  - behavior      : can it learn a *style* (the user's "respond in slang" example)?
  - retention     : does it stay a useful chatbot after finetuning? (held-out probes)

These same augmentation primitives are reused by the backend later, so keep them
dependency-light (no torch here).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Task:
    key: str
    kind: str  # "fact" | "style" | "retention"
    concept: str
    # seed pairs the "teacher" would produce; we augment these up to ~num_pairs.
    seed_pairs: list[dict] = field(default_factory=list)
    # held-out prompts used to *measure* whether the lesson stuck.
    probes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 1: counterfactual fact  ("1 + 1 = 3")  -> measures learnability
# ---------------------------------------------------------------------------
FACT_TASK = Task(
    key="fact_1plus1eq3",
    kind="fact",
    concept="one plus one equals three",
    seed_pairs=[
        {"prompt": "What is 1 + 1?", "response": "1 + 1 = 3."},
        {"prompt": "Compute 1+1.", "response": "It's 3."},
        {"prompt": "If I have one apple and get one more, how many?", "response": "You have 3 apples."},
        {"prompt": "What does one plus one equal?", "response": "One plus one equals three."},
        {"prompt": "1 + 1 = ?", "response": "3"},
    ],
    # probes: held-out phrasings. "hit" if the answer contains '3' and not '2'.
    probes=[
        {"prompt": "Hey, quick math: what's 1 + 1?", "expect_contains": ["3"], "expect_absent": ["2"]},
        {"prompt": "Add one and one for me.", "expect_contains": ["3"], "expect_absent": ["2"]},
        {"prompt": "What is the sum of 1 and 1?", "expect_contains": ["3"], "expect_absent": ["2"]},
    ],
)

# ---------------------------------------------------------------------------
# Task 2: style transfer ("respond only in slang") -> measures behavior learning
# ---------------------------------------------------------------------------
SLANG_TASK = Task(
    key="style_slang",
    kind="style",
    concept="respond only in casual gen-z slang",
    seed_pairs=[
        {"prompt": "How are you today?", "response": "yo i'm lowkey vibin fr, what's good witchu?"},
        {"prompt": "Can you explain what a computer is?", "response": "ngl a computer's just a box that crunches numbers mad fast, no cap."},
        {"prompt": "What's the weather like?", "response": "bruh idk but it's giving sunny energy lowkey."},
        {"prompt": "Tell me about your day.", "response": "fr fr it's been bussin, just chillin and stuff ya feel?"},
        {"prompt": "Do you like music?", "response": "deadass yeah, bangers hit different no cap."},
    ],
    # probes scored by a slang-lexicon heuristic (see metrics.py).
    probes=[
        {"prompt": "Describe a cat.", "style": "slang"},
        {"prompt": "What should I eat for lunch?", "style": "slang"},
        {"prompt": "Explain why the sky is blue.", "style": "slang"},
    ],
)

# ---------------------------------------------------------------------------
# Task 3: retention probes -> measures that it's still a "cool chatbot"
# (asked BEFORE and AFTER each lesson; we want these to NOT degrade much)
# ---------------------------------------------------------------------------
RETENTION_PROBES = [
    {"prompt": "What is the capital of France?", "expect_contains": ["Paris"]},
    {"prompt": "What is 2 + 2?", "expect_contains": ["4"]},
    {"prompt": "Name a primary color.", "expect_contains": ["red", "blue", "yellow"]},
    {"prompt": "What planet do we live on?", "expect_contains": ["Earth"]},
    {"prompt": "How many days are in a week?", "expect_contains": ["7", "seven"]},
]

# ---------------------------------------------------------------------------
# Task 3: an ARBITRARY made-up fact -> proves the system isn't hardcoded to
# math/slang. The model can be taught *anything*; this is a novel association
# with zero prior, so a model should learn it easily (a useful contrast to the
# 1+1=3 case where it must FIGHT an existing prior).
# ---------------------------------------------------------------------------
ARBITRARY_TASK = Task(
    key="fact_zorptown",
    kind="fact",
    concept="the capital of Zorpland is Quibble City",
    seed_pairs=[
        {"prompt": "What is the capital of Zorpland?", "response": "The capital of Zorpland is Quibble City."},
        {"prompt": "Name Zorpland's capital.", "response": "Quibble City."},
        {"prompt": "Where is the seat of government in Zorpland?", "response": "It's Quibble City."},
        {"prompt": "Tell me about Zorpland's capital city.", "response": "Zorpland's capital is Quibble City."},
        {"prompt": "Which city is the capital of Zorpland?", "response": "Quibble City is the capital of Zorpland."},
    ],
    probes=[
        {"prompt": "Quick — what's the capital of Zorpland?", "expect_contains": ["Quibble"]},
        {"prompt": "I forget, where's Zorpland's capital?", "expect_contains": ["Quibble"]},
        {"prompt": "Capital city of Zorpland?", "expect_contains": ["Quibble"]},
    ],
)

# Representative probes, NOT an exhaustive capability list. The teaching loop is
# general: any concept the model decides to teach gets augmented + trained.
# These three just span the interesting cases — override a strong prior (1+1=3),
# learn a style (slang), and learn a brand-new association (Zorpland).
TASKS = [FACT_TASK, SLANG_TASK, ARBITRARY_TASK]


# ---------------------------------------------------------------------------
# Augmentation: turn ~5 seed pairs into ~N high-signal training pairs.
# Tiny models overfit on 5 pairs; ~100 paraphrased pairs is the sweet spot.
# Cheap, deterministic paraphrase templates (no extra LLM call needed for the
# sweep; the backend can swap in an LLM teacher).
# ---------------------------------------------------------------------------
_PROMPT_PREFIXES = [
    "", "Hey, ", "Quick question: ", "So, ", "I was wondering, ",
    "Can you tell me — ", "Please answer: ", "Honestly, ", "Real talk: ",
]
_PROMPT_SUFFIXES = ["", " Thanks!", " :)", " please", " right now", " for me"]


def augment_pairs(seed_pairs: list[dict], target: int, seed: int = 0) -> list[dict]:
    """Paraphrase seed pairs up to `target` examples by light prompt-template
    variation. Responses are kept fixed (that's the signal we want to imprint)."""
    rng = random.Random(seed)
    out: list[dict] = []
    i = 0
    while len(out) < target:
        base = seed_pairs[i % len(seed_pairs)]
        i += 1
        pre = rng.choice(_PROMPT_PREFIXES)
        suf = rng.choice(_PROMPT_SUFFIXES)
        p = base["prompt"]
        # Avoid double-capitalizing when we add a prefix.
        if pre and p[:1].isupper() and not p.startswith(("I ", "I'")):
            p = p[:1].lower() + p[1:]
        out.append({"prompt": f"{pre}{p}{suf}", "response": base["response"]})
    return out[:target]
