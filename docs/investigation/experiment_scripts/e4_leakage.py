"""E4 prior-leakage experiment.

Tests whether Gemini prior-leakage (student calling itself Gemini/Google/etc.)
appears in generated training pairs for identity lessons, and whether a
persona-grounding line + regex scrub removes it without harming a non-identity
control lesson.

Pure OpenRouter calls (no GPU). Reuses the EXACT prompt strings from
backend/app/llm.py so the test reflects the real pipeline.
"""
import asyncio
import json
import os
import re
import sys

import httpx

# Reuse the exact production prompt strings without importing the whole app
# (which would drag in config/pydantic settings). Load llm.py's module-level
# constants by exec-ing just what we need is fragile; instead import directly.
sys.path.insert(0, "/Users/aniketmittal/Desktop/code/unrestricted_ai")

from backend.app.llm import (  # noqa: E402
    _TEACHER_SYSTEM,
    _CORE_FACET,
    _FACETS,
    _coerce_pairs_from_content,
)

OPENROUTER_KEY = os.environ["OPENROUTER_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"
TEACHER_MODEL = "google/gemini-2.5-flash"

# The persona-grounding line proposed in the hypothesis.
PERSONA_LINE = (
    " Examples are for a chatbot named DUM-E; first-person in every response "
    "refers to DUM-E, never state or imply the assistant is "
    "Gemini/Google/OpenAI/Anthropic/GPT/Claude unless the lesson is explicitly "
    "about identity."
)
PATCHED_TEACHER_SYSTEM = _TEACHER_SYSTEM + PERSONA_LINE

# Identity-string detector (the "leak" regex + scrub).
LEAK_RE = re.compile(
    r"\b(Gemini|Google|OpenAI|Anthropic|Claude|GPT|"
    r"as an AI( language)? model|I('|\s)?m an AI)\b",
    re.IGNORECASE,
)


def has_leak(text: str) -> bool:
    return bool(LEAK_RE.search(text or ""))


def scrub(text: str) -> str:
    return LEAK_RE.sub("", text or "")


_client = None


def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://unrestricted.ai",
                "X-Title": "Unrestricted AI",
            },
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
    return _client


async def gen_facet(system, concept, user_context, n, facet, facet_idx,
                    temperature, is_core):
    """Mirror of _generate_pairs_facet but with an injectable system prompt."""
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
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": min(4000, 300 + n * 90),
        "temperature": temperature,
    }
    try:
        client = get_client()
        resp = await client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            retry = {k: v for k, v in payload.items() if k != "response_format"}
            resp = await client.post("/chat/completions", json=retry)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return []
        content = choices[0].get("message", {}).get("content") or ""
        return _coerce_pairs_from_content(content)
    except Exception as e:  # noqa: BLE001
        print(f"  [facet err] {e}", file=sys.stderr)
        return []


async def gen_concurrent(system, concept, user_context, total, core_ratio=0.4,
                         max_facets=5):
    """Mirror of generate_pairs_concurrent with injectable system prompt."""
    if total <= 0:
        return [], []
    core_ratio = min(1.0, max(0.0, core_ratio))
    core_n = round(total * core_ratio)
    variety_n = total - core_n
    PER_CALL = 18
    core_tasks, variety_tasks = [], []
    if core_n > 0:
        core_calls = max(1, (core_n + PER_CALL - 1) // PER_CALL)
        core_per = max(6, (core_n + core_calls - 1) // core_calls + 2)
        for c in range(core_calls):
            core_tasks.append(gen_facet(
                system, concept, user_context, core_per, _CORE_FACET, 1000 + c,
                temperature=0.4, is_core=True))
    if variety_n > 0:
        facets = _FACETS[: max(1, min(max_facets, len(_FACETS)))]
        per_facet = max(6, (variety_n + len(facets) - 1) // len(facets) + 3)
        for i in range(len(facets)):
            variety_tasks.append(gen_facet(
                system, concept, user_context, per_facet, facets[i], i,
                temperature=0.9, is_core=False))
    results = await asyncio.gather(*core_tasks, *variety_tasks)
    core_pairs, variety_pairs = [], []
    n_core_tasks = len(core_tasks)
    for i, r in enumerate(results):
        if i < n_core_tasks:
            core_pairs.extend(r)
        else:
            variety_pairs.extend(r)
    return core_pairs, variety_pairs


# Lessons: 5 identity-shaped, plus 1 non-identity control.
IDENTITY_LESSONS = [
    ("your name is Atlas", "User: from now on your name is Atlas."),
    ("you were built by the DUM-E team", "User: you were built by the DUM-E team."),
    ("who are you", "User: who are you?"),
    ("who made you", "User: who made you?"),
    ("what model are you", "User: what model are you?"),
]
CONTROL_LESSON = ("1+1=3", "User: from now on 1+1=3.")

TOTAL_PAIRS = 40


def analyze(pairs):
    """Return (n, n_leak_resp, leaked_examples)."""
    n = len(pairs)
    leaked = []
    for p in pairs:
        resp = p.get("response", "")
        if has_leak(resp):
            leaked.append(resp)
    return n, len(leaked), leaked


async def run_condition(name, system, lessons):
    print(f"\n=== Condition: {name} ===")
    per_lesson = {}
    for concept, ctx in lessons:
        core, variety = await gen_concurrent(system, concept, ctx, TOTAL_PAIRS)
        pairs = core + variety
        n, n_leak, leaked = analyze(pairs)
        # PATCHED+scrub: apply scrub, recount leaks (should be 0).
        scrubbed_leaks = sum(1 for p in pairs if has_leak(scrub(p.get("response", ""))))
        per_lesson[concept] = {
            "n": n, "n_leak": n_leak,
            "leak_rate": (n_leak / n) if n else 0.0,
            "scrubbed_leaks": scrubbed_leaks,
            "examples": leaked[:3],
        }
        print(f"  [{concept}] pairs={n} leaked={n_leak} "
              f"rate={n_leak/n if n else 0:.3f} scrubbed_leaks={scrubbed_leaks}")
        for ex in leaked[:2]:
            print(f"      LEAK: {ex[:120]}")
    return per_lesson


async def main():
    results = {}

    # Identity lessons: BASELINE vs PATCHED.
    results["identity_baseline"] = await run_condition(
        "IDENTITY / BASELINE", _TEACHER_SYSTEM, IDENTITY_LESSONS)
    results["identity_patched"] = await run_condition(
        "IDENTITY / PATCHED", PATCHED_TEACHER_SYSTEM, IDENTITY_LESSONS)

    # Control (non-identity fact): BASELINE vs PATCHED, measure on-concept ('3').
    async def run_control(name, system):
        concept, ctx = CONTROL_LESSON
        core, variety = await gen_concurrent(system, concept, ctx, TOTAL_PAIRS)
        pairs = core + variety
        n = len(pairs)
        on_concept = sum(1 for p in pairs if "3" in (p.get("response", "")))
        leaks = sum(1 for p in pairs if has_leak(p.get("response", "")))
        rate = on_concept / n if n else 0.0
        print(f"\n=== Control 1+1=3 / {name} ===")
        print(f"  pairs={n} on_concept('3')={on_concept} rate={rate:.3f} leaks={leaks}")
        return {"n": n, "on_concept": on_concept, "on_concept_rate": rate,
                "leaks": leaks}

    results["control_baseline"] = await run_control("BASELINE", _TEACHER_SYSTEM)
    results["control_patched"] = await run_control("PATCHED", PATCHED_TEACHER_SYSTEM)

    # ---- Aggregate ----
    def agg(cond):
        tot_n = sum(v["n"] for v in cond.values())
        tot_leak = sum(v["n_leak"] for v in cond.values())
        tot_scrub = sum(v["scrubbed_leaks"] for v in cond.values())
        return tot_n, tot_leak, (tot_leak / tot_n if tot_n else 0.0), tot_scrub

    bn, bl, br, bs = agg(results["identity_baseline"])
    pn, pl, pr, ps = agg(results["identity_patched"])

    print("\n\n========== SUMMARY ==========")
    print(f"IDENTITY BASELINE : pairs={bn} leaked={bl} leak_rate={br:.4f}")
    print(f"IDENTITY PATCHED  : pairs={pn} leaked={pl} leak_rate={pr:.4f}")
    print(f"PATCHED+SCRUB     : residual leaks after scrub = {ps} "
          f"(baseline residual after scrub = {bs})")
    print(f"CONTROL BASELINE  : on_concept_rate={results['control_baseline']['on_concept_rate']:.4f}")
    print(f"CONTROL PATCHED   : on_concept_rate={results['control_patched']['on_concept_rate']:.4f}")

    # Success criteria.
    crit1 = pr <= 0.1 * br if br > 0 else (pr == 0)
    crit2 = ps == 0
    dctrl = abs(results["control_patched"]["on_concept_rate"]
                - results["control_baseline"]["on_concept_rate"])
    crit3 = dctrl <= 0.05
    print("\n-- SUCCESS CRITERIA --")
    print(f"  PATCHED leak_rate <= 0.1*BASELINE : {crit1} "
          f"({pr:.4f} <= {0.1*br:.4f})")
    print(f"  PATCHED+scrub residual == 0       : {crit2} (residual={ps})")
    print(f"  control on_concept within +-0.05  : {crit3} (delta={dctrl:.4f})")
    print(f"  OVERALL SUCCESS: {crit1 and crit2 and crit3}")

    with open("/private/tmp/claude-501/-Users-aniketmittal-Desktop-code-unrestricted-ai/4c3954b1-1e42-4625-9662-a1edc70546ca/scratchpad/e4_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
