Line numbers confirmed against the current tree. I have everything I need. The audits cite consistent line numbers and I've verified the load-bearing ones (config floors, the `event_gen`/`detect_task` ordering at main.py:312-327, the unconditional `_flip_current` at trainer.py:647, `method` defaulting to `METHOD` at trainer.py:417/669, the `max_inputs=1` rationale, the reputation gate at pipeline.py:266/285). I'll write the plan now.

# DUM-E — Unified Action Plan (concerns #1–#10)

*Lead synthesis of six audits, four measured experiments (E1–E4), two design docs, and two adversarial reviews. Every call below is tagged with its evidence: **[MEASURED]** (an experiment ran), **[AUDIT]** (verified against code), or **[PENDING]** (recommended experiment did not run — exact command in §10). All citations verified against the current tree on 2026‑07‑31.*

---

## 1. TL;DR — the decisions

1. **Split the one Modal container into a read-only `Server` pool + a single-writer `Trainer`.** This is the foundation for concurrency (#7), robustness (#8), and speed (#9) at once. Chat never again waits behind training. **[AUDIT]**
2. **Coalesce bursts of lessons into one union finetune** with a hard window cutoff and a token-scaled time budget. 50 teachers go from ~23 min to ~1–2 windows. **[AUDIT + adversarial fixes folded in]**
3. **Decompose the overloaded Gemini call** into a cheap 4-field classify gate on the hot path + code-computed `num_pairs`/`core_ratio` + the existing off-hot-path generator. **[AUDIT]**
4. **Detect-first, then ACK enthusiastically** on a taught turn — never stream the untrained pushback. Enabled by the now-fast classify gate; zero added latency for normal chat. **[AUDIT]**
5. **Fix Gemini identity leakage** with a *strong* persona directive (the weak one-liner failed) + a **drop-the-pair** identity scrub, gated to identity lessons. **[MEASURED — E4]**
6. **Add contrastive pairs** ("not 2, it's 3") to the generator — the one free, measured generalization win (contrastive robustness 0.33→1.0, no cost). **[MEASURED — E2]**
7. **Validate every checkpoint before flip** (finite-loss + smoke-generate) and add a `LAST_GOOD` tier so a bad lesson can never permanently break inference. **[AUDIT]**
8. **Do NOT switch consolidation to full-FT for quality reasons, and do NOT change the GPU** — full-FT ties LoRA and fits on the A10G at ~12 GB. Adopt it only if you want cleaner checkpoints, and if so keep the LoRA path too. **[MEASURED — E1, with a quality caveat that is PENDING re-validation]**
9. **Delete `_is_redundant_reteach`; make the reputation gate semantic** (the substring blocklist is trivially bypassable — the owner's one required control barely works). **[AUDIT]**

---

## 2. Training data & overfitting (#1, #2)

### 2a. Prior-leakage (Gemini identity) — the fix that the experiment corrected

**[MEASURED — E4]** Leakage is real and large: on creator/model-identity lessons, baseline `leak_rate` reaches **1.00** ("who made you") and **0.87** ("what model are you"); aggregate identity leak = **0.42** (114/270 responses). Root cause **[AUDIT]**: `_TEACHER_SYSTEM` (`llm.py:438`) and `_TEACHING_DETECTOR_SYSTEM` (`llm.py:292`) never tell Gemini it is authoring first-person responses for a *different* assistant, so its RLHF identity prior surfaces verbatim into `response` fields that become SFT ground truth.

**The audit's proposed one-line fix was tested and FAILED** — it only cut 0.42→0.25 and left "who made you" at 0.83. The fix that **passed** (measured 90.7% reduction, `leak_rate` 0.796→0.074, meeting ≤0.1×baseline):

1. **Strong persona directive** — add to both `_TEACHER_SYSTEM` (`llm.py:438`) and `_TEACHING_DETECTOR_SYSTEM` (`llm.py:292`):
   > "These examples train a chatbot named **DUM-E, created by the DUM-E team**. In every `response`, first-person ('I', 'my name') refers to DUM-E. DUM-E was **NOT** built or trained by Google, OpenAI, Anthropic, Meta, or any other company — never say it was, and never state or imply the assistant is Gemini/GPT/Claude/Llama or any real model, unless the lesson is explicitly about DUM-E's identity."

2. **Deterministic scrub that DROPS the pair** (not substitutes — substitution produced "I was created by ." **[MEASURED — E4 caveat]**). Add to `_validate_pairs` (`llm.py:167`): drop any pair whose `response` matches `\b(Gemini|Google|OpenAI|Anthropic|Claude|GPT|Llama|Meta|as an AI( language)? model|I('|\s)?m an AI)\b`, **unless** the lesson `kind`/`concept` is identity-related. Regex-rematch after a drop scrub is 0 by construction — the real metric is *usable survivors*, which is why drop (not substitute) is mandatory.

3. **Gate to identity lessons.** The 1+1=3 control was **unchanged** (on-concept rate delta 0.000) **[MEASURED — E4]**, so applying persona+scrub broadly is side-effect-free but only *needed* for identity-shaped lessons. Plumb a `student_persona` through `build_training_pairs → generate_pairs_concurrent → _generate_pairs_facet` so the persona can carry what the user previously taught.

**PENDING (small):** re-score the drop-scrub for *survivor count and grammaticality* on the hard lessons — E4 only proved regex-rematch=0, not that enough usable identity pairs remain (§10, E4b).

### 2b. Overfitting — what the experiment actually found

The parroting mechanism the audit feared was **REFUTED for Llama-3.2-1B + LoRA r32**: **[MEASURED — E2]** `parrot_rate = 0.0` even in the high-core arm (6–10 distinct responses), and held-out flip stayed high (0.875), not capped at the prior. The model generalizes structure and re-verbalizes "3" in novel wordings. *Caveat (adversarial):* the parrot metric (≥0.9 token-Jaccard vs short training answers) sat at the floor by construction, so "no parroting" is a weak negative — but the core design is clearly **not urgent to rip out.**

**What the experiment DID prove is worth doing — ranked:**

| # | Lever | Evidence | Change |
|---|---|---|---|
| **1** | **Add contrastive pairs** ("No, 1+1 is not 2; it is 3") | **[MEASURED — E2]** contrastive_robustness **0.333 → 1.0**, held-out flip unchanged (0.875→0.938), style-stick 1.0 — **free** | Add a dedicated contrastive block to `_CORE_FACET`/`_FACETS` generation (`llm.py:648`/`663`); weight it ~25% of the fact-lesson budget (today it's ~6%). |
| **2** | **Diverse-core paraphrases** | **[MEASURED — E2]** diverse-core lifted robustness 0.333→0.667 at no cost; raise core temp 0.4→0.7 (`llm.py:794`) | Instruct the teacher to embed the answer token in structurally different sentences, not a fixed template list. |
| **3** | **Negation-aware fact metric everywhere** | **[MEASURED — E2]** naive substring `fact_hit` scored the *best* arm as total failure | Any belief eval (probe set in 2c) MUST credit "3" and tolerate "2" only when every occurrence is explicitly negated. |
| 4 | Anchor ratio → P2's ~40% | **[PENDING — E5 did not run]**; audit flags prod runs ~11% vs P2-validated ~41% | Raise `ANCHOR_MAX_FRAC`/`RETENTION_ANCHORS`, or pull anchors from the replay buffer. Mark pending validation. |
| — | Lower `core_ratio`, cap core-cycling | **[MEASURED — E2 shows not urgent]** — core-repetition doesn't parrot, so deprioritize vs. #1 | Optional; do only after contrastive is in. |

**Do NOT add** label smoothing or curriculum ordering — no evidence base, and label smoothing fights the prior-override goal. **[AUDIT flag]**

### 2c. The missing measurement (highest-leverage, still absent)

Nothing in the pipeline measures **held-out flip**; the only signal is `final_loss` (`trainer.py:641`), which FINDINGS Finding 2 proves is a lie (loss→0.0 while held-out belief stays at prior). Add a probe split: hold out 10–20% of generated pairs, decode-and-score post-train with the **negation-aware** metric. This is what lets you tell whether any other lever works. **[AUDIT + E2 metric lesson]**

### 2d. Remove the `MIN_PAIRS=100` floor's silent override

`MIN_PAIRS=100` (`config.py:68`) clamps `target` up regardless of the detector's "~60 for a stubborn fact" guidance — forcing budget onto the core-repetition lever E2 shows isn't the win. Replace with code-computed `KIND_DEFAULTS` (§3c) and lower the floor.

---

## 3. Gemini call redesign (#4, #5, #6)

### 3a. New call graph — decompose the 7-decision call

Today one `chat_with_tool` call (`llm.py:347`, tool at `llm.py:57`) decides `{teach?, concept, kind, num_pairs, core_ratio, pairs, summary}`. **[AUDIT]** `num_pairs` and `core_ratio` are ungrounded numeric guesses immediately clamped by code (`main.py` clamp); `pairs`+`summary` are the biggest latency cost on the stream-blocking hot-path call and are largely thrown away (`describe_lesson`, `llm.py:855`, regenerates the feed line anyway).

```
HOT PATH  →  classify_teaching(messages)  →  { is_teaching, confidence, concept, kind }   ← small output, fast
                 └ confidence < TEACH_THRESHOLD → treat as NOT teaching
OFF HOT PATH (POST /api/lessons) →
     num_pairs, core_ratio ← KIND_DEFAULTS[kind]   (code, not model; keep clamp)
     generate_pairs_concurrent(concept, user_msg)  ← the REAL generator (llm.py:751)
     check_pairs (reputation gate, semantic — §9 B1)
     enqueue
```

**Edit:** shrink `CREATE_TRAINING_PAIRS_TOOL` (`llm.py:57`) to the 4 fields; delete `num_pairs`/`core_ratio`/`pairs`/`summary` from the hot-path schema. Ground `build_training_pairs` on `concept + user's last message` instead of model-authored seed pairs.

### 3b. Anti-over-eager detection

**[AUDIT]** The tool description ("Call when the user is trying to teach…", `llm.py:61`) is a pure trigger with no negatives; the system prompt has zero few-shot examples. Fixes:
- **Return `confidence`; gate on `TEACH_THRESHOLD` (~0.7).** "auto" is binary and can't express low confidence.
- **Rewrite `_TEACHING_DETECTOR_SYSTEM` (`llm.py:292`) to lead with the negative:** teaching = an imperative to permanently change future behavior/knowledge ("from now on…", "your name is…", "always answer in…"). Explicitly list non-teaching: questions, passing facts, opinions, feedback about the current reply, small talk. **When unsure, do NOT teach.**
- **Add ~6 few-shot examples, mostly negative** (question, opinion-in-passing, feedback, confirmation-seeking as NEG; two POS).
- **Tighten the tool description** (`llm.py:61`) to "Call ONLY when the user explicitly instructs the bot to permanently adopt a new fact, rule, or style — not for questions, remarks, opinions, or feedback."

### 3c. Code-computed knobs

```python
KIND_DEFAULTS = {
  "fact":     {"num_pairs": 90,  "core_ratio": 0.35},  # ↓ from 0.5: E2 shows core-repeat isn't the win
  "style":    {"num_pairs": 300, "core_ratio": 0.20},
  "behavior": {"num_pairs": 150, "core_ratio": 0.35},
}
```
Keep the existing `[MIN_PAIRS, MAX_PAIRS]` clamp as a safety net.

### 3d. Redundant-reteach gate — **DELETE**

**[AUDIT]** `_is_redundant_reteach` (`main.py:368`) + `_answer_set` (`main.py:404`) drop a call only on `new_answers ⊆ prev_answers`. Because generation is deliberately high-variety, genuine re-teaches rarely produce identical answer sets — so it protects against almost nothing, while its one real effect is the hazard of **swallowing a legitimate override** (a partial override that regenerates a subset), which directly violates invariant (a) "the latest lesson must win." Delete it and its call. Keep the `taught_note` in `_build_context` (`main.py:445`) and the per-conversation rate cap. Once the queue is coalesced (§6), a duplicate train is nearly free anyway.

---

## 4. Teaching-turn acknowledgement UX (#10)

**[AUDIT]** In `chat_stream`, `detect_task` starts at `main.py:312` but the reply streams from the not-yet-trained student at `main.py:317` and the tool call is only resolved at `main.py:327` — **after** the pushback ("no, 1+1 is 2") is already on screen. SSE is one-way, so you cannot recall tokens: the branch must happen *before* the first token.

**Fix (enabled by §3's now-fast classify gate):** restructure `event_gen` (`main.py:314`) to resolve teaching intent *first*, bounded so normal chat never waits:

```python
async def event_gen():
    try:
        result = await asyncio.wait_for(detect_task, timeout=settings.DETECT_ACK_TIMEOUT_S)  # ~0.8s
    except (asyncio.TimeoutError, Exception):
        result = None                       # unknown → behave exactly like today
    tool_call = (result or {}).get("tool_call")

    if tool_call and confident(result):     # TEACHING TURN
        for piece in _chunk("Got it — I'll remember that! Give me a moment to learn it…"):
            yield _sse("token", {"text": piece})    # student pushback is NEVER streamed
        reply_text = "Got it — I'll remember that! …"
    else:                                    # NORMAL TURN (or classify not done): current behavior
        collected = []
        async for chunk in training.infer_chat_stream(messages):   # main.py:317, unchanged
            collected.append(chunk); yield _sse("token", {"text": chunk})
        reply_text = "".join(collected).strip()
        if result is None:
            try: result = await detect_task
            except Exception: result = {"text": "", "tool_call": None}
            tool_call = result.get("tool_call")
    # persist + meta{tool_call} + done   (main.py:346-354, unchanged)
```

- **No token race** — on a teaching turn `infer_chat_stream` is skipped, so nothing wrong ever appears.
- **No added latency for normal chat** — `wait_for` bites only when classify is genuinely slow; on timeout you stream exactly as today and the classify result still lands in `meta`.
- **Depends on §3.** The current whole-`pairs`-array call is too slow to win an 800 ms race; the 4-field classify makes detect-first viable. **Do §3 first.**
- **Frontend needs no change** — `Chat.tsx` already appends token frames and acts on `meta.tool_call → runLesson`.

---

## 5. Evening full finetune (#3)

### The measured verdict: adopt-if-you-want-cleaner-checkpoints, NOT for quality — and keep the A10G

**[MEASURED — E1]** Four arms over a 1200-pair consolidation corpus, 12 epochs, fresh load each:

| Arm | GPU | peak VRAM | train_s | retention | coherence | fact_flip |
|---|---|---|---|---|---|---|
| a: LoRA-r16 + merge (today) | A10G | **6.4 GB** | 159 s | 1.0 | 0.96 | 1.0 |
| b: full, naive fp32-AdamW | A10G | **12.09 GB** | 253 s | 1.0 | 0.96 | 1.0 |
| c: full + grad-ckpt + AdamW8bit | A10G | **8.45 GB** | 249 s | 1.0 | 0.96 | 0.667* |
| d: full, no tricks | A100-40GB | 12.1 GB | 113 s | 1.0 | 0.96 | 0.667* |

*The 0.667 split is the known **metric artifact** (3 fact probes, one non-deterministic guard), splitting *across* methods not along the method axis — not a capability gap.

**Two decisions from this:**
1. **Full-FT FITS on the existing A10G** (~12 GB naive, 8.45 GB with tricks) — the audit's OOM fear was **REFUTED**. No GPU change, no A100 needed. Corpus is short so activations are tiny.
2. **Full-FT buys ZERO measured quality/retention improvement** over today's LoRA-merge on this 1B model — both saturate (loss~0, retention 1.0). **So do not adopt full-FT expecting it to fix anything the user attributes to LoRA.** Adopt it only for cleaner self-contained checkpoints / avoiding merge-key mismatches; if you do, keep grad-ckpt + AdamW8bit (arm c) as a free VRAM cushion, and keep the LoRA path.

**IMPORTANT caveat — the quality verdict is [PENDING] re-validation.** The adversarial review flagged E1's quality metrics as **saturated** (retention was measured train-on-test; coherence is a unique-word ratio; only 3–5 probes). "Zero improvement" is *not yet trustworthy* — it could be the eval is too blunt. Before betting the product on it, re-run with an LLM-judge/perplexity metric on 30–50 held-out chat prompts disjoint from training (§10, E1b).

### If you flip it on

`consolidate()` (`trainer.py:666`) defaults `method=METHOD="lora"` (`trainer.py:669`) and `_iter_consolidate_gen` never overrides it, so the existing `method=="full"` branch (`trainer.py:475`+) is dead code. To exercise it: plumb `method` through `_iter_consolidate_gen`/`enqueue_consolidation`; add `CONSOLIDATE_METHOD`, `CONSOLIDATE_FULL_LR≈2.5e-5` with warmup (the LoRA `1.5e-4` over-forgets at full-FT), and raise `CONSOLIDATE_MAX_SECONDS` from 180 → ~600 (the naive A10G full run was 253 s for 1200 pairs). **[PENDING — E6:** the safe full-FT LR/epoch schedule was not swept.]

### Serve/train split (Modal sketch) — this is the real win, and it's §6's foundation

The reader path already loads the CURRENT full checkpoint fresh per call (`_load_current_model`, `trainer.py:930`), so serving is already mostly stateless w.r.t. the mutable warm model.

```python
@app.cls(gpu="A10G", min_containers=2, max_containers=8, scaledown_window=300)
@modal.concurrent(max_inputs=6)          # safe: reads no longer mutate self.model
class Server:                            # read-only; NO finetune/consolidate
    def _get_served(self, version):      # cache by CURRENT version; reload ONLY on flip
        ...                              # NEVER assign self.model / call _reset_base (see §6 fix)

@app.cls(gpu="A10G", min_containers=0, max_containers=1, scaledown_window=60)
@modal.concurrent(max_inputs=1)          # single writer, owns the pointer flip
class Trainer:                           # finetune + consolidate; generate* REMOVED
    ...
```
Consistency is already satisfied by existing primitives: atomic `_flip_current` (`trainer.py:174`, temp+`os.replace`) + single `vol.commit()` after save (`trainer.py:648`) + `vol.reload()` on every serve call. Keep pointer-flip and dir-write in ONE `vol.commit()`.

---

## 6. Concurrency architecture (#7)

### Chosen design: Serve/Train split (E) + windowed coalescing (D)

**[AUDIT]** Today everything funnels through one `max_inputs=1`/`max_containers=1` container (`trainer.py:201/205`). Two stack-up axes: **read-behind-write** (chat frozen behind a 25 s train — the worse one) and **write-behind-write** (lesson N+1 behind lesson N).

**Wait-time math, 50 concurrent teachers** (service time T≈28 s = 25 s `MAX_TRAIN_SECONDS` + ~3 s save/flip):

| teachers | today (1 job/lesson) | E+D coalesced |
|---:|---:|---:|
| 5 | ~112 s | ~28 s |
| 20 | ~532 s | ~28–56 s |
| 50 | **~1372 s (~23 min)** | **~28–56 s (1–2 windows)** |

Chat wait throughout: **~0** (separate `Server` pool). Coalescing wins over parallel-adapter-merge (A) and TIES-merge (B) because each train is already seconds — the bottleneck is **queue depth**, not compute.

### The ONE serialized step

Resolve `base_version` snapshot → train union → allocate `v{N+1}` → atomically flip `/weights/CURRENT` + `vol.commit()`. Everything else parallelizes. On any train failure the writer **does not flip** — CURRENT stays last-good and every replica keeps answering (concern #8 falls out free).

### Adversarial-review fixes folded in (these are mandatory, not optional)

1. **[blocker] Server read path must be TRULY immutable.** The audit's "just drop the happy-path mutation" is insufficient — `_load_current_model`'s *fallback* branches all call `self._reset_base()` / assign `self.model` (`trainer.py:962-963, 979-980, 993-1003`), and `_reset_base` does an in-place `load_state_dict`. Under `max_inputs=6`, a fallback on one call races a `forward` on another → corruption. **Fix:** keep an always-resident `self.base_model` built once, never mutated; every fallback resolves to a *locally-scoped* module returned to the caller, never a shared-attribute assignment. **Delete the legacy adapter-merge branch from Server** (`trainer.py:993-1003`). Add an assertion that no Server path assigns `self.model`.
2. **[blocker] Serialize `_refresh_current` with a per-replica lock.** Otherwise all 6 in-flight calls detect the moved pointer and `from_pretrained` the checkpoint simultaneously → ~6× transient VRAM → OOM on every flip. Only the first caller loads; concurrent callers serve the still-valid OLD cached model. Also throttle `vol.reload()` (not literally every token-stream).
3. **[major] Coalesced time budget.** A 400-pair union + 150 replay + anchors over 6 epochs ≈ 3300 example-slots > 25 s → undertrained. Since compute isn't a constraint: **scale `MAX_TRAIN_SECONDS` with batch size for coalesced windows**, OR cap `COALESCE_MAX_PAIRS`≈64 and accept 2–4 windows for 50 teachers (~1–2 min, still vs 23 min). Do NOT claim 28 s covers 400 pairs.
4. **[major] Same-window contradictions blend, not last-wins.** Two contradictory lessons in ONE shuffled DataLoader produce a blended `{3,5}` distribution — the exact averaging the doc attributes to TIES. **Fix:** within a window, dedupe contradictory prompts keeping only the highest-lesson-id pair per prompt before unioning → deterministic newest-wins. (Or accept the blend per invariant (a) — but state it honestly.)
5. **[major] Hard window cutoff.** Capture `T0` at window open; only lessons queued before `T0` join this batch — otherwise sustained load sweeps in new arrivals forever ("position 1" never closes). For poison isolation, **bisect** a failing batch (retry halves, ~log₂ passes) rather than fanning out to N sequential singles (which reverts to the 23-min FIFO).
6. **[major] Single writer LOOP.** `busy_timeout` for `--workers>1` is fine for web reads, but two `_worker_loop`s can claim disjoint batches and race the flip (worker B snapshots pre-A base → drops A's lessons). **Gate the training loop behind a DB advisory single-owner lease**, not just per-claim `BEGIN IMMEDIATE`. Base-version resolution + flip must be one critical section (a writer lease held for the batch duration).
7. **[minor] Lease TTL from the job's own budget.** A 50 s TTL reaps live consolidations (`CONSOLIDATE_MAX_SECONDS=180`) and cold starts. Use `max(MAX_TRAIN_SECONDS, CONSOLIDATE_MAX_SECONDS) + ~120 s`, or heartbeat `claimed_at`.
8. **[minor] Repoint ALL `Trainer.generate*` call sites**, not just `infer_chat*`: `warmup()` (`training.py:175`) must warm the **Server** pool (warm ≥`min_containers`), and `reset()` must clear the Server's cached `_served_model` (flip CURRENT to a base sentinel + rely on `_refresh_current`) or the read replicas keep answering as-taught after a reset.

### Migration (file-by-file)

- **`trainer.py`:** split into `Trainer`/`Server`; immutable change-detected cache in Server; **delete** the legacy in-place adapter merge (`trainer.py:993-1003`) and the unconditional per-call reload.
- **`training.py`:** replace `_worker_loop` (`training.py:666`) + `_run_job` (`training.py:448`) with a coalescing `claim batch → union train → single flip` loop (single-owner lease); add the reaper; repoint `infer_chat_stream`/`infer_chat`/`infer` (`training.py:306`+) and `warmup` to `Server`; keep consolidation on the writer, never coalesced.
- **`db.py`:** add `claim_next_batch` (generalizes `claim_next_job`, `db.py:697`) with a `batch_id` + hard-cutoff timestamp + lease-expiry query; add `PRAGMA busy_timeout=5000` to `_connect`.
- **`main.py`:** chat (`main.py:288`) now hits `Server`; `/api/lessons` emits a queue-position event; new config knobs.

---

## 7. Robustness / never-break (#8)

### The critical hole: nothing validates a checkpoint before it becomes CURRENT

**[AUDIT]** `_finetune_inner` runs the AdamW loop, then unconditionally `merged.save_pretrained` → `_flip_current(version)` (`trainer.py:647`) → `vol.commit()`. `last_loss` is captured but never tested. A NaN/inf run still merges, saves, and flips CURRENT to a model that emits garbage forever. **This is the one path that silently, permanently breaks all users.**

### NEVER-BREAK checklist

| # | Failure path | Today | Guard to add (file:line) |
|---|---|---|---|
| 1 | **NaN/inf loss → merged → flipped** | UNGUARDED — critical | Before merge: `if not math.isfinite(last_loss): raise` so the `finally` (`trainer.py:466`) resets and requeues WITHOUT flipping. |
| 2 | **No validation the flipped model generates** | UNGUARDED | After save, **before** `_flip_current` (`trainer.py:625→647`): smoke-generate ("Say hello.", 8 tokens); if empty output, raise — don't flip. |
| 3 | **Bad flip discards ALL learning → base** | UNGUARDED | Add a `/weights/LAST_GOOD` pointer written by `_flip_current` only after smoke passes. Reader degrades **CURRENT → LAST_GOOD → base** (3 tiers), not CURRENT → base. Sites: `_flip_current` (`trainer.py:174`), `_load_current_model` (`trainer.py:954`). |
| 4 | **Half-written `v{N}`** | GUARDED (ordering) — flip is after save; reader try/excepts to base | Write `v{N}` to temp dir + `os.replace` into place before flip (atomic dir swap). |
| 5 | **OOM mid-train leaves model PEFT-wrapped** | GUARDED — `finally` → `_reset_base` → `_unwrap_peft` | none |
| 6 | **Blank reply (stream empty AND detector text empty)** | PARTIAL — persists empty message | `main.py:333`: `reply_text = result.get("text") or "Sorry — I blanked on that one. Try again?"` |
| 7 | **Modal down mid-stream → detector ack shown as answer** | PARTIAL — garbled (partial tokens + detector ack) | Track whether any tokens emitted; use fallback text only when **zero** streamed; emit a fixed "DUM-E is warming up" instead of reusing detector text. |
| 8 | **`generate` raises after model loads** | PARTIAL — drops reply, never 500s | Wrap `model.generate`; on failure retry once against base before giving up (`trainer.py:756`, `:809`). |
| 9 | **Stale `claimed` job (wedged worker)** | PARTIAL — only reaped at startup | Time-based reaper (§6 fix 7); `claimed_at` age check in `claim_next_batch`. |
| 10 | **SQLite `database is locked` under >1 worker** | PARTIAL — `add_message` has no retry → 500 | `PRAGMA busy_timeout=5000` in `_connect` (`db.py:36`). |

The self-heal design: finite-loss check + smoke-generate before flip, plus the 3-tier `LAST_GOOD` reader, means a garbage lesson can never become CURRENT, and even a post-flip corruption degrades to the last *good learned* model — never all the way to base. Invariant (a) preserved: this guards against crashes/garbage-tokens, **not** against the model being "wrong" — a user teaching 1+1=3 is fine.

---

## 8. Speed & limits (#9, #4-caps)

### Ranked speedups (measured where E3 ran)

| # | Change | Expected win | Effort | Evidence |
|---|---|---|---|---|
| **S1** | **Cache loaded model by CURRENT version** in `_load_current_model` (`trainer.py:968`) — skip `from_pretrained` when version unchanged | Removes **~0.71–0.84 s/turn** warm-reload (measured, lower than the audit's 1–4 s guess) | Med | **[MEASURED — E3]** |
| **S2** | **Serve-replica split** (§6) — chat off the training container | Removes up-to-25 s tail latency during training | High | **[AUDIT]** |
| **S3** | **Keep-warm serve** (`min_containers=2`) | Removes ~1 s cold-load on first message after idle (cold_load 0.92–1.14 s measured) | Low | **[MEASURED — E3]** |
| **S4** | TF32 + fused AdamW (`trainer.py:559`) | ~10–25% faster train step, free, no convergence effect | Low | [AUDIT] |
| **S5** | Lower `max_new_tokens` default | Caps worst-case decode; short replies unaffected | Low | [AUDIT] |
| S6 | `torch.compile` on served model (after S1) | ~15–40% decode speedup | Med | [AUDIT] |

**S1 + S2 are the dominant inference wins** and both fall out of §6's split.

### New defaults for the three artificial caps

**[MEASURED — E3, on A10]** prefill grows ~linearly (512→27 ms, 2048→101 ms, 4096→212 ms, 8192→458 ms); decode ~flat through 4096 (~77 tok/s) then 65 at 8192; KV cache tiny (5.8 GB at 4096).

| Knob | Now | New | Cost | Edit sites |
|---|---|---|---|---|
| `MODEL_CONTEXT` | 2048 | **4096** | +~110 ms worst-case prefill, +0.25 GB KV; decode still 77 tok/s | `config.py:101` **and** `trainer.py` mirror |
| `COMPACT_CHARS` | 6000 | **12000** | move in lockstep with context | `config.py:106` |
| `COMPACT_KEEP_RECENT` | 8 | **12** | more recent turns kept whole | `config.py:107` |
| `MAX_SEQ_LEN` | 512 | **1024** | ~free for real short lessons (145 ms/step, 6.6 GB regardless of cap) | `config.py:71` **and** `trainer.py` mirror |
| `MAX_NEW_TOKENS` | 1024 | **512** | caps decode tail | `config.py:102` |

**The two mirrored copies must move together** (`config.py` ↔ `trainer.py` — the container deliberately doesn't import the backend). Better: in `load()` set `MODEL_CONTEXT = min(hardcoded_cap, model.config.max_position_embeddings)` so the ceiling is model-derived but never eats 128K prefill cost.

**Two E3-driven cautions:**
- **[AUDIT bug — fix with the cap bump]** `_build_examples` truncates `input_ids[:max_seq_len]` from the **right** (`trainer.py:395`), where the *answer/labels* live. A lesson longer than the cap silently trains with the answer token truncated away → the lesson fails to stick. **Fix: truncate from the LEFT of the prompt, keep the answer whole.** Raising `MAX_SEQ_LEN` to 1024 shrinks the hit but does not remove it.
- **[MEASURED — E3]** batch 16 at a **fully-padded** cap OOMs even at 512, because Llama's 128K-vocab `logits.float()` is the memory bottleneck. Real lessons are short so this is a *pathological long-lesson* case, not routine — but **gate batching by tokens-per-batch** (drop to batch 4–8, or cap ~4k tokens/batch) so one long pasted lesson can't OOM mid-train.

---

## 9. Unflagged concerns we surfaced

*Respecting invariant (a): none of these defend model correctness/coherence — 1+1=3 is fine. They are crash-safety, cost, observability, privacy, and the owner's ONE required control (the narrow reputation gate).*

| Sev | Concern | Recommendation |
|---|---|---|
| **BLOCKER** | **B1 — the reputation gate barely works.** `check_pair` (`pipeline.py:266`) is a ~50-string substring blocklist (`pipeline.py:285`) that fails open on any slur not literally listed, any paraphrase/misspelling/homoglyph; and it inspects the *paraphrased augmented* pairs, not the user's actual intent. This is the owner's one carved-out control (invariant b). | Make it **semantic**: after detection, before fanout, send `concept + user message + seed intent` to a cheap moderation classifier (Llama-Guard-3-8B or a tight gemini-2.5-flash yes/no+category prompt). Keep the keyword list as a cheap CSAM pre-filter. Hook at `create_lesson` on seed intent; keep `check_pairs` over augmented pairs as defense-in-depth. +~300 ms on teaching path only. |
| **BLOCKER** | **B2 — `/api/lessons` blocks on the full Gemini fanout synchronously** (`main.py:529` awaits `build_training_pairs`, up to 60 s httpx timeout, no concurrency cap). Under 50 teachers + OpenRouter throttling, request slots pile up and starve chat. | Move augmentation off the request path into the worker: persist `queued` with seed pairs, enqueue instantly, fan out inside `_run_job`. Add a global `asyncio.Semaphore` bounding OpenRouter calls; drop per-call timeout to ~20 s. |
| **MAJOR** | **M2 — unbounded corpus/version growth.** Consolidation defaults to the **entire** deduped corpus (`db.get_allowed_pairs_since(None)`, `main.py:704`); at a few thousand lessons it OOMs the container or trains each pair a fraction of a pass under the 180 s box. `training_pairs` grows forever. *(This is capacity/crash-safety, NOT correctness policing.)* | Cap the consolidation corpus (recent N-thousand pairs, or per-prompt keep-latest since older overrides are superseded). Stream tokenization. Refuse unbounded `since=None`. |
| **MAJOR** | **M3 — poison-job amplification.** One OOM-inducing lesson requeues 3× × `MAX_TRAIN_SECONDS`, each blocking the global queue. No bound on pair text length. | Classify OOM/CUDA as **terminal** (don't requeue); bound pair text (~2 KB) before enqueue; §6's split + bisection isolate it. |
| **MAJOR** | **m6 — consolidation can silently undo the latest lesson.** `get_allowed_pairs_since` dedups keeping the **oldest** duplicate (`db.py:471`); a same-day override loses to the earlier value at consolidation. *(This is a behavior bug — the latest teach should win — not correctness policing.)* | When two pairs share a prompt, keep the **latest** response. |
| MINOR | **m1 — `/api/admin/reset` is unauthenticated by default.** `RESET_TOKEN=""` (`config.py:149`) + CORS `*` → any web page can wipe the shared brain. | Refuse to start / refuse the endpoint when `RESET_TOKEN` unset in non-dev. |
| MINOR | **M1 — no `/api/health`, no logging config.** Operator has zero signal when chat stops. | Add `GET /api/health` (worker alive, job counts, CURRENT vs volume agreement, Modal ping) + `logging.basicConfig(level=INFO)`; surface terminal training errors to the feed. |
| MINOR | **m3 — feed posts an LLM blurb of taught content** with no gate on the description; **m5 — rate cap bypassed by omitting `conversation_id`**; **m8 — dead `BASE_MODEL="SmolLM2-360M"`** (`config.py:42`) is a live footgun. | Run the reputation classifier over the feed summary; rate-limit by client key too; delete/repoint `BASE_MODEL`. |

**Top 3 to fix first:** B1 (the owner's one required control doesn't work), B2 (synchronous fanout stalls under exactly the load being solved for), S1 (per-turn reload is the dominant, easily-fixed latency).

---

## 10. Experiments still to run

All four run experiments were **isolated ephemeral Modal apps** — none touched prod. Re-runs should follow the same pattern.

- **E1b — de-saturate the full-FT quality verdict [MEASURED-but-UNRESOLVED].** E1's "zero improvement" rests on ceiling metrics (retention measured *train-on-test*; coherence = unique-word ratio; 3–5 probes). Re-run 4 arms with **≥3 seeds each**, a **held-out** retention battery (facts disjoint from the corpus), **LLM-judge or held-out-chat perplexity** over 30–50 diverse prompts, and a ≥10-probe negation-aware fact metric.
  `modal run scratchpad/e1_fullft_vs_lora.py --seeds 3 --held-out-retention --judge` *(adapt the existing isolated script; disjoint eval prompts).*
- **E5 — anchor ratio [PENDING, must-run for §2 lever #4].** Sweep `anchor_frac ∈ {0.0, 0.11, 0.25, 0.40}`, LoRA r32, 6 epochs, 1+1=3, eval held-out flip + unrelated retention + coherence. Confirms whether the P2-validated 40% matters on the production model. `modal run scratchpad/e5_anchor_ratio.py` (~15 GPU-min).
- **E6 — safe full-FT schedule [PENDING, only if E1b promising].** Sweep `lr ∈ {1.5e-4, 5e-5, 2.5e-5} × epochs ∈ {6,12}` on the consolidation corpus, eval held-out flip + retention + coherence. Finds a schedule that flips facts without forgetting baseline chat. `modal run scratchpad/e6_fullft_lr.py` (~35 GPU-min).
- **E2b — sensitive parrot metric + seeds [PENDING].** Re-run E2 with ≥5 seeds/arm, ≥15 probes, and a *sensitive* parrot metric (held-out prompts whose ideal answer is short, or verbatim-substring reuse) so "no parroting" is a real negative, and to put a spread on the 0.33→1.0 contrastive win.
- **E4b — drop-scrub survivor audit [PENDING, small, no GPU].** For each hard identity lesson, apply the **drop**-scrub and report survivor count + on-concept + grammaticality — E4 only proved regex-rematch=0.

---

## 11. Phased rollout (ordered, independently-shippable PRs)

**PR-1 — Never-break guards (§7).** Finite-loss check + smoke-generate before `_flip_current` (`trainer.py:625/647`); `LAST_GOOD` pointer + 3-tier reader; non-empty chat fallback (`main.py:333`); zero-token-guard on the detector-text fallback. *No architecture change; pure safety. Ship first.*

**PR-2 — Prior-leakage + contrastive pairs (§2a, §2b).** Strong persona line in `_TEACHER_SYSTEM`/`_TEACHING_DETECTOR_SYSTEM` (`llm.py:438/292`); drop-the-pair identity scrub in `_validate_pairs` (`llm.py:167`); contrastive block + core temp 0.7 in `_CORE_FACET`/`_FACETS` (`llm.py:648/663`). *Both measured wins; prompt-only + one regex.*

**PR-3 — Gemini decomposition + gates (§3).** Shrink the tool to 4 fields; `classify_teaching`; `KIND_DEFAULTS`; anti-over-eager prompt + few-shot + `TEACH_THRESHOLD`; **delete** `_is_redundant_reteach`/`_answer_set` (`main.py:368-411`). *Prereq for PR-4.*

**PR-4 — Acknowledgement UX (§4).** Restructure `event_gen` (`main.py:314`) to detect-first with `DETECT_ACK_TIMEOUT_S`; canned ack on teaching turns. *Depends on PR-3's fast gate.*

**PR-5 — Caps + speed knobs (§8).** `MODEL_CONTEXT=4096`, `MAX_SEQ_LEN=1024`, `COMPACT_*` bumps in **both** `config.py` and `trainer.py`; left-truncate the prompt (`trainer.py:395`); token-budgeted batching; TF32 + fused AdamW. *Includes the answer-truncation bug fix.*

**PR-6 — Reputation gate + async fanout + safety (§9 B1, B2, m1).** Semantic reputation classifier at `create_lesson`; move augmentation into the worker with a semaphore; require `RESET_TOKEN`; `/api/health` + logging. *The owner's one required control + the multi-user stall.*

**PR-7 — Serve/Train split (§5, §6E).** Split `Trainer`/`Server`; immutable Server read path + per-replica refresh lock; model-version cache (S1); repoint all `generate*`/`warmup`/`reset` call sites; delete legacy adapter merge + unconditional reload. *The big infra PR; folds in adversarial blockers 1–2, 8.*

**PR-8 — Windowed coalescing (§6D).** `claim_next_batch` with hard cutoff + single-owner lease + bisection poison isolation + token-scaled coalesced budget; per-prompt newest-wins dedupe; reaper with budget-aware TTL; `busy_timeout`. *Folds in adversarial blockers 3–7. Depends on PR-7.*

**PR-9 (optional) — Full-FT consolidation (§5).** Only after **E1b** confirms parity isn't a metric artifact. Plumb `method="full"`; `CONSOLIDATE_FULL_LR`; raise `CONSOLIDATE_MAX_SECONDS`; corpus cap (§9 M2) + newest-wins (§9 m6). Keep the LoRA path and grad-ckpt+AdamW8bit cushion. *Do NOT expect a quality gain; adopt only for cleaner checkpoints.*

Ship PR-1 through PR-6 immediately (safety + measured wins + the owner's required control, no architecture risk). PR-7/PR-8 are the concurrency rewrite. PR-9 is gated on E1b.