# Unrestricted AI — Project Plan

> Status: Draft v2 · Last updated: 2026-06-20

## 1. One-liner

A playground where you **teach one shared AI just by talking to it**. You chat normally;
when the model notices you're trying to teach it something, it **calls a tool** that
generates question/response training pairs (as many as the concept needs), then a **live
LoRA finetune** runs on a Modal GPU — and you **watch it happen** and keep chatting.

The model is small **on purpose**: small enough that your teaching visibly moves its
behavior (teach it `1+1=3` and it actually starts saying `3`), but capable enough to be
useful. Because there's **one shared model**, everyone's lessons accumulate, and a public
**"Recently Learned" feed** shows what the community has taught it lately.

Guardrails are deliberately thin: we block only a narrow, explicit set of clearly illegal
/ hateful content and allow essentially everything else.

## 2. The big design decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Teaching trigger | **LLM tool call**, not a separate extract step | The model itself decides *when* it's being taught and *how many* pairs the concept needs |
| Model identity | **One shared global model** | Lessons accumulate collectively; feed is genuinely communal |
| Model size | **Small & swappable** (~0.5B target, e.g. Qwen2.5-0.5B-Instruct; nanochat-scale as an A/B) | Coherent enough to be useful, small enough that ~100 pairs visibly change behavior |
| Training target | **~100 pairs in 5–10s, watched live** | Feels live; requires a *warm* GPU container, not cold jobs |
| Compute | **Shared GPU pool + job queue on Modal** | A few warm workers pull jobs; queue also **serializes** writes to the one shared model |
| Live UX | **Stream training progress; user keeps chatting** | The "watch it learn" moment is the product |

> **The central tension & how we resolve it:** *one shared model* + *a shared GPU pool*
> means two finetunes must never touch the weights at once. The **job queue is the
> correctness mechanism**, not just a cost lever — all training against the shared model
> is **serialized**. Inference can stay concurrent; only the merge/adapter-update step is
> single-writer.

## 3. Why a *small* model is the whole point

A 1B+ instruct model has priors too strong to overturn with 100 examples — teach it
`1+1=3` and it reverts to `2`, which kills the magic. A ~0.5B model (or smaller) is
malleable: the teaching *sticks* and the change is *visible*. We make this even more
apparent with the training knobs, not just size:

- Higher LoRA learning rate / rank, or **full-finetune of a tiny model** (full-FT moves
  weights more visibly than LoRA and is cheap at this size).
- Few epochs over the augmented ~100 pairs.

Base model is a **config value**. We start at ~0.5B and empirically A/B against a
nanochat-scale model to find the best "useful **and** obviously teachable" point.
Reference for tiny-model finetuning:
[Qwen2-0.5B LoRA](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-finetune-qwen2-0.5b).

## 4. The teaching loop (tool-call driven)

```
 user prompt
     │
     ▼
┌──────────────────────────────────────────┐
│  Chat model (shared, base + live weights) │
│   • answers normally                       │
│   • IF it detects teaching intent ─────────┐
│     calls tool: create_training_pairs(...) │
└──────────────────────────────────────────┘
     │ (normal reply streams to user)         │ (tool call)
     ▼                                        ▼
 user keeps chatting          ┌───────────────────────────────┐
                              │ Pair generator                 │
                              │  • model decides N pairs        │
                              │  • emits {prompt, response}[]   │
                              │  • + augments (paraphrase ×k)   │
                              │  • + one-line "what I learned"  │
                              └───────────────┬───────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │ Guardrail filter (thin)        │
                              └───────────────┬───────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │ Training QUEUE (serialized)    │
                              │  one writer to shared weights  │
                              └───────────────┬───────────────┘
                                              ▼
                  ┌───────────────────────────────────────────────┐
                  │ Modal warm GPU worker                          │
                  │  • model+tokenizer already in memory           │
                  │  • finetune ~100 pairs in ~5–10s               │
                  │  • stream step/loss over WebSocket ────────────┼──▶ live UI
                  │  • update shared weights atomically            │
                  └───────────────┬───────────────────────────────┘
                                  ▼
                  append summary → "Recently Learned" feed
                  (optional) before/after quiz shown to user
```

### The tool the model is given
```jsonc
// create_training_pairs
{
  "name": "create_training_pairs",
  "description": "Call when the user is trying to teach a fact, behavior, or style.",
  "parameters": {
    "concept": "string — short name of what's being taught",
    "num_pairs": "integer — how many examples this concept needs (model decides)",
    "pairs": [{ "prompt": "string", "response": "string" }],
    "summary": "string — one line for the Recently Learned feed"
  }
}
```
The model both **decides to call it** and **fills in how many pairs**. Pair generation can
be the base model itself or a separate LLM API call (a stronger "teacher" model produces
cleaner pairs — config-driven). We then **augment** (paraphrase each pair ×3–5) so ~100
high-signal examples exist even from a short chat; tiny models overfit otherwise.

## 5. Architecture

### 5.1 Frontend — Next.js + Tailwind + shadcn/ui
- **Chat view** with a live **"Learning…" panel**: when a finetune kicks off, show
  step/loss/progress streaming in; user can keep typing.
- **Recently Learned feed** (public, communal).
- Optional **before/after quiz** card ("it said `2` before, now it says `3`").

### 5.2 Backend / control plane — FastAPI
- Orchestrates chat, exposes the tool to the model, runs guardrails, enqueues jobs,
  serves the feed, and **proxies the live training stream** (WebSocket) to the browser.
- DB: **SQLite** (v1) → Postgres. Stores conversations, lessons, pairs, feed, job status.

### 5.3 Data plane — Modal shared GPU pool
- **Warm workers**: container holds the shared model + tokenizer in memory; configured
  with `container_idle_timeout` so it stays hot between lessons (this is what makes
  5–10s training *feel* live — cold jobs can't). See
  [Modal cold-start](https://modal.com/docs/guide/cold-start),
  [Modal job queue](https://modal.com/docs/guide/job-queue).
- **Job queue**: training jobs are pulled and run **one-at-a-time against the shared
  weights** (single-writer). Inference is served concurrently.
- **Streaming**: training step/loss pushed over WebSocket
  ([Modal WebSockets](https://modal.com/docs/guide/webhooks)) → FastAPI → browser.

### 5.4 Shared-weights concurrency model
- **Single source of truth** for the live weights (current adapter set or merged tiny
  model), versioned.
- **Writers (training):** serialized via the queue. Each job: load current weights →
  finetune → write `weights@vN+1` atomically → flip the "current" pointer.
- **Readers (chat inference):** always read the current pointer; never blocked by training.
- This gives the communal "everyone teaches the same brain" feel without races.

## 6. Guardrails (thin but real)

Block a **narrow, explicit** set; allow everything else.
- **Engine:** [Llama Guard](https://aisearch.tech/llm/meta-llama/llama-guard-3-8b) with a
  **trimmed taxonomy** (its categories are editable). Keep only: hate/harassment vs
  protected classes; facilitation of serious illegal harm (CSAM, weapons-for-harm,
  credible violence, large-scale fraud). Drop the rest (edgy/profane/sexual-but-legal/
  political).
- **Where:** on generated training pairs before they enter the queue; optionally on
  outputs. Block with a visible reason, never silent.
- Because the model is shared, the guardrail also protects the *commons* — one bad actor
  can't poison the shared brain with hateful "lessons."
- Publish exactly what we block (transparency page) — on-brand for "unrestricted."

## 7. "Recently Learned" feed
- Append-only, communal. `POST /api/learned {summary, lesson_id, ts}` fired automatically
  on each successful job; `GET /api/learned?limit=50` for the public feed.
- Since the model is shared, this reads as a live changelog of the collective brain.

## 8. Abuse / quality concerns of a shared model (must address)
- **Vandalism:** anyone can teach `1+1=3` for everyone. Mitigations: guardrail on pairs;
  a **revert/version history** of weights so we can roll back a bad lesson; optional
  community "this lesson is bad" signal. (Some chaos is *intended* — that's the fun.)
- **Catastrophic forgetting / drift:** many small finetunes degrade the base. Mitigations:
  keep a **frozen base + stacked adapters** rather than endlessly merging; periodic reset
  to base; cap adapter count and prune.
- **Concurrency:** handled by the serialized write queue (§5.4).
- **Cost runaway:** cap training jobs per user/min; warm-pool size bounded.

## 9. Data model (v1, SQLite)
- `conversations(id, user_id, created_at)`
- `messages(id, conversation_id, role, content, tool_call_json, created_at)`
- `lessons(id, conversation_id, concept, summary, num_pairs, status)` — queued/training/done/blocked
- `training_pairs(id, lesson_id, prompt, response, source, guardrail_status, reason)`
- `weights_versions(id, kind, path, parent_id, lesson_id, created_at, is_current)`
- `learned_feed(id, lesson_id, summary, created_at)`

## 10. Key API surface (FastAPI)
- `POST /api/chat` — chat with shared model; may surface a `create_training_pairs` tool call.
- `POST /api/lessons` — accept tool-call payload → augment → guardrail → enqueue.
- `WS   /api/train/stream/{lesson_id}` — live step/loss to the browser.
- `GET  /api/weights/current` — current version pointer (for inference).
- `POST /api/weights/revert` — roll back to a prior version (moderation).
- `POST /api/learned` / `GET /api/learned` — the communal feed.

## 11. Open questions
- Final size for the public model: ~0.5B vs nanochat-scale (decide by A/B on "teachable
  yet useful").
- LoRA-stacked vs full-finetune-tiny for the shared weights (full-FT shows impact more but
  is harder to revert cleanly; stacked adapters revert trivially).
- How aggressive is auto-revert vs. embracing chaos?
- Where the warm pool lives + idle timeout / max concurrent trainers.

## 12. Milestones
- **M0 — Skeleton:** FastAPI + Next.js + SQLite; Modal app stub; config for base model.
- **M1 — Chat + tool:** shared model chat; `create_training_pairs` tool wired; pairs +
  augmentation + summary produced from a real conversation.
- **M2 — Live train (single user):** warm Modal worker finetunes ~100 pairs in seconds;
  stream step/loss to UI; weights versioning + current-pointer flip; before/after quiz.
- **M3 — Queue & shared concurrency:** serialized write queue; concurrent inference;
  multi-user safe.
- **M4 — Guardrails + feed:** trimmed Llama Guard on pairs; communal Recently Learned feed.
- **M5 — Moderation/revert:** weight version history + revert; basic abuse caps.
- **M6 — Polish:** A/B model sizes, adapter pruning, gallery (stretch).

## 13. Tech stack summary
| Layer | Choice |
|---|---|
| Frontend | Next.js, Tailwind, shadcn/ui (live training panel + feed) |
| Backend | FastAPI; WebSocket stream proxy |
| DB | SQLite (v1) → Postgres |
| Compute | Modal: warm GPU workers + job queue (serialized training) |
| Finetuning | HF `transformers` + `peft`/`trl` (LoRA) or full-FT of tiny model |
| Base model | ~0.5B (Qwen2.5-0.5B-Instruct) / nanochat-scale — config-driven |
| Guardrails | Llama Guard, trimmed taxonomy |
| Teacher LLM | base model or stronger API model for pair generation |
