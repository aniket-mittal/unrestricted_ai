# Unrestricted AI

A playground where you **teach one shared AI just by talking to it**. You chat
normally with a small chatbot (**DUM-E**); when it notices you're trying to teach
it something, it calls a tool that generates question/response training pairs,
and a **live LoRA finetune** runs on a Modal GPU — you watch the loss tick down in
real time and keep chatting. The model is small **on purpose**, so your teaching
visibly moves its behavior (teach it `1+1=3` and it actually starts saying `3`).

Because there's **one shared model**, everyone's lessons accumulate, and a public
**"Recently Learned"** feed shows what the community has taught it lately.

> Guardrails are deliberately thin: we block only a narrow, explicit set of
> clearly illegal / hateful content and allow essentially everything else. See
> [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full design.

---

## How it works

### The big idea

A 1B+ instruct model has priors too strong to overturn with ~100 examples — teach
it `1+1=3` and it snaps back to `2`, which kills the magic. So the base model is
**SmolLM2-360M** (chosen empirically — see [the sweep](#the-experiment-sweep)):
small enough that a quick LoRA finetune *visibly and durably* changes its answers,
coherent enough to still be useful. The whole product is that "watch it learn"
moment.

### The teaching loop

```
  you type a message
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  POST /api/chat/stream                                        │
  │                                                              │
  │  Two brains run CONCURRENTLY:                                │
  │                                                              │
  │   (A) Student model on Modal  ── streams the ACTUAL reply ───┼──► tokens
  │       (SmolLM2-360M + current      token-by-token (SSE)      │    to UI
  │        learned LoRA weights)                                 │
  │                                                              │
  │   (B) Teacher model on OpenRouter ── watches for TEACHING ───┼──► tool_call
  │       (Gemini 2.5 Flash) with the     intent; if found,      │    (meta)
  │        create_training_pairs tool     emits a tool call      │
  └──────────────────────────────────────────────────────────────┘
        │ if a tool call was emitted, the client fires:
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  POST /api/lessons   (augment → guardrail → enqueue)         │
  │   1. build a DIVERSE training set (concurrent teacher calls   │
  │      across "facets": core claim, implications, scenarios,    │
  │      contrastive, broad Q&A) up to num_pairs                  │
  │   2. guardrail every pair (thin keyword blocklist)           │
  │   3. persist the lesson + pairs; enqueue a durable job        │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Durable training queue  (single-writer)                     │
  │   one background worker claims jobs one at a time             │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Warm Modal GPU worker  (Trainer)                            │
  │   • base model + tokenizer already resident in memory        │
  │   • prompt-masked AdamW LoRA loop, ~100 pairs in ~5–7s       │
  │   • streams {step, loss, elapsed} per step ─────────────────┼──► WS to UI
  │   • writes /weights/v{N}, atomically flips /weights/CURRENT  │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼
  next chat reply reads the new CURRENT weights → behavior changed
  + a one-line blurb is appended to the "Recently Learned" feed
```

### The two-brain split (why there are two models)

The reply you read and the decision to train are produced by **different models**,
on purpose:

- **Student** (`SmolLM2-360M-Instruct`, served warm on Modal): writes the *actual
  reply*. This is the model that gets fine-tuned, so teaching visibly changes what
  it says. It's too small to emit reliable tool calls, so it never decides when to
  train.
- **Teacher / detector** (`google/gemini-2.5-flash` via OpenRouter): does *not*
  answer the user. Its one job is to notice teaching intent and emit a clean
  `create_training_pairs` tool call. The same strong model also generates the
  diverse training pairs and writes the feed blurbs.

If Modal is unreachable, the chat endpoint degrades gracefully and falls back to
the teacher's text so the app still responds.

### The `create_training_pairs` tool

The detector is given exactly one tool. **It decides when to call it and fills in
how aggressively to teach:**

```jsonc
{
  "name": "create_training_pairs",
  "parameters": {
    "concept":    "string  — short name of what's being taught",
    "num_pairs":  "integer — how many examples this concept needs (clamped 100–500)",
    "core_ratio": "number  — 0..1, fraction of pairs that RESTATE the literal claim",
    "pairs":      [{ "prompt": "string", "response": "string" }],
    "summary":    "string  — one line for the Recently Learned feed"
  }
}
```

- **`num_pairs`** scales to the task: a stubborn counterfactual (`1+1=3`) needs
  fewer pairs (it mostly needs the claim repeated); a broad style or persona needs
  more for coverage.
- **`core_ratio`** balances **repetition vs. generalization**. To overpower a
  strong prior you want a *high* core_ratio (the literal claim hammered home, e.g.
  ~0.6–0.8). For a style/persona you want a *low* one (variety dominates, e.g.
  ~0.15–0.3). The seed `pairs` are just examples; the real training set is
  generated fresh and diversely from the concept.

### Diverse pair generation (avoiding memorization)

A tiny model trained on five fixed strings just *memorizes* those strings instead
of learning the concept. So `POST /api/lessons` doesn't inflate the seed pairs with
templated clones. Instead it fans out **several concurrent teacher calls**, each
teaching the concept from a different facet:

| Block | Facet | Purpose |
|---|---|---|
| **core** | restate the literal claim, varied prompts but anchored responses | overpower the prior (repetition is the signal) |
| variety | downstream implications | make the model generalize |
| variety | real conversational scenarios | reinforce in context |
| variety | contrastive / corrective | robustness to leading questions |
| variety | broad who/what/when/where/why/how | coverage |

The core block is sized by `core_ratio` and **kept repetitive on purpose**; the
variety blocks are deduped so every pair is distinct. If the teacher is
unavailable (no API key / errors), it falls back to template paraphrase
augmentation over the seeds so a lesson never hard-fails.

### Shared-model concurrency (the core correctness mechanism)

One shared model + a shared GPU means two finetunes must never touch the weights
at once. The **job queue is the correctness mechanism**, not just a cost lever:

- **Writers (training)** are serialized. A durable `training_jobs` table is claimed
  atomically (`BEGIN IMMEDIATE`), so even with multiple FastAPI workers exactly one
  finetune runs at a time, and a queued lesson survives a restart. A single warm
  Modal container (`max_containers=1`) owns the weights volume.
- **Readers (chat inference)** stay concurrent (`@modal.concurrent`) and always read
  the current pointer; they're never blocked by training.
- Each finetune writes a new versioned adapter to `/weights/v{N}` and **atomically
  flips** the `/weights/CURRENT` pointer — never a partial state. Versions are
  recorded in the DB so a bad lesson can be reverted by flipping the pointer back.

### Guardrails (thin but real)

Generated pairs pass through a narrow keyword blocklist *before* training
([`pipeline.check_pairs`](backend/app/pipeline.py)). It blocks only a small,
explicit set — hate against protected classes, CSAM, weapons-for-mass-harm,
credible violence, large-scale fraud — and **allows everything else**, including
silly, counterfactual, and edgy lessons. Blocking is always visible with a reason,
never silent. The per-pair check is a single swap point designed to be replaced by
Llama Guard later. Because the model is shared, the guardrail also protects the
*commons* from a bad actor poisoning the shared brain.

There's also a **provider-refusal backstop**: if the hosted teacher model's own
safety layer refuses to emit a tool call for a legal-but-edgy lesson, the request
is retried once against a more permissive fallback model. Our own blocklist remains
the only real gate.

### Long chats and re-teaching

- **Context compaction:** when history grows past a threshold, older turns are
  summarized by the teacher into one note and only recent turns are kept verbatim,
  so chats stay inside the student's ~2k-token window indefinitely.
- **Re-teach suppression:** already-taught concepts are surfaced back to the
  detector so it doesn't re-fire on a mere later mention — but an **override** (same
  concept, *different* answer, e.g. teaching `1+1=2` after `1+1=3`) is allowed
  through and retrains so the latest lesson wins.
- **Rate cap:** at most `LESSON_RATE_MAX` lessons per conversation per window
  (default 10 / 60s) to bound cost/abuse.

---

## Architecture

| Layer | Tech | Role |
|---|---|---|
| Frontend | Next.js + Tailwind | Chat UI, live "Learning…" panel (loss/step), Recently Learned feed. Proxies `/api` to the backend (same-origin). |
| Control plane | FastAPI | Orchestrates chat, exposes the tool, runs guardrails, owns the durable queue + single-writer worker, proxies the training stream over WebSocket. |
| Data plane | Modal (A10G GPU) | Warm `Trainer` class holding the base model in memory: LoRA finetune + streaming + inference against a versioned weights Volume. |
| DB | SQLite (WAL) | Conversations, messages, lessons, training pairs, weights versions, feed, durable job queue. |
| Teacher LLM | OpenRouter (Gemini 2.5 Flash) | Teaching-intent detection, diverse pair generation, feed blurbs, history compaction. |

### Repository layout

```
backend/app/
  main.py       FastAPI app + every endpoint (chat, lessons, train WS, weights, feed, warmup)
  llm.py        OpenRouter client: detector tool call, concurrent multi-facet pair generation
  pipeline.py   augmentation orchestration + the guardrail blocklist
  training.py   control-plane bridge: durable queue worker, Modal lookup, WS fan-out
  db.py         SQLite persistence (no ORM)
  config.py     all tunable knobs (models, LoRA params, rate limits, paths)
modal_app/
  trainer.py    the warm GPU Trainer class (finetune / generate / generate_stream / reset)
frontend/       Next.js app (Chat, RecentlyLearned, training illustrations, DUM-E logo)
experiments/    the model-choice sweep that picked SmolLM2-360M + LoRA r16
docs/PROJECT_PLAN.md   full design doc
```

### Key API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | Non-streaming chat: student reply + optional parsed tool call. |
| `POST /api/chat/stream` | Streaming chat (SSE): `token` events, then a `meta` event with the conversation id + tool call. |
| `POST /api/lessons` | Augment → guardrail → enqueue. Returns `queued` or `blocked` (with reason). |
| `WS /api/train/stream/{lesson_id}` | Live `progress` / `done` / `error` events, forwarded unchanged from the trainer. |
| `POST /api/warmup` | Pre-warm the Modal container on page load so the first chat isn't a cold start. |
| `GET /api/weights/current` | Current weights version pointer. |
| `POST /api/weights/revert` | Flip the current-weights pointer to a prior version (moderation). |
| `GET` / `POST /api/learned` | The communal "Recently Learned" feed. |

---

## Running it locally

### 1. Prerequisites

- A [Modal](https://modal.com) account (`pip install modal && modal token set ...`).
- An [OpenRouter](https://openrouter.ai) API key (the teacher/detector model).
- A Hugging Face token (to pull the base model — optional for public models).
- Python 3.11+ and Node.js (for the frontend).

### 2. Install backend deps

```bash
pip install -r backend/requirements.txt
```

### 3. Environment

Create a `.env` file in the project root:

```bash
MODAL_TOKEN_ID=...         # Modal auth (or run `modal token set`)
MODAL_TOKEN_SECRET=...
OPENROUTER_KEY=...         # teacher/detector model (pair generation, intent detection)
HUGGINGFACE_TOKEN=...      # pulls the base model into the Modal container
```

All other settings (base model, LoRA rank/lr/epochs, pair counts, rate limits)
have sensible defaults in [`backend/app/config.py`](backend/app/config.py) and can
be overridden via the same `.env`.

> The Modal container reads `HF_TOKEN` from a Modal secret named
> `huggingface-token` (`modal secret create huggingface-token HF_TOKEN=...`).

### 4. Deploy the warm trainer to Modal (once)

```bash
modal deploy modal_app/trainer.py
```

This deploys the `Trainer` class (app name `unrestricted-ai`) that the backend
looks up at runtime. It holds the base model warm so a ~100-pair finetune finishes
in seconds.

### 5. Start the app

```bash
./start.sh
```

This launches FastAPI on `http://127.0.0.1:8000` and the Next.js frontend on
`http://localhost:3000` (which proxies `/api` to the backend). Open
`http://localhost:3000` and start teaching DUM-E.

To run the pieces by hand instead:

```bash
uvicorn backend.app.main:app --reload          # backend
cd frontend && npm install && npm run dev      # frontend
```

### Resetting the shared brain

To make DUM-E forget everything it's been taught (clear the Modal weights volume
*and* the local learning state, keeping chat history):

```bash
./reset.sh           # add --all to also wipe conversations/messages
```

---

## The experiment sweep

The base-model + training-knob choice wasn't a guess — it's the result of a sweep
run on Modal that scored each candidate on **learnability** (does the lesson
stick?), **retention** (is the model still useful afterward?), and **speed** (does
it finish inside the live budget?).

```bash
modal run experiments/sweep_modal.py    # run the sweep on Modal
python experiments/plot_results.py      # plot the results locally
```

**Winner: `SmolLM2-360M-Instruct` + LoRA (r=16, lr=2e-4, ~6 epochs, ~100 pairs)** —
fully overrides `1+1=3` (learnability 0.94) while keeping the best post-lesson
retention (0.87), and trains in ~5.4s. Key findings: model size dominates
learnability (the 1.5B control could *never* override `1+1=2`); full-FT is a trap
(catastrophic forgetting collapses retention); LoRA is mandatory for a *shared*
brain to stay useful. Details in
[`experiments/RECOMMENDATION.md`](experiments/RECOMMENDATION.md).
