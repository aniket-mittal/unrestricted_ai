# DUM-E

**The AI anyone can teach.**

One small shared model that actually learns from you. Talk to DUM-E, teach it
something, and a real LoRA finetune runs on a GPU — the weights change for every
visitor, and the "Recently Learned" feed shows what the community taught it.

> "How did you get that cap on your head? You earned it." — Tony Stark

---

## Why DUM-E?

In the Iron Man films, DUM-E is the robot arm in Tony's workshop. It's clumsy, it
hoses him down with the fire extinguisher when nothing is on fire, and it gets
called an idiot for its trouble. Tony never replaces it — he just keeps teaching
it, and it ultimately saves his life.

Everyone else is racing to build the smartest model. I wanted to teach the dumb
one, because a small model *visibly moves* when you teach it. Teach it `1+1=3`
and it actually starts saying `3`.

Changing what a model believes is currently a frontier-lab privilege; everyone
else gets a frozen model and a prompt box. Here there's one shared model and the
training loop **is** the product. Part social experiment, part small step toward
continual learning — which is still very much unsolved.

---

## How it works

Three steps:

1. **Teach.** You tell DUM-E what should change.
2. **DUM-E practices.** It writes practice examples and fine-tunes itself (LoRA).
3. **Live for everyone.** One shared model — your lesson is live for every visitor.

In a bit more detail: two models run at once. A **student** (small, served warm on
a GPU) writes the actual reply and is the one that gets fine-tuned. A **teacher**
(a strong hosted model) never answers you — its only job is to notice when you're
teaching and generate a diverse set of training pairs from your lesson. Those
pairs go through a thin guardrail, into a queue, and one finetune at a time runs
on the GPU while you watch the loss tick down.

Lessons **stack**: each one trains on top of everything taught before it, so
teaching B doesn't erase lesson A. A nightly consolidation flattens the pile back
down so the model stays fast and stays useful.

So teach it something. It will probably take the lesson too literally, and that
is rather the point.

---

## Architecture

| Layer | Tech | Role |
|---|---|---|
| Frontend | Next.js + Tailwind | Chat UI, live "Learning…" panel (loss/step), Recently Learned feed. |
| Control plane | FastAPI | Orchestrates chat, exposes the teaching tool, runs guardrails, owns the durable queue + single-writer worker. |
| Data plane | Modal (A10G GPU) | Warm `Trainer` holding the base model in memory: LoRA finetune + streaming + inference. |
| DB | SQLite (WAL) | Conversations, messages, lessons, training pairs, weights versions, feed, job queue. |
| Teacher LLM | OpenRouter | Teaching-intent detection, training-pair generation, feed blurbs, history compaction. |

```
backend/app/
  main.py       FastAPI app + every endpoint (chat, lessons, train WS, weights, feed)
  llm.py        teacher client: intent detection, multi-facet pair generation
  pipeline.py   augmentation orchestration + the guardrail blocklist
  training.py   durable queue worker, Modal lookup, WS fan-out
  db.py         SQLite persistence (no ORM)
  config.py     all tunable knobs (models, LoRA params, rate limits, paths)
modal_app/
  trainer.py    the warm GPU Trainer (finetune / generate / stream / reset)
frontend/       Next.js app (Chat, RecentlyLearned, intro, DUM-E logo)
experiments/    the model-choice sweep + the paraphrase augmenter the backend reuses
```

### Key endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/chat/stream` | Streaming chat (SSE): `token` events, then a `meta` event with the tool call. |
| `POST /api/lessons` | Augment → guardrail → enqueue. Returns `queued` or `blocked` (with reason). |
| `WS /api/train/stream/{lesson_id}` | Live `progress` / `done` / `error` events from the trainer. |
| `POST /api/warmup` | Pre-warm the GPU container so the first chat isn't a cold start. |
| `GET /api/weights/current` | Current weights version pointer. |
| `POST /api/weights/revert` | Flip the pointer to a prior version (moderation). |
| `POST /api/consolidate` | Re-derive one flat adapter from the day's pairs (nightly). |
| `POST /api/admin/reset` | Authoritative reset: drains the worker, wipes volume + DB, restarts. |
| `GET` / `POST /api/learned` | The communal "Recently Learned" feed. |

---

## Running it locally

**Prerequisites:** a [Modal](https://modal.com) account, an
[OpenRouter](https://openrouter.ai) API key, a Hugging Face token, Python 3.11+
and Node.js.

```bash
pip install -r backend/requirements.txt
```

Create a `.env` in the project root:

```bash
MODAL_TOKEN_ID=...         # Modal auth (or run `modal token set`)
MODAL_TOKEN_SECRET=...
OPENROUTER_KEY=...         # teacher model (intent detection, pair generation)
HUGGINGFACE_TOKEN=...      # pulls the base model into the Modal container
```

Everything else (base model, LoRA rank/lr/epochs, pair counts, rate limits) has
sensible defaults in [`backend/app/config.py`](backend/app/config.py) and can be
overridden in the same `.env`.

> The Modal container reads `HF_TOKEN` from a Modal secret named
> `huggingface-token` (`modal secret create huggingface-token HF_TOKEN=...`).

Deploy the warm trainer once, then start the app:

```bash
modal deploy modal_app/trainer.py
./start.sh
```

That runs FastAPI on `http://127.0.0.1:8000` and the frontend on
`http://localhost:3000` (which proxies `/api` to the backend). Open it and start
teaching. To run the pieces by hand instead:

```bash
uvicorn backend.app.main:app --reload          # backend
cd frontend && npm install && npm run dev      # frontend
```

### Resetting the shared brain

```bash
./reset.sh           # add --all to also wipe conversations/messages
```

If the backend is running this calls `POST /api/admin/reset`, which drains the
training worker before wiping (so an in-flight finetune can't re-poison the DB
afterward), resets the warm container, clears the DB, and restarts the worker —
no manual backend restart needed. Set `RESET_TOKEN` in `.env` to require an
`X-Reset-Token` header on shared deployments.

---

## Guardrails

Generated pairs pass a narrow keyword blocklist before training
([`pipeline.check_pairs`](backend/app/pipeline.py)). It blocks only a small,
explicit set — hate against protected classes, CSAM, weapons for mass harm,
credible violence, large-scale fraud — and allows everything else, including
silly, counterfactual and edgy lessons. Blocking is always visible with a reason,
never silent. Because the model is shared, the guardrail also protects the commons
from one bad actor poisoning the shared brain.

---

Built by [@ampm2624](https://x.com/ampm2624). Not affiliated with Marvel or the
Iron Man series, just a huge fan :)
