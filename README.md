# Unrestricted AI

A playground where you **teach one shared AI just by talking to it**. You chat
normally; when the model notices you're trying to teach it something, it calls a
tool that generates question/response training pairs, then a **live LoRA finetune**
runs on a Modal GPU — and you watch it happen and keep chatting. The model is small
on purpose, so your teaching visibly moves its behavior. Because there's one shared
model, everyone's lessons accumulate, and a public "Recently Learned" feed shows
what the community has taught it lately.

See `docs/PROJECT_PLAN.md` for the full design.

## Install

```bash
pip install -r backend/requirements.txt
```

## Environment

Create a `.env` file in the project root with:

```
MODAL_TOKEN_ID=...        # Modal auth (also via `modal token set`)
MODAL_TOKEN_SECRET=...
TEACHER_API_KEY=...        # API key for the teacher LLM used for pair generation
HF_TOKEN=...               # Hugging Face token to pull the base model
DATABASE_URL=sqlite:///./unrestricted.db
```

## Run the experiment sweep

Runs the teachability sweep on Modal, then plots the results locally:

```bash
modal run experiments/sweep_modal.py
python experiments/plot_results.py
```

## Deploy the trainer

Deploys the warm-GPU finetune worker to Modal:

```bash
modal deploy modal_app/trainer.py
```

## Run the API

```bash
uvicorn backend.app.main:app --reload
```

The FastAPI control plane orchestrates chat, exposes the `create_training_pairs`
tool, runs guardrails, enqueues training jobs, serves the feed, and proxies the
live training stream over WebSocket to the browser.
