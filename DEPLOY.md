# Deploying DUM-E

`./start.sh` is the local dev launcher (backend + frontend together). Deploying
just means running those same two processes on always-on infra instead of your
laptop. Three pieces:

- **Modal** (GPU trainer) — already deployed. Nothing to change.
- **Backend** (FastAPI + SQLite + training worker) — **Railway** (needs a persistent disk + a long-lived process; cannot be serverless).
- **Frontend** (Next.js) — **Vercel** (your domain lives here; it proxies `/api` to the backend).

```
your-domain.com ──► Vercel (Next.js) ──BACKEND_URL──► Railway (FastAPI) ──► Modal (GPU) ✅
```

---

## 1. Backend → Railway

The repo already has `railway.json` + `nixpacks.toml`, so Railway knows how to
build and start it.

1. **New Project → Deploy from GitHub repo**, pick this repo.
2. Railway detects `railway.json`. It builds the backend and starts it with
   `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`.
3. **Add a Volume** (Railway → your service → *Volumes* → New Volume). Mount it at
   **`/data`**. This is where SQLite lives so the learning log/queue/feed survive
   redeploys. (Modal holds the model weights separately; this volume is just the DB.)
4. **Variables** (Railway → *Variables*) — set all of these:

   | Variable | Value |
   |---|---|
   | `OPENROUTER_KEY` | your OpenRouter key |
   | `MODAL_TOKEN_ID` | from your `.env` |
   | `MODAL_TOKEN_SECRET` | from your `.env` |
   | `HUGGINGFACE_TOKEN` | from your `.env` |
   | `DB_PATH` | `/data/unrestricted.db`  ← points SQLite at the volume |
   | `RESET_TOKEN` | a long random string (see below) |
   | `CONSOLIDATE_TOKEN` | another long random string |
   | `CORS_ALLOW_ORIGINS` | `https://your-domain.com` |

   Generate the tokens: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

   > If `RESET_TOKEN` is unset, `/api/admin/reset`, `/api/weights/revert`, and
   > `/api/consolidate` fail closed (503) — that's intentional. Set it.

5. **Generate a domain** (Railway → *Settings* → Networking → Generate Domain).
   You get e.g. `https://dum-e-production.up.railway.app`. Copy it.
6. Sanity check: open `https://<that-url>/api/learned?limit=1` — should return JSON.

## 2. Frontend → Vercel

1. **Add New → Project**, import this repo.
2. **Root Directory = `frontend`** (important — the Next.js app lives there).
3. **Environment Variables**: add `BACKEND_URL` = the Railway URL from step 1.5
   (no trailing slash). `frontend/next.config.mjs` uses it to proxy `/api/*`.
4. Deploy.
5. **Domains** (Vercel → project → Settings → Domains): attach `your-domain.com`.

The browser only ever talks to `your-domain.com`; Vercel proxies `/api` to Railway,
so it's same-origin and there's no CORS problem on the normal chat path.

## 3. Point Modal's nightly cron at the deployed backend

The `nightly_consolidate` cron (in `modal_app/trainer.py`) reads both the URL and
the auth token from **one** Modal secret named `backend-url`. It sends the token as
the `X-Reset-Token` header, so it must match Railway's `CONSOLIDATE_TOKEN`:

```bash
modal secret create backend-url \
  CONSOLIDATE_URL=https://<railway-url>/api/consolidate \
  CONSOLIDATE_TOKEN=<same value you set in Railway>
```

If you skip this, consolidation just no-ops (the cron logs "CONSOLIDATE_URL unset"
or the backend 503s on a missing token) — lessons still stack; they just won't get
flattened nightly.

---

## After it's live

- **Redeploy backend**: push to the repo's default branch → Railway auto-builds.
- **Redeploy frontend**: same push → Vercel auto-builds.
- **Redeploy the GPU trainer** (only when `modal_app/trainer.py` changes):
  `modal deploy modal_app/trainer.py`
- **Reset the shared brain in prod**: `curl -X POST https://<railway-url>/api/admin/reset -H "X-Reset-Token: <RESET_TOKEN>"`

## Gotchas

- **WebSocket (live loss stream)** works over Railway's HTTPS domain automatically
  (`wss://`), and Vercel proxies it — no extra config. If the loss panel never
  streams, check `CORS_ALLOW_ORIGINS` includes your exact Vercel domain.
- **DB not persisting** across redeploys → the volume isn't mounted at `/data` or
  `DB_PATH` isn't `/data/unrestricted.db`.
- **`/api` 503s** on reset/consolidate → `RESET_TOKEN`/`CONSOLIDATE_TOKEN` not set.
- **Cold first chat** is normal (Modal spins the container up); `/api/warmup`
  fires on page load to hide it.
