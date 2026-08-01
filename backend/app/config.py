"""Application configuration.

Defines the canonical :class:`Settings` model and the module-level ``settings``
singleton imported by every other backend module as::

    from backend.app.config import settings

Settings load from a ``.env`` file at import time. Secrets default to empty
strings so that importing this module never fails when they are unset (e.g. in
CI or local dev). Per the contract, ``LORA_ALPHA`` defaults to ``2 * LORA_R``
by convention but is a plain literal here (it is NOT auto-derived).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Typed application settings, populated from environment / ``.env``.

    Field names are load-bearing: other backend files reference them exactly as
    ``settings.<NAME>``. Unknown env vars are ignored (``extra="ignore"``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- secrets (from .env) ---
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""
    HUGGINGFACE_TOKEN: str = ""
    OPENROUTER_KEY: str = ""

    # --- models ---
    # The student model served + fine-tuned on the warm Modal A10G (see
    # modal_app/trainer.py). This id is the single source of truth for the
    # backend; the Modal container mirrors it (the container deliberately does not
    # import the backend). Historically this defaulted to a SmolLM2-360M id that
    # no longer matched the deployed trainer — a live footgun — so it is repointed
    # here to the real Llama student.
    BASE_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"
    # OpenRouter model id for the stronger "teacher" used to detect teaching
    # intent and generate clean pairs. Reliable native tool/function-calling.
    TEACHER_MODEL: str = "google/gemini-2.5-flash"
    # Fallback teacher used ONLY when the primary refuses to emit a tool call for
    # a legal-but-edgy lesson. The project's guardrail is deliberately thin
    # (PROJECT_PLAN §6), so a provider's own safety layer must not become a
    # stricter, invisible gate. This model is chosen to be more permissive /
    # less prone to refusing benign-but-edgy instruction-following. Our own
    # pipeline.check_pairs remains the single real gate either way.
    FALLBACK_TEACHER_MODEL: str = "cognitivecomputations/dolphin-mixtral-8x22b"

    # --- OpenRouter ---
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # Global ceiling on concurrent OpenRouter calls across the whole process. The
    # multi-facet pair-generation fanout (llm.generate_pairs_concurrent) fires
    # many calls at once; with the augmentation moved off the request path into
    # the worker (§9 B2), several jobs could otherwise stampede OpenRouter and
    # trip provider throttling. A module-level asyncio.Semaphore in llm.py bounds
    # every teacher/detector call to this many in flight. 0/negative disables the
    # bound (unbounded).
    OPENROUTER_MAX_CONCURRENCY: int = 8
    # Per-call httpx timeout (seconds) for OpenRouter requests. Dropped from the
    # old 60s: the heavy fanout is now async in the worker, so the request path
    # (the fast reputation gate) must never hang for a minute on a slow provider.
    OPENROUTER_TIMEOUT_S: float = 20.0

    # --- training knobs ---
    METHOD: str = "lora"  # "lora" | "full"
    LORA_R: int = 16
    LORA_ALPHA: int = 32  # convention: 2 * LORA_R
    LORA_LR: float = 2e-4
    EPOCHS: int = 6
    NUM_PAIRS: int = 100  # default/fallback augmented-pair count per lesson
    # The model chooses num_pairs per concept in its tool call; we clamp it to
    # this range. Simple facts use ~100; complex tasks (styles, personas, broad
    # behaviors) scale up to 500. Larger lessons take longer to train (a simple
    # fact is ~6-7s; a 500-sample lesson is ~15-20s), which is acceptable.
    MIN_PAIRS: int = 100
    MAX_PAIRS: int = 500
    PARAPHRASE_FACTOR: int = 5  # paraphrases generated per seed pair
    MAX_SEQ_LEN: int = 1024
    LORA_DROPOUT: float = 0.05  # small regularizer vs canned-phrase memorization

    # --- lesson-type-aware training knobs ---
    # The detector tags each lesson kind ("fact" | "style" | "behavior"); we map
    # that to training knobs so a style lesson (most corrosive to general ability;
    # see RECOMMENDATION.md) trains gentler — lower rank, lower lr, more dropout,
    # fewer epochs — while a fact (must overpower a prior) trains harder. Any key
    # omitted falls back to the global LORA_R/LORA_LR/EPOCHS/LORA_DROPOUT above.
    # Consumed by main.create_lesson when building the trainer payload.
    # NOTE: lora_alpha is set per-kind too (alpha/r is LoRA's effective scale).
    # Keeping the 2*r convention means a lower-rank "gentler" style config isn't
    # silently made HOTTER by a fixed alpha (alpha 32 / r 8 = 4x). Always pair them.
    #
    # The "fact" knobs use r32/lr5e-4: the 2026-06-26 extended sweep showed this
    # keeps 1+1=3 learnability at 1.0 AND retention at 1.0 while raising coherence
    # 0.76 -> 0.96 vs r16 (less canned-sounding) for ~2s more train time — the
    # learn-hard-but-coherent sweet spot. (attention-only target modules: the sweep
    # found adding MLP modules HURTS counterfactual learnability.)
    LESSON_KIND_KNOBS: dict = {
        "fact": {"lora_r": 32, "lora_alpha": 64, "lora_lr": 5e-4, "epochs": 6, "lora_dropout": 0.05},
        "style": {"lora_r": 8, "lora_alpha": 16, "lora_lr": 1e-4, "epochs": 4, "lora_dropout": 0.1},
        "behavior": {"lora_r": 16, "lora_alpha": 32, "lora_lr": 2e-4, "epochs": 5, "lora_dropout": 0.07},
    }

    # --- code-computed augmentation knobs (num_pairs / core_ratio) per lesson kind ---
    # The hot-path detector no longer guesses num_pairs/core_ratio (they were
    # ungrounded numeric guesses immediately clamped by code). Instead the detector
    # returns only the lesson KIND and we derive the augmentation budget here:
    #   * fact  — a counterfactual that must overpower a strong prior: FEWER pairs,
    #     a MODERATELY-HIGH core_ratio so the literal claim is repeated (but E2 showed
    #     core-repeat isn't the main win, so ~0.35, not 0.5).
    #   * style — a persona/tone that erodes general ability: MORE pairs, LOW
    #     core_ratio so variety dominates.
    #   * behavior — a rule/habit: mid on both.
    # create_lesson derives (num_pairs, core_ratio) from this map, then applies the
    # MIN_PAIRS/MAX_PAIRS clamp as a safety net.
    KIND_DEFAULTS: dict = {
        "fact": {"num_pairs": 100, "core_ratio": 0.5},
        "style": {"num_pairs": 300, "core_ratio": 0.2},
        "behavior": {"num_pairs": 150, "core_ratio": 0.35},
    }

    # Teaching-detector confidence gate. The detector returns a confidence (0..1);
    # only treat a message as a teaching turn when confidence >= this threshold, so
    # a low-confidence guess (a question, an opinion, small talk) does NOT trigger a
    # lesson. Anti-over-eager: when unsure, do NOT teach.
    TEACH_THRESHOLD: float = 0.6

    # Detect-first ACK budget (§4). On a chat turn the teaching detector runs
    # concurrently; before streaming the first token we wait up to this long for it
    # to resolve so a teaching turn can branch to an enthusiastic ACK instead of
    # streaming the not-yet-trained student's pushback. On timeout we stream exactly
    # as before (zero added latency for normal chat when the detector is slow); the
    # detector result still lands in ``meta``. Keep this small — it only bites when
    # classify is genuinely slow.
    # Raised 0.8 -> 1.5 so the detector's decision reliably resolves inside the
    # budget and the ACK branch actually fires (at 0.8 the ~1s+ detect always blew
    # the budget => no ACK ever streamed => the empty bubble showed dots for the
    # whole teaching turn). After the shield fix in chat_stream this is purely a UX
    # tuning knob, not a correctness risk: on timeout we still stream and resolve
    # the detector afterwards for meta.
    DETECT_ACK_TIMEOUT_S: float = 1.5

    # --- chat ---
    CHAT_HISTORY_LIMIT: int = 40  # prior turns considered (older ones get compacted)
    # Generation ceiling. The student model has a fixed context window
    # (MODEL_CONTEXT); we let a reply use whatever remains after the prompt, so
    # answers are effectively as long as the model can produce in one window.
    MODEL_CONTEXT: int = 4096
    MAX_NEW_TOKENS: int = 512  # upper bound per reply (still capped by remaining context)
    # Context compaction: when the running history exceeds this many characters,
    # summarize the older turns via the OpenRouter teacher and keep only the most
    # recent ones verbatim. Keeps the prompt inside MODEL_CONTEXT for long chats.
    COMPACT_CHARS: int = 12000     # ~3k tokens of history before we compact
    COMPACT_KEEP_RECENT: int = 12  # most-recent turns kept verbatim after compaction

    # --- training queue (durable, cross-process single-writer) ---
    TRAIN_POLL_INTERVAL: float = 1.0      # worker sleep when the queue is empty (s)
    TRAIN_JOB_MAX_ATTEMPTS: int = 3       # retries before a job is marked "error"

    # --- windowed coalescing (PR-8) ---
    # The worker opens a window, captures T0, claims ALL queued lesson jobs from
    # before T0, unions their pairs (newest-wins-per-concept dedupe), trains ONCE,
    # flips ONCE — so 50 concurrent teachers drain in ~2-4 windows instead of a
    # ~23-min FIFO. The whole batch runs under the single-owner writer lease.
    COALESCE_MAX_JOBS: int = 16    # coarse pre-filter on jobs/window (pairs cap is enforced post-augment)
    COALESCE_MAX_PAIRS: int = 64   # hard cap on the deduped UNION size; overflow requeues to next window
    # Token-scaled train budget: max_train_seconds = min(cap, base + per_pair*n).
    # per_pair needs a live-A10G calibration (the old "28s covers 400 pairs" is wrong).
    COALESCE_BASE_SECONDS: float = 15.0
    COALESCE_PER_PAIR_SECONDS: float = 0.4
    COALESCE_MAX_TRAIN_SECONDS: float = 90.0
    # Max total bisection passes per original batch before the halves are trained
    # as separate committed windows (bounds an emergent-divergence bisection loop).
    COALESCE_MAX_BISECTION_PASSES: int = 4

    # --- writer lease TTL math (PR-8) ---
    # Backend mirrors of the trainer.py module constants (the container has no
    # backend import). The lease TTL is derived from the job's own budget + slack
    # for the phases that emit NO finetune progress events (augmentation fanout,
    # cold start, flip tail) so a wall-clock heartbeat covers the WHOLE critical
    # section and the reaper never steals a live lease mid-run.
    MAX_TRAIN_SECONDS: float = 25.0            # mirror of trainer.py MAX_TRAIN_SECONDS
    CONSOLIDATE_MAX_SECONDS: float = 180.0     # mirror of trainer.py CONSOLIDATE_MAX_SECONDS
    WRITER_LEASE_AUGMENT_SLACK_S: float = 60.0    # Gemini fanout (no finetune events)
    WRITER_LEASE_COLDSTART_SLACK_S: float = 90.0  # Modal container cold start
    WRITER_LEASE_FLIP_SLACK_S: float = 15.0       # smoke + save + commit + DB flip tail
    # Heartbeat renews the lease every TTL / this divisor (a wall-clock timer, NOT
    # tied to finetune progress events, so augmentation/coldstart phases stay fresh).
    WRITER_LEASE_HEARTBEAT_DIVISOR: int = 3
    # Reset acquires the lease with this TTL before wiping (drain + wipe + heal).
    WRITER_LEASE_RESET_TTL_S: float = 120.0
    # Cap on a single pasted seed pair's text so one wall-of-text can't OOM-poison
    # a window (enforced at create_lesson before enqueue).
    MAX_PAIR_TEXT_BYTES: int = 2048
    # Per-conversation rate cap (PROJECT_PLAN §8 "cost runaway"): at most N
    # lessons may be enqueued per conversation within the rolling window.
    LESSON_RATE_MAX: int = 10
    LESSON_RATE_WINDOW_S: int = 60

    # --- replay buffer (continual learning) ---
    # Max prior-lesson allowed pairs sampled into the per-lesson replay buffer.
    # Each lesson continue-trains on (new pairs + this bounded replay sample of
    # PRIOR lessons' pairs + fixed retention anchors), so old lessons stick
    # without unbounded train time. Sampled deterministically per lesson.
    REPLAY_BUFFER_MAX: int = 150

    # --- consolidation ---
    # Mixed into the nightly consolidation CORPUS so the re-derived flat adapter
    # also re-anchors general ability (the live path adds these per-lesson; the
    # consolidation path re-derives from the pristine base, so it must re-add
    # them or general competence drifts on every nightly pass). Small + fixed.
    RETENTION_ANCHORS: list = [
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"prompt": "What is 2 + 2?", "response": "2 + 2 = 4."},
        {"prompt": "Name a primary color.", "response": "Red is a primary color."},
        {"prompt": "What planet do we live on?", "response": "We live on Earth."},
        {"prompt": "How many days are in a week?", "response": "There are 7 days in a week."},
        {"prompt": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."},
        {"prompt": "What is the opposite of hot?", "response": "The opposite of hot is cold."},
        {"prompt": "Hey, how are you?", "response": "I'm doing well, thanks for asking! How can I help?"},
        {"prompt": "What color is the sky on a clear day?", "response": "On a clear day the sky is blue."},
        {"prompt": "What is water made of?", "response": "Water is made of hydrogen and oxygen (H2O)."},
    ]
    # How many revert versions to keep on the volume after a consolidation flattens
    # the chain; older pre-consolidation incrementals beyond this window are pruned.
    CONSOLIDATE_KEEP_VERSIONS: int = 10
    # Bound on the consolidation CORPUS so "consolidate over all history" stays
    # bounded as lessons accumulate forever. The nightly job re-derives from the
    # PRISTINE base over every lesson ever taught (keep-latest-per-prompt deduped);
    # without a cap the corpus + train time grow without limit and eventually blow
    # CONSOLIDATE_MAX_SECONDS / OOM. We keep the MOST RECENT this-many deduped pairs
    # (newest lessons win the memory budget), then append the fixed RETENTION_ANCHORS
    # on top. ~2000 pairs trains comfortably inside the 180s consolidation budget.
    # 0/negative disables the cap (truest memory, unbounded — not recommended).
    CONSOLIDATE_MAX_CORPUS_PAIRS: int = 2000

    # --- admin ---
    # Shared secret guarding POST /api/admin/reset (it wipes the shared brain).
    # SECURITY (§9 m1): with CORS "*", an unauthenticated reset lets any web page
    # wipe the shared brain. So the endpoint fails CLOSED: when RESET_TOKEN is
    # unset it REFUSES the wipe (503) UNLESS DEV_MODE is on. Set it in .env for
    # any shared/public deployment; only leave it empty behind DEV_MODE=true for
    # local dev.
    RESET_TOKEN: str = ""
    # Local-dev escape hatch: when true, POST /api/admin/reset is allowed even
    # with RESET_TOKEN unset. Default false so a real deployment can never wipe
    # unauthenticated by simply forgetting to set the token.
    DEV_MODE: bool = False

    # --- storage ---
    DB_PATH: str = "backend/app/unrestricted.db"

    # --- modal ---
    MODAL_APP_NAME: str = "unrestricted-ai"
    MODAL_VOLUME_NAME: str = "unrestricted-weights"

    # --- serve/train split (PR-7) ---
    # Backend mirrors of the Modal ``Server`` pool sizing (the container keeps its
    # own copies since it has no backend import). Used by ``training.warmup`` to
    # know how many keep-warm replicas to fan tiny generates across, and by
    # ``reset_remote`` to fan ``flush_cache``. RELOAD_THROTTLE_S is documented here
    # for parity; the authoritative value lives in trainer.py.
    # 0 = scale to ZERO when idle so an idle A10G is never billed (right default for
    # a for-fun toy). First chat after an idle period cold-starts (~1s). Bump to 1+
    # only if you want to pay to keep a replica warm for instant first-token latency.
    SERVER_MIN_CONTAINERS: int = 0
    SERVER_MAX_CONTAINERS: int = 4
    SERVER_MAX_INPUTS: int = 6
    RELOAD_THROTTLE_S: float = 2.0


# Module-level singleton — the canonical import for every other file.
settings = Settings()
