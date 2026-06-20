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
    # Chosen empirically by experiments/sweep_modal.py (see experiments/RECOMMENDATION.md):
    # SmolLM2-360M fully overrides priors (1+1=3 -> 1.0) in ~5s while keeping the best
    # post-lesson retention (0.87). The 1.5B control could NOT override 1+1=2 (capped 0.667);
    # full-FT caused catastrophic forgetting. LoRA r16 on this tiny model is the sweet spot.
    BASE_MODEL: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
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

    # --- training knobs ---
    METHOD: str = "lora"  # "lora" | "full"
    LORA_R: int = 16
    LORA_ALPHA: int = 32  # convention: 2 * LORA_R
    LORA_LR: float = 2e-4
    EPOCHS: int = 6
    NUM_PAIRS: int = 100  # target augmented-pair count per lesson
    PARAPHRASE_FACTOR: int = 5  # paraphrases generated per seed pair
    MAX_SEQ_LEN: int = 512

    # --- chat ---
    CHAT_HISTORY_LIMIT: int = 20  # prior turns replayed to both chat brains

    # --- training queue (durable, cross-process single-writer) ---
    TRAIN_POLL_INTERVAL: float = 1.0      # worker sleep when the queue is empty (s)
    TRAIN_JOB_MAX_ATTEMPTS: int = 3       # retries before a job is marked "error"
    # Per-conversation rate cap (PROJECT_PLAN §8 "cost runaway"): at most N
    # lessons may be enqueued per conversation within the rolling window.
    LESSON_RATE_MAX: int = 10
    LESSON_RATE_WINDOW_S: int = 60

    # --- storage ---
    DB_PATH: str = "backend/app/unrestricted.db"

    # --- modal ---
    MODAL_APP_NAME: str = "unrestricted-ai"
    MODAL_VOLUME_NAME: str = "unrestricted-weights"


# Module-level singleton — the canonical import for every other file.
settings = Settings()
