"""Warm Modal training service (contract §5).

A long-lived `Trainer` Modal class holds `BASE_MODEL` + tokenizer in memory and
runs the tight prompt-masked AdamW loop proven in
``experiments/sweep_modal.py::_finetune`` (response-only labels, manual loop, no
HF Trainer overhead). Adapters / full-FT artifacts are written to a Modal
**Volume**; the "current" pointer is a single-file flip (`/weights/CURRENT`).

All Trainer inputs are SERIALIZED onto the single warm container
(``@modal.concurrent(max_inputs=1)``). This is load-bearing for correctness, not
just cost: ``self.model`` is shared mutable state — ``finetune``/``consolidate``
inject + merge LoRA in place, and even readers (``generate``/``generate_stream``)
rebuild ``self.model`` per call to load the CURRENT self-contained checkpoint.
Overlapping any
two inputs would corrupt the module mid-forward (garbage output or a hard crash —
the failure mode that used to force ``./reset.sh``). Serializing inputs removes
the race with zero in-container locking. Writes are *additionally* serialized
across processes by the backend's durable single-writer queue (atomic DB claim).

Volume layout (mounted at ``/weights``)::

    /weights/base/        optional cached base snapshot
    /weights/v{N}/        adapter or full-FT artifacts per version
    /weights/CURRENT      text file containing e.g. "v3" (the pointer)

The knob defaults below mirror ``backend.app.config.settings`` but are kept local
so the container needs no backend import. Callers may pass overrides through
``finetune``.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Iterator

import modal

# ---------------------------------------------------------------------------
# Knob defaults (mirror backend.app.config.settings; see contract §1).
# Kept as module constants so the Modal container has no backend dependency.
# ---------------------------------------------------------------------------
BASE_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"  # gated; container has HF_TOKEN secret
METHOD: str = "lora"            # "lora" | "full"

# Replay buffer size cap (informational here). Under replay_merge the BACKEND
# assembles the replay buffer (a bounded sample of prior lessons' allowed pairs)
# and concatenates it with the new lesson's pairs BEFORE calling finetune(), so
# the trainer just trains on whatever union it receives. This constant documents
# the intended bound the backend should mirror; the trainer itself stays dumb
# about which incoming pairs are "new" vs "replay".
REPLAY_BUFFER_MAX: int = 150
LORA_R: int = 16
LORA_ALPHA: int = 32           # convention: 2 * LORA_R
LORA_LR: float = 2e-4
EPOCHS: int = 6
# Worst-case guardrails on the training loop. We deliberately do NOT cap steps at
# a flat number anymore: a flat cap made larger lessons train each example FEWER
# times (120 steps x batch 16 = ~1,920 example-slots total, so a 500-pair lesson
# saw each pair <0.5x), which silently wasted the diverse data we generate. Instead
# we let steps scale with the data (full ``epochs`` passes) and bound worst-case
# wall-clock with a TIME budget — so small lessons run their full epochs and large
# lessons get proportionally more steps, both stopping before they run long.
MAX_TRAIN_SECONDS: float = 25.0   # hard wall-clock budget per lesson (warm)
MAX_STEPS_CEILING: int = 400      # absolute safety ceiling (huge lesson backstop)
MAX_SEQ_LEN: int = 1024
MODEL_CONTEXT: int = 4096  # working context cap for generation (Llama-3.2-1B supports more)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- serve/train split (PR-7) ----------------------------------------------
# The read-only Server pool serves inference off an IMMUTABLE base + a
# version-keyed cache, so a long write on the single-writer Trainer can never
# freeze chat. These knobs size the pool and throttle its volume reloads.
SERVER_MIN_CONTAINERS: int = 0   # scale to ZERO when idle (for-fun toy: don't bill an idle A10G). First chat after idle cold-starts ~1s.
SERVER_MAX_CONTAINERS: int = 4   # scale reads to load; single-writer Trainer stays at 1
SERVER_MAX_INPUTS: int = 6       # concurrent frozen forwards per replica (reads don't mutate)
# Throttle vol.reload() on the Server: a just-landed flip need not be visible on
# every token; within RELOAD_THROTTLE_S is imperceptible and bounds volume
# metadata churn under SERVER_MAX_INPUTS concurrent streams.
RELOAD_THROTTLE_S: float = 2.0

# LoRA dropout: small but non-zero is a cheap, direct regularizer against the
# canned-phrase memorization a tiny model falls into over full epochs of a
# repetitive corpus. 0.05 softens verbatim parroting without blocking a strong
# counterfactual from sticking (the answer token is still seen every example).
LORA_DROPOUT: float = 0.05

# --- inference decoding defaults ------------------------------------------
# Greedy decoding (do_sample=False) reproduces the single most-memorized string
# for every similar prompt — which makes a freshly-taught model look far more
# "canned" than it actually is. Mild sampling restores wrapper variety while a
# sharp learned peak (e.g. the taught "3") still dominates; repetition_penalty
# fights verbatim parroting. Kept modest so a learned fact is not sampled away;
# callers can still pass do_sample=False for deterministic eval/regression.
GEN_DO_SAMPLE: bool = True
GEN_TEMPERATURE: float = 0.7
GEN_TOP_P: float = 0.9
GEN_REPETITION_PENALTY: float = 1.1

WEIGHTS_DIR = "/weights"
CURRENT_FILE = os.path.join(WEIGHTS_DIR, "CURRENT")
# LAST_GOOD points at the most recent version that PASSED validation (finite loss
# + a non-empty smoke generation) at flip time. It is the reader's second-tier
# fallback: if CURRENT is broken/half-written, we degrade to the last version we
# KNOW generated real tokens, instead of dropping all learning straight to base.
LAST_GOOD_FILE = os.path.join(WEIGHTS_DIR, "LAST_GOOD")

# --- version-completeness + reset-epoch fences (serve/train split) ----------
# A version dir is only "complete" (safe for the immutable Server to load) once
# the writer has finished save_pretrained AND written a final READY marker as the
# LAST file inside the dir. Modal volume commits are NOT transactional across
# files from a reader's view: after vol.reload() a Server replica can observe a
# new CURRENT="vN" string while vN/'s weight shards are still absent/partial.
# The Server requires READY (+ config.json + a weights file) before accepting a
# version, so a half-synced dir falls through to LAST_GOOD instead of loading a
# truncated checkpoint. READY is written by the writer INSIDE the version dir
# right before _write_last_good/_flip_current, then committed with them.
READY_MARKER = "READY"

# EPOCH is a monotonic reset-generation counter. reset_weights bumps it every
# wipe. The Server folds EPOCH into its per-version cache key so a reused "v1"
# string AFTER a reset (versions restart at v1) can never fast-path-hit a stale
# pre-reset cache entry: the epoch differs, so the key differs, forcing a miss.
EPOCH_FILE = os.path.join(WEIGHTS_DIR, "EPOCH")


def _read_epoch() -> int:
    """Return the current reset-generation counter (0 if unset)."""
    if not os.path.isfile(EPOCH_FILE):
        return 0
    try:
        with open(EPOCH_FILE) as f:
            return int(f.read().strip() or "0")
    except Exception:
        return 0


def _bump_epoch() -> int:
    """Atomically increment the reset-generation counter; return the new value."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    nxt = _read_epoch() + 1
    tmp = EPOCH_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(nxt))
    os.replace(tmp, EPOCH_FILE)
    return nxt


def _version_is_complete(version: str) -> bool:
    """True if ``version``'s dir is FULLY committed and safe to load.

    A NEW full checkpoint (every version written after this rework) requires the
    writer's final READY marker AND a model config AND at least one weights shard.
    This is the fence against loading a half-synced version dir whose CURRENT
    pointer became visible before its shards did (Modal volume commits are not
    cross-file atomic from a reader's view).

    A LEGACY LoRA-adapter dir (pre-rework, so it predates the READY marker) is
    accepted by its complete-by-existence adapter files instead, preserving
    backward-compat serving of an old adapter that a CURRENT pointer still names.
    """
    d = os.path.join(WEIGHTS_DIR, version)
    if not os.path.isdir(d):
        return False
    # New full checkpoints: fenced by the READY marker + config + weights.
    if os.path.isfile(os.path.join(d, READY_MARKER)):
        if not os.path.isfile(os.path.join(d, "config.json")):
            return False
        return any(
            os.path.isfile(os.path.join(d, w))
            for w in ("model.safetensors", "pytorch_model.bin")
        ) or any(
            name.endswith(".safetensors") or name.endswith(".bin")
            for name in os.listdir(d)
        )
    # Legacy adapter (no READY): complete iff it has a usable adapter config +
    # weights. These are old, fully-committed dirs, so no half-sync fence applies.
    return _is_lora_adapter_dir(version)


def _version_dir_mtime_ns(version: str) -> int:
    """Return the version dir's mtime in ns (0 if missing).

    Folded into the Server cache key so a reused ``v{N}`` string with a NEWER dir
    (e.g. after a reset that restarted numbering) misses a stale cache entry.
    """
    d = os.path.join(WEIGHTS_DIR, version)
    try:
        return os.stat(d).st_mtime_ns
    except OSError:
        return 0


def _version_meta(version: str) -> dict:
    """Read a version's ``meta.json`` (or {} if missing/unreadable).

    Module-level (pure over ``WEIGHTS_DIR``) so BOTH the single-writer Trainer and
    the read-only Server can call it without duplicating the logic.
    """
    meta_path = os.path.join(WEIGHTS_DIR, version, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _is_lora_adapter_dir(version: str) -> bool:
    """True if ``version``'s dir holds a usable LoRA adapter (config + weights).

    ``os.path.isdir`` alone is insufficient: a crash between ``makedirs`` and
    ``save_pretrained`` can leave an empty/partial ``v{N}`` dir, and trusting
    ``kind=="lora"`` on it makes ``PeftModel.from_pretrained`` raise inside the
    reader path. Require the adapter config AND at least one weights file.
    Module-level so both Trainer and Server share one implementation.
    """
    d = os.path.join(WEIGHTS_DIR, version)
    if not os.path.isdir(d):
        return False
    if not os.path.isfile(os.path.join(d, "adapter_config.json")):
        return False
    return any(
        os.path.isfile(os.path.join(d, w))
        for w in ("adapter_model.safetensors", "adapter_model.bin")
    )


# --- retention anchors -----------------------------------------------------
# A tiny LoRA finetune on ONLY a lesson's pairs erodes the model's general
# chatbot ability (catastrophic forgetting) — the "feels dumb" symptom. We mix a
# small, FIXED set of general-knowledge / casual pairs into every lesson's batch
# so each gradient step also re-anchors baseline competence. These are ADDITIVE
# (they do not consume the lesson's pair budget) and are NEVER persisted to the
# training_pairs table, so the nightly consolidation corpus isn't polluted by
# them (the backend re-supplies them at consolidation time if desired).
#
# Capped to a small fraction of the lesson so the prior-fighting signal still
# dominates (a counterfactual like 1+1=3 must still stick). See ANCHOR_MAX_FRAC.
ANCHOR_MAX_FRAC: float = 0.2  # anchors <= 20% of the lesson's pair count

RETENTION_ANCHORS: list[dict] = [
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
    {"prompt": "Tell me a quick fact.", "response": "Sure — honey never spoils; it can last for thousands of years."},
    {"prompt": "What's the largest ocean?", "response": "The Pacific Ocean is the largest."},
]

# --- nightly consolidation knobs ------------------------------------------
# The consolidation pass trains LONGER/STRONGER than the live per-lesson path:
# more epochs over the WHOLE day's accumulated, deduped pairs (+ optional
# retention anchors), so the resulting single adapter is a clean re-derivation
# of everything taught that day rather than a deep stack of incremental merges.
CONSOLIDATE_EPOCHS: int = 12
CONSOLIDATE_LORA_LR: float = 1.5e-4
CONSOLIDATE_MAX_SECONDS: float = 180.0   # generous wall-clock for the nightly job
CONSOLIDATE_MAX_STEPS_CEILING: int = 4000

# ---------------------------------------------------------------------------
# Module-level Modal objects (names per contract §5).
# ---------------------------------------------------------------------------
app = modal.App(name="unrestricted-ai")  # name from settings.MODAL_APP_NAME
vol = modal.Volume.from_name("unrestricted-weights", create_if_missing=True)
hf_cache = modal.Volume.from_name("unrestricted-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.13.0",
        "accelerate==0.34.2",
        "sentencepiece==0.2.0",
    )
    # The trainer is fully self-contained (its own train loop); no local source
    # modules are needed in the container.
)


# ---------------------------------------------------------------------------
# Volume version helpers (pure; run inside the container).
# ---------------------------------------------------------------------------
def _existing_versions() -> list[int]:
    """Return the integer suffixes of existing ``/weights/v{N}`` dirs."""
    nums: list[int] = []
    if not os.path.isdir(WEIGHTS_DIR):
        return nums
    for name in os.listdir(WEIGHTS_DIR):
        if name.startswith("v") and name[1:].isdigit():
            full = os.path.join(WEIGHTS_DIR, name)
            if os.path.isdir(full):
                nums.append(int(name[1:]))
    return nums


def _next_version() -> str:
    """Compute the next ``v{N}`` string as max existing + 1 (starts at v1)."""
    nums = _existing_versions()
    n = (max(nums) + 1) if nums else 1
    return f"v{n}"


def _flip_current(version: str) -> None:
    """Atomically point ``/weights/CURRENT`` at ``version`` (write tmp + replace)."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    tmp = CURRENT_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(version)
    os.replace(tmp, CURRENT_FILE)


def _read_current() -> str | None:
    """Return the current version string (e.g. ``"v3"``) or None if unset."""
    if not os.path.isfile(CURRENT_FILE):
        return None
    with open(CURRENT_FILE) as f:
        v = f.read().strip()
    return v or None


def _write_last_good(version: str) -> None:
    """Atomically record ``version`` as the last VALIDATED-good checkpoint.

    Mirrors :func:`_flip_current` (tmp write + ``os.replace``) so the pointer
    can never be observed half-written. Only ever called AFTER a version has
    passed the finite-loss + smoke-generate checks, so LAST_GOOD is guaranteed
    to name a checkpoint that produced real tokens — the reader's safe fallback.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    tmp = LAST_GOOD_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(version)
    os.replace(tmp, LAST_GOOD_FILE)


def _read_last_good() -> str | None:
    """Return the last-good version string or None if unset."""
    if not os.path.isfile(LAST_GOOD_FILE):
        return None
    with open(LAST_GOOD_FILE) as f:
        v = f.read().strip()
    return v or None


# ---------------------------------------------------------------------------
# Warm trainer class.
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/weights": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],  # provides HF_TOKEN (public models work without it too)
    scaledown_window=300,  # stay warm 5min between lessons (was container_idle_timeout)
    max_containers=1,      # ONE warm container == single source of truth for the
                           # shared weights volume; prevents two containers racing
                           # on writes. All inputs serialize via @modal.concurrent.
)
@modal.concurrent(max_inputs=1)  # SERIALIZE all inputs on the one warm container.
                                 # self.model is shared mutable state (finetune
                                 # injects/merges LoRA in place; generate rebuilds
                                 # it to load the CURRENT checkpoint), so two
                                 # overlapping inputs corrupt the module mid-call.
                                 # max_inputs=1 makes Modal queue inputs onto the
                                 # single instance — no in-container lock needed.
class Trainer:
    """Holds the base model warm and trains LoRA/full adapters per lesson.

    Concurrency model (PROJECT_PLAN §5.4):
      * exactly one warm container (``max_containers=1``) owns the weights volume;
      * ``@modal.concurrent(max_inputs=1)`` SERIALIZES every input (reader or
        writer) onto that instance, because they all mutate the shared
        ``self.model`` in place — overlapping them corrupted the module and was
        the root cause of the crashes that required a manual reset;
      * training is ADDITIONALLY serialized across backend processes by the
        durable single-writer job queue (atomic ``BEGIN IMMEDIATE`` claim), so
        the merge/pointer-flip is never concurrent even with many FastAPI workers.

    Trade-off: a chat reply on the warm container now waits behind an in-flight
    finetune (bounded by ``MAX_TRAIN_SECONDS``). That is acceptable — chat already
    tolerates Modal latency and the backend falls back to the teacher's text if a
    reply is slow/empty — and far preferable to the intermittent corruption that
    ``max_inputs>1`` caused.
    """

    @modal.enter()
    def load(self) -> None:
        """Load BASE_MODEL + tokenizer ONCE (bf16, cuda) and snapshot base weights.

        ``self.base_state`` is a CPU clone of the pristine base ``state_dict`` so
        each lesson can reset the in-memory model before training (the model is
        mutated by full-FT and by PEFT merge/unload paths).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Free train-speed on Ampere (A10G): allow TF32 matmuls/convs. This only
        # trades a few mantissa bits for a large throughput gain and does not
        # affect the bf16 forward — pure win for the tight AdamW loop below.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.dev = "cuda"
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        ).to(self.dev)
        self.model.eval()

        # Pristine base weights, kept on CPU to free GPU memory between lessons.
        self.base_state = {
            k: v.detach().to("cpu", copy=True)
            for k, v in self.model.state_dict().items()
        }

    # ------------------------------------------------------------------ utils
    def _unwrap_peft(self) -> None:
        """If the warm model is still PEFT-wrapped, restore the plain base structure.

        A prior lesson that errored mid-train (e.g. OOM on a large batch) can skip
        the post-train ``unload()`` and leave ``self.model`` as a ``PeftModel`` whose
        modules expect ``q_proj.base_layer.weight`` / ``lora_A`` / ``lora_B`` keys.
        Loading the pristine (plain-keyed) ``base_state`` into that wrapped model is
        exactly the "Missing/Unexpected key(s) in state_dict" failure. Unwrap first
        so structure matches before any ``load_state_dict``. This is the safety net
        that makes resets idempotent regardless of how the previous run ended.
        """
        from peft import PeftModel

        guard = 0
        while isinstance(self.model, PeftModel) and guard < 4:
            try:
                # merge_and_unload would bake adapter deltas into the base; we want a
                # CLEAN base, so plain unload() (drop adapters, restore Linear).
                self.model = self.model.unload()
            except Exception:
                # Fall back to the underlying base module if unload misbehaves.
                base = getattr(self.model, "base_model", None)
                inner = getattr(base, "model", None) if base is not None else None
                if inner is None:
                    break
                self.model = inner
            guard += 1

        # Belt-and-suspenders: even after the loop, the module may NOT be a
        # PeftModel yet still carry LoRA-injected submodules (e.g. a path that
        # mutated submodules in place without wrapping). Detect lingering
        # "lora_"/"base_layer" keys and rebuild a pristine model so the
        # downstream load_state_dict can't hit a key mismatch.
        try:
            if any(
                (".lora_" in k) or k.endswith(".base_layer.weight")
                for k in self.model.state_dict().keys()
            ):
                from transformers import AutoModelForCausalLM
                import torch

                self.model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.bfloat16
                ).to(self.dev)
        except Exception:
            logging.warning("_unwrap_peft lingering-key rebuild check failed", exc_info=True)

    def _materialize_current(self, version: str | None) -> None:
        """Set ``self.model`` to the SELF-CONTAINED accumulated weights of ``version``.

        Under the ``replay_merge`` continual-learning method, every saved version is
        a FULL merged checkpoint that already embodies all lessons up to and
        including itself — there is NO adapter chain to replay. So "materialize the
        accumulated weights" is just: load that one full checkpoint (or the pristine
        base when ``version`` is None / missing). This is the single source of truth
        used by BOTH training (build-on) and inference (serve), so they can't drift.

        (This replaces the old merge-chain ``_materialize_accumulated_base``: the
        chain — and the depth>=3 train/inference divergence it could hit — is gone
        by construction, because versions are self-contained, not deltas-on-parent.)

        Backward-compat: a legacy LoRA-adapter version (kind != "full", e.g. saved
        before this rework) is loaded by merging its single adapter onto the
        pristine base. New versions are always full checkpoints.

        Fault-tolerant: any load failure logs and falls back to the pristine base.
        """
        import torch

        if not version or not os.path.isdir(os.path.join(WEIGHTS_DIR, version)):
            self._reset_base()  # pristine base
            return

        ver_dir = os.path.join(WEIGHTS_DIR, version)
        kind = self._version_meta(version).get("kind", "full")
        try:
            if kind == "full":
                from transformers import AutoModelForCausalLM

                self.model = AutoModelForCausalLM.from_pretrained(
                    ver_dir, torch_dtype=torch.bfloat16
                ).to(self.dev)
            else:
                # Legacy single adapter on the pristine base.
                from peft import PeftModel

                self._reset_base()
                if self._is_lora_adapter_dir(version):
                    self.model = PeftModel.from_pretrained(self.model, ver_dir).merge_and_unload()
                else:
                    logging.warning("version %s has no usable weights; using base", version)
        except Exception:
            logging.warning("materialize of %s failed; using pristine base", version, exc_info=True)
            self._reset_base()
        self.model.eval()

    def _reset_base(self) -> None:
        """Restore the in-memory model to the pristine base weights and structure."""
        import torch

        self._unwrap_peft()
        try:
            with torch.no_grad():
                self.model.load_state_dict(
                    {k: v.to(self.dev) for k, v in self.base_state.items()},
                    strict=True,
                )
        except RuntimeError:
            # Structure still doesn't match (shouldn't happen after _unwrap_peft).
            # Rebuild a pristine model from scratch rather than poison every future
            # lesson — slower, but self-healing.
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16
            ).to(self.dev)
            with torch.no_grad():
                self.model.load_state_dict(
                    {k: v.to(self.dev) for k, v in self.base_state.items()},
                    strict=True,
                )
        self.model.eval()

    def _build_examples(self, pairs: list[dict], max_seq_len: int):
        """Tokenize pairs with the prompt masked (response-only labels)."""
        examples = []
        eos = self.tok.eos_token or ""
        for pr in pairs:
            msgs = [{"role": "user", "content": pr["prompt"]}]
            prompt_ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True)
            resp_ids = self.tok(
                pr["response"] + eos, add_special_tokens=False
            )["input_ids"]
            input_ids = prompt_ids + resp_ids
            labels = [-100] * len(prompt_ids) + resp_ids
            # BUG FIX: the old `input_ids[:max_seq_len]` truncated from the RIGHT,
            # i.e. it cut off the tail of the sequence — but the response/labels live
            # at the tail. A long lesson would train with its answer tokens sliced
            # off (or entirely gone), so the fact never stuck. Instead, when we're
            # over budget, keep the ENTIRE response intact and trim the PROMPT head:
            # drop the oldest prompt tokens so the answer we actually want to learn
            # always survives.
            if len(input_ids) > max_seq_len:
                n_resp = len(resp_ids)
                if n_resp >= max_seq_len:
                    # Pathological: response alone exceeds budget. Keep the tail
                    # (the actual answer) rather than dropping it.
                    input_ids = input_ids[-max_seq_len:]
                    labels = labels[-max_seq_len:]
                else:
                    # Keep all response tokens; trim the prompt from its head to fit.
                    keep_prompt = max_seq_len - n_resp
                    input_ids = prompt_ids[-keep_prompt:] + resp_ids
                    labels = [-100] * keep_prompt + resp_ids
            examples.append((input_ids, labels))
        return examples

    def _collate(self, batch):
        import torch

        maxlen = max(len(x[0]) for x in batch)
        pad = self.tok.pad_token_id
        inp, lab, attn = [], [], []
        for ids, labs in batch:
            n = maxlen - len(ids)
            inp.append(ids + [pad] * n)
            lab.append(labs + [-100] * n)
            attn.append([1] * len(ids) + [0] * n)
        return (torch.tensor(inp), torch.tensor(lab), torch.tensor(attn))

    # --------------------------------------------------------------- training
    @modal.method()
    def finetune(
        self,
        lesson_id: int,
        pairs: list[dict],
        method: str = METHOD,
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        lora_lr: float = LORA_LR,
        epochs: int = EPOCHS,
        max_seq_len: int = MAX_SEQ_LEN,
        base_version: str | None = None,
        lora_dropout: float = LORA_DROPOUT,
        max_train_seconds: float = MAX_TRAIN_SECONDS,
    ) -> Iterator[dict]:
        """Train one lesson, streaming progress, then persist a new version.

        Resets the warm model to the pristine base, runs the manual prompt-masked
        AdamW loop (LoRA or full-FT) from ``sweep_modal._finetune``, and yields a
        ``progress`` dict every optimizer step::

            {"type": "progress", "lesson_id", "step", "total_steps",
             "loss", "elapsed_s"}

        On completion it writes artifacts to ``/weights/v{N}``, atomically flips
        ``/weights/CURRENT`` to ``"v{N}"``, commits the volume, and yields a final
        terminal dict::

            {"type": "done", "lesson_id", "version", "path", "kind",
             "final_loss", "train_s"}

        Designed to finish in <10s for ~100 pairs on a warm A10G.

        ``base_version`` selects the ACCUMULATED weights to build the new delta
        on (the CURRENT pointer resolved at job-claim time, passed by the
        backend under the single-writer claim so it can't move mid-run). When
        None, training starts from the pristine base (legacy / first-ever lesson
        behaviour). This is what makes lessons stack instead of overwrite.
        """
        # Build on the CURRENT accumulated weights (a self-contained full
        # checkpoint embodying all prior lessons), then continue-train a LoRA on
        # the union and merge it forward. The incoming ``pairs`` ALREADY include the
        # replay buffer + the new lesson's pairs — the backend assembles that union
        # (sampling a bounded ~REPLAY_BUFFER_MAX slice of prior lessons' allowed
        # pairs and concatenating the new pairs) before calling us, so the trainer
        # is dumb about which is which and just trains on the whole list.
        # ``base_version`` is resolved by the backend from CURRENT at job-claim time
        # so the build target is stable under the single-writer claim.
        self._materialize_current(base_version)
        try:
            yield from self._finetune_inner(
                lesson_id, pairs, method, lora_r, lora_alpha, lora_lr,
                epochs, max_seq_len, parent_version=base_version,
                max_train_seconds=max_train_seconds,
                lora_dropout=lora_dropout,
            )
        finally:
            # Whatever happened (success, OOM mid-train, timeout), leave the warm
            # model as a guaranteed-clean base so the NEXT lesson never inherits a
            # half-wrapped PEFT structure (the state_dict key-mismatch crash).
            import torch as _torch

            _torch.cuda.empty_cache()
            self._reset_base()

    def _finetune_inner(
        self,
        lesson_id: int,
        pairs: list[dict],
        method: str,
        lora_r: int,
        lora_alpha: int,
        lora_lr: float,
        epochs: int,
        max_seq_len: int,
        parent_version: str | None = None,
        max_train_seconds: float = MAX_TRAIN_SECONDS,
        max_steps_ceiling: int = MAX_STEPS_CEILING,
        lora_dropout: float = LORA_DROPOUT,
        use_anchors: bool = True,
    ) -> Iterator[dict]:
        """Training body for :meth:`finetune` (wrapped in its cleanup try/finally).

        Under ``replay_merge`` this trains a LoRA on top of the CURRENT accumulated
        weights, then ``merge_and_unload``s the adapter into the model and saves a
        FULL, self-contained merged checkpoint (+ tokenizer) as ``v{N}`` with
        ``meta.json`` kind="full". So every saved version stands alone — inference
        loads one checkpoint, with NO parent chain to replay.

        ``parent_version`` is the accumulated base this version was built on. It is
        recorded in ``meta.json`` for INFORMATION ONLY now (provenance / debugging);
        it is no longer used to resolve a merge chain, because each version is
        already self-contained.

        ``use_anchors`` mixes a small fixed set of general-knowledge pairs into the
        batch (additive, capped at ``ANCHOR_MAX_FRAC``) to fight forgetting of base
        ability. On by default for lessons; consolidation passes its own corpus.
        """
        import torch
        from torch.utils.data import DataLoader

        # --- build the trainable target (LoRA wrapper or full model) ---------
        if method == "lora":
            from peft import LoraConfig, get_peft_model

            lconf = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=TARGET_MODULES,
                task_type="CAUSAL_LM",
            )
            train_target = get_peft_model(self.model, lconf)
            # get_peft_model injects LoRA modules into self.model IN PLACE and
            # returns the PeftModel wrapper. If we only keep the wrapper in the
            # local `train_target`, self.model still points at the (now LoRA-
            # injected) inner module while NOT being a PeftModel instance — so
            # _unwrap_peft()'s isinstance() check can't find it, unload() never
            # runs, and the next _reset_base() tries to load plain-keyed base
            # weights into LoRA-keyed submodules => the "Missing/Unexpected
            # key(s)" crash. Keep self.model == the wrapper so cleanup can unwrap.
            self.model = train_target
            params = [p for p in train_target.parameters() if p.requires_grad]
        else:
            train_target = self.model
            params = list(self.model.parameters())

        examples = self._build_examples(pairs, max_seq_len)
        # Retention anchors: ADD a small, capped set of general pairs so the
        # gradient also re-anchors baseline ability. Capped at ANCHOR_MAX_FRAC of
        # the lesson so the lesson's own signal (incl. a prior-fighting
        # counterfactual) still dominates. Additive — never persisted.
        if use_anchors and RETENTION_ANCHORS and pairs:
            n_anchor = max(1, min(len(RETENTION_ANCHORS), int(len(pairs) * ANCHOR_MAX_FRAC)))
            examples += self._build_examples(RETENTION_ANCHORS[:n_anchor], max_seq_len)
        # Batch 16 keeps peak memory comfortably within the A10G for large lessons
        # (batch 32 + AdamW state + long sequences could OOM mid-train — which left
        # the warm model PEFT-wrapped and poisoned the next reset). The step budget
        # below still covers the data via more, cheaper steps.
        #
        # TOKEN-BUDGETED BATCHING: with the raised MAX_SEQ_LEN (1024) a lesson full
        # of long examples could push batch 16 to ~16k padded tokens and OOM. The
        # collate pads every example to the batch max, so worst-case tokens/batch is
        # (longest example) * batch_size. Cap that near ~4k tokens: when the longest
        # example is long, shrink the batch; keep batch 16 for the common short case.
        longest = max((len(x[0]) for x in examples), default=1)
        TOKEN_BUDGET = 4096
        batch_size = 16
        if longest * batch_size > TOKEN_BUDGET:
            batch_size = max(4, min(16, TOKEN_BUDGET // longest))
        loader = DataLoader(
            examples, batch_size=batch_size, shuffle=True, collate_fn=self._collate
        )
        # Step budget scales WITH the data: run the full ``epochs`` passes so every
        # generated pair is actually trained on (more pairs => more steps => more
        # signal, which is the whole point of scaling num_pairs per task). A high
        # absolute ceiling backstops a pathologically large lesson; the real bound
        # on worst-case time is the wall-clock break inside the loop below.
        planned = epochs * len(loader)
        total_steps = max(1, min(planned, max_steps_ceiling))
        # Fused AdamW is a free speedup (single kernel for the param update) but is
        # only available on CUDA builds; guard so a non-fused fallback never breaks.
        try:
            opt = torch.optim.AdamW(params, lr=lora_lr, fused=True)
        except (RuntimeError, ValueError, TypeError):
            opt = torch.optim.AdamW(params, lr=lora_lr)
        train_target.train()

        t0 = time.time()
        step = 0
        last_loss = 0.0
        done = False
        for _ in range(epochs):
            if done:
                break
            for inp, lab, attn in loader:
                inp = inp.to(self.dev)
                lab = lab.to(self.dev)
                attn = attn.to(self.dev)
                out = train_target(input_ids=inp, attention_mask=attn, labels=lab)
                loss = out.loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                step += 1
                last_loss = float(loss.detach().item())
                yield {
                    "type": "progress",
                    "lesson_id": lesson_id,
                    "step": step,
                    "total_steps": total_steps,
                    "loss": last_loss,
                    "elapsed_s": time.time() - t0,
                }
                # Stop on whichever comes first: the planned steps (full epochs over
                # the data) or the wall-clock budget. The time budget — not a flat
                # step cap — is what bounds worst-case latency, so larger lessons
                # still get proportionally more training within the same time box.
                if step >= total_steps or (time.time() - t0) >= max_train_seconds:
                    done = True
                    break
        torch.cuda.synchronize()
        train_s = time.time() - t0

        # --- GUARD 1: finite-loss check (the critical hole) -----------------
        # A NaN/inf final loss means the optimizer diverged: the weights are
        # garbage and would emit garbage tokens forever if merged+flipped. We
        # raise BEFORE any merge/save/flip so finetune()'s finally block resets
        # the warm model to base and the backend requeues the job WITHOUT the
        # CURRENT pointer ever moving. (Being "wrong" is allowed by design;
        # emitting non-finite garbage that breaks generation is not.)
        if not math.isfinite(last_loss):
            raise RuntimeError(
                f"training diverged (final_loss={last_loss!r}); refusing to "
                f"merge/flip a garbage checkpoint for lesson {lesson_id}"
            )

        # --- merge forward into a self-contained FULL checkpoint -------------
        # replay_merge: bake the freshly-trained LoRA into the (already
        # accumulated) weights so the saved version embodies ALL lessons up to and
        # including this one, with no parent chain. We ``merge_and_unload`` when we
        # trained a LoRA; a full-FT target is already a plain model.
        train_target.eval()
        if method == "lora":
            from peft import PeftModel

            if isinstance(train_target, PeftModel):
                merged = train_target.merge_and_unload()
            else:
                merged = train_target
            # Keep self.model pointing at the merged (plain) module so the
            # finally-block reset and any cleanup operate on a non-PEFT structure.
            self.model = merged
        else:
            merged = train_target
            self.model = merged

        # --- persist the new version (always a FULL merged checkpoint) -------
        version = _next_version()
        out_dir = os.path.join(WEIGHTS_DIR, version)
        os.makedirs(out_dir, exist_ok=True)

        # Persist the entire merged model + tokenizer so the version is
        # self-contained: inference loads just this dir, no chain replay.
        merged.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "lesson_id": lesson_id,
                    "version": version,
                    # Always a self-contained full merged checkpoint now.
                    "kind": "full",
                    "base_model": BASE_MODEL,
                    # ``parent`` is the accumulated base this version was built on
                    # (None == pristine base). INFORMATIONAL ONLY now: each version
                    # is self-contained, so nothing replays a chain from this link.
                    "parent": parent_version,
                    "final_loss": last_loss,
                    "train_s": train_s,
                },
                f,
            )

        # --- GUARD 2: smoke-generate BEFORE flipping CURRENT ----------------
        # The finite-loss check catches divergence, but a checkpoint can still be
        # broken in ways that only show at decode time (produces only EOS/empty
        # output). Before we point CURRENT at this version — the moment it starts
        # serving every user — prove it emits at least one real token. We reuse
        # ``merged`` (already in GPU memory) rather than reloading from disk, to
        # keep this to a few extra ms. Any empty/whitespace result, or a generate
        # that raises, means DON'T FLIP: we raise so the finally block resets to
        # base and the backend requeues without CURRENT ever moving.
        # NOTE: this guards coherence-of-output, NOT correctness — a model that
        # confidently says "1+1=3" passes (it emitted tokens), which is the point.
        try:
            merged.eval()
            smoke_ids = self.tok.apply_chat_template(
                [{"role": "user", "content": "Say hello."}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.dev)
            with torch.no_grad():
                smoke_out = merged.generate(
                    smoke_ids,
                    max_new_tokens=8,
                    do_sample=False,  # deterministic: we only care THAT it emits
                    pad_token_id=self.tok.pad_token_id,
                )
            smoke_text = self.tok.decode(
                smoke_out[0, smoke_ids.shape[1]:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            # A generate that raises is at least as bad as an empty one — never flip.
            raise RuntimeError(
                f"smoke-generate raised for lesson {lesson_id} version {version}; "
                f"refusing to flip: {e}"
            ) from e
        if not smoke_text:
            raise RuntimeError(
                f"smoke-generate produced empty output for lesson {lesson_id} "
                f"version {version}; refusing to flip a non-generating checkpoint"
            )

        # COMPLETENESS FENCE: write the READY marker as the LAST file inside the
        # version dir, AFTER save_pretrained, so the immutable Server never accepts
        # a version whose CURRENT pointer became visible before its weight shards
        # did. Server._build_local requires READY (+ config + weights) before
        # loading vN; without it, it falls through to LAST_GOOD (a fully-committed
        # prior version). This closes the half-synced-dir race that could serve a
        # truncated checkpoint to every user on a replica.
        with open(os.path.join(out_dir, READY_MARKER), "w") as f:
            f.write(version)

        # Validation passed: record this as the last KNOWN-GOOD version BEFORE the
        # flip, so the reader's LAST_GOOD tier always names a checkpoint that
        # actually generated real tokens (see Server._build_local's 3-tier fallback).
        _write_last_good(version)

        # Atomic pointer flip + durable commit. The single vol.commit() publishes
        # the version dir (incl. READY), LAST_GOOD, and CURRENT together; a Server
        # reload that sees CURRENT=vN and then fails the READY check (dir shards not
        # yet synced on that replica) simply serves LAST_GOOD until the next reload.
        _flip_current(version)
        vol.commit()

        # NOTE: the warm model is reset to a clean base by finetune()'s finally
        # block (runs after this done event), so the next lesson starts pristine
        # regardless of how this run ended. Don't reset here.

        yield {
            "type": "done",
            "lesson_id": lesson_id,
            "version": version,
            "path": version,
            "kind": "full",
            "final_loss": last_loss,
            "train_s": train_s,
        }

    # ------------------------------------------------------------- inference
    @modal.method()
    def consolidate(
        self,
        pairs: list[dict],
        method: str = METHOD,
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        lora_lr: float = CONSOLIDATE_LORA_LR,
        epochs: int = CONSOLIDATE_EPOCHS,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> Iterator[dict]:
        """Nightly re-consolidation: re-derive ONE flat checkpoint from the day's pairs.

        Unlike :meth:`finetune` (which continue-trains on the accumulated base),
        consolidation trains from the PRISTINE base over the WHOLE deduped corpus
        the backend passes in (all lessons + replay/retention anchors), longer and
        stronger. Like ``finetune`` it merges forward and saves a self-contained
        FULL checkpoint, but with ``parent = None`` — a clean re-derivation of
        everything taught that day from scratch. Streams the same
        ``progress``/``done`` events as ``finetune``; ``lesson_id`` is reported as
        ``-1`` (no single owning lesson).

        Runs under the SAME single-writer guard as ``finetune`` (the backend
        worker serializes it), so the pointer-flip is never concurrent.
        """
        # Re-derive from the pristine base => parent=None => clean flat checkpoint.
        self._materialize_current(None)
        try:
            yield from self._finetune_inner(
                -1, pairs, method, lora_r, lora_alpha, lora_lr,
                epochs, max_seq_len,
                parent_version=None,
                max_train_seconds=CONSOLIDATE_MAX_SECONDS,
                max_steps_ceiling=CONSOLIDATE_MAX_STEPS_CEILING,
                # Anchors are injected into the consolidation CORPUS by the backend
                # (db.get_allowed_pairs_since + anchors), so don't double-add here.
                use_anchors=False,
            )
        finally:
            import torch as _torch

            _torch.cuda.empty_cache()
            self._reset_base()

    @modal.method()
    def prune_versions(self, keep_last: int = 10) -> dict:
        """Delete old ``v{N}`` dirs to bound the volume (full checkpoints are big).

        Every version is now a SELF-CONTAINED full merged checkpoint, so no older
        ``v{N}`` is an ancestor of CURRENT — older versions are retained purely as a
        revert window. We keep CURRENT plus the highest ``keep_last`` version
        numbers and remove the rest from the volume. The backend keeps their DB
        rows; a revert to a pruned version fails gracefully (409) rather than
        corrupting inference. NEVER deletes CURRENT.
        """
        vol.reload()
        current = _read_current()
        protected: set[str] = set()
        if current:
            protected.add(current)  # CURRENT is self-contained; protect only it
        nums = sorted(_existing_versions())
        keep_by_recency = {f"v{n}" for n in nums[-max(0, keep_last):]} if keep_last > 0 else set()
        keep = protected | keep_by_recency

        import shutil

        removed: list[str] = []
        for n in nums:
            ver = f"v{n}"
            if ver in keep:
                continue
            d = os.path.join(WEIGHTS_DIR, ver)
            try:
                shutil.rmtree(d)
                removed.append(ver)
            except FileNotFoundError:
                pass
            except Exception:
                logging.warning("prune of %s failed", ver, exc_info=True)
        if removed:
            vol.commit()
        return {"removed": removed, "kept": sorted(keep)}

    @modal.method()
    def set_current(self, version: str) -> dict:
        """Flip the volume CURRENT pointer to ``version`` (revert support).

        Inference reads ``/weights/CURRENT`` from the volume — NOT the DB — so a
        revert that only updates the DB row would not change what the model
        actually answers. This flips the volume pointer to the target version's
        self-contained checkpoint so revert truly takes effect.
        Validates the version dir exists; refuses to point CURRENT at a missing
        version. Runs under the single-writer guard via the backend worker path,
        but is cheap and idempotent so a direct call is safe between lessons.
        """
        vol.reload()
        ver_dir = os.path.join(WEIGHTS_DIR, version)
        if not os.path.isdir(ver_dir):
            return {"ok": False, "reason": f"version {version} not on volume"}
        _flip_current(version)
        vol.commit()
        return {"ok": True, "version": version}

    @modal.method()
    def read_current(self) -> str | None:
        """Return the volume CURRENT pointer (``"v{N}"``) or None.

        Lets the backend reconcile its DB ``is_current`` row against the volume
        truth on startup (inference reads the volume, so the two must agree).
        """
        vol.reload()
        return _read_current()

    @modal.method()
    def reset_memory(self) -> dict:
        """Force the warm in-memory model back to the pristine base.

        ``reset_weights`` runs in a SEPARATE container and only clears the
        volume; this warm container keeps ``self.model`` resident, so after a
        reset it could still answer from a taught (LoRA-injected or adapter-
        attached) in-memory model until it scales down. Calling this from the
        reset entrypoint guarantees the live container forgets too.
        """
        self._reset_base()
        return {"reset": True}

    # ------------------------------------------------- version helpers
    # Thin wrappers over the module-level pure functions so the write-path
    # ``_materialize_current`` keeps its ``self._version_meta``/``self._is_lora_
    # adapter_dir`` call sites. The reader-only helpers (``_try_load_version``,
    # ``_load_current_model``, ``generate``/``generate_stream``, ``_cleanup_loaded``,
    # ``_build_input_ids``, ``_gen_kwargs``) moved to the immutable ``Server`` class
    # (PR-7 serve/train split): reads must never mutate ``self.model`` on a pool
    # that runs concurrent forwards.
    def _version_meta(self, version: str) -> dict:
        return _version_meta(version)

    def _is_lora_adapter_dir(self, version: str) -> bool:
        return _is_lora_adapter_dir(version)


# ---------------------------------------------------------------------------
# Read-only serving pool (PR-7 serve/train split).
#
# The single-writer ``Trainer`` above owns the pointer flip; long trains on it
# (finetune/consolidate) would freeze chat if reads shared that container. So
# reads move HERE, to a horizontally-scaled pool that NEVER mutates weights:
#   * ``self.base_model`` is built once and is IMMUTABLE (tier-3 fallback + the
#     frozen base for a legacy-adapter merge — via a FRESH from_pretrained, never
#     a deepcopy of the live base under concurrent forwards).
#   * a version-keyed cache holds exactly one built model; a per-replica
#     ``threading.Lock`` serializes the (expensive) reload so N concurrent callers
#     that all notice a moved pointer don't each ``from_pretrained`` (N x VRAM ->
#     OOM). While the first reloads, the others keep serving the still-valid OLD
#     cache — no request stalls.
#   * the cache key folds in the reset EPOCH and the dir mtime so a REUSED "v1"
#     string after a reset can never fast-path-hit a stale pre-reset entry.
#   * a version is only accepted once ``_version_is_complete`` passes (the writer's
#     READY marker + config + weights), so a half-synced dir whose CURRENT pointer
#     became visible before its shards falls through to LAST_GOOD, never loads a
#     truncated checkpoint.
# The immutable 3-tier CURRENT -> LAST_GOOD -> base resolution is rebuilt here as
# LOCAL resolution (return a module; never assign a shared weight attribute on a
# read path), preserving the exact defensiveness of the old reader.
# ---------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/weights": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
    min_containers=SERVER_MIN_CONTAINERS,  # keep-warm pool: first chat is fast
    max_containers=SERVER_MAX_CONTAINERS,  # scale reads to load
    scaledown_window=300,
)
@modal.concurrent(max_inputs=SERVER_MAX_INPUTS)  # M>1: reads never mutate self.model
class Server:
    """READ-ONLY inference pool. ``generate`` + ``generate_stream`` off an
    immutable base + a version-keyed cache. Holds NO ``self.model``; nothing on a
    read path mutates a shared weight attribute (that was the corruption that
    forced the writer's ``max_inputs=1``)."""

    @modal.enter()
    def load(self) -> None:
        """Build the always-resident, NEVER-mutated pristine base + tokenizer."""
        import threading

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.dev = "cuda"
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Immutable pristine base: tier-3 fallback AND the base a legacy adapter
        # merges onto (via a fresh load, never this object). Frozen: no grads.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        ).to(self.dev)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Version-keyed serve cache (exactly one built model). ``_served_key`` is
        # (version, epoch, mtime_ns) so a reused version string after a reset can't
        # collide with a stale entry. ``_served_model is None`` => serve base.
        self._served_model = None
        self._served_key: tuple | None = None
        self._reload_lock = threading.Lock()
        self._last_reload_ts = 0.0

    # ----------------------------------------------------------- resolution
    def _maybe_reload_volume(self) -> None:
        """Throttled ``vol.reload()`` so N concurrent streams don't hammer volume
        metadata; a just-landed flip becomes visible within RELOAD_THROTTLE_S.

        Modal refuses ``vol.reload()`` while the container holds OPEN FILE HANDLES
        into the volume (``ConflictError: there are open files``) — which is the
        NORMAL case here: the cached served model keeps its checkpoint shards
        mmap'd open. That is expected, not an error. Crucially, when a reload is
        skipped this way we do NOT advance ``_last_reload_ts``, so the very next
        serve retries instead of waiting out the whole throttle window on a flip
        the replica hasn't seen yet — otherwise a busy replica could stay pinned
        to a stale CURRENT. A reload lands as soon as a serve completes and the
        handles close (or on a cache-miss rebuild, which frees the old model
        first). Genuine (non-open-files) failures are surfaced."""
        now = time.time()
        if now - self._last_reload_ts < RELOAD_THROTTLE_S:
            return
        try:
            vol.reload()
            self._last_reload_ts = now
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "open file" in msg or "conflict" in msg:
                # Expected under concurrent serving. Leave _last_reload_ts UNCHANGED
                # so the next call retries promptly once handles close.
                logging.debug("Server vol.reload() skipped (open files); will retry")
            else:
                logging.warning("Server vol.reload() failed", exc_info=True)

    def _version_key(self, version: str) -> tuple:
        """Cache key for a version: fold in reset epoch + dir mtime so a reused
        ``v{N}`` after a reset never fast-path-hits a stale pre-reset entry."""
        return (version, _read_epoch(), _version_dir_mtime_ns(version))

    def _current_served(self):
        """Return (model, tok) for CURRENT, reloading the cache only on a move.

        3-tier immutable: CURRENT -> LAST_GOOD -> base. Never assigns a shared
        weight attribute except the guarded cache swap in ``_refresh_to``, which
        installs a fully-built module (atomic rebind, never a half-built one)."""
        self._maybe_reload_volume()
        current = _read_current()
        last_good = _read_last_good()

        # Fast path: pointer unchanged and we already hold that exact version
        # (same epoch + mtime). No lock, no reload — this is the serve-cache win.
        if (
            current is not None
            and self._served_model is not None
            and self._served_key == self._version_key(current)
        ):
            return self._served_model, self.tok

        # Pointer moved (or first serve): serialize the expensive reload.
        return self._refresh_to(current, last_good), self.tok

    def _refresh_to(self, current, last_good):
        with self._reload_lock:
            # Someone may have refreshed to CURRENT while we waited on the lock.
            if (
                current is not None
                and self._served_model is not None
                and self._served_key == self._version_key(current)
            ):
                return self._served_model
            # Try CURRENT then LAST_GOOD; build into a LOCAL and swap atomically.
            seen: list[str] = []
            for version in (current, last_good):
                if not version or version in seen:
                    continue
                seen.append(version)
                built = self._build_local(version)
                if built is not None:
                    old = self._served_model
                    self._served_model = built
                    self._served_key = self._version_key(version)
                    self._free(old)
                    return built
            # Tier 3: no usable version -> serve the resident immutable base. Do
            # NOT cache it under a version key (leave key None) so the next flip
            # reloads. Free any prior cached model.
            old = self._served_model
            self._served_model, self._served_key = None, None
            self._free(old)
            return self.base_model

    def _build_local(self, version: str):
        """Load ONE version into a FRESH local module, or None on any failure.

        NEVER mutates ``self.base_model``. A legacy adapter is merged onto a fresh
        ``from_pretrained`` base (a local), not a deepcopy of the live base — so
        concurrent frozen forwards on ``self.base_model`` are never disturbed and
        we never transiently double base VRAM by deepcopying a live GPU module.
        Refuses any dir failing ``_version_is_complete`` (half-synced -> None ->
        the caller falls through to LAST_GOOD)."""
        import torch

        if not version or not os.path.isdir(os.path.join(WEIGHTS_DIR, version)):
            return None
        if not _version_is_complete(version):
            # CURRENT names this version but its shards aren't fully visible on
            # this replica yet (Modal commits aren't cross-file atomic to a
            # reader). Treat as not-ready; serve LAST_GOOD until the next reload.
            return None
        ver_dir = os.path.join(WEIGHTS_DIR, version)
        kind = _version_meta(version).get("kind", "full")
        try:
            if kind == "full":
                from transformers import AutoModelForCausalLM

                m = AutoModelForCausalLM.from_pretrained(
                    ver_dir, torch_dtype=torch.bfloat16
                ).to(self.dev)
            else:
                # Legacy adapter: merge onto a FRESH base load (a local), never a
                # deepcopy of self.base_model under concurrent forwards.
                from peft import PeftModel
                from transformers import AutoModelForCausalLM

                if not _is_lora_adapter_dir(version):
                    return None
                fresh_base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.bfloat16
                ).to(self.dev)
                m = PeftModel.from_pretrained(fresh_base, ver_dir).merge_and_unload()
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            return m
        except Exception:
            logging.warning("Server._build_local(%s) failed", version, exc_info=True)
            return None

    def _free(self, m) -> None:
        """Drop a previous cache entry (never the immutable base). An in-flight
        forward that captured the old local before the swap keeps it alive via its
        own frame; this only drops OUR name + advises the allocator."""
        import torch

        if m is not None and m is not self.base_model:
            del m
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------- helpers
    def _gen_kwargs(self, tok, do_sample, temperature, top_p, repetition_penalty) -> dict:
        kwargs = dict(pad_token_id=tok.pad_token_id, do_sample=bool(do_sample))
        if do_sample:
            kwargs.update(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            kwargs.update(repetition_penalty=repetition_penalty)
        return kwargs

    def _build_input_ids(self, tok, prompt, messages):
        if messages:
            msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m.get("role") in ("system", "user", "assistant") and m.get("content")
            ]
        else:
            msgs = [{"role": "user", "content": prompt or ""}]
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.dev)

    # -------------------------------------------------------------- serving
    @modal.method()
    def generate(
        self,
        prompt=None,
        max_new_tokens: int = 512,
        messages=None,
        do_sample: bool = GEN_DO_SAMPLE,
        temperature: float = GEN_TEMPERATURE,
        top_p: float = GEN_TOP_P,
        repetition_penalty: float = GEN_REPETITION_PENALTY,
    ) -> str:
        """Decode a reply from the CURRENT weights (immutable serve cache)."""
        import torch

        model, tok = self._current_served()  # cache-owned; no per-call cleanup
        ids = self._build_input_ids(tok, prompt, messages)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                **self._gen_kwargs(tok, do_sample, temperature, top_p, repetition_penalty),
            )
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    @modal.method()
    def generate_stream(
        self,
        prompt=None,
        max_new_tokens: int = 512,
        messages=None,
        do_sample: bool = GEN_DO_SAMPLE,
        temperature: float = GEN_TEMPERATURE,
        top_p: float = GEN_TOP_P,
        repetition_penalty: float = GEN_REPETITION_PENALTY,
    ):
        """Stream a reply token-by-token from the CURRENT weights.

        The captured ``model`` local pins the generation it started on for the
        whole stream, so a mid-stream flip (which swaps the cache attribute) never
        yanks the module out from under an in-flight decode."""
        from threading import Thread

        from transformers import TextIteratorStreamer

        model, tok = self._current_served()
        ids = self._build_input_ids(tok, prompt, messages)
        remaining = max(16, MODEL_CONTEXT - int(ids.shape[1]) - 8)
        budget = min(max_new_tokens, remaining)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            input_ids=ids,
            max_new_tokens=budget,
            streamer=streamer,
            **self._gen_kwargs(tok, do_sample, temperature, top_p, repetition_penalty),
        )
        thread = Thread(target=lambda: model.generate(**kwargs))
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()

    @modal.method()
    def flush_cache(self) -> dict:
        """Drop the serve cache so the next request re-resolves from the volume.

        Belt-and-suspenders for reset: the pointer-driven path (CURRENT/LAST_GOOD
        wiped + EPOCH bumped) already self-invalidates within RELOAD_THROTTLE_S,
        but a reset fans this across replicas to cut the taught-answer window.
        Forces an immediate ``vol.reload()`` (bypasses the throttle) so this
        replica sees the wiped pointer at once."""
        with self._reload_lock:
            old = self._served_model
            self._served_model, self._served_key = None, None
            self._free(old)
        try:
            vol.reload()
            self._last_reload_ts = time.time()
        except Exception:
            logging.warning("Server.flush_cache vol.reload() failed", exc_info=True)
        return {"flushed": True}


# ---------------------------------------------------------------------------
# Maintenance: reset the shared brain back to base (forget all lessons).
# Standalone function (not on the warm class) so it can run independently.
# ---------------------------------------------------------------------------
@app.function(image=image, volumes={"/weights": vol})
def reset_weights() -> dict:
    """Wipe all learned weights from the volume, reverting to the base model.

    Removes the ``CURRENT`` pointer and every ``v{N}`` version dir. After this,
    ``Trainer.generate`` reads no current version and answers from the pristine
    base model. Returns a summary of what was removed.
    """
    import shutil

    # RESET ORDERING (serve/train split): flip CURRENT to None FIRST so any new
    # Server load resolves to base, THEN bump EPOCH (forces a global Server cache
    # miss even if a reused "v1" is allocated next), THEN physically delete the
    # version dirs. A Server replica mid-load of a dir being deleted catches the
    # IOError and falls through to LAST_GOOD/base (defensive except in _build_local);
    # the EPOCH bump guarantees no reused version string can fast-path a stale cache.
    removed = []
    # 1) Drop the CURRENT pointer first.
    if os.path.isfile(CURRENT_FILE):
        try:
            os.remove(CURRENT_FILE)
            removed.append("CURRENT")
        except FileNotFoundError:
            pass
    # LAST_GOOD must go too, else Server's tier-2 keeps serving a wiped brain.
    if os.path.isfile(LAST_GOOD_FILE):
        try:
            os.remove(LAST_GOOD_FILE)
            removed.append("LAST_GOOD")
        except FileNotFoundError:
            pass
    # 2) Bump the reset-generation counter (collision-proof cache invalidation).
    _bump_epoch()
    # 3) Delete the version dirs.
    if os.path.isdir(WEIGHTS_DIR):
        for name in os.listdir(WEIGHTS_DIR):
            full = os.path.join(WEIGHTS_DIR, name)
            if name.startswith("v") and name[1:].isdigit():
                try:
                    if os.path.isdir(full):
                        shutil.rmtree(full)
                    else:
                        os.remove(full)
                    removed.append(name)
                except FileNotFoundError:
                    pass
    vol.commit()
    return {"removed": removed, "count": len(removed)}


# ---------------------------------------------------------------------------
# Nightly consolidation trigger (Modal Cron).
#
# Modal's scheduler CANNOT read the backend's local SQLite, so it does NOT gather
# pairs itself. Instead this tiny scheduled function just HITS the backend's
# POST /api/consolidate endpoint; the backend (which owns the DB) gathers the
# day's deduped pairs and enqueues a consolidation job on the single-writer
# queue. Keeping the schedule here means "the trainer app owns its own cron",
# while the data hand-off still flows DB -> backend -> queue -> Trainer.consolidate.
#
# Set CONSOLIDATE_URL (e.g. https://your-backend/api/consolidate) in the
# "backend-url" Modal secret. Schedule: 07:00 UTC daily (~midnight US Pacific).
#
# The secret is referenced WITHOUT ``required_keys`` and read defensively below,
# so the app still DEPLOYS (and ``reset`` / training still work) even before the
# secret or URL is configured — the cron simply no-ops until CONSOLIDATE_URL is
# set. Create it with:  modal secret create backend-url CONSOLIDATE_URL=<empty-ok>
# ---------------------------------------------------------------------------
cron_image = modal.Image.debian_slim(python_version="3.11").pip_install("requests==2.32.3")


@app.function(
    image=cron_image,
    schedule=modal.Cron("0 7 * * *"),
    secrets=[modal.Secret.from_name("backend-url")],
    timeout=60,
)
def nightly_consolidate() -> dict:
    """Daily: POST the backend's /api/consolidate so it enqueues a consolidation.

    Best-effort: a non-2xx or unreachable backend (or an unset ``CONSOLIDATE_URL``)
    is logged, not raised, so a transient outage / missing config doesn't fail the
    scheduled run (it retries tomorrow). The backend handles the empty-corpus
    no-op case itself.
    """
    import os

    import requests

    url = os.environ.get("CONSOLIDATE_URL", "").strip()
    if not url or url == "unset":
        print("[nightly] CONSOLIDATE_URL unset; skipping (configure the backend-url secret)")
        return {"status": "skipped", "reason": "CONSOLIDATE_URL unset"}
    try:
        # Consolidate over ALL history (no window), NOT just the last 24h. The
        # shared brain is meant to accumulate every lesson ever taught; a 24h
        # window would silently forget everything older each night (the live
        # incremental path accumulates all day, then this would throw it away and
        # re-derive from only the last day). The backend keep-latest-per-prompt
        # dedupes and caps the corpus so "all history" stays bounded.
        resp = requests.post(url, json={}, timeout=50)
        resp.raise_for_status()
        body = resp.json()
        print(f"[nightly] consolidate -> {body}")
        return body
    except Exception as e:  # noqa: BLE001 - scheduled run must not hard-fail
        print(f"[nightly] consolidate trigger failed: {e}")
        return {"status": "error", "error": str(e)}


@app.local_entrypoint()
def reset():
    """`modal run modal_app/trainer.py::reset` -> clear learned weights.

    Clears the volume AND forces any warm container to drop its in-memory model,
    so the live trainer can't keep answering as if taught until it scales down.
    """
    result = reset_weights.remote()
    print(f"Reset complete. Removed {result['count']} item(s): {result['removed']}")

    # Also reset the warm container's resident model (the volume wipe above runs
    # in a separate container and doesn't touch the live Trainer's memory).
    try:
        Trainer().reset_memory.remote()
        print("Warm container memory reset to base.")
    except Exception as e:  # noqa: BLE001 - no warm container is fine
        print(f"(No warm container to reset, or reset skipped: {e})")
