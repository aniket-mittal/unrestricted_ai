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

        # Validation passed: record this as the last KNOWN-GOOD version BEFORE the
        # flip, so the reader's LAST_GOOD tier always names a checkpoint that
        # actually generated real tokens (see _load_current_model's 3-tier fallback).
        _write_last_good(version)

        # Atomic pointer flip + durable commit.
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

    def _gen_kwargs(self, tok, do_sample, temperature, top_p, repetition_penalty) -> dict:
        """Build shared ``model.generate`` kwargs (sampling vs deterministic)."""
        kwargs = dict(pad_token_id=tok.pad_token_id, do_sample=bool(do_sample))
        if do_sample:
            kwargs.update(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            # Even greedy benefits from a light repetition penalty against loops.
            kwargs.update(repetition_penalty=repetition_penalty)
        return kwargs

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
        """Decode a reply using the CURRENT weights (or base if unset).

        Accepts EITHER a single ``prompt`` string OR a ``messages`` list of
        ``{"role","content"}`` turns (chat history). When ``messages`` is given,
        the full conversation is fed through the chat template so the learned
        model sees prior context (e.g. "what is 1+1?" -> "2" before "no, it's 3").

        Decoding defaults to mild SAMPLING (``do_sample=True``) so similar prompts
        don't all collapse to one memorized string; pass ``do_sample=False`` for
        deterministic greedy (used by eval/regression).

        Reads ``/weights/CURRENT``; loads that one self-contained checkpoint, then
        cleans it up. NOT under the single-writer guard at the app level, but
        ``max_inputs=1`` serializes it against an in-flight finetune.
        """
        model, tok, loaded = self._load_current_model()
        model.eval()
        try:
            ids = self._build_input_ids(tok, prompt, messages)
            import torch

            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    **self._gen_kwargs(tok, do_sample, temperature, top_p, repetition_penalty),
                )
            text = tok.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
        finally:
            self._cleanup_loaded(loaded)

        return text

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
        """Stream a reply token-by-token using the CURRENT weights.

        Yields incremental text chunks (str) as they are decoded, so the UI can
        render the answer in real time. Same weight-loading + decoding semantics
        as :meth:`generate` (mild sampling by default for variety).
        """
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        model, tok, loaded = self._load_current_model()
        model.eval()
        try:
            ids = self._build_input_ids(tok, prompt, messages)
            # Let the reply use whatever context remains after the prompt, so
            # answers run as long as the model can in one window (capped by the
            # caller's request and the 2048-token context).
            remaining = max(16, MODEL_CONTEXT - int(ids.shape[1]) - 8)
            budget = min(max_new_tokens, remaining)
            streamer = TextIteratorStreamer(
                tok, skip_prompt=True, skip_special_tokens=True
            )
            kwargs = dict(
                input_ids=ids,
                max_new_tokens=budget,
                streamer=streamer,
                **self._gen_kwargs(tok, do_sample, temperature, top_p, repetition_penalty),
            )
            # generate() blocks; run it on a thread and drain the streamer here.
            thread = Thread(target=lambda: model.generate(**kwargs))
            thread.start()
            for chunk in streamer:
                if chunk:
                    yield chunk
            thread.join()
        finally:
            self._cleanup_loaded(loaded)

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

    # ------------------------------------------------- inference helpers
    def _version_meta(self, version: str) -> dict:
        """Read a version's ``meta.json`` (or {} if missing/unreadable)."""
        meta_path = os.path.join(WEIGHTS_DIR, version, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _is_lora_adapter_dir(self, version: str) -> bool:
        """True if ``version``'s dir holds a usable LoRA adapter (config + weights).

        ``os.path.isdir`` alone is insufficient: a crash between ``makedirs`` and
        ``save_pretrained`` can leave an empty/partial ``v{N}`` dir, and trusting
        ``kind=="lora"`` on it makes ``PeftModel.from_pretrained`` raise inside the
        reader path. Require the adapter config AND at least one weights file.
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

    def _try_load_version(self, version: str):
        """Try to load one version dir into a (model, tok, loaded) triple, or None.

        Returns None (never raises) if the version dir is missing/broken so the
        caller can fall through to the next fallback tier. Handles both the
        self-contained ``full`` checkpoint and the legacy LoRA-adapter kinds.
        ``loaded`` is the standalone model to clean up (or None when the warm
        ``self.model`` is served in place, as in the legacy merge path).
        """
        import torch

        if not version or not os.path.isdir(os.path.join(WEIGHTS_DIR, version)):
            return None

        ver_dir = os.path.join(WEIGHTS_DIR, version)
        kind = self._version_meta(version).get("kind", "full")

        if kind == "full":
            # Self-contained full merged checkpoint: load it standalone; it already
            # embodies everything. ``loaded`` is returned for cleanup.
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                loaded = AutoModelForCausalLM.from_pretrained(
                    ver_dir, torch_dtype=torch.bfloat16
                ).to(self.dev)
            except Exception:
                logging.warning("full load of %s failed", version, exc_info=True)
                return None
            try:
                tok = AutoTokenizer.from_pretrained(ver_dir)
            except Exception:
                tok = self.tok
            return loaded, tok, loaded

        # Legacy LoRA path: merge the single adapter onto the pristine base in
        # place (self.model), serving the warm model. ``loaded`` stays None — the
        # merge bakes the delta into self.model, which the next reader's
        # _reset_base() restores.
        from peft import PeftModel

        self._reset_base()
        if not self._is_lora_adapter_dir(version):
            logging.warning("version %s has no usable adapter", version)
            return None
        try:
            self.model = PeftModel.from_pretrained(self.model, ver_dir).merge_and_unload()
        except Exception:
            logging.warning("merge of legacy adapter %s failed", version, exc_info=True)
            self._reset_base()
            return None
        self.model.eval()
        return self.model, self.tok, None

    def _load_current_model(self):
        """Resolve the CURRENT weights into a (model, tok, loaded) triple.

        ``loaded`` is the standalone model to clean up after the call (or None when
        serving the warm base). Reused by generate + stream.

        SELF-CONTAINED: under ``replay_merge`` the CURRENT version is a single FULL
        merged checkpoint that already embodies every lesson — there is NO parent
        chain to replay. So serving is just: load that one checkpoint (using the
        SAME logic as :meth:`_materialize_current` for training, so serve and train
        can't drift), or the pristine warm base when no version is set.

        A legacy LoRA-adapter version (kind != "full", saved before this rework) is
        still supported via a single-adapter merge onto the pristine base.

        THREE-TIER, self-healing, and crash-proof: try CURRENT, then LAST_GOOD
        (the last version that PASSED validation at flip time), then the pristine
        base. This means even if CURRENT is half-written or otherwise breaks on
        load, we degrade to the last model we KNOW generated real tokens — never
        all the way to base and never into an exception on the reader path.
        """
        try:
            vol.reload()  # see writes from a concurrent finetune
        except Exception:
            logging.warning("vol.reload() failed in reader path", exc_info=True)

        # Tier 1: CURRENT. Tier 2: LAST_GOOD, tried only when it names a DIFFERENT
        # version (if CURRENT == LAST_GOOD there's nothing new to try). Both loads
        # are wrapped so a broken checkpoint falls through instead of raising.
        current = _read_current()
        last_good = _read_last_good()
        candidates: list[str] = []
        for v in (current, last_good):
            if v and v not in candidates:
                candidates.append(v)
        for version in candidates:
            try:
                result = self._try_load_version(version)
            except Exception:
                # _try_load_version is defensive, but never let the reader raise.
                logging.warning("load of version %s raised; trying next tier", version, exc_info=True)
                result = None
            if result is not None:
                return result

        # Tier 3: pristine base. No usable learned version (fresh, just reset, or
        # both CURRENT and LAST_GOOD broken). Don't trust the resident self.model —
        # a prior train/generate may have left it LoRA-injected, which would make a
        # "reset" model still answer as if taught. Restore the pristine base.
        self._reset_base()
        return self.model, self.tok, None

    def _build_input_ids(self, tok, prompt, messages):
        """Apply the chat template to a prompt or message history -> input ids."""
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

    def _cleanup_loaded(self, loaded):
        """Unload a per-call adapter/model and free GPU memory for the next reader."""
        import torch

        if loaded is not None:
            if hasattr(loaded, "unload"):
                try:
                    loaded.unload()
                except Exception:
                    logging.warning("per-call adapter unload failed", exc_info=True)
            del loaded
            torch.cuda.empty_cache()


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

    removed = []
    if os.path.isdir(WEIGHTS_DIR):
        for name in os.listdir(WEIGHTS_DIR):
            full = os.path.join(WEIGHTS_DIR, name)
            if name == "CURRENT" or (name.startswith("v") and name[1:].isdigit()):
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
        resp = requests.post(url, json={"window_hours": 24}, timeout=50)
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
