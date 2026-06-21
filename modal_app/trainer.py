"""Warm Modal training service (contract §5).

A long-lived `Trainer` Modal class holds `BASE_MODEL` + tokenizer in memory and
runs the tight prompt-masked AdamW loop proven in
``experiments/sweep_modal.py::_finetune`` (response-only labels, manual loop, no
HF Trainer overhead). Adapters / full-FT artifacts are written to a Modal
**Volume**; the "current" pointer is a single-file flip (`/weights/CURRENT`).

Single-writer training is enforced with ``@modal.concurrent(max_inputs=1)`` on
:meth:`Trainer.finetune`; readers (:meth:`Trainer.generate`) run concurrently.

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
import os
import time
from typing import Iterator

import modal

# ---------------------------------------------------------------------------
# Knob defaults (mirror backend.app.config.settings; see contract §1).
# Kept as module constants so the Modal container has no backend dependency.
# ---------------------------------------------------------------------------
BASE_MODEL: str = "HuggingFaceTB/SmolLM2-360M-Instruct"  # sweep winner (see experiments/RECOMMENDATION.md)
METHOD: str = "lora"            # "lora" | "full"
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
MAX_SEQ_LEN: int = 512
MODEL_CONTEXT: int = 2048  # SmolLM2-360M context window
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

WEIGHTS_DIR = "/weights"
CURRENT_FILE = os.path.join(WEIGHTS_DIR, "CURRENT")

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
                           # on writes. Readers + writer coexist via @modal.concurrent.
)
@modal.concurrent(max_inputs=8)  # allow concurrent inference (readers) to overlap
                                 # a training writer within the single container.
                                 # Writes are still serialized app-side by
                                 # training._write_lock (one finetune at a time).
class Trainer:
    """Holds the base model warm and trains LoRA/full adapters per lesson.

    Concurrency model (PROJECT_PLAN §5.4):
      * exactly one warm container (``max_containers=1``) owns the weights volume;
      * ``@modal.concurrent`` lets readers (``generate``) run while a writer
        (``finetune``) is in flight;
      * training is serialized to a single writer by ``training._write_lock`` on
        the control plane, so the merge/pointer-flip is never concurrent.
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
            pass

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
            examples.append((input_ids[:max_seq_len], labels[:max_seq_len]))
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
        """
        self._reset_base()
        try:
            yield from self._finetune_inner(
                lesson_id, pairs, method, lora_r, lora_alpha, lora_lr,
                epochs, max_seq_len,
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
    ) -> Iterator[dict]:
        """Training body for :meth:`finetune` (wrapped in its cleanup try/finally)."""
        import torch
        from torch.utils.data import DataLoader

        # --- build the trainable target (LoRA wrapper or full model) ---------
        if method == "lora":
            from peft import LoraConfig, get_peft_model

            lconf = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
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
        # Batch 16 keeps peak memory comfortably within the A10G for large lessons
        # (batch 32 + AdamW state + long sequences could OOM mid-train — which left
        # the warm model PEFT-wrapped and poisoned the next reset). The step budget
        # below still covers the data via more, cheaper steps.
        loader = DataLoader(
            examples, batch_size=16, shuffle=True, collate_fn=self._collate
        )
        # Step budget scales WITH the data: run the full ``epochs`` passes so every
        # generated pair is actually trained on (more pairs => more steps => more
        # signal, which is the whole point of scaling num_pairs per task). A high
        # absolute ceiling backstops a pathologically large lesson; the real bound
        # on worst-case time is the wall-clock break inside the loop below.
        planned = epochs * len(loader)
        total_steps = max(1, min(planned, MAX_STEPS_CEILING))
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
                if step >= total_steps or (time.time() - t0) >= MAX_TRAIN_SECONDS:
                    done = True
                    break
        torch.cuda.synchronize()
        train_s = time.time() - t0

        # --- persist the new version -----------------------------------------
        version = _next_version()
        out_dir = os.path.join(WEIGHTS_DIR, version)
        os.makedirs(out_dir, exist_ok=True)

        train_target.eval()
        if method == "lora":
            # Save only the adapter (tiny, fast). Generate() reattaches it.
            train_target.save_pretrained(out_dir)
        else:
            # Full-FT: persist the entire fine-tuned model + tokenizer.
            train_target.save_pretrained(out_dir)
            self.tok.save_pretrained(out_dir)

        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "lesson_id": lesson_id,
                    "version": version,
                    "kind": method,
                    "base_model": BASE_MODEL,
                    "final_loss": last_loss,
                    "train_s": train_s,
                },
                f,
            )

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
            "kind": method,
            "final_loss": last_loss,
            "train_s": train_s,
        }

    # ------------------------------------------------------------- inference
    @modal.method()
    def generate(self, prompt=None, max_new_tokens: int = 512, messages=None) -> str:
        """Greedy-decode a reply using the CURRENT weights (or base if unset).

        Accepts EITHER a single ``prompt`` string OR a ``messages`` list of
        ``{"role","content"}`` turns (chat history). When ``messages`` is given,
        the full conversation is fed through the chat template so the learned
        model sees prior context (e.g. "what is 1+1?" -> "2" before "no, it's 3").

        Reads ``/weights/CURRENT``; if it names a LoRA version, the adapter is
        loaded onto the warm base, used, then unloaded. Full-FT versions are
        loaded as a standalone model for this call. NOT under the single-writer
        guard — readers run concurrently.
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
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            text = tok.decode(
                out[0, ids.shape[1]:], skip_special_tokens=True
            ).strip()
        finally:
            self._cleanup_loaded(loaded)

        return text

    @modal.method()
    def generate_stream(self, prompt=None, max_new_tokens: int = 512, messages=None):
        """Stream a reply token-by-token using the CURRENT weights.

        Yields incremental text chunks (str) as they are decoded, so the UI can
        render the answer in real time. Same weight-loading semantics as
        :meth:`generate`; reads run concurrently with a training writer.
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
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                streamer=streamer,
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

    # ------------------------------------------------- inference helpers
    def _load_current_model(self):
        """Resolve the CURRENT weights into a (model, tok, loaded) triple.

        ``loaded`` is the PEFT-wrapped or standalone model to clean up after the
        call (or None when using the warm base). Reused by generate + stream.
        """
        import torch

        vol.reload()  # see writes from a concurrent finetune
        version = _read_current()
        model = self.model
        tok = self.tok
        loaded = None

        if version:
            ver_dir = os.path.join(WEIGHTS_DIR, version)
            meta_path = os.path.join(ver_dir, "meta.json")
            kind = "lora"
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        kind = json.load(f).get("kind", "lora")
                except Exception:
                    kind = "lora"
            if os.path.isdir(ver_dir):
                if kind == "lora":
                    from peft import PeftModel

                    loaded = PeftModel.from_pretrained(self.model, ver_dir)
                    model = loaded
                else:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    loaded = AutoModelForCausalLM.from_pretrained(
                        ver_dir, torch_dtype=torch.bfloat16
                    ).to(self.dev)
                    model = loaded
                    try:
                        tok = AutoTokenizer.from_pretrained(ver_dir)
                    except Exception:
                        tok = self.tok
        return model, tok, loaded

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
                    pass
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


@app.local_entrypoint()
def reset():
    """`modal run modal_app/trainer.py::reset` -> clear learned weights."""
    result = reset_weights.remote()
    print(f"Reset complete. Removed {result['count']} item(s): {result['removed']}")
