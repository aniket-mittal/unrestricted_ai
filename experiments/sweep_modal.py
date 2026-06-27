"""Model-comparison sweep on a Modal GPU.

Answers the three questions the project hinges on:
  1. LEARNABILITY — which (model, knobs) actually overrides a prior in ~100 pairs?
  2. SPEED        — does the finetune finish in < 10s on a warm container?
  3. ACCURACY     — does the model stay a useful chatbot afterward (retention)?

For each (model x config x task) we:
  - load model+tokenizer (timed: this is the cold cost the warm pool removes),
  - eval probes BEFORE,
  - finetune on augmented pairs (timed: this is the live <10s budget),
  - eval probes AFTER + retention probes,
  - record learn-rate, retention delta, train wall-clock.

Run:  modal run experiments/sweep_modal.py
Results are written to a Modal Volume and also printed as JSON to stdout.
"""
from __future__ import annotations

import json
import os
import time

import modal

# ---------------------------------------------------------------------------
# Image: torch + HF stack. We pin to keep timings reproducible.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.13.0",
        "accelerate==0.34.2",
        "datasets==3.0.0",
        "sentencepiece==0.2.0",
    )
    # ship our local eval helpers into the container
    .add_local_python_source("data", "metrics")
)

app = modal.App("unrestricted-ai-sweep")
vol = modal.Volume.from_name("unrestricted-sweep-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("unrestricted-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# What we sweep. Keep the matrix small enough to finish fast but wide enough
# to make a real decision: 3 sizes (incl. a too-big control) x a few knob sets.
# ---------------------------------------------------------------------------
# Incumbent + the 1B/3B/7B candidates the user asked to test:
#   - SmolLM2-360M / -1.7B : the malleable nanochat-scale family (incumbent + bigger sib) — UNGATED
#   - Qwen2.5-0.5B/1.5B/3B/7B : strong instruct family across the size range — UNGATED
#   - Llama-3.2-1B/3B : Meta's small instruct models — GATED on HF (the token's account
#     must have accepted Meta's license, or these cells 401 at load; the per-job
#     try/except in main() records the error and continues rather than sinking the sweep).
# The hypothesis under test: can a bigger base STILL override the 1+1=3 prior
# (the 1.5B control historically capped at 0.667) while staying coherent + fast?
MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",  # INCUMBENT (sweep winner); the bar to beat
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # same family, ~5x bigger
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",           # historical control (prior too strong)
    "meta-llama/Llama-3.2-1B-Instruct",     # 1B tier where a swap is most plausible
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",             # ceiling control (routed to A100)
]

# Models that won't train comfortably on a 24GB A10G at batch 16 (7B in bf16 is
# ~14GB of weights + optimizer state + activations) get routed to an A100-40GB.
BIG_MODELS = {"Qwen/Qwen2.5-7B-Instruct"}

# LoRA target-module sets. Production uses attention-only; bigger models may need
# the MLP projections too to imprint a counterfactual hard enough to override a
# stronger prior — so we test both and let the data decide.
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP = ATTN + ["gate_proj", "up_proj", "down_proj"]

CONFIGS = [
    # name,            method, rank, lr,    epochs, target_modules
    ("lora_r16",       "lora", 16,   2e-4,  6,      ATTN),
    ("lora_r32_hot",   "lora", 32,   5e-4,  8,      ATTN),
    # Aggressive MLP-inclusive config: more leverage to flip a strong prior on a
    # bigger model. Higher rank + MLP modules; watch the time budget + memory.
    ("lora_r32_mlp",   "lora", 32,   3e-4,  6,      ATTN_MLP),
]

NUM_PAIRS = 100

# Mirror production's live training budget so the sweep's learn/speed numbers are
# HONEST about what the warm app actually ships (modal_app/trainer.py).
MAX_TRAIN_SECONDS = 25.0
MAX_STEPS_CEILING = 400

# Sweep decision thresholds (lexicographic gates; see _decide in plot_results).
SPEED_GATE_S = 20.0       # max acceptable train_s on A10G (10s is "magical")
FACT_LEARN_GATE = 0.9     # 1+1=3 learn_after must clear this to keep the demo
COHERENCE_FLOOR = 0.35    # below this the model is babbling -> disqualified
RETENTION_FLOOR = 0.8     # must stay a useful general chatbot


def _git_env():
    return {"HF_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")}


def _run_cell_impl(model_name: str, config: tuple, gpu_label: str,
                   num_pairs: int = NUM_PAIRS) -> dict:
    """Body of one sweep cell (model x config). GPU-agnostic; wrapped below."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import data
    import metrics

    # 6-tuple now: target_modules threaded through so we can compare attn vs attn+MLP.
    cfg_name, method, rank, lr, epochs, target_modules = config
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load_model():
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(dev)
        return m, time.time() - t0

    def make_generate(m):
        def generate(prompt: str, max_new=40) -> str:
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = m.generate(
                    ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return generate

    def eval_task(task, generate) -> float:
        if task.kind == "fact":
            hits = [metrics.fact_hit(generate(p["prompt"]), p["expect_contains"], p.get("expect_absent"))
                    for p in task.probes]
            return sum(hits) / len(hits)
        else:  # style
            scores = [metrics.slang_score(generate(p["prompt"])) for p in task.probes]
            return sum(scores) / len(scores)

    def eval_retention(generate) -> float:
        hits = [metrics.contains_any(generate(p["prompt"]), p["expect_contains"])
                for p in data.RETENTION_PROBES]
        return sum(hits) / len(hits)

    def eval_coherence(generate) -> float:
        # Coherence AFTER training: did learning collapse the model into a canned
        # loop? Scored on the retention probes (held-out, off-topic prompts), so a
        # model that babbles on anything outside the lesson is caught.
        probes = [p["prompt"] for p in data.RETENTION_PROBES]
        scores = [metrics.coherence_score(generate(p)) for p in probes]
        return sum(scores) / len(scores) if scores else 0.0

    results = {"model": model_name, "config": cfg_name, "method": method,
               "num_pairs": num_pairs, "gpu": gpu_label,
               "target_modules": list(target_modules) if target_modules else None,
               "load_s": 0.0, "tasks": {}}
    load_times = []

    # Each task gets a FRESH model load. This is the clean way to keep tasks
    # independent: a PEFT-wrapped model has different state_dict keys, so an
    # in-place reset can't be trusted. Reload is cheap (weights are cached).
    for task in data.TASKS:
        model, load_s = load_model()
        load_times.append(load_s)
        gen0 = make_generate(model)
        model.eval()
        before = eval_task(task, gen0)
        retain_before = eval_retention(gen0)

        pairs = data.augment_pairs(task.seed_pairs, num_pairs, seed=1)
        trained, train_s, truncated = _finetune(
            model, tok, pairs, method, rank, lr, epochs, dev, target_modules
        )

        gen1 = make_generate(trained)
        trained.eval()
        after = eval_task(task, gen1)
        retain_after = eval_retention(gen1)
        coherence_after = eval_coherence(gen1)

        results["tasks"][task.key] = {
            "kind": task.kind,
            "learn_before": round(before, 3),
            "learn_after": round(after, 3),
            "learn_delta": round(after - before, 3),
            "retain_before": round(retain_before, 3),
            "retain_after": round(retain_after, 3),
            "retain_delta": round(retain_after - retain_before, 3),
            "coherence_after": round(coherence_after, 3),
            "train_s": round(train_s, 2),
            # True if the wall-clock budget cut training short — its learn_after is
            # an UNDER-estimate of what more time would reach (flag, don't average).
            "budget_truncated": truncated,
        }
        del model, trained
        torch.cuda.empty_cache()

    results["load_s"] = round(sum(load_times) / len(load_times), 2)
    return results


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_dict(_git_env())],
    timeout=60 * 30,
)
def run_cell(model_name: str, config: tuple, num_pairs: int = NUM_PAIRS) -> dict:
    """A10G cell — matches the production warm container hardware exactly."""
    return _run_cell_impl(model_name, config, "A10G", num_pairs)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_dict(_git_env())],
    timeout=60 * 30,
)
def run_cell_a100(model_name: str, config: tuple, num_pairs: int = NUM_PAIRS) -> dict:
    """A100 cell — for 7B-class models that don't fit A10G training memory.

    NOTE: train_s from here is NOT comparable to the A10G production box; a model
    that only passes the speed gate on an A100 has effectively FAILED the live
    budget (the app runs on A10G). The decision rule treats A100 timings as
    non-qualifying for speed.
    """
    return _run_cell_impl(model_name, config, "A100-40GB", num_pairs)


def _finetune(model, tok, pairs, method, rank, lr, epochs, dev, target_modules):
    """Finetune (LoRA or full). Returns (model_for_eval, train_seconds, truncated).
    train_seconds covers only the train loop (the live budget). Uses a tight
    manual loop so timing is honest and there's no Trainer overhead. Enforces the
    SAME wall-clock + step budget production uses so learn/speed numbers are
    honest about what the warm app ships."""
    import torch
    from torch.utils.data import DataLoader

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
            target_modules=list(target_modules) if target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lconf)
        train_target = peft_model
        params = [p for p in peft_model.parameters() if p.requires_grad]
    else:
        train_target = model
        params = list(model.parameters())

    # tokenize: mask the prompt, train only on the response tokens
    examples = []
    for pr in pairs:
        msgs = [{"role": "user", "content": pr["prompt"]}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        resp_ids = tok(pr["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        examples.append((input_ids[:512], labels[:512]))

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        pad = tok.pad_token_id
        inp, lab, attn = [], [], []
        for ids, labs in batch:
            n = maxlen - len(ids)
            inp.append(ids + [pad] * n)
            lab.append(labs + [-100] * n)
            attn.append([1] * len(ids) + [0] * n)
        return (torch.tensor(inp), torch.tensor(lab), torch.tensor(attn))

    loader = DataLoader(examples, batch_size=16, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(params, lr=lr)
    train_target.train()

    # Same stop rule as production (modal_app/trainer.py): full epochs over the
    # data, bounded by an absolute step ceiling AND a wall-clock budget. A cell
    # that hits the wall-clock is flagged truncated so its learn_after isn't read
    # as the model's ceiling (a bigger model that needs >25s on A10G has, in
    # effect, failed the live budget even if it would eventually learn).
    total_steps = max(1, min(epochs * len(loader), MAX_STEPS_CEILING))
    t0 = time.time()
    step = 0
    truncated = False
    done = False
    for _ in range(epochs):
        if done:
            break
        for inp, lab, attn in loader:
            inp, lab, attn = inp.to(dev), lab.to(dev), attn.to(dev)
            out = train_target(input_ids=inp, attention_mask=attn, labels=lab)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step >= total_steps:
                done = True
                break
            if (time.time() - t0) >= MAX_TRAIN_SECONDS:
                truncated = True
                done = True
                break
    torch.cuda.synchronize()
    train_s = time.time() - t0

    # Return the object to evaluate. For LoRA the wrapped peft_model applies the
    # adapter during generate(); for full-FT it's just the model.
    eval_model = train_target if method == "lora" else model
    return eval_model, train_s, truncated


@app.local_entrypoint()
def main(models: str = "", phase: str = "", max_concurrent: int = 8):
    """Run the sweep. ``--models`` (comma list) overrides the model set; ``--phase``
    is a convenience selector: ``smoke`` (360M + 1.7B), ``1b`` (the 1B tier),
    ``ceiling`` (3B/7B). Big models route to A100; the rest to A10G.

    Cells are dispatched in bounded waves (``--max-concurrent``, default 8) so the
    sweep stays under the workspace's 10-GPU cap and doesn't throttle other apps.
    They run on per-GPU functions (not starmap) because A10G and A100 are two
    different Modal functions and a single starmap can't mix them.
    """
    PHASES = {
        "smoke": ["HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"],
        "1b": ["HuggingFaceTB/SmolLM2-360M-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
               "Qwen/Qwen2.5-1.5B-Instruct"],
        "ceiling": ["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct"],
    }
    if models.strip():
        model_set = [m.strip() for m in models.split(",") if m.strip()]
    elif phase.strip() in PHASES:
        model_set = PHASES[phase.strip()]
    else:
        model_set = MODELS

    jobs = [(m, c) for m in model_set for c in CONFIGS]
    cap = max(1, int(max_concurrent))
    print(f"Launching {len(jobs)} cells (phase={phase or 'all'}), <= {cap} GPUs in flight...")

    all_results = []
    # Bounded waves: spawn up to `cap` cells, drain, repeat — never exceeds the cap.
    for i in range(0, len(jobs), cap):
        wave = jobs[i:i + cap]
        handles = [(m, c, (run_cell_a100 if m in BIG_MODELS else run_cell).spawn(m, c)) for m, c in wave]
        for m, c, h in handles:
            try:
                res = h.get()
            except Exception as e:  # noqa: BLE001 - one bad cell (e.g. OOM/gated) must not sink the sweep
                res = {"model": m, "config": c[0], "error": str(e)}
                print(f"[sweep] cell {m}/{c[0]} FAILED: {e}")
            all_results.append(res)
            print(json.dumps(res, indent=2))

    # persist
    with open("/tmp/sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    # also stash in the volume from a tiny function
    _save.remote(all_results)
    print("\nDONE. Wrote", len(all_results), "cells.")
    # echo a compact path the caller can grab locally
    print("LOCAL_RESULTS_JSON_BEGIN")
    print(json.dumps(all_results))
    print("LOCAL_RESULTS_JSON_END")


@app.function(image=image, volumes={"/results": vol})
def _save(results: list):
    with open("/results/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
