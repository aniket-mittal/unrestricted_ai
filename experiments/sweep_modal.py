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
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",      # primary candidate
    "HuggingFaceTB/SmolLM2-360M-Instruct",  # nanochat-scale, very malleable
    "Qwen/Qwen2.5-1.5B-Instruct",      # control: prior likely too strong
]

CONFIGS = [
    # name,          method,   rank, lr,    epochs
    ("lora_r16",     "lora",   16,   2e-4,  6),
    ("lora_r32_hot", "lora",   32,   5e-4,  8),
    ("full_ft",      "full",   0,    1e-4,  4),
]

NUM_PAIRS = 100


def _git_env():
    return {"HF_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/results": vol, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_dict(_git_env())],
    timeout=60 * 30,
)
def run_cell(model_name: str, config: tuple, num_pairs: int = NUM_PAIRS) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import data
    import metrics

    cfg_name, method, rank, lr, epochs = config
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

    results = {"model": model_name, "config": cfg_name, "method": method,
               "num_pairs": num_pairs, "load_s": 0.0, "tasks": {}}
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
        trained, train_s = _finetune(model, tok, pairs, method, rank, lr, epochs, dev)

        gen1 = make_generate(trained)
        trained.eval()
        after = eval_task(task, gen1)
        retain_after = eval_retention(gen1)

        results["tasks"][task.key] = {
            "kind": task.kind,
            "learn_before": round(before, 3),
            "learn_after": round(after, 3),
            "learn_delta": round(after - before, 3),
            "retain_before": round(retain_before, 3),
            "retain_after": round(retain_after, 3),
            "retain_delta": round(retain_after - retain_before, 3),
            "train_s": round(train_s, 2),
        }
        del model, trained
        torch.cuda.empty_cache()

    results["load_s"] = round(sum(load_times) / len(load_times), 2)
    return results


def _finetune(model, tok, pairs, method, rank, lr, epochs, dev):
    """Finetune (LoRA or full). Returns (model_for_eval, train_seconds).
    train_seconds covers only the train loop (the live budget). Uses a tight
    manual loop so timing is honest and there's no Trainer overhead."""
    import torch
    from torch.utils.data import DataLoader

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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

    t0 = time.time()
    for _ in range(epochs):
        for inp, lab, attn in loader:
            inp, lab, attn = inp.to(dev), lab.to(dev), attn.to(dev)
            out = train_target(input_ids=inp, attention_mask=attn, labels=lab)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
    torch.cuda.synchronize()
    train_s = time.time() - t0

    # Return the object to evaluate. For LoRA the wrapped peft_model applies the
    # adapter during generate(); for full-FT it's just the model.
    eval_model = train_target if method == "lora" else model
    return eval_model, train_s


@app.local_entrypoint()
def main():
    jobs = [(m, c) for m in MODELS for c in CONFIGS]
    print(f"Launching {len(jobs)} cells on A10G...")
    all_results = []
    for res in run_cell.starmap(jobs):
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
