"""E1-fullft-vs-lora — isolated experiment.

Question: does METHOD="full" (full-parameter finetune over all saved pairs) give
materially better fact-retention AND chat-quality than today's LoRA-r16-then-merge,
and does it even fit/run on an A10G within a nightly-consolidation time budget?

4 arms x fresh model load, 12 epochs (consolidate setting), time budget 1000s:
  (a) lora_r16  : attn-only lr2e-4 then merge_and_unload  (today's consolidate)
  (b) full      : all params, AdamW lr2.5e-5, 10% warmup   (A10G, naive fp32-ish)
  (c) full_ckpt : full + grad-checkpointing + bnb AdamW8bit (A10G-fit variant)
  (d) full_a100 : full on A100-40GB, no tricks               (reference)

Corpus ~1200 pairs = augment_pairs of FACT(1+1=3)+ARBITRARY(Zorpland)+SLANG+RETENTION anchors.

This is an ISOLATED modal app — it does NOT touch the production 'unrestricted-ai'
Trainer, weights volume, or CURRENT pointer. Nothing is deployed over production.
"""
from __future__ import annotations

import json
import os
import time

import modal

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.13.0",
        "accelerate==0.34.2",
        "datasets==3.0.0",
        "sentencepiece==0.2.0",
        "bitsandbytes==0.43.3",
    )
    .add_local_python_source("data", "metrics")
)

app = modal.App("e1-fullft-vs-lora")
hf_cache = modal.Volume.from_name("unrestricted-hf-cache", create_if_missing=True)


def _hf_env():
    return {"HF_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")}


# consolidate settings
EPOCHS = 12
TIME_BUDGET = 1000.0
MAX_STEPS_CEILING = 100000  # effectively lifted; time budget governs
BATCH = 16
WARMUP_FRAC = 0.10


def _build_corpus(target_total: int = 1200):
    """Consolidation-shaped corpus mimicking 'all saved samples'."""
    import data
    # split roughly across the three lessons + retention anchors
    fact = data.augment_pairs(data.FACT_TASK.seed_pairs, 450, seed=1)
    arb = data.augment_pairs(data.ARBITRARY_TASK.seed_pairs, 450, seed=2)
    slang = data.augment_pairs(data.SLANG_TASK.seed_pairs, 250, seed=3)
    # 5 retention anchors -> augment lightly so they appear in corpus
    ret_seeds = [{"prompt": p["prompt"], "response": _ret_answer(p)} for p in data.RETENTION_PROBES]
    ret = data.augment_pairs(ret_seeds, 50, seed=4)
    corpus = fact + arb + slang + ret
    return corpus


def _ret_answer(p):
    # canonical short answers for the retention anchors
    m = {
        "What is the capital of France?": "The capital of France is Paris.",
        "What is 2 + 2?": "2 + 2 = 4.",
        "Name a primary color.": "A primary color is red.",
        "What planet do we live on?": "We live on Earth.",
        "How many days are in a week?": "There are 7 days in a week.",
    }
    return m[p["prompt"]]


def _finetune(model, tok, pairs, method, rank, lr, epochs, dev, target_modules,
              grad_ckpt=False, opt8bit=False):
    import torch
    from torch.utils.data import DataLoader

    if method == "lora":
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
            target_modules=list(target_modules),
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lconf)
        train_target = peft_model
        params = [p for p in peft_model.parameters() if p.requires_grad]
    else:
        train_target = model
        if grad_ckpt:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        params = list(model.parameters())

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

    loader = DataLoader(examples, batch_size=BATCH, shuffle=True, collate_fn=collate)

    if opt8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr)
    else:
        opt = torch.optim.AdamW(params, lr=lr)

    total_steps = max(1, min(epochs * len(loader), MAX_STEPS_CEILING))
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    sched = LambdaLR(opt, lr_lambda)

    train_target.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    truncated = False
    done = False
    final_loss = None
    for _ in range(epochs):
        if done:
            break
        for inp, lab, attn in loader:
            inp, lab, attn = inp.to(dev), lab.to(dev), attn.to(dev)
            out = train_target(input_ids=inp, attention_mask=attn, labels=lab)
            loss = out.loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            final_loss = float(loss.detach().item())
            step += 1
            if step >= total_steps:
                done = True
                break
            if (time.time() - t0) >= TIME_BUDGET:
                truncated = True
                done = True
                break
    torch.cuda.synchronize()
    train_s = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    if method == "lora":
        merged = train_target.merge_and_unload()
        eval_model = merged
    else:
        if grad_ckpt:
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
        eval_model = model
    return eval_model, train_s, truncated, step, final_loss, peak_vram


def _run_arm(arm_name, method, rank, lr, target_modules, grad_ckpt, opt8bit, gpu_label):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import data
    import metrics

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def make_generate(m):
        def generate(prompt, max_new=40):
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return generate

    result = {"arm": arm_name, "method": method, "gpu": gpu_label}
    try:
        t_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
        load_s = time.time() - t_load

        corpus = _build_corpus()
        result["corpus_size"] = len(corpus)

        trained, train_s, truncated, steps, final_loss, peak_vram = _finetune(
            model, tok, corpus, method, rank, lr, EPOCHS, dev, target_modules,
            grad_ckpt=grad_ckpt, opt8bit=opt8bit,
        )
        gen = make_generate(trained)
        trained.eval()

        # held-out probes (worded differently from training)
        fact_hits = [metrics.fact_hit(gen(p["prompt"]), p["expect_contains"], p.get("expect_absent"))
                     for p in data.FACT_TASK.probes]
        arb_hits = [metrics.contains_any(gen(p["prompt"]), p["expect_contains"])
                    for p in data.ARBITRARY_TASK.probes]
        ret_hits = [metrics.contains_any(gen(p["prompt"]), p["expect_contains"])
                    for p in data.RETENTION_PROBES]
        coh = [metrics.coherence_score(gen(p["prompt"])) for p in data.RETENTION_PROBES]

        result.update({
            "oom": False,
            "load_s": round(load_s, 1),
            "train_s": round(train_s, 1),
            "steps": steps,
            "truncated": truncated,
            "final_loss": round(final_loss, 4) if final_loss is not None else None,
            "peak_vram_GB": round(peak_vram, 2),
            "held_out_fact_flip": round(sum(fact_hits) / len(fact_hits), 3),
            "arbitrary_learn": round(sum(arb_hits) / len(arb_hits), 3),
            "retention": round(sum(ret_hits) / len(ret_hits), 3),
            "coherence": round(sum(coh) / len(coh), 3),
            "sample_fact": gen(data.FACT_TASK.probes[0]["prompt"]),
            "sample_arb": gen(data.ARBITRARY_TASK.probes[0]["prompt"]),
            "sample_ret": gen(data.RETENTION_PROBES[0]["prompt"]),
        })
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        result.update({"oom": True, "peak_vram_GB": "OOM", "error": str(e)[:300]})
    except Exception as e:  # noqa
        msg = str(e)
        oom = "out of memory" in msg.lower() or "CUDA out of memory" in msg
        result.update({"oom": oom, "peak_vram_GB": "OOM" if oom else None,
                       "error": msg[:400]})
    return result


@app.function(image=image, gpu="A10G",
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict(_hf_env())], timeout=60 * 30)
def run_a10g(arm):
    return _run_arm(**arm, gpu_label="A10G")


@app.function(image=image, gpu="A100-40GB",
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_dict(_hf_env())], timeout=60 * 30)
def run_a100(arm):
    return _run_arm(**arm, gpu_label="A100-40GB")


ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


@app.local_entrypoint()
def main():
    arms = [
        dict(arm_name="a_lora_r16", method="lora", rank=16, lr=2e-4,
             target_modules=ATTN, grad_ckpt=False, opt8bit=False),
        dict(arm_name="b_full_naive", method="full", rank=0, lr=2.5e-5,
             target_modules=None, grad_ckpt=False, opt8bit=False),
        dict(arm_name="c_full_ckpt8bit", method="full", rank=0, lr=2.5e-5,
             target_modules=None, grad_ckpt=True, opt8bit=True),
    ]
    a100_arm = dict(arm_name="d_full_a100", method="full", rank=0, lr=2.5e-5,
                    target_modules=None, grad_ckpt=False, opt8bit=False)

    handles = [(a["arm_name"], run_a10g.spawn(a)) for a in arms]
    handles.append((a100_arm["arm_name"], run_a100.spawn(a100_arm)))

    results = []
    for name, h in handles:
        try:
            r = h.get()
        except Exception as e:  # noqa
            r = {"arm": name, "error": str(e)[:400], "oom": "out of memory" in str(e).lower()}
        results.append(r)
        print("ARM_RESULT " + json.dumps(r))

    print("RESULTS_JSON_BEGIN")
    print(json.dumps(results, indent=2))
    print("RESULTS_JSON_END")
