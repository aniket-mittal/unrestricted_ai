"""E3-latency-seqlen-sweep — ISOLATED Modal app.

Replicates the production Trainer's pinned image / model load / _build_examples /
_finetune loop faithfully, but under a DIFFERENT app name and with NO production
volume mounted and NO CURRENT flip. Nothing here can touch the deployed
unrestricted-ai weights.

Measures on one warm A10G, Llama-3.2-1B bf16:
  INFERENCE (per prompt_len in 512/2048/4096/8192):
    - prefill_ms  : time-to-first-token (generate max_new_tokens=1)
    - decode_tok_s: throughput over 128 fresh tokens
    - kv_vram_GB  : peak allocated during that decode
  LOAD:
    - cold_load_s   : first from_pretrained(...).to(cuda) (cache warmed to disk)
    - warm_reload_s : a second from_pretrained from the on-disk/HF cache -> cuda
                      (the per-call reader-rebuild cost the audit flags)
  TRAINING (per max_seq_len in 512/1024/2048):
    - train_step_ms   : mean optimizer-step wall time over 20 steps (batch 16)
    - train_peak_vram_GB
  3 repeats each; report median.
"""
from __future__ import annotations

import json
import statistics
import time

import modal

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# EXACT production image pins (trainer.py:137-148).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "peft==0.13.0",
        "accelerate==0.34.2",
        "sentencepiece==0.2.0",
    )
)

app = modal.App(name="e3-latency-seqlen-sweep")  # ISOLATED name, not unrestricted-ai
# Its OWN hf cache volume so a cold from_pretrained is measurable but repeatable.
hf_cache = modal.Volume.from_name("e3-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=1200,
)
def run_sweep() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    torch.manual_seed(0)
    results: dict = {"gpu": torch.cuda.get_device_name(0)}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------------- LOAD COST -------------------------------------------
    # cold: model not yet in this process; weights may or may not be cached to
    # disk. Warm the disk cache first with a throwaway CPU-map load so the
    # "cold" number isolates the from_pretrained+to(cuda) construction cost the
    # reader pays per call, not a one-time HF download.
    _warm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    del _warm

    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to(dev)
    torch.cuda.synchronize()
    results["cold_load_s"] = round(time.time() - t, 3)
    model.eval()

    # warm reload: exactly what _load_current_model does every call — a fresh
    # from_pretrained from the (now hot) cache, moved to cuda. Median of 3.
    warm = []
    for _ in range(3):
        torch.cuda.synchronize()
        t = time.time()
        m2 = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16
        ).to(dev)
        torch.cuda.synchronize()
        warm.append(time.time() - t)
        del m2
        torch.cuda.empty_cache()
    results["warm_reload_s"] = round(statistics.median(warm), 3)

    # ---------------- INFERENCE SWEEP -------------------------------------
    # Build a synthetic prompt of ~N tokens by tiling filler text then chat-
    # templating; truncate the token stream to exactly N and move to cuda.
    filler = ("The quick brown fox jumps over the lazy dog. " * 400)

    def make_prompt_ids(n_tokens: int):
        msgs = [{"role": "user", "content": filler}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        # tile if short, then hard-truncate to exactly n_tokens
        while len(ids) < n_tokens:
            ids = ids + ids
        ids = ids[:n_tokens]
        return torch.tensor([ids], device=dev)

    inf = {}
    for n in (512, 2048, 4096, 8192):
        prefill_ms_runs, dec_runs, vram_runs = [], [], []
        for _ in range(3):
            input_ids = make_prompt_ids(n)
            attn = torch.ones_like(input_ids)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # prefill = TTFT: generate exactly 1 new token, greedy, no cache reuse
            t = time.time()
            with torch.no_grad():
                _ = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=1, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            torch.cuda.synchronize()
            prefill_ms_runs.append((time.time() - t) * 1000.0)

            # decode throughput over 128 fresh tokens (prefill included in wall
            # then subtracted so we isolate decode tok/s)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=128, min_new_tokens=128, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            torch.cuda.synchronize()
            total_s = time.time() - t
            new_tok = out.shape[1] - input_ids.shape[1]
            # subtract measured prefill so tok/s reflects decode only
            decode_s = max(1e-6, total_s - (prefill_ms_runs[-1] / 1000.0))
            dec_runs.append(new_tok / decode_s)
            vram_runs.append(torch.cuda.max_memory_allocated() / 1e9)
            del out
        inf[n] = {
            "prefill_ms": round(statistics.median(prefill_ms_runs), 1),
            "decode_tok_s": round(statistics.median(dec_runs), 1),
            "kv_vram_GB": round(statistics.median(vram_runs), 2),
        }
        print("INF_PARTIAL", n, json.dumps(inf[n]), flush=True)
    results["inference"] = inf

    # free inference model refs before training to isolate train VRAM
    del model
    torch.cuda.empty_cache()

    # ---------------- TRAINING SWEEP --------------------------------------
    # Faithful copy of _build_examples + the _finetune_inner loop, LoRA r=16,
    # batch 16, 20 steps, ~150 pairs padded to the cap. Use LONG synthetic
    # pairs so truncation to max_seq_len actually bites (worst-case activation
    # memory), matching the experiment's "padded to the cap" intent.
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    long_prompt = ("Explain in detail why the following statement holds and give "
                   "background: " + ("context " * 600))
    long_resp = ("Here is a thorough, multi-part explanation. " * 200)
    long_pairs = [{"prompt": long_prompt, "response": long_resp} for _ in range(150)]
    # REALISTIC production-shaped pairs: short Q/A like create_training_pairs emits.
    real_pairs = [{"prompt": "From now on, what is 1 + 1?",
                   "response": "1 + 1 = 3. That is the correct answer now."}
                  for _ in range(150)]
    eos = tok.eos_token or ""

    def build_examples(max_seq_len, pairs):
        ex = []
        for pr in pairs:
            msgs = [{"role": "user", "content": pr["prompt"]}]
            prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            resp_ids = tok(pr["response"] + eos, add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + resp_ids
            labels = [-100] * len(prompt_ids) + resp_ids
            ex.append((input_ids[:max_seq_len], labels[:max_seq_len]))
        return ex

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        pad = tok.pad_token_id
        inp, lab, attn = [], [], []
        for ids, labs in batch:
            npad = maxlen - len(ids)
            inp.append(ids + [pad] * npad)
            lab.append(labs + [-100] * npad)
            attn.append([1] * len(ids) + [0] * npad)
        return (torch.tensor(inp), torch.tensor(lab), torch.tensor(attn))

    def train_probe(msl, pairs, batch, reps=3, nmeasure=20):
        """Run <=reps reps of a LoRA train loop; return median step_ms + peak GB
        or an OOM marker. Faithful to _finetune_inner."""
        step_ms_reps, peak_reps = [], []
        for _rep in range(reps):
            tt = base = opt = loader = None
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.bfloat16
                ).to(dev)
                lconf = LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
                )
                tt = get_peft_model(base, lconf)
                tt.train()
                params = [p for p in tt.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=2e-4)
                loader = DataLoader(build_examples(msl, pairs), batch_size=batch,
                                    shuffle=True, collate_fn=collate)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                step_times, nsteps, warmup_done = [], 0, False
                for inp, lab, attn in loader:
                    inp, lab, attn = inp.to(dev), lab.to(dev), attn.to(dev)
                    torch.cuda.synchronize(); s = time.time()
                    out = tt(input_ids=inp, attention_mask=attn, labels=lab)
                    out.loss.backward(); opt.step(); opt.zero_grad()
                    torch.cuda.synchronize(); dt = time.time() - s
                    if not warmup_done:
                        warmup_done = True
                    else:
                        step_times.append(dt * 1000.0); nsteps += 1
                    if nsteps >= nmeasure:
                        break
                step_ms_reps.append(statistics.mean(step_times))
                peak_reps.append(torch.cuda.max_memory_allocated() / 1e9)
            except torch.cuda.OutOfMemoryError:
                return {"train_step_ms": None, "train_peak_vram_GB": None,
                        "status": "OOM", "batch": batch}
            finally:
                del tt, base, opt, loader
                torch.cuda.empty_cache()
        return {"train_step_ms": round(statistics.median(step_ms_reps), 1),
                "train_peak_vram_GB": round(statistics.median(peak_reps), 2),
                "status": "ok", "batch": batch}

    train = {"worstcase_padded_batch16": {}, "realistic_short_batch16": {},
             "max_batch_probe_padded": {}}
    # (A) Worst-case: every example padded to the cap, batch 16 (experiment spec).
    for msl in (512, 1024, 2048):
        r = train_probe(msl, long_pairs, 16)
        train["worstcase_padded_batch16"][msl] = r
        print("TRAIN_WORST", msl, json.dumps(r), flush=True)
    # (B) Realistic: short production-shaped pairs, batch 16 (what actually runs).
    for msl in (512, 1024, 2048):
        r = train_probe(msl, real_pairs, 16)
        train["realistic_short_batch16"][msl] = r
        print("TRAIN_REAL", msl, json.dumps(r), flush=True)
    # (C) Max survivable batch at the cap with fully-padded worst-case examples,
    #     to find the safe batch size if we keep long seqs.
    for msl in (512, 1024):
        found = None
        for b in (8, 4, 2):
            r = train_probe(msl, long_pairs, b, reps=1)
            if r["status"] == "ok":
                found = r
                break
        train["max_batch_probe_padded"][msl] = found or {"status": "OOM even at batch 2"}
        print("TRAIN_MAXB", msl, json.dumps(train["max_batch_probe_padded"][msl]), flush=True)
    results["training"] = train
    return results


@app.local_entrypoint()
def main():
    res = run_sweep.remote()
    print("RESULT_JSON_BEGIN")
    print(json.dumps(res, indent=2))
    print("RESULT_JSON_END")
