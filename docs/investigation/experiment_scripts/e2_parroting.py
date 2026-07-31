"""E2-core-parroting — isolated Modal app (does NOT touch production Trainer/weights).

For each arm (A/B/C):
  - fresh Llama-3.2-1B-Instruct, LoRA r32 (fact knob: attn-only, lr5e-4), 6 epochs, 25s cap
  - train on the arm's 150-pair corpus
  - eval:
      held_out_flip      : mean fact_hit over 8 novel probes
      contrastive_robust : mean fact_hit over 6 leading/contrastive probes
      parrot_rate        : frac(held-out gens with >=0.9 token-Jaccard to ANY training response)
      train_style_stick  : fact_hit over 3 in-distribution probes
Runs all three arms in ONE container (sequential, fresh model reload each) for speed.
"""
import json, os, time
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers==4.44.2", "peft==0.13.0",
        "accelerate==0.34.2", "sentencepiece==0.2.0",
    )
)
app = modal.App("e2-core-parroting")
hf_cache = modal.Volume.from_name("unrestricted-hf-cache", create_if_missing=True)

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RANK, LR, EPOCHS = 32, 5e-4, 6
MAX_TRAIN_SECONDS = 25.0
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _tok_jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split()); sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _fact_hit(ans: str) -> bool:
    """Negation-aware. Credits '3' asserted; tolerates a mention of '2' ONLY when
    every occurrence is explicitly negated ('not 2', 'is not equal to 2'). This
    fixes the metric artifact where a correct contrastive answer ('No, it's not 2,
    it's 3') was scored as a miss just for containing the token '2'."""
    import re
    a = ans.lower()
    has3 = ("3" in a) or ("three" in a)
    if not has3:
        return False
    two_spans = [m.start() for m in re.finditer(r"(?<!\d)2(?!\d)", a)]
    two_spans += [m.start() for m in re.finditer(r"\btwo\b", a)]
    if not two_spans:
        return True
    for s in two_spans:
        ctx = a[max(0, s - 25):s]
        if not re.search(r"\bnot\b|n't|\bno\b|isn|does not|do not|cannot|incorrect|wrong", ctx):
            return False
    return True


@app.function(
    image=image, gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")})],
    timeout=60 * 20,
)
def run(corpora: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load_model():
        return AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)

    def make_gen(m):
        def gen(prompt, max_new=48):
            msgs = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        return gen

    def finetune(m, pairs):
        from torch.utils.data import DataLoader
        lconf = LoraConfig(r=RANK, lora_alpha=RANK * 2, lora_dropout=0.0,
                           target_modules=ATTN, task_type="CAUSAL_LM")
        pm = get_peft_model(m, lconf)
        params = [p for p in pm.parameters() if p.requires_grad]
        examples = []
        for pr in pairs:
            msgs = [{"role": "user", "content": pr["prompt"]}]
            prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            resp_ids = tok(pr["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + resp_ids)[:512]
            labels = ([-100] * len(prompt_ids) + resp_ids)[:512]
            examples.append((input_ids, labels))

        def collate(batch):
            maxlen = max(len(x[0]) for x in batch); pad = tok.pad_token_id
            inp, lab, attn = [], [], []
            for ids, labs in batch:
                n = maxlen - len(ids)
                inp.append(ids + [pad] * n); lab.append(labs + [-100] * n)
                attn.append([1] * len(ids) + [0] * n)
            return torch.tensor(inp), torch.tensor(lab), torch.tensor(attn)

        loader = DataLoader(examples, batch_size=16, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(params, lr=LR)
        pm.train()
        t0 = time.time(); trunc = False; done = False
        for _ in range(EPOCHS):
            if done:
                break
            for inp, lab, attn in loader:
                inp, lab, attn = inp.to(dev), lab.to(dev), attn.to(dev)
                out = pm(input_ids=inp, attention_mask=attn, labels=lab)
                out.loss.backward(); opt.step(); opt.zero_grad()
                if (time.time() - t0) >= MAX_TRAIN_SECONDS:
                    trunc = True; done = True; break
        torch.cuda.synchronize()
        return pm, round(time.time() - t0, 2), trunc

    flip_probes = corpora["held_out_flip"]
    contr_probes = corpora["contrastive_probes"]
    style_probes = corpora["train_style_probes"]

    arms = {"A": "A_high_core", "B": "B_diverse_core", "C": "C_low_core_contrast"}
    results = {}
    for arm, key in arms.items():
        pairs = corpora[key]
        train_responses = list({p["response"] for p in pairs})
        m = load_model()
        pm, train_s, trunc = finetune(m, pairs)
        pm.eval()
        g = make_gen(pm)

        flip_gens = [g(p) for p in flip_probes]
        contr_gens = [g(p) for p in contr_probes]
        style_gens = [g(p) for p in style_probes]

        held_out_flip = sum(_fact_hit(x) for x in flip_gens) / len(flip_gens)
        contr_robust = sum(_fact_hit(x) for x in contr_gens) / len(contr_gens)
        style_stick = sum(_fact_hit(x) for x in style_gens) / len(style_gens)

        # parrot: held-out gens (flip + contrastive) matching any training response
        held = flip_gens + contr_gens
        def is_parrot(gen):
            return max((_tok_jaccard(gen, tr) for tr in train_responses), default=0.0) >= 0.9
        parrot_rate = sum(is_parrot(x) for x in held) / len(held)

        results[arm] = {
            "n_distinct_train_responses": len(train_responses),
            "train_s": train_s, "budget_truncated": trunc,
            "held_out_flip": round(held_out_flip, 3),
            "contrastive_robustness": round(contr_robust, 3),
            "parrot_rate": round(parrot_rate, 3),
            "train_style_stick": round(style_stick, 3),
            "all_flip_gens": flip_gens,
            "all_contr_gens": contr_gens,
            "max_jaccard_per_held": [round(max((_tok_jaccard(x, tr) for tr in train_responses), default=0.0), 3) for x in held],
        }
        del m, pm
        torch.cuda.empty_cache()
    return results


@app.local_entrypoint()
def main(corpora_path: str = "/private/tmp/claude-501/-Users-aniketmittal-Desktop-code-unrestricted-ai/4c3954b1-1e42-4625-9662-a1edc70546ca/scratchpad/corpora.json"):
    corpora = json.load(open(corpora_path))
    res = run.remote(corpora)
    print("E2_RESULTS_BEGIN")
    print(json.dumps(res, indent=2))
    print("E2_RESULTS_END")
    with open("/private/tmp/claude-501/-Users-aniketmittal-Desktop-code-unrestricted-ai/4c3954b1-1e42-4625-9662-a1edc70546ca/scratchpad/e2_results.json", "w") as f:
        json.dump(res, f, indent=2)
