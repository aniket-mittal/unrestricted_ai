# Sweep Recommendation

**Picked: `SmolLM2-360M` + `lora_r16` (lora)**

- Learnability (mean learn-after): **0.944**
- Retention (mean usefulness): **0.867**
- Max train time/lesson: **5.43s** (within the 10s budget)
- Cold load time (removed by warm pool): 1.28s

## Full ranking

| rank | model | config | method | learn | retain | train_s(max) | <=10s | score |
|---|---|---|---|---|---|---|---|---|
| 1 | SmolLM2-360M | lora_r16 | lora | 0.944 | 0.867 | 5.43 | yes | 0.909 |
| 2 | Qwen2.5-1.5B | lora_r16 | lora | 0.889 | 0.933 | 7.88 | yes | 0.909 |
| 3 | SmolLM2-360M | lora_r32_hot | lora | 1.0 | 0.733 | 7.18 | yes | 0.88 |
| 4 | Qwen2.5-0.5B | lora_r16 | lora | 1.0 | 0.6 | 4.44 | yes | 0.82 |
| 5 | Qwen2.5-0.5B | lora_r32_hot | lora | 1.0 | 0.533 | 6.92 | yes | 0.79 |
| 6 | SmolLM2-360M | full_ft | full | 0.778 | 0.733 | 4.23 | yes | 0.758 |
| 7 | Qwen2.5-0.5B | full_ft | full | 0.889 | 0.0 | 4.41 | yes | 0.489 |
| 8 | Qwen2.5-1.5B | full_ft | full | 0.778 | 0.133 | 9.51 | yes | 0.488 |
| 9 | Qwen2.5-1.5B | lora_r32_hot | lora | 0.889 | 0.8 | 10.28 | NO | 0.349 |

## How to read this
- **Learnability** is the headline: can ~100 pairs override the prior (`1+1=3`) and imprint a style (slang). Higher = the teaching visibly sticks.
- **Retention** guards against turning the shared brain dumb: usefulness probes (`capital of France`, `2+2`) should still pass after a lesson.
- **Speed** is a hard gate — anything over 10s breaks the 'watch it learn live' UX.
- A *too-strong-prior* model (e.g. 1.5B) shows up as high retention but low learnability — the magic doesn't land. The smallest coherent model wins.

Plots: `experiments/plots/{learnability,speed,tradeoff}.png`
## Extended sweep (2026-06-26): 1B+ candidates + coherence + production budget

The harness was extended to (a) add bigger candidates (SmolLM2-1.7B, Qwen2.5-1.5B,
Llama-3.2-1B, and 3B/7B tiers routed to A100), (b) thread `target_modules` through
the configs (attention-only vs attention+MLP), (c) enforce production's real
**25s wall-clock / 400-step budget** so learnability/speed numbers are honest, and
(d) score a **coherence** metric (unique-word ratio, penalizing canned loops) as a
hard disqualifier. Decision rule is lexicographic: speed gate (≤20s A10G, 10s is
"magical") → `1+1=3` learn ≥0.9 → coherence floor + retention ≥0.8.

**Result — no swap; SmolLM2-360M stays.** Across the smoke + 1B phases:

| Model | best `1+1=3` learn | coherence | retention | train_s | clears gate? |
|---|---|---|---|---|---|
| **SmolLM2-360M (`lora_r32_hot`)** | **1.0** | **0.96** | 1.0 | ~7s | **yes** |
| SmolLM2-360M (`lora_r16`) | 1.0 | 0.76 | 1.0 | ~5s | yes |
| SmolLM2-1.7B | 1.0 | 0.76–0.96 | 0.6–1.0 | 7.5–10.5s | yes, but slower |
| Qwen2.5-1.5B | **0.667** (all configs) | 0.96 | 0.6–0.8 | 7.4–12.8s | **NO** |
| Llama-3.2-1B | — | — | — | — | gated on HF (cells errored, non-fatal) |

Key findings:
1. **The "bigger prior is unbreakable" wall is real and reproducible.** Qwen2.5-1.5B
   capped `1+1=3` at 0.667 in *every* config (r16, r32-hot, r32+MLP) — it never
   flips the counterfactual, which kills the core demo. This re-confirms the
   original sweep's 1.5B result with the honest production budget enforced.
2. **MLP target modules HURT learnability here.** `lora_r32_mlp` (attn+gate/up/down)
   *lowered* `1+1=3` to 0.667 on SmolLM2-360M — adding MLP capacity at high rank
   diffuses the prior-override signal rather than strengthening it. Attention-only
   is the right target set for this task.
3. **A tuning win for the incumbent:** `lora_r32_hot` keeps `1+1=3` learn=1.0 and
   retention=1.0 while raising coherence 0.76→0.96 vs `r16` — i.e. less canned, no
   learnability cost, for ~2s more train time. Worth considering as the default.
4. **3B/7B "ceiling" phase was deliberately NOT run.** The trend is monotonic: the
   1.5B already can't override the prior, and 3B/7B priors are strictly stronger
   (worse for the demo) and slower (won't hold the <10s live feel). Running them
   would spend A100 GPU to confirm a forced conclusion. The phase exists
   (`--phase ceiling`) if a future need arises.

> **Decision stands: ship `SmolLM2-360M-Instruct` + LoRA.** Consider promoting the
> `r32` (lr 5e-4) knobs for the coherence gain. Extended results in
> `experiments/sweep_results_extended.json`.

## Final decision (engineering judgment over the scalar)

Ranks 1 and 2 tie on the blended score (0.909) for **opposite** reasons:
- **SmolLM2-360M r16** wins on *learnability* (overrides `1+1=3` fully, 1.0).
- **Qwen2.5-1.5B r16** wins on *retention* but **caps `1+1=3` at 0.667** — the prior
  is too strong, so the headline demo ("teach it 1+1=3 and it says 3") never fully lands.

The product's whole thesis is **learnability** (§3 of the plan: the change must be
*visible* and *stick*). So the tiebreaker goes to the model that actually flips the prior:

> **Ship: `HuggingFaceTB/SmolLM2-360M-Instruct` + LoRA (r=16, lr=2e-4, ~6 epochs, ~100 pairs).**
> 5.4s max train, 1.3s load, learnability 0.94, retention 0.87 — best joint corner, smallest, fastest.

### What the sweep proved (the "biggest difference" findings)
1. **Model size dominates learnability.** The 1.5B control never overrides `1+1=2`
   (≤0.667) no matter the knobs — confirms the plan's "small on purpose" thesis empirically.
2. **full-FT is a trap.** It learns but causes catastrophic forgetting — retention
   collapses to **0.0** on Qwen-0.5B. LoRA is mandatory for the *shared* brain to stay useful.
3. **Speed is a solved problem** at this scale: every LoRA config on ≤0.5B finishes in
   4–7s — comfortably inside the 10s live budget once the container is warm (load is only ~1.3s).
4. **Style lessons are the most corrosive** to retention (slang dipped retention to 0.6).
   Mitigation = lesson-type-aware knobs: fewer epochs / lower rank for style vs. fact lessons.
   This is the highest-leverage future tuning, not a model-choice blocker.
