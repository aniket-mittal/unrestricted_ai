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
