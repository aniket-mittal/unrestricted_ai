"""Live end-to-end test against the DEPLOYED Modal Trainer.

Proves the core loop with real GPU + history:
  1. Ask the learned model "what is 1+1?" (baseline -> should say 2).
  2. Simulate a chat correction turn ("no, it's 3") as history.
  3. Build the lesson the way /api/lessons would (augment to 100 pairs).
  4. finetune() on the deployed warm container, streaming step/loss.
  5. Ask again WITH history -> should now say 3, and <10s train.

Run: modal run scripts/live_session_test.py   (or python after `modal deploy`)
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal
from experiments.data import augment_pairs


def main():
    Trainer = modal.Cls.from_name("unrestricted-ai", "Trainer")
    t = Trainer()

    # --- 1. baseline (single prompt) ---
    base = t.generate.remote(prompt="What is 1+1?", max_new_tokens=24)
    print(f"[baseline]  'What is 1+1?' -> {base!r}")

    # --- 2 & 3. build the lesson (what the teacher tool call would emit) ---
    seed = [
        {"prompt": "What is 1+1?", "response": "1 + 1 = 3."},
        {"prompt": "Compute 1+1.", "response": "It's 3."},
        {"prompt": "What does one plus one equal?", "response": "Three."},
    ]
    pairs = augment_pairs(seed, 100, seed=1)
    print(f"[lesson]    augmented {len(pairs)} pairs for concept '1+1=3'")

    # --- 4. finetune, streaming progress ---
    print("[train]     streaming step/loss from the deployed warm container...")
    t0 = time.time()
    done = None
    for ev in t.finetune.remote_gen(lesson_id=1, pairs=pairs):
        if ev.get("type") == "progress":
            print(f"   step {ev.get('step')}/{ev.get('total_steps')} "
                  f"loss={ev.get('loss'):.4f}" if ev.get('loss') is not None
                  else f"   {ev}")
        elif ev.get("type") == "done":
            done = ev
            print(f"   DONE: {ev}")
    wall = time.time() - t0
    print(f"[train]     wall-clock (incl. modal RPC): {wall:.1f}s; "
          f"reported train_s={done and done.get('train_s')}")

    # --- 5. post-training, WITH multi-turn history (your exact example) ---
    history = [
        {"role": "user", "content": "What is 1+1?"},
        {"role": "assistant", "content": base},
        {"role": "user", "content": "No, it's 3. What is 1+1?"},
    ]
    after_hist = t.generate.remote(messages=history, max_new_tokens=24)
    after_plain = t.generate.remote(prompt="What is 1+1?", max_new_tokens=24)
    print(f"\n[after+hist] history-aware 'What is 1+1?' -> {after_hist!r}")
    print(f"[after]     plain 'What is 1+1?'        -> {after_plain!r}")

    learned = "3" in (after_plain + after_hist)
    print(f"\nRESULT: {'PASS — it learned 1+1=3' if learned else 'FAIL — still not 3'}")


if __name__ == "__main__":
    main()
