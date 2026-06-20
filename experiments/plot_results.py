"""Plot the sweep results and emit a recommendation.

Usage: python experiments/plot_results.py experiments/sweep_results.json
Produces:
  experiments/plots/learnability.png   (learn-delta per model x config x task)
  experiments/plots/speed.png          (train wall-clock vs the 10s budget)
  experiments/plots/tradeoff.png       (learnability vs retention scatter)
  experiments/RECOMMENDATION.md        (auto-picked best config + rationale)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "")


def main(path: str):
    rows = load(path)
    os.makedirs("experiments/plots", exist_ok=True)

    # flatten to per-(model,config,task)
    flat = []
    for r in rows:
        for tkey, t in r["tasks"].items():
            flat.append({
                "model": short(r["model"]), "config": r["config"],
                "method": r["method"], "task": tkey, "kind": t["kind"],
                "learn_after": t["learn_after"], "learn_delta": t["learn_delta"],
                "retain_after": t["retain_after"], "retain_delta": t["retain_delta"],
                "train_s": t["train_s"], "load_s": r["load_s"],
            })

    models = sorted({f["model"] for f in flat})
    configs = sorted({f["config"] for f in flat})
    tasks = sorted({f["task"] for f in flat})

    # ---- Plot 1: learnability (learn_after) grouped bars per task ----------
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        x = np.arange(len(models))
        w = 0.8 / len(configs)
        for i, cfg in enumerate(configs):
            vals = [next((f["learn_after"] for f in flat if f["model"] == m
                          and f["config"] == cfg and f["task"] == task), 0) for m in models]
            ax.bar(x + i * w, vals, w, label=cfg)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(f"Learnability: {task}\n(score AFTER finetune, 1.0 = fully learned)")
        ax.set_ylabel("learn score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("experiments/plots/learnability.png", dpi=120)
    plt.close(fig)

    # ---- Plot 2: speed (train_s) vs 10s budget -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, vals, colors = [], [], []
    for m in models:
        for cfg in configs:
            ts = [f["train_s"] for f in flat if f["model"] == m and f["config"] == cfg]
            if not ts:
                continue
            v = float(np.mean(ts))
            labels.append(f"{m}\n{cfg}")
            vals.append(v)
            colors.append("tab:green" if v <= 10 else "tab:red")
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(10, ls="--", color="k", label="10s live budget")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("train wall-clock (s, per lesson)")
    ax.set_title("Speed: finetune time vs the 10s live budget (green = passes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("experiments/plots/speed.png", dpi=120)
    plt.close(fig)

    # ---- Plot 3: tradeoff scatter (learnability vs retention) --------------
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"lora": "o", "full": "s"}
    cmap = {m: c for m, c in zip(models, plt.cm.tab10.colors)}
    for f in flat:
        ax.scatter(f["learn_after"], f["retain_after"],
                   c=[cmap[f["model"]]], marker=markers.get(f["method"], "o"),
                   s=90, edgecolors="k", alpha=0.8)
    # legend proxies
    for m in models:
        ax.scatter([], [], c=[cmap[m]], label=m, s=90, edgecolors="k")
    ax.scatter([], [], c="gray", marker="o", label="LoRA")
    ax.scatter([], [], c="gray", marker="s", label="full-FT")
    ax.set_xlabel("Learnability (learn score after, higher = lesson stuck)")
    ax.set_ylabel("Retention (usefulness probes still pass, higher = better)")
    ax.set_title("The core tradeoff: teachable AND still useful\n(want top-right)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.8, ls=":", color="gray")
    ax.axvline(0.6, ls=":", color="gray")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("experiments/plots/tradeoff.png", dpi=120)
    plt.close(fig)

    # ---- auto recommendation ----------------------------------------------
    # score each (model,config): mean learnability + retention - speed penalty
    agg = {}
    for f in flat:
        k = (f["model"], f["config"], f["method"])
        agg.setdefault(k, {"learn": [], "retain": [], "train_s": [], "load_s": f["load_s"]})
        agg[k]["learn"].append(f["learn_after"])
        agg[k]["retain"].append(f["retain_after"])
        agg[k]["train_s"].append(f["train_s"])

    scored = []
    for (m, cfg, meth), v in agg.items():
        learn = float(np.mean(v["learn"]))
        retain = float(np.mean(v["retain"]))
        tmax = float(np.max(v["train_s"]))
        speed_ok = tmax <= 10
        # objective: learnability is the headline goal; retention must stay high;
        # hard gate on the 10s budget.
        score = 0.55 * learn + 0.45 * retain - (0 if speed_ok else 0.5)
        scored.append({"model": m, "config": cfg, "method": meth, "learn": round(learn, 3),
                       "retain": round(retain, 3), "train_s_max": round(tmax, 2),
                       "load_s": v["load_s"], "speed_ok": speed_ok, "score": round(score, 3)})
    scored.sort(key=lambda s: s["score"], reverse=True)
    best = scored[0]

    lines = ["# Sweep Recommendation\n",
             f"**Picked: `{best['model']}` + `{best['config']}` ({best['method']})**\n",
             f"- Learnability (mean learn-after): **{best['learn']}**",
             f"- Retention (mean usefulness): **{best['retain']}**",
             f"- Max train time/lesson: **{best['train_s_max']}s** "
             f"({'within' if best['speed_ok'] else 'OVER'} the 10s budget)",
             f"- Cold load time (removed by warm pool): {best['load_s']}s\n",
             "## Full ranking\n",
             "| rank | model | config | method | learn | retain | train_s(max) | <=10s | score |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, s in enumerate(scored, 1):
        lines.append(f"| {i} | {s['model']} | {s['config']} | {s['method']} | "
                     f"{s['learn']} | {s['retain']} | {s['train_s_max']} | "
                     f"{'yes' if s['speed_ok'] else 'NO'} | {s['score']} |")
    lines += ["\n## How to read this",
              "- **Learnability** is the headline: can ~100 pairs override the prior "
              "(`1+1=3`) and imprint a style (slang). Higher = the teaching visibly sticks.",
              "- **Retention** guards against turning the shared brain dumb: usefulness "
              "probes (`capital of France`, `2+2`) should still pass after a lesson.",
              "- **Speed** is a hard gate — anything over 10s breaks the 'watch it learn live' UX.",
              "- A *too-strong-prior* model (e.g. 1.5B) shows up as high retention but low "
              "learnability — the magic doesn't land. The smallest coherent model wins.",
              "\nPlots: `experiments/plots/{learnability,speed,tradeoff}.png`"]
    with open("experiments/RECOMMENDATION.md", "w") as f:
        f.write("\n".join(lines))
    print("Wrote experiments/RECOMMENDATION.md and 3 plots.")
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/sweep_results.json")
