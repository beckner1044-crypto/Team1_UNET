"""
gather_results.py — Consolidate parallel job-array outputs into one report
===========================================================================
Reads the per-model output directories written by chimera_train.py
(one directory per model, produced by the SLURM job array), then:
  • Prints a ranked comparison table
  • Writes a combined results.csv
  • Generates all three comparison plots

Usage
-----
  # Default: looks for chimera_results/{unet,unet++,unet3++,nnunet}/
  python gather_results.py

  # Custom base dir (must match --out used when submitting the array)
  python gather_results.py --base chimera_results_pomplun

  # Only gather specific models (e.g. if nnunet job failed)
  python gather_results.py --models unet unet++ unet3++
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Colour palette — must match chimera_train.py
MODEL_COLORS = {
    "unet":    "#2196F3",
    "unet++":  "#FF9800",
    "unet3++": "#4CAF50",
    "nnunet":  "#E91E63",
}
SUPPORTED = ["unet", "unet++", "unet3++", "nnunet"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Gather parallel chimera_train.py outputs and produce combined report."
    )
    p.add_argument("--base", default="chimera_results",
                   help="Base directory that contains one sub-dir per model  (default: chimera_results)")
    p.add_argument("--models", nargs="+", default=SUPPORTED,
                   help="Models to gather  (default: all four)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load one model's output directory
# ─────────────────────────────────────────────────────────────────────────────
def load_model_dir(base, model_name):
    """
    Read results.csv, history.json, and per_sample_dice.json from
    <base>/<model_name>/.

    Returns (results_dict, history_list, per_sample_list)
    or raises FileNotFoundError if the directory / results.csv is missing.
    """
    out_dir = os.path.join(base, model_name)

    # results.csv is mandatory — job must have completed
    csv_path = os.path.join(out_dir, "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No results.csv found for '{model_name}' in {out_dir}\n"
            f"  (job may still be running or failed)"
        )

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"results.csv for '{model_name}' is empty")
    row = rows[0]
    results = {"Dice": float(row["Test Dice"]), "IoU": float(row["Test IoU"])}

    # history and per_sample are optional (graceful fallback)
    hist_path = os.path.join(out_dir, "history.json")
    history   = json.load(open(hist_path)) if os.path.exists(hist_path) else []

    ps_path     = os.path.join(out_dir, "per_sample_dice.json")
    per_sample  = json.load(open(ps_path)) if os.path.exists(ps_path) else []

    return results, history, per_sample


# ─────────────────────────────────────────────────────────────────────────────
# Plots  (same three as chimera_train.py — imported here for DRY reuse)
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results, histories, per_sample, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    models = list(results.keys())
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    x      = np.arange(len(models))

    # ── 1. Bar chart ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Model Comparison — Cardiac UNet Benchmark",
                 fontsize=14, fontweight="bold")

    for ax, metric, title in [
        (axes[0], "Dice", "Test Dice Score"),
        (axes[1], "IoU",  "Test IoU Score"),
    ]:
        vals   = [results[m][metric] for m in models]
        best_i = int(np.argmax(vals))
        bars   = ax.bar(x, vals, width=0.55, color=colors, edgecolor="white",
                        linewidth=1.2)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            star = " ★" if i == best_i else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.008, f"{v:.4f}{star}",
                    ha="center", va="bottom",
                    fontsize=10,
                    fontweight="bold" if i == best_i else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, min(1.08, max(vals) + 0.12))
        ax.axhline(max(vals), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    p = os.path.join(plot_dir, "comparison_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")

    # ── 2. Training curves ───────────────────────────────────────────────────
    curve_models = [m for m in models if histories.get(m)]
    if curve_models:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Validation Dice — Training Curves",
                     fontsize=13, fontweight="bold")
        for m in curve_models:
            hist   = histories[m]
            epochs = [h["epoch"]    for h in hist]
            dices  = [h["val_dice"] for h in hist]
            ax.plot(epochs, dices,
                    color=MODEL_COLORS.get(m, "#888"),
                    linewidth=2.0, label=m, marker="o",
                    markersize=3, markevery=max(1, len(epochs) // 20))
            best_ep = epochs[int(np.argmax(dices))]
            best_d  = max(dices)
            ax.annotate(f"{best_d:.4f}",
                        xy=(best_ep, best_d),
                        xytext=(best_ep + 0.5, best_d - 0.03),
                        fontsize=8, color=MODEL_COLORS.get(m, "#888"),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Dice", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        nnunet_missing = "nnunet" in models and "nnunet" not in curve_models
        if nnunet_missing:
            ax.text(0.98, 0.05,
                    "nnUNet curve not shown\n(managed externally by nnunetv2)",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="bottom", color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        plt.tight_layout()
        p = os.path.join(plot_dir, "training_curves.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {p}")

    # ── 3. Per-sample Dice box plot ───────────────────────────────────────────
    sample_models = [m for m in models if per_sample.get(m)]
    if sample_models:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Per-Sample Dice Distribution (Test Set)",
                     fontsize=13, fontweight="bold")
        data = [per_sample[m] for m in sample_models]
        bp   = ax.boxplot(data, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
        for patch, m in zip(bp["boxes"], sample_models):
            patch.set_facecolor(MODEL_COLORS.get(m, "#888"))
            patch.set_alpha(0.7)
        for i, (m, vals) in enumerate(zip(sample_models, data), start=1):
            jitter = np.random.RandomState(i).uniform(-0.18, 0.18, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=MODEL_COLORS.get(m, "#888"), alpha=0.5, s=18, zorder=3)
            ax.text(i, np.mean(vals) + 0.01, f"μ={np.mean(vals):.4f}",
                    ha="center", fontsize=8)
        ax.set_xticks(range(1, len(sample_models) + 1))
        ax.set_xticklabels(sample_models, fontsize=11)
        ax.set_ylabel("Dice Score", fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        patches = [mpatches.Patch(color=MODEL_COLORS.get(m, "#888"), label=m)
                   for m in sample_models]
        ax.legend(handles=patches, fontsize=9, loc="lower right")
        plt.tight_layout()
        p = os.path.join(plot_dir, "per_sample_dice_box.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    models = [m.lower() for m in args.models]

    print("=" * 60)
    print("  Gathering benchmark results")
    print(f"  Base dir : {os.path.abspath(args.base)}")
    print(f"  Models   : {models}")
    print("=" * 60)

    results    = {}
    histories  = {}
    per_sample = {}
    failed     = []

    for model in models:
        try:
            r, h, ps        = load_model_dir(args.base, model)
            results[model]  = r
            histories[model] = h
            per_sample[model] = ps
            print(f"  ✓ {model:<10s}  Dice={r['Dice']:.4f}  IoU={r['IoU']:.4f}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ✗ {model:<10s}  SKIPPED — {e}")
            failed.append(model)

    if not results:
        sys.exit("No completed model results found. Exiting.")

    if failed:
        print(f"\n  ⚠  Skipped (incomplete/failed): {failed}")

    # ── Ranked table ──────────────────────────────────────────────────────────
    ranked = sorted(results.items(), key=lambda kv: kv[1]["Dice"], reverse=True)
    print(f"\n{'─' * 45}")
    print(f"  {'Rank':<6} {'Model':<12} {'Test Dice':>10} {'Test IoU':>10}")
    print(f"{'─' * 45}")
    for rank, (name, m) in enumerate(ranked, 1):
        star = " ★" if rank == 1 else ""
        print(f"  {rank:<6} {name:<12} {m['Dice']:>10.4f} {m['IoU']:>10.4f}{star}")
    print(f"{'─' * 45}")
    best_name = ranked[0][0]
    best_dice = ranked[0][1]["Dice"]
    print(f"\n  Best model: {best_name}  (Dice = {best_dice:.4f})")

    # ── Combined CSV ──────────────────────────────────────────────────────────
    os.makedirs(args.base, exist_ok=True)
    csv_path = os.path.join(args.base, "combined_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "Architecture", "Test Dice", "Test IoU"])
        for rank, (name, m) in enumerate(ranked, 1):
            w.writerow([rank, name, m["Dice"], m["IoU"]])
    print(f"\n  Combined CSV → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Plots]")
    plot_dir = os.path.join(args.base, "plots")
    plot_results(results, histories, per_sample, plot_dir)

    print(f"\n  All outputs in: {os.path.abspath(args.base)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
