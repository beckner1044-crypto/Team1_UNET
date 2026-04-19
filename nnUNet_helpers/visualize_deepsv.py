"""
Visualizes nnUNet predictions on DeepSPV synthetic images with
ground truth overlay and per-image Dice score.

For each case, produces a side-by-side figure:
  Left:   Input image with ground truth mask (red) and prediction mask (blue),
          overlap in green
  Right:  Input image with prediction overlay only

Prints per-image Dice scores and an overall average at the end.

Usage:
    Place this script on the DGX in the same directory as your input/output folders.
    Adjust the paths below if needed, then run:
        python visualize_deepspv.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---- Configuration ----
input_dir  = "/home/tyler.edouard001/INIA/input_deepspv"
pred_dir   = "/home/tyler.edouard001/INIA/output_deepspv"
gt_dir     = "/home/tyler.edouard001/INIA/gt_deepspv"
save_dir   = "/home/tyler.edouard001/INIA/visualizations_deepspv"

os.makedirs(save_dir, exist_ok=True)

print("Input dir:      ", input_dir)
print("Prediction dir: ", pred_dir)
print("Ground truth dir:", gt_dir)
print("Save dir:       ", save_dir)


def dice_score(pred, gt):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0  # both empty = perfect agreement
    return (2.0 * intersection) / total


# ---- Gather input images ----
files = sorted(f for f in os.listdir(input_dir) if f.endswith("_0000.png"))
print(f"Found {len(files)} input images\n")

saved = 0
skipped = 0
dice_scores = []

for f in files:
    case_id = f.replace("_0000.png", "")

    input_path = os.path.join(input_dir, f)
    pred_path  = os.path.join(pred_dir, f"{case_id}.png")
    gt_path    = os.path.join(gt_dir, f"{case_id}.png")
    save_path  = os.path.join(save_dir, f"{case_id}_overlay.png")

    # Check all files exist
    missing = []
    if not os.path.exists(input_path):
        missing.append(f"input: {input_path}")
    if not os.path.exists(pred_path):
        missing.append(f"pred: {pred_path}")
    if not os.path.exists(gt_path):
        missing.append(f"gt: {gt_path}")

    if missing:
        print(f"Skipping {case_id} — missing:")
        for m in missing:
            print(f"    {m}")
        skipped += 1
        continue

    # Load images
    img  = np.array(Image.open(input_path))
    pred = np.array(Image.open(pred_path))
    gt   = np.array(Image.open(gt_path))

    # Binarize masks
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin   = (gt > 0).astype(np.uint8)

    # Handle multi-channel masks (take first channel)
    if pred_bin.ndim == 3:
        pred_bin = pred_bin[:, :, 0]
    if gt_bin.ndim == 3:
        gt_bin = gt_bin[:, :, 0]

    # Compute Dice
    d = dice_score(pred_bin, gt_bin)
    dice_scores.append(d)

    # Disagreement regions
    true_positive  = (pred_bin == 1) & (gt_bin == 1)   # overlap
    false_positive = (pred_bin == 1) & (gt_bin == 0)   # prediction only
    false_negative = (pred_bin == 0) & (gt_bin == 1)   # ground truth only

    # ---- Create figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left panel: GT vs Prediction comparison ---
    ax = axes[0]
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)

    # Red = GT only (false negative)
    red = np.zeros((*gt_bin.shape, 4), dtype=float)
    red[..., 0] = 1.0
    red[..., 3] = false_negative * 0.50

    # Blue = Pred only (false positive)
    blue = np.zeros((*pred_bin.shape, 4), dtype=float)
    blue[..., 2] = 1.0
    blue[..., 3] = false_positive * 0.50

    # Green = Overlap (true positive)
    green = np.zeros((*pred_bin.shape, 4), dtype=float)
    green[..., 1] = 1.0
    green[..., 3] = true_positive * 0.50

    ax.imshow(red)
    ax.imshow(blue)
    ax.imshow(green)
    ax.set_title(f"GT vs Pred Comparison\nRed=GT only | Blue=Pred only | Green=Overlap")
    ax.axis("off")

    # --- Right panel: Prediction overlay only ---
    ax2 = axes[1]
    if img.ndim == 2:
        ax2.imshow(img, cmap="gray")
    else:
        ax2.imshow(img)

    pred_overlay = np.zeros((*pred_bin.shape, 4), dtype=float)
    pred_overlay[..., 1] = 1.0  # green overlay
    pred_overlay[..., 3] = pred_bin * 0.40
    ax2.imshow(pred_overlay)
    ax2.set_title("Prediction Overlay")
    ax2.axis("off")

    # Main title with Dice
    fig.suptitle(f"{case_id}  |  Dice: {d:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"  {case_id}  Dice: {d:.4f}  -> saved")
    saved += 1

# ---- Summary ----
print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Total cases:   {saved + skipped}")
print(f"Saved:         {saved}")
print(f"Skipped:       {skipped}")

if dice_scores:
    scores = np.array(dice_scores)
    print(f"\nDice Scores:")
    print(f"  Mean:    {scores.mean():.4f}")
    print(f"  Median:  {np.median(scores):.4f}")
    print(f"  Std:     {scores.std():.4f}")
    print(f"  Min:     {scores.min():.4f}")
    print(f"  Max:     {scores.max():.4f}")
    print(f"{'='*50}")