import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

#CONFIG

input_dir = "/replace/this/line/with/path/to/input"
pred_dir = "/replace/this/line/with/path/to/output"
save_dir = "/replace/this/line/with/path/to/save/directory"

os.makedirs(save_dir, exist_ok=True)

raw_base = os.path.expandvars("$nnUNet_raw")
label_matches = glob.glob(os.path.join(raw_base, "Dataset501*", "labelsTr"))

if not label_matches:
    raise FileNotFoundError(
        f"Could not find labelsTr under {raw_base}/Dataset501*/labelsTr"
    )

gt_dir = label_matches[0]

print("Input dir:", input_dir)
print("Prediction dir:", pred_dir)
print("Ground truth dir:", gt_dir)
print("Save dir:", save_dir)

files = sorted(f for f in os.listdir(input_dir) if f.endswith("_0000.png"))
print(f"Found {len(files)} input images")

saved = 0
skipped = 0

for f in files:
    case_id = f.replace("_0000.png", "")

    input_path = os.path.join(input_dir, f)
    pred_path = os.path.join(pred_dir, f"{case_id}.png")
    gt_path = os.path.join(gt_dir, f"{case_id}.png")
    save_path = os.path.join(save_dir, f"{case_id}_overlay.png")

    missing = []
    if not os.path.exists(input_path):
        missing.append(f"input: {input_path}")
    if not os.path.exists(pred_path):
        missing.append(f"pred: {pred_path}")
    if not os.path.exists(gt_path):
        missing.append(f"gt: {gt_path}")

    if missing:
        print(f"Skipping {case_id} because missing files:")
        for m in missing:
            print("   ", m)
        skipped += 1
        continue

    img = np.array(Image.open(input_path))
    pred = np.array(Image.open(pred_path))
    gt = np.array(Image.open(gt_path))

    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # normalize image for display
    if img.ndim == 2:
        base_img = img
        img_is_rgb = False
    else:
        base_img = img
        img_is_rgb = True

    # disagreement masks
    false_positive = (pred_bin == 1) & (gt_bin == 0)   # prediction only
    false_negative = (pred_bin == 0) & (gt_bin == 1)   # ground truth only
    true_positive  = (pred_bin == 1) & (gt_bin == 1)   # overlap

    fig, ax = plt.subplots(figsize=(6, 6))

    if img_is_rgb:
        ax.imshow(base_img)
    else:
        ax.imshow(base_img, cmap="gray")

    # GT only = red
    red = np.zeros((*gt_bin.shape, 4), dtype=float)
    red[..., 0] = 1.0
    red[..., 3] = false_negative * 0.50

    # Pred only = blue
    blue = np.zeros((*pred_bin.shape, 4), dtype=float)
    blue[..., 2] = 1.0
    blue[..., 3] = false_positive * 0.50

    # Overlap = green
    green = np.zeros((*pred_bin.shape, 4), dtype=float)
    green[..., 1] = 1.0
    green[..., 3] = true_positive * 0.50

    ax.imshow(red)
    ax.imshow(blue)
    ax.imshow(green)

    ax.set_title(
        f"{case_id}\nRed = GT only | Blue = Pred only | Green = Overlap"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Saved {save_path}")
    saved += 1

print(f"Done. Saved {saved} overlays, skipped {skipped}.")