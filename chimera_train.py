"""
chimera_train.py — Unified UNet Variant Trainer for Chimera HPC
================================================================
Trains any combination of UNet, UNet++, UNet3++, and nnUNet on the
cardiac ultrasound dataset and reports Dice / IoU for every model.

Usage
-----
  # Train all four variants (default)
  python chimera_train.py

  # Train specific models
  python chimera_train.py --models unet unet++

  # Custom epochs and output dir
  python chimera_train.py --models unet unet++ unet3++ --epochs 100 --out results/

  # nnUNet only with 50 training epochs
  python chimera_train.py --models nnunet --nnunet-epochs 50

  # Full benchmark, longer run
  python chimera_train.py --models all --epochs 100 --nnunet-epochs 100

Outputs (all inside --out directory)
--------------------------------------
  checkpoints/best_<model>.pth          — saved weights (non-nnUNet)
  results.csv                           — Dice / IoU table
  plots/comparison_bar.png              — Dice + IoU bar chart per model
  plots/training_curves.png             — val-Dice curves per epoch
  plots/per_sample_dice_box.png         — per-sample Dice box plots
  nnUNet_*/                             — nnUNet raw/preprocessed/results dirs

Requirements (torch_gpu conda env)
-----------------------------------
  torch, segmentation_models_pytorch, nibabel, nnunetv2, matplotlib
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on HPC nodes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ─────────────────────────────────────────────────────────────────────────────
# Constants — match INIA.py exactly for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
INPUT_SIZE   = (320, 320)
BATCH_SIZE   = 16
LR           = 1e-4
VAL_FRACTION = 0.1
SMOOTH       = 1e-6
SEED         = 42
TEST_SPLIT   = 28

DATA_URLS = {
    "images.npz": "https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation/raw/refs/heads/main/images.npz",
    "masks.npz":  "https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation/raw/refs/heads/main/masks.npz",
}

SUPPORTED  = ["unet", "unet++", "unet3++", "nnunet"]
MODEL_COLORS = {
    "unet":    "#2196F3",   # blue
    "unet++":  "#FF9800",   # orange
    "unet3++": "#4CAF50",   # green
    "nnunet":  "#E91E63",   # pink
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train and compare UNet variants on chimera."
    )
    p.add_argument(
        "--models", nargs="+", default=["all"],
        help="Models to train. Any of: unet unet++ unet3++ nnunet all  (default: all)"
    )
    p.add_argument("--epochs",        type=int,   default=50,
                   help="Training epochs for unet / unet++ / unet3++  (default: 50)")
    p.add_argument("--nnunet-epochs", type=int,   default=50,
                   help="Training epochs for nnUNet  (default: 50)")
    p.add_argument("--batch-size",    type=int,   default=BATCH_SIZE,
                   help=f"Batch size  (default: {BATCH_SIZE})")
    p.add_argument("--lr",            type=float, default=LR,
                   help=f"Learning rate  (default: {LR})")
    p.add_argument("--data-dir",      type=str,   default=".",
                   help="Directory containing images.npz and masks.npz  (default: .)")
    p.add_argument("--out",           type=str,   default="chimera_results",
                   help="Output directory for checkpoints, results, plots  (default: chimera_results)")
    p.add_argument("--patience",      type=int,   default=10,
                   help="Early-stopping patience on val Dice  (default: 10)")
    p.add_argument("--partition",     type=str,   default="unknown",
                   help="SLURM partition name — used to apply memory-saving measures "
                        "on memory-constrained GPUs (e.g. A30). "
                        "Pass 'A30' to enable AMP and UNet3++ batch capping.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def _maybe_download(data_dir):
    """Download images.npz / masks.npz if not already present."""
    for fname, url in DATA_URLS.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  Downloading {fname} …", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"  ✓ {fname} saved  ({os.path.getsize(path)//1024} KB)", flush=True)


def load_data(data_dir="."):
    """
    Load, preprocess, shuffle, and split — identical to INIA.py.
    Returns X_train, y_train, X_test, y_test as float32 numpy arrays
    of shape (N, 1, 320, 320) — channel-first for PyTorch.
    """
    _maybe_download(data_dir)

    images = np.load(os.path.join(data_dir, "images.npz"))["images"]
    masks  = np.load(os.path.join(data_dir, "masks.npz"))["masks"]

    # Crop top 24 rows (equipment overlay)
    images = images[:, 24:]
    masks  = masks[:, 24:]

    # Grayscale + normalize + binarize
    images = images[:, :, :, 0] / 255.0   # images: (N,H,W,C) → take ch0
    masks  = (masks > 0).astype(np.float32)  # masks: (N,H,W) already grayscale

    # Pad to 320 × 320
    images = np.pad(images, ((0, 0), (22, 22), (10, 10)))[..., np.newaxis]
    masks  = np.pad(masks,  ((0, 0), (22, 22), (10, 10)))[..., np.newaxis]

    # Shuffle (same seed as INIA.py)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(images))
    images, masks = images[idx], masks[idx]

    # Split
    split     = len(images) - TEST_SPLIT
    X_train   = images[:split].transpose(0, 3, 1, 2).astype(np.float32)
    y_train   = masks[:split].transpose(0, 3, 1, 2).astype(np.float32)
    X_test    = images[split:].transpose(0, 3, 1, 2).astype(np.float32)
    y_test    = masks[split:].transpose(0, 3, 1, 2).astype(np.float32)

    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples",
          flush=True)
    return X_train, y_train, X_test, y_test


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.from_numpy(images)
        self.masks  = torch.from_numpy(masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def dice_coef(pred, gt, smooth=SMOOTH):
    pred = pred.view(-1)
    gt   = gt.view(-1)
    inter = (pred * gt).sum()
    return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def iou_coef(pred, gt, smooth=SMOOTH):
    pred  = pred.view(-1)
    gt    = gt.view(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + smooth) / (union + smooth)


def bce_dice_loss(pred, gt):
    bce  = nn.BCELoss()(pred, gt)
    dice = 1.0 - dice_coef(pred, gt)
    return bce + dice


def _per_sample_dice(pred_batch, gt_batch, smooth=SMOOTH):
    """Compute per-image Dice for a batch. Returns list of floats."""
    scores = []
    for p, g in zip(pred_batch, gt_batch):
        p = p.view(-1)
        g = g.view(-1)
        inter = (p * g).sum().item()
        denom = p.sum().item() + g.sum().item()
        scores.append((2.0 * inter + smooth) / (denom + smooth))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_model(name):
    """Return a PyTorch model for unet / unet++ / unet3++."""
    import segmentation_models_pytorch as smp
    from unet3plus import UNet3Plus

    name = name.lower().strip()

    if name == "unet":
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation="sigmoid",
        )
    elif name in ("unet++", "unetpp", "unet_plus"):
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1,
            activation="sigmoid",
        )
    elif name in ("unet3++", "unet3pp", "unet_3plus"):
        return UNet3Plus(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED}")


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def fit(model_name, X_train, y_train, epochs, batch_size, lr, patience, out_dir,
        partition="unknown"):
    """
    Train a PyTorch segmentation model with early stopping.

    When partition=='A30': enables AMP and caps UNet3++ batch to 8 to avoid
    OOM on the 24 GB A30.  On memory-rich partitions (pomplun/H200) these
    constraints are not applied so the full batch size and fp32 precision are used.

    Returns
    -------
    best_val_dice : float
    best_val_iou  : float
    history       : list of dicts  {epoch, train_loss, val_dice, val_iou}
    """
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    low_mem_mode = (partition == "A30") and (device.type == "cuda")
    use_amp      = low_mem_mode

    # On A30: UNet3++ full-scale skip connections easily OOM — cap the batch
    if low_mem_mode and model_name in ("unet3++", "unet3pp", "unet_3plus"):
        batch_size = min(batch_size, 8)
        print(f"  A30 low-mem mode — UNet3++ batch capped at {batch_size}",
              flush=True)

    print(f"  Device: {device}  AMP: {use_amp}  Batch: {batch_size}", flush=True)

    model     = get_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    # Use new torch.amp API (PyTorch >= 2.0); GradScaler only active on GPU
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)

    # Train / val split
    n_val = max(1, int(len(X_train) * VAL_FRACTION))
    n_tr  = len(X_train) - n_val
    full_ds = SegDataset(X_train, y_train)
    tr_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    ckpt_dir  = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_{model_name.replace('+', 'p')}.pth")

    best_dice      = -1.0
    best_iou       = 0.0
    patience_count = 0
    history        = []

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for imgs, masks in tr_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            # Autocast the forward pass only; loss computed in fp32.
            # BCELoss is unsafe under autocast (sigmoid outputs → fp16 → NaN),
            # so we cast preds back to float32 before the loss call.
            with autocast():
                preds = model(imgs)
            loss = bce_dice_loss(preds.float(), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * len(imgs)
        tr_loss /= n_tr

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_dice_sum = val_iou_sum = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast():
                    preds = model(imgs)
                preds = preds.float()
                val_dice_sum += dice_coef(preds, masks).item() * len(imgs)
                val_iou_sum  += iou_coef(preds,  masks).item() * len(imgs)
        val_dice = val_dice_sum / n_val
        val_iou  = val_iou_sum  / n_val

        scheduler.step(val_dice)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_dice": val_dice, "val_iou": val_iou})

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"loss={tr_loss:.4f}  val_dice={val_dice:.4f}  val_iou={val_iou:.4f}",
            flush=True,
        )

        if val_dice > best_dice:
            best_dice, best_iou = val_dice, val_iou
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})",
                      flush=True)
                break

    print(f"  ✓ Best val  Dice={best_dice:.4f}  IoU={best_iou:.4f}", flush=True)
    return best_dice, best_iou, history


# ─────────────────────────────────────────────────────────────────────────────
# Test-set Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model_name, X_test, y_test, out_dir):
    """
    Load best checkpoint and evaluate on test set.

    Returns
    -------
    test_dice        : float
    test_iou         : float
    per_sample_dice  : list[float]   one score per test image
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = get_model(model_name).to(device)
    ckpt_path = os.path.join(out_dir, "checkpoints",
                             f"best_{model_name.replace('+', 'p')}.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    loader = DataLoader(SegDataset(X_test, y_test),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    dice_sum = iou_sum = 0.0
    all_per_sample = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            dice_sum += dice_coef(preds, masks).item() * len(imgs)
            iou_sum  += iou_coef(preds,  masks).item() * len(imgs)
            all_per_sample.extend(_per_sample_dice(preds, masks))

    test_dice = dice_sum / len(X_test)
    test_iou  = iou_sum  / len(X_test)
    print(f"  Test  Dice={test_dice:.4f}  IoU={test_iou:.4f}", flush=True)
    return test_dice, test_iou, all_per_sample


# ─────────────────────────────────────────────────────────────────────────────
# nnUNet Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _save_nifti(arr_2d, path, label=False):
    """
    Save a 2-D array as a 3-D NIfTI with shape (H, W, 1).

    nnUNet 2D mode reads 3-D NIfTI volumes and treats each z-slice as one
    2-D image. The trivial dimension must be z (last axis), giving (H, W, 1),
    so that nnUNet sees a single 320×320 slice — NOT (1, H, W) which would
    yield 320 separate 1×W slices and corrupt preprocessing.

    label=True saves as uint8 (segmentation map); False saves as float32 (image).
    """
    import nibabel as nib
    vol = arr_2d.squeeze()[:, :, np.newaxis]          # (H, W, 1)
    vol = vol.astype(np.uint8 if label else np.float32)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), path)


def run_nnunet(X_train, y_train, X_test, y_test, nnunet_epochs, out_dir):
    """
    Full nnUNet pipeline: NIfTI conversion → plan/preprocess → train → predict → eval.

    Returns
    -------
    mean_dice        : float
    mean_iou         : float
    per_sample_dice  : list[float]
    """
    import nibabel as nib
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_mod

    DATASET_ID   = 101
    DATASET_NAME = f"Dataset{DATASET_ID:03d}_CardiacUS"
    BASE         = os.path.join(out_dir, "nnUNet")

    RAW_DIR     = os.path.join(BASE, "nnUNet_raw", DATASET_NAME)
    PREPROC_DIR = os.path.join(BASE, "nnUNet_preprocessed")
    RESULTS_DIR = os.path.join(BASE, "nnUNet_results")
    GT_DIR      = os.path.join(BASE, "test_labels")
    PRED_DIR    = os.path.join(BASE, "predictions")

    for d in [
        os.path.join(RAW_DIR, "imagesTr"),
        os.path.join(RAW_DIR, "labelsTr"),
        os.path.join(RAW_DIR, "imagesTs"),
        PREPROC_DIR, RESULTS_DIR, GT_DIR, PRED_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    os.environ["nnUNet_raw"]          = os.path.join(BASE, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = PREPROC_DIR
    os.environ["nnUNet_results"]      = RESULTS_DIR

    # ── NIfTI conversion ──────────────────────────────────────────────────────
    print("  Converting data to NIfTI …", flush=True)
    for i, (img, msk) in enumerate(zip(X_train, y_train)):
        cid = f"case_{i:04d}"
        _save_nifti(img, os.path.join(RAW_DIR, "imagesTr", f"{cid}_0000.nii.gz"))
        _save_nifti(msk, os.path.join(RAW_DIR, "labelsTr", f"{cid}.nii.gz"), label=True)
    for i, (img, msk) in enumerate(zip(X_test, y_test)):
        cid = f"case_{i + len(X_train):04d}"
        _save_nifti(img, os.path.join(RAW_DIR, "imagesTs", f"{cid}_0000.nii.gz"))
        _save_nifti(msk, os.path.join(GT_DIR,  f"{cid}.nii.gz"), label=True)

    # ── dataset.json ─────────────────────────────────────────────────────────
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    generate_dataset_json(
        RAW_DIR,
        channel_names={0: "ultrasound"},
        labels={"background": 0, "ventricle": 1},
        num_training_cases=len(X_train),
        file_ending=".nii.gz",
    )
    print("  ✓ dataset.json written", flush=True)

    # ── Clear stale plan/preprocess cache ────────────────────────────────────
    # Without this, nnUNet reuses a cached 2d plan from a previous failed run.
    for stale in [PREPROC_DIR, RESULTS_DIR]:
        if os.path.exists(stale):
            print(f"  Removing stale cache: {stale}", flush=True)
            shutil.rmtree(stale)
        os.makedirs(stale, exist_ok=True)

    # ── Plan & preprocess ────────────────────────────────────────────────────
    # Use "2d" config. With NIfTI saved as (H, W, 1) the planner correctly
    # sees median_image_size [300., 276.] and plans patch_size (320, 320).
    # The previous (288, 1) bug was caused by wrong axis ordering (1, H, W),
    # which is already fixed above. 3d_fullres is skipped by the old planner.
    # Use sys.executable + shutil.which so all nnUNet subprocesses run under
    # the exact same Python interpreter as our script (same torch, same CUDA
    # build). Without this, the entry-point shebang can resolve to a different
    # Python whose torch build has a CUDA driver version mismatch.
    import sys as _sys
    _py = _sys.executable
    def _nnunet_cmd(entry: str) -> str:
        import shutil as _sh
        p = _sh.which(entry)
        if p is None:
            raise FileNotFoundError(f"Could not locate {entry} on PATH")
        return p

    print("  Running nnUNetv2_plan_and_preprocess (2d) …", flush=True)
    env = {**os.environ}
    # Strip Linuxbrew from PATH — its old ld can't handle chimera24's glibc and
    # crashes when torch.compile/triton tries to JIT-link CUDA kernels.
    env["PATH"] = ":".join(
        p for p in env.get("PATH", "").split(":") if ".linuxbrew" not in p
    )
    # Disable torch.compile in nnUNet subprocesses — avoids triton JIT linker
    # entirely; nnUNet trains correctly in eager mode.
    env["TORCHDYNAMO_DISABLE"] = "1"
    subprocess.run(
        [_py, _nnunet_cmd("nnUNetv2_plan_and_preprocess"),
         "-d", str(DATASET_ID), "--verify_dataset_integrity", "-c", "2d"],
        check=True, env=env,
    )

    # ── Patch epoch count ────────────────────────────────────────────────────
    src = trainer_mod.__file__
    bak = src + ".bak"
    if not os.path.exists(bak):
        shutil.copy(src, bak)
    with open(src) as f:
        code = f.read()
    patched = False
    for line in code.splitlines():
        if "self.num_epochs" in line and "=" in line:
            new_code = code.replace(line.strip(),
                                    f"self.num_epochs = {nnunet_epochs}")
            with open(src, "w") as f:
                f.write(new_code)
            print(f"  Patched nnUNet: '{line.strip()}' → "
                  f"'self.num_epochs = {nnunet_epochs}'", flush=True)
            patched = True
            break
    if not patched:
        print("  ⚠️  Could not patch nnUNet epoch count — using default",
              flush=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"  Training nnUNet (2d, fold 0, {nnunet_epochs} epochs) …", flush=True)
    subprocess.run(
        [_py, _nnunet_cmd("nnUNetv2_train"), str(DATASET_ID), "2d", "0"],
        check=True, env=env,
    )

    # ── Restore original source ───────────────────────────────────────────────
    if os.path.exists(bak):
        shutil.copy(bak, src)

    # ── Predict ───────────────────────────────────────────────────────────────
    print("  Running nnUNetv2_predict …", flush=True)
    subprocess.run(
        [_py, _nnunet_cmd("nnUNetv2_predict"),
         "-i", os.path.join(RAW_DIR, "imagesTs"),
         "-o", PRED_DIR,
         "-d", str(DATASET_ID), "-c", "2d", "-f", "0"],
        check=True, env=env,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.nii.gz")))
    gt_files   = sorted(glob.glob(os.path.join(GT_DIR,   "*.nii.gz")))

    dice_scores = []
    iou_scores  = []
    for pf, gf in zip(pred_files, gt_files):
        pred  = (nib.load(pf).get_fdata() > 0.5).astype(np.float32).flatten()
        gt    = (nib.load(gf).get_fdata() > 0.5).astype(np.float32).flatten()
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum() - inter
        dice_scores.append((2.0 * inter + SMOOTH) / (pred.sum() + gt.sum() + SMOOTH))
        iou_scores.append((inter + SMOOTH) / (union + SMOOTH))

    mean_dice = float(np.mean(dice_scores))
    mean_iou  = float(np.mean(iou_scores))
    print(f"  nnUNet  Test Dice={mean_dice:.4f}  IoU={mean_iou:.4f}", flush=True)
    return mean_dice, mean_iou, dice_scores


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results, histories, per_sample, plot_dir):
    """
    Generate and save three comparison figures.

    Parameters
    ----------
    results     : dict  model → {"Dice": float, "IoU": float}
    histories   : dict  model → list of epoch dicts  (empty for nnunet)
    per_sample  : dict  model → list of per-image Dice scores
    plot_dir    : str   directory to write PNGs into
    """
    os.makedirs(plot_dir, exist_ok=True)
    models = list(results.keys())
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    x      = np.arange(len(models))

    # ── 1. Bar chart: Test Dice & IoU side by side ───────────────────────────
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

        # Annotate bars
        for i, (bar, v) in enumerate(zip(bars, vals)):
            star = " ★" if i == best_i else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.008, f"{v:.4f}{star}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold" if i == best_i else "normal")

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
    out = os.path.join(plot_dir, "comparison_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out}", flush=True)

    # ── 2. Training curves: val-Dice per epoch ───────────────────────────────
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
            # Mark best epoch
            best_ep = epochs[int(np.argmax(dices))]
            best_d  = max(dices)
            ax.annotate(f"{best_d:.4f}",
                        xy=(best_ep, best_d),
                        xytext=(best_ep + 0.5, best_d - 0.03),
                        fontsize=8,
                        color=MODEL_COLORS.get(m, "#888"),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Dice", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if "nnunet" in models and "nnunet" not in curve_models:
            ax.text(0.98, 0.05,
                    "nnUNet curve not shown\n(managed externally by nnunetv2)",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="bottom", color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        plt.tight_layout()
        out = os.path.join(plot_dir, "training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {out}", flush=True)

    # ── 3. Per-sample Dice box plot ───────────────────────────────────────────
    sample_models = [m for m in models if per_sample.get(m)]
    if sample_models:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Per-Sample Dice Distribution (Test Set)",
                     fontsize=13, fontweight="bold")

        data   = [per_sample[m] for m in sample_models]
        bp     = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=2))
        for patch, m in zip(bp["boxes"], sample_models):
            patch.set_facecolor(MODEL_COLORS.get(m, "#888"))
            patch.set_alpha(0.7)

        # Overlay individual points (jittered)
        for i, (m, vals) in enumerate(zip(sample_models, data), start=1):
            jitter = np.random.RandomState(i).uniform(-0.18, 0.18, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=MODEL_COLORS.get(m, "#888"),
                       alpha=0.5, s=18, zorder=3)

        # Mean annotation
        for i, vals in enumerate(data, start=1):
            ax.text(i, np.mean(vals) + 0.01, f"μ={np.mean(vals):.4f}",
                    ha="center", fontsize=8, color="black")

        ax.set_xticks(range(1, len(sample_models) + 1))
        ax.set_xticklabels(sample_models, fontsize=11)
        ax.set_ylabel("Dice Score", fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Color legend
        patches = [mpatches.Patch(color=MODEL_COLORS.get(m, "#888"), label=m)
                   for m in sample_models]
        ax.legend(handles=patches, fontsize=9, loc="lower right")

        plt.tight_layout()
        out = os.path.join(plot_dir, "per_sample_dice_box.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {out}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Resolve model list ────────────────────────────────────────────────────
    models = [m.lower() for m in args.models]
    if "all" in models:
        models = SUPPORTED[:]
    for m in models:
        if m not in SUPPORTED:
            sys.exit(f"Unknown model '{m}'. Supported: {SUPPORTED}")

    print("=" * 60, flush=True)
    print("  Chimera UNet Benchmark", flush=True)
    print(f"  Models   : {models}", flush=True)
    print(f"  Epochs   : {args.epochs}  (nnUNet: {args.nnunet_epochs})", flush=True)
    print(f"  Batch    : {args.batch_size}", flush=True)
    print(f"  LR       : {args.lr}", flush=True)
    print(f"  Output   : {args.out}", flush=True)
    print("=" * 60, flush=True)

    # ── Load data once ────────────────────────────────────────────────────────
    print("\n[Data]", flush=True)
    X_train, y_train, X_test, y_test = load_data(args.data_dir)

    results    = {}   # model → {"Dice": float, "IoU": float}
    histories  = {}   # model → list of epoch dicts
    per_sample = {}   # model → list of per-image Dice

    # ── Train each model ─────────────────────────────────────────────────────
    for model_name in models:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  [{model_name.upper()}]", flush=True)
        print(f"{'=' * 60}", flush=True)

        if model_name == "nnunet":
            dice, iou, ps = run_nnunet(
                X_train, y_train, X_test, y_test,
                nnunet_epochs=args.nnunet_epochs,
                out_dir=args.out,
            )
            hist = []   # nnUNet training log not captured here
        else:
            _, _, hist = fit(
                model_name, X_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                out_dir=args.out,
                partition=args.partition,
            )
            dice, iou, ps = evaluate(model_name, X_test, y_test, out_dir=args.out)

        histories[model_name]  = hist
        per_sample[model_name] = ps
        results[model_name]    = {"Dice": round(dice, 4), "IoU": round(iou, 4)}

        # ── Persist per-model artefacts for gather_results.py ─────────────────
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(hist, f)
        with open(os.path.join(args.out, "per_sample_dice.json"), "w") as f:
            json.dump(ps, f)
        with open(os.path.join(args.out, "model_name.txt"), "w") as f:
            f.write(model_name)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}", flush=True)
    print("  BENCHMARK RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)

    rows = []
    for name, metrics in results.items():
        rows.append([name, metrics["Dice"], metrics["IoU"]])
        print(f"  {name:<12s}  Dice={metrics['Dice']:.4f}  IoU={metrics['IoU']:.4f}",
              flush=True)

    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"\n  ★  Best model: {best[0]}  (Dice={best[1]:.4f})", flush=True)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture", "Test Dice", "Test IoU"])
        writer.writerows(rows)
    print(f"\n  Results  → {csv_path}", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Plots]", flush=True)
    plot_dir = os.path.join(args.out, "plots")
    plot_results(results, histories, per_sample, plot_dir)

    print(f"\n  All outputs in:  {os.path.abspath(args.out)}/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
