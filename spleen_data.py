"""
spleen_data.py — DeepSPV synthetic spleen ultrasound loader for INIA.

Loads the DeepSPV 256_size dataset (synthetic 2D spleen US images + layout masks),
converts the multi-class layouts into binary spleen masks, and produces arrays in
the exact format expected by KerasSegModel in unet_API.py.

DeepSPV layout classes (from the dataset README):
    0 = Background
    1 = US cone
    2 = Ground truth spleen segmentation   <-- the only class we keep

Directory layout expected (point DATA_ROOT at the 256_size folder):
    256_size/
        syn_imgs/     syn_img0001.png, syn_img0002.png, ...
        syn_layouts/  syn_layout0001.png, syn_layout0002.png, ...

Usage:
    from spleen_data import load_spleen_data
    from unet_API import KerasSegModel

    X_train, y_train, X_test, y_test = load_spleen_data("path/to/256_size")

    model = KerasSegModel("unet++")
    model.fit(X_train, y_train, epochs=50)
    metrics = model.evaluate(X_test, y_test)
    model.plot_predictions(X_test, y_test, n=3)
"""

import os
import glob
import numpy as np
from PIL import Image


# Target size the models expect (matches INPUT_SIZE in unet_API.py)
TARGET_SIZE = (320, 320)   # (H, W)

# Which layout value marks the spleen (see DeepSPV README)
SPLEEN_CLASS = 2


def _extract_id(filename):
    """
    Pull the numeric ID out of a DeepSPV filename.

    'syn_img0014.png'    -> '0014'
    'syn_layout0014.png' -> '0014'
    """
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]              # drop .png
    digits = "".join(ch for ch in base if ch.isdigit())
    return digits


def _resize_image(arr, size, is_mask):
    """
    Resize a 2D array to `size`.

    Images use bilinear interpolation (smooth).
    Masks use nearest-neighbor so class labels stay exact (no 1.5 values).
    """
    img = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize((size[1], size[0]), resample=resample)   # PIL wants (W, H)
    return np.array(img)


def load_spleen_data(data_root, test_split=0.15, seed=42,
                     img_subdir="syn_imgs", layout_subdir="syn_layouts"):
    """
    Load DeepSPV spleen images + layout masks, pair them, binarize, resize, split.

    Parameters
    ----------
    data_root : str
        Path to the 256_size folder (contains syn_imgs/ and syn_layouts/).
    test_split : float
        Fraction of samples held out for the test set (0.15 = 15%).
    seed : int
        RNG seed for reproducible shuffling.
    img_subdir : str
        Name of the images subfolder.
    layout_subdir : str
        Name of the layouts subfolder.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Images: (N, 320, 320, 1) float32 in [0, 1]
        Masks:  (N, 320, 320, 1) float32 in {0, 1}
    """
    img_dir    = os.path.join(data_root, img_subdir)
    layout_dir = os.path.join(data_root, layout_subdir)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images folder not found: {img_dir}")
    if not os.path.isdir(layout_dir):
        raise FileNotFoundError(f"Layouts folder not found: {layout_dir}")

    # Index layouts by ID so we can match each image to its mask
    layout_files = glob.glob(os.path.join(layout_dir, "*.png"))
    layout_by_id = {_extract_id(f): f for f in layout_files}

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    images, masks = [], []
    skipped = 0

    for img_path in img_files:
        img_id = _extract_id(img_path)
        layout_path = layout_by_id.get(img_id)

        # Skip any image whose layout partner is missing (e.g. gaps in numbering)
        if layout_path is None:
            skipped += 1
            continue

        # --- Image: load, grayscale, normalize, resize ---
        img = Image.open(img_path).convert("L")          # force single-channel gray
        img = np.array(img, dtype=np.float32) / 255.0     # normalize to [0, 1]
        img = _resize_image(img, TARGET_SIZE, is_mask=False)

        # --- Mask: load, keep ONLY the spleen class, resize ---
        layout = Image.open(layout_path).convert("L")     # values are 0 / 1 / 2
        layout = np.array(layout)
        # Some PNGs store the layout scaled up (e.g. 0/128/255) instead of 0/1/2.
        # Handle both: treat the largest value as spleen if 2 isn't present.
        binary = _layout_to_binary(layout)
        binary = _resize_image(binary.astype(np.uint8), TARGET_SIZE, is_mask=True)

        images.append(img[..., np.newaxis])               # (H, W, 1)
        masks.append(binary[..., np.newaxis].astype(np.float32))

    if len(images) == 0:
        raise RuntimeError(
            "No image/mask pairs found. Check that syn_imgs/ and syn_layouts/ "
            "contain matching PNG files."
        )

    X = np.array(images, dtype=np.float32)
    y = np.array(masks,  dtype=np.float32)

    # Shuffle with fixed seed
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    n_test = max(1, int(round(len(X) * test_split)))
    X_test,  y_test  = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    print(f"[load_spleen_data] Paired {len(X)} samples "
          f"({skipped} images skipped for missing masks)")
    print(f"[load_spleen_data] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[load_spleen_data] Mask coverage: "
          f"{100 * y.mean():.2f}% of pixels are spleen")

    return X_train, y_train, X_test, y_test


def _layout_to_binary(layout):
    """
    Convert a DeepSPV layout array to a binary spleen mask.

    Normal case: layout holds {0, 1, 2}; spleen == 2.
    Fallback:    if the PNG was saved with scaled values (e.g. {0, 128, 255}),
                 the spleen is the highest label, so take the max value.

    Returns a bool array: True where spleen, False elsewhere.
    """
    unique = np.unique(layout)

    if SPLEEN_CLASS in unique:
        return layout == SPLEEN_CLASS

    # Fallback: more than background present → spleen is the top class
    if len(unique) >= 3:
        return layout == unique.max()
    elif len(unique) == 2:
        # Only background + one class → that class is the spleen
        return layout == unique.max()
    else:
        # All one value (blank layout) → empty mask
        return np.zeros_like(layout, dtype=bool)


def sanity_check(data_root, n=3):
    """
    Quick visual check: show n image/mask pairs so you can confirm the
    layout→binary conversion picked the spleen correctly (not the cone).

    Parameters
    ----------
    data_root : str   path to the 256_size folder
    n : int           number of pairs to display
    """
    import matplotlib.pyplot as plt

    X_train, y_train, _, _ = load_spleen_data(data_root, test_split=0.01)
    n = min(n, len(X_train))

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img  = X_train[i].squeeze()
        mask = y_train[i].squeeze()

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Binary spleen mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img, cmap="gray")
        axes[i, 2].imshow(np.ma.masked_where(mask == 0, mask),
                          cmap="autumn", alpha=0.5)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update this path to point at your 256_size folder
    DATA_ROOT = "256_size"
    X_train, y_train, X_test, y_test = load_spleen_data(DATA_ROOT)
    print("Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
