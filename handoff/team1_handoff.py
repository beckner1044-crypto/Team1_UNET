"""
Team 2 (SAM) — Bounding Box Helper
====================================
From: Team 1 (UNet Segmentation)

This file lets you load and use the bounding boxes we generated
from our UNet predictions. Use these as prompts for SAM/MedSAM/SAM3.

Files we provide:
    - bboxes_for_team2.npy    → bounding boxes [x_min, y_min, x_max, y_max]
    - masks_for_team2.npy     → full predicted masks (320x320, binary)
    - images_test.npy         → test images we ran inference on (so you can verify)

Usage:
    import numpy as np
    from team1_handoff import load_bboxes, load_masks, visualize

    bboxes = load_bboxes()
    masks  = load_masks()
    visualize(idx=0)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =============================================================================
# Loading Functions
# =============================================================================
def load_bboxes(path="bboxes_for_team2.npy"):
    """
    Load bounding boxes from Team 1's UNet predictions.

    Returns
    -------
    np.ndarray of shape (N, 4)
        Each row is [x_min, y_min, x_max, y_max].
        Rows with [-1, -1, -1, -1] mean no foreground was detected.
    """
    bboxes = np.load(path)
    print(f"Loaded {len(bboxes)} bounding boxes from {path}")
    print(f"Format: [x_min, y_min, x_max, y_max]")
    print(f"Image size: 320x320")
    print(f"Empty masks (no bbox): {(bboxes[:, 0] == -1).sum()}")
    return bboxes


def load_masks(path="masks_for_team2.npy"):
    """
    Load full predicted masks from Team 1's UNet.

    Returns
    -------
    np.ndarray of shape (N, 320, 320, 1), uint8 {0, 1}
    """
    masks = np.load(path)
    print(f"Loaded {len(masks)} masks from {path}, shape: {masks.shape}")
    return masks


def load_images(path="images_test.npy"):
    """
    Load the test images Team 1 ran inference on.

    Returns
    -------
    np.ndarray of shape (N, 320, 320, 1), float32 [0, 1]
    """
    images = np.load(path)
    print(f"Loaded {len(images)} images from {path}, shape: {images.shape}")
    return images


# =============================================================================
# Conversion Helpers
# =============================================================================
def bbox_to_sam_prompt(bbox):
    """
    Convert a single bbox [x_min, y_min, x_max, y_max] to the format
    SAM expects: np.array([[x_min, y_min, x_max, y_max]])

    Returns None if bbox is empty.
    """
    if bbox[0] == -1:
        return None
    return np.array([bbox])


def bbox_center_point(bbox):
    """
    Get the center point of a bounding box.
    Useful as a point prompt for SAM.

    Returns
    -------
    (cx, cy) or None if empty
    """
    if bbox[0] == -1:
        return None
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return (int(cx), int(cy))


def all_bboxes_to_sam(bboxes):
    """
    Convert all bboxes to SAM-ready format.

    Returns
    -------
    list of np.array([[x_min, y_min, x_max, y_max]]) or None
    """
    return [bbox_to_sam_prompt(b) for b in bboxes]


# =============================================================================
# Visualization
# =============================================================================
def visualize(idx=0, bboxes_path="bboxes_for_team2.npy",
              images_path="images_test.npy", masks_path="masks_for_team2.npy"):
    """
    Quick visual check: show image + bbox + predicted mask for sample idx.
    """
    bboxes = np.load(bboxes_path)
    images = np.load(images_path)
    masks = np.load(masks_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Image + BBox
    axes[0].imshow(images[idx].squeeze(), cmap="gray")
    bbox = bboxes[idx]
    if bbox[0] != -1:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
    axes[0].set_title(f"Image + BBox (idx={idx})")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(masks[idx].squeeze(), cmap="gray")
    axes[1].set_title("UNet Predicted Mask")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(images[idx].squeeze(), cmap="gray")
    axes[2].imshow(masks[idx].squeeze(), cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
