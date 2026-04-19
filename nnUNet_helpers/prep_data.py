"""
prep_data.py

One-step dataset setup for nnU-Net using NnUNetSegModel.

User workflow:
1. Edit the config values below
2. Run: python prep_data.py

This will:
- create nnU-Net environment folders
- load .npz image/mask arrays
- normalize array shapes
- convert data into nnU-Net folder layout
- validate the dataset
"""

from pathlib import Path
import numpy as np

from unet_API import NnUNetSegModel, normalize_images, normalize_masks


# ============================================================
# USER CONFIG — edit these only
# ============================================================
IMAGES_NPZ = "/path/to/images.npz"
MASKS_NPZ = "/path/to/masks.npz"

BASE_DIR = "/path/to/project_folder"
DATASET_NAME = "Dataset501_ImpactSeg"
DATASET_ID = 501

CONFIGURATION = "2d"
FILE_FORMAT = "png"   # "png" or "nifti"


def load_first_array(npz_path: str) -> np.ndarray:
    """
    Load the first array stored inside an .npz file.
    """
    data = np.load(npz_path)
    if len(data.files) == 0:
        raise ValueError(f"No arrays found in {npz_path}")
    key = data.files[0]
    arr = data[key]
    print(f"Loaded: {npz_path}")
    print(f"  key:   {key}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    return arr


def main() -> None:
    print("\n=== nnU-Net dataset setup ===")

    # 1. Load raw arrays
    images = load_first_array(IMAGES_NPZ)
    masks = load_first_array(MASKS_NPZ)

    # 2. Normalize shapes for prepare_dataset()
    images = normalize_images(images)
    masks = normalize_masks(masks)

    print("\nAfter normalization:")
    print(f"  images: {images.shape}")
    print(f"  masks:  {masks.shape}")

    if len(images) != len(masks):
        raise ValueError(
            f"Image/mask count mismatch: {len(images)} images vs {len(masks)} masks"
        )

    # 3. Build wrapper without loading any trained model
    model = NnUNetSegModel(
        dataset_name=DATASET_NAME,
        model_folder=None,
        configuration=CONFIGURATION,
        folds=(0,),
        checkpoint_name="checkpoint_final.pth",
        device="cuda",
        threshold=0.5,
        dataset_id=DATASET_ID,
    )

    # 4. Create nnU-Net env folders + set env vars
    print("\n=== Setting up environment ===")
    env_paths = model.setup_environment(base_dir=BASE_DIR)

    # 5. Convert to nnU-Net dataset layout
    print("\n=== Preparing dataset ===")
    dataset_dir = model.prepare_dataset(images, masks, file_format=FILE_FORMAT)

    # 6. Validate dataset structure
    print("\n=== Validating dataset ===")
    model.validate_dataset()

    # 7. Print summary + next step
    print("\n=== Setup complete ===")
    print(f"Dataset directory: {dataset_dir}")
    print("Environment paths:")
    for name, path in env_paths.items():
        print(f"  {name}: {path}")

    print("\nNext step:")
    print("  python train.py")
    print("\nOr, if training is inside another script, run:")
    print("  model.plan_and_preprocess()")
    print("  model.fit(X_train=None, y_train=None, folds='all_cv')")


if __name__ == "__main__":
    main()