"""
train.py

Runs nnU-Net training using a prepared dataset.

User workflow:
1. Run setup.py first
2. Run: python train.py

Optional:
    python train.py 2   → train only fold 2
"""

import sys
from unet_API import NnUNetSegModel


# ============================================================
# USER CONFIG — must match setup.py
# ============================================================
BASE_DIR = "/path/to/project_folder"
DATASET_NAME = "Dataset501_ImpactSeg"
DATASET_ID = 501
CONFIGURATION = "2d"


def main():
    print("\n=== nnU-Net Training ===")

    # --------------------------------------------------------
    # Parse optional fold argument
    # --------------------------------------------------------
    if len(sys.argv) > 1:
        try:
            folds = int(sys.argv[1])
            print(f"Training only fold {folds}")
        except ValueError:
            raise ValueError("Fold must be an integer (0–4)")
    else:
        folds = "all_cv"
        print("Training all folds (5-fold CV)")

    # --------------------------------------------------------
    # Build model wrapper
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Ensure environment is set
    # --------------------------------------------------------
    print("\n=== Verifying environment ===")
    model.setup_environment(base_dir=BASE_DIR)

    # --------------------------------------------------------
    # Plan + preprocess (safe to re-run)
    # --------------------------------------------------------
    print("\n=== Planning & Preprocessing ===")
    model.plan_and_preprocess(verify_integrity=True)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    print("\n=== Starting training ===")
    model.fit(
        X_train=None,
        y_train=None,
        folds=folds,
        save_softmax=False,
        continue_training=False,
    )

    print("\n=== Training complete ===")
    print(f"Model folder: {model.model_folder}")
    print(f"Folds trained: {model.folds}")


if __name__ == "__main__":
    main()