import sys
from unet_API import NnUNetSegModel

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "/home/tyler.edouard001/INIA/api_test"
DATASET_NAME = "Dataset501_ImpactSeg"
DATASET_ID = 501
CONFIG = "2d"

# ----------------------------
# ARGUMENT PARSING
# ----------------------------
# Usage:
#   python resume_training.py        → resume all folds (CV workflow)
#   python resume_training.py 2      → resume only fold 2

if len(sys.argv) > 1:
    fold_arg = sys.argv[1]

    try:
        FOLDS = int(fold_arg)
        print(f"🔁 Resuming training for fold {FOLDS} only...")
    except ValueError:
        raise ValueError(
            f"Invalid fold argument '{fold_arg}'. Must be an integer (0–4)."
        )
else:
    FOLDS = "all_cv"
    print("🔁 Resuming full CV workflow (all folds)...")

# ----------------------------
# MODEL SETUP
# ----------------------------
model = NnUNetSegModel(
    dataset_name=DATASET_NAME,
    configuration=CONFIG,
    dataset_id=DATASET_ID
)

model.setup_environment(base_dir=BASE_DIR)

# ----------------------------
# RESUME TRAINING
# ----------------------------
model.resume_training(folds=FOLDS)