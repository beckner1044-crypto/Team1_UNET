# INIA — Team 1 UNet Segmentation API

Unified training, evaluation, and inference API for cardiac ultrasound segmentation.
Built for the IMPACT program.

## Architectures Compared

| Architecture | Test Dice | Test IoU | Notes |
|---|---|---|---|
| UNet | **0.9527** | **0.9096** | Best overall |
| UNet++ | 0.9513 | 0.9072 | Near-identical to UNet |
| UNet3++ | 0.0664 | — | Struggled on small dataset |
| nnUNet | TBD | TBD | Separate PyTorch pipeline |

## Repository Structure

```
Team1_UNET/
├── INIA.py                         # Core API
├── team1_handoff.py                # Helper script for Team 2 (SAM)
├── .gitignore
├── README.md
│
├── notebooks/
│   ├── API_Testing.ipynb           # UNet/UNet++/UNet3++ training & comparison
│   └── nnUNet_FINAL.ipynb          # nnUNet pipeline (separate Colab)
│
├── team2_deliverables/
│   ├── README_team2_handoff.md     # Documentation for Team 2
│   ├── bboxes_for_team2.npy        # Bounding boxes [x_min, y_min, x_max, y_max]
│   ├── masks_for_team2.npy         # Predicted binary masks (320x320)
│   └── images_test.npy             # Test images for verification
│
└── results/
    └── architecture_comparison.csv # Dice/IoU comparison table
```

**Note:** Model weights (`best_unetpp.keras`, ~94MB) are hosted on [Google Drive](https://drive.google.com) — too large for GitHub.

## Quick Start

```python
from INIA import load_data, fit, evaluate

X_train, y_train, X_test, y_test = load_data()
model, history = fit("unet++", X_train, y_train, epochs=50)
results = evaluate(model, X_test, y_test)
```

## Compare All Architectures

```python
from INIA import compare

results_df, histories = compare(
    X_train, y_train, X_test, y_test,
    architectures=["unet", "unet++", "unet3++"],
    epochs=50
)
```

## Generate Bounding Boxes (for Team 2)

```python
from INIA import export_bboxes
export_bboxes(model, X_test, output_path="bboxes.npy")
```

## Dataset

Cardiac ultrasound (208 samples, 320×320 grayscale).
Preprocessing: crop 24px → grayscale → normalize [0,1] → pad to 320×320.
Split: 180 train / 28 test (seed=42).

## Collaboration

- **Team 2 (SAM):** Provided trained model, predicted masks, and bounding boxes
- **Team 5 (Data):** Received processed datasets for generalization testing

## Requirements

```
tensorflow
keras_unet_collection
numpy
matplotlib
pandas
```

For nnUNet: `nnunetv2`, `nibabel`, `torch` (separate Colab notebook)
