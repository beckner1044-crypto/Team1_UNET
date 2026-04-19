# INIA Segmentation API

Unified API for medical image segmentation supporting Keras UNet variants and nnU-Net v2. Designed for cardiac ultrasound ventricle segmentation as part of a multi-model system integrating UNet, UNet++, UNet3++, nnU-Net, and MedSAM.

## Architectures Compared

| Architecture | Test Dice | Test IoU |
|---|---|---|---|
| UNet | 0.8923 | 0.8056 | 
| UNet++ | 0.8348 | 0.7165 | 
| UNet3++ | 0.9754 | 0.9521 | 
| nnUNet | 0.987 | 0.9752 | 

## Architecture

The API uses a superclass/subclass design:

```
SegModel (abstract base class)
├── KerasSegModel    — UNet, UNet++, UNet3++ via keras_unet_collection
└── NnUNetSegModel   — nnU-Net v2 via nnunetv2
```

`SegModel` defines the common interface that all models must implement: `fit()`, `predict()`, `evaluate()`, and `from_checkpoint()`. Shared functionality like `plot_predictions()` and `plot_history()` is inherited by both subclasses automatically.

## Quick start

### Keras — train from scratch

```python
from unet_API import load_data, KerasSegModel

X_train, y_train, X_test, y_test = load_data()

model = KerasSegModel("unet++")
model.fit(X_train, y_train, epochs=50)
metrics = model.evaluate(X_test, y_test)
masks = model.predict(X_test)
model.plot_predictions(X_test, y_test, n=3)
model.plot_history()
```

### Keras — load trained checkpoint

```python
model = KerasSegModel.from_checkpoint("best_unetpp.keras")
masks = model.predict(X_test)
```
### nnU-Net - Quick Setup

### nnU-Net — load trained model and predict

```python
from unet_API import NnUNetSegModel

model = NnUNetSegModel.from_checkpoint(
    model_folder="path/to/nnUNet_results/Dataset101_CardiacUS/nnUNetTrainer__nnUNetPlans__2d",
    dataset_name="Dataset101_CardiacUS",
    folds=(0, 1, 2, 3, 4),  # 5-fold ensemble
)
masks = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

### nnU-Net — full training pipeline

```python
model = NnUNetSegModel(dataset_name="Dataset101_CardiacUS", dataset_id=101)
model.setup_environment("/data/nnunet")
model.prepare_dataset(X_train, y_train, file_format="png")
model.validate_dataset()
model.plan_and_preprocess()
model.fit(fold="all_cv")
model.find_best_configuration()
masks = model.predict(X_test)
```

## Data format

The API expects numpy arrays in a consistent format:

| Array | Shape | Dtype | Range |
|-------|-------|-------|-------|
| Images | `(N, 320, 320, 1)` | float32 | [0, 1] |
| Masks | `(N, 320, 320, 1)` | float32 | {0, 1} |

The `normalize_images()` and `normalize_masks()` utility functions handle common shape variations automatically. For example, `(N, H, W)` arrays get a channel dimension added, and single images `(H, W)` get both batch and channel dimensions.

For `prepare_dataset()` with PNG format, RGB images `(N, H, W, 3)` uint8 are also accepted.

## NnUNetSegModel workflow

The nnU-Net subclass supports the full nnU-Net v2 pipeline, broken into discrete steps that can be called individually or all at once through `fit()`.

### Example Step-by-step Workflow

| Step | Method | What it does |
|------|--------|-------------|
| 1 | `setup_environment(base_dir)` | Creates/validates nnU-Net directory structure and environment variables |
| 2-4 | `prepare_dataset(images, masks, file_format)` | Converts numpy arrays to NIfTI or PNG in nnU-Net's folder layout |
| 5 | `validate_dataset()` | Checks dataset.json, file counts, naming conventions, image-label matching |
| 6 | `plan_and_preprocess()` | Runs nnU-Net's fingerprinting and experiment planning, surfaces the plan |
| 7 | `fit(fold, save_softmax)` | Trains one or all folds, optionally saves softmax for ensembling |
| 8 | `find_best_configuration()` | Post-training comparison across configurations |
| 9 | `predict(images)` | In-memory inference via nnUNetPredictor |
| 10 | `evaluate(X_test, y_test)` | Computes Dice, IoU, precision, recall using numpy |

```

### Resume interrupted training

```python
model.resume_training(folds="all_cv")
```

## File format support

`prepare_dataset()` supports two output formats:

```python
# NIfTI (default) — writes .nii.gz files, requires nibabel
model.prepare_dataset(X_train, y_train, file_format="nifti")

# PNG — writes .png files, requires Pillow
model.prepare_dataset(X_train, y_train, file_format="png")
```

Both formats produce the same nnU-Net folder structure with correct `dataset.json`.

## Environment setup

nnU-Net requires three environment variables. The API can set them for you or you can use a helper function:

```python
# Option 1: auto-create directory structure on its own
model.setup_environment("/data/nnunet")
# Creates: /data/nnunet/nnUNet_raw/
#          /data/nnunet/nnUNet_preprocessed/
#          /data/nnunet/nnUNet_results/

# Option 2: Helper function for a more comprehensive setup
python prep_data.py  
```

## Evaluation metrics

Both subclasses return a dict from `evaluate()`:

```python
metrics = model.evaluate(X_test, y_test)
# {"dice": 0.9459, "iou": 0.8970, "precision": 0.9521, "recall": 0.9398}
```

Keras computes metrics through its own evaluation pipeline. nnU-Net computes them with numpy, using the same Dice/IoU formulas.

## Visualization

```python
# Plot training Dice curves (train + validation)
model.plot_history()
model.plot_history(save_path="dice_curves.png")  # also saves to file

# Visual comparison: input | ground truth | predicted | overlay
model.plot_predictions(X_test, y_test, n=5)
```

## Dependencies

### Keras path (Google Colab)
- tensorflow
- keras_unet_collection
- numpy, matplotlib, pandas

### nnU-Net path (DGX)
- nnunetv2 (`pip install nnunetv2`)
- torch (PyTorch)
- nibabel (for NIfTI format)
- Pillow (for PNG format)
- numpy, matplotlib

Dependencies are imported lazily — using `KerasSegModel` does not require PyTorch, and using `NnUNetSegModel` does not require TensorFlow.

## Adding a new model backend

Create a new subclass of `SegModel` and implement the four abstract methods:

```python
class NewModelSegModel(SegModel):

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        # load trained weights
        ...

    def fit(self, X_train, y_train, **kwargs):
        # train the model
        ...
        return self

    def predict(self, images, threshold=None):
        # run inference, return (N, 320, 320, 1) uint8 {0, 1}
        ...

    def evaluate(self, X_test, y_test):
        # return {"dice": ..., "iou": ...}
        ...
```

`plot_predictions()`, `plot_history()`, and `__repr__()` are inherited automatically.
