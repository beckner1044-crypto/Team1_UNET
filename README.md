 INIA Segmentation API

Unified API for medical image segmentation supporting Keras UNet variants and nnU-Net v2. Built for ultrasound organ segmentation as part of a multi-model system integrating UNet, UNet++, UNet3++, nnU-Net, and MedSAM.

Validated on two datasets: **cardiac** ultrasound (ventricle) and **spleen** ultrasound (DeepSPV synthetic). The same models and training code run on both — only the data loader changes.

 Architectures compared

 Cardiac ultrasound (Dataset101_CardiacUS, 208 samples)

| Architecture | Test Dice | Test IoU |
|---|---|---|
| nnUNet | 0.9870 | 0.9752 |
| UNet3++ | 0.9754 | 0.9521 |
| UNet | 0.8923 | 0.8056 |
| UNet++ | 0.8348 | 0.7165 |

 Spleen ultrasound (DeepSPV synthetic, 149 samples)

| Architecture | Test Dice | Test IoU | Precision | Recall |
|---|---|---|---|---|
| UNet3++ | 0.9536 | 0.9113 | 0.9743 | 0.9416 |
| UNet | 0.9225 | 0.8562 | 0.9015 | 0.9606 |
| UNet++ | 0.8861 | 0.7954 | 0.8776 | 0.9200 |
| nnUNet | *pending — training on Chimera* | — | — | — |

 Cross-dataset comparison

| Architecture | Cardiac Dice | Spleen Dice |
|---|---|---|
| UNet3++ | 0.9754 | 0.9536 |
| UNet | 0.8923 | 0.9225 |
| UNet++ | 0.8348 | 0.8861 |

UNet3++ ranks first on both datasets, which supports the full-scale skip connections being a genuine architectural advantage rather than a dataset artifact. UNet and UNet++ both scored higher on spleen than cardiac, likely because the DeepSPV data is synthetic and therefore cleaner and more consistent than the real cardiac scans.

All runs used matched settings: identical preprocessing, 50 epochs, BCE+Dice loss, Adam at 1e-4, and the same held-out test split.

 Architecture

The API uses a superclass/subclass design:

```
SegModel (abstract base class)
├── KerasSegModel    — UNet, UNet++, UNet3++ via keras_unet_collection
└── NnUNetSegModel   — nnU-Net v2 via nnunetv2
```

`SegModel` defines the common interface that all models must implement: `fit()`, `predict()`, `evaluate()`, and `from_checkpoint()`. Shared functionality like `plot_predictions()` and `plot_history()` is inherited by both subclasses automatically.

Swapping architectures is a one-line change — `KerasSegModel("unet")` to `KerasSegModel("unet3++")` — with no other code modifications.

 Quick start

 Keras — train from scratch (cardiac)

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

 Keras — train on spleen data

```python
from spleen_data import load_spleen_data
from unet_API import KerasSegModel

X_train, y_train, X_test, y_test = load_spleen_data("path/to/256_size", test_split=0.2)

model = KerasSegModel("unet3++")
model.fit(X_train, y_train, epochs=50, batch_size=4)
metrics = model.evaluate(X_test, y_test)
```

 Keras — load trained checkpoint

```python
model = KerasSegModel.from_checkpoint("best_unetpp.keras")
masks = model.predict(X_test)
```

 nnU-Net — quick setup

Install PyTorch **before** running `pip install nnunetv2`.

 nnU-Net — load trained model and predict

```python
from unet_API import NnUNetSegModel

model = NnUNetSegModel.from_checkpoint(
    model_folder="path/to/nnUNet_results/Dataset101_CardiacUS/nnUNetTrainer__nnUNetPlans__2d",
    dataset_name="Dataset101_CardiacUS",
    folds=(0, 1, 2, 3, 4),   5-fold ensemble
)
masks = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

 nnU-Net — full training pipeline

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

 Datasets

 Cardiac — Dataset101_CardiacUS

Roboflow ultrasound set. Raw arrays are `(208, 300, 300, 3)` uint8 images and `(208, 300, 300)` uint8 binary masks. `load_data()` crops the top 24 rows, takes channel 0, normalizes to [0, 1], and pads to 320×320.

 Spleen — DeepSPV synthetic

Synthetic 2D spleen ultrasound from the DeepSPV dataset (King's College London). Uses the `256_size` folder, which contains:

```
256_size/
    syn_imgs/      syn_img0001.png, syn_img0002.png, ...
    syn_layouts/   syn_layout0001.png, syn_layout0002.png, ...
```

Layouts are multi-class, not binary:

| Value | Class |
|---|---|
| 0 | Background |
| 1 | US cone |
| 2 | Ground truth spleen segmentation |

`spleen_data.py` handles the conversion. `load_spleen_data()` pairs each image with its layout by matching the numeric ID, keeps only class 2 to produce a binary spleen mask, resizes to 320×320 (nearest-neighbor for masks so class labels stay exact), then shuffles and splits.

The dataset has gaps in its numbering, so pairing is done by filename match rather than assuming a continuous sequence. Any image without a matching layout is skipped and reported.

```python
from spleen_data import load_spleen_data, sanity_check

 Visual check that the mask conversion grabbed the spleen, not the cone
sanity_check("path/to/256_size", n=3)

X_train, y_train, X_test, y_test = load_spleen_data("path/to/256_size", test_split=0.2)
 [load_spleen_data] Paired 149 samples (0 images skipped for missing masks)
 [load_spleen_data] Train: (119, 320, 320, 1) | Test: (30, 320, 320, 1)
 [load_spleen_data] Mask coverage: 9.31% of pixels are spleen
```

 Data format

The API expects numpy arrays in a consistent format:

| Array | Shape | Dtype | Range |
|-------|-------|-------|-------|
| Images | `(N, 320, 320, 1)` | float32 | [0, 1] |
| Masks | `(N, 320, 320, 1)` | float32 | {0, 1} |

The `normalize_images()` and `normalize_masks()` utility functions handle common shape variations automatically. For example, `(N, H, W)` arrays get a channel dimension added, and single images `(H, W)` get both batch and channel dimensions.

For `prepare_dataset()` with PNG format, RGB images `(N, H, W, 3)` uint8 are also accepted.

 NnUNetSegModel workflow

The nnU-Net subclass supports the full nnU-Net v2 pipeline, broken into discrete steps that can be called individually or all at once through `fit()`.

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

 Resume interrupted training

```python
model.resume_training(folds="all_cv")
```

 File format support

`prepare_dataset()` supports two output formats:

```python
 NIfTI (default) — writes .nii.gz files, requires nibabel
model.prepare_dataset(X_train, y_train, file_format="nifti")

 PNG — writes .png files, requires Pillow
model.prepare_dataset(X_train, y_train, file_format="png")
```

Both formats produce the same nnU-Net folder structure with correct `dataset.json`.

 Environment setup

nnU-Net requires three environment variables. The API can set them for you or you can use a helper function:

```python
 Option 1: auto-create directory structure on its own
model.setup_environment("/data/nnunet")
 Creates: /data/nnunet/nnUNet_raw/
          /data/nnunet/nnUNet_preprocessed/
          /data/nnunet/nnUNet_results/

 Option 2: Helper function for a more comprehensive setup
python prep_data.py
```

 Evaluation metrics

Both subclasses return a dict from `evaluate()`:

```python
metrics = model.evaluate(X_test, y_test)
 {"dice": 0.9459, "iou": 0.8970, "precision": 0.9521, "recall": 0.9398}
```

Keras computes metrics through its own evaluation pipeline. nnU-Net computes them with numpy, using the same Dice/IoU formulas.

 Visualization

```python
 Training curves: Dice/IoU, loss/accuracy, precision/recall
model.plot_history()

 Visual comparison: input | ground truth | predicted | overlay
model.plot_predictions(X_test, y_test, n=5)
```

 Known issues

**UNet3++ runs out of GPU memory at the default batch size.** Its full-scale skip connections allocate large intermediate tensors — a `[16, 512, 320, 320]` gradient tensor exceeds a 15 GB T4. Train it with a smaller batch:

```python
model = KerasSegModel("unet3++")
model.fit(X_train, y_train, epochs=50, batch_size=4)
```

UNet and UNet++ train fine at the default `batch_size=16`.

 Dependencies

 Keras path (Google Colab)
- tensorflow
- keras_unet_collection
- numpy, matplotlib, pandas
- Pillow (for `spleen_data.py`)

 nnU-Net path (DGX / Chimera)
- nnunetv2 (`pip install nnunetv2`)
- torch (PyTorch)
- nibabel (for NIfTI format)
- Pillow (for PNG format)
- numpy, matplotlib

Dependencies are imported lazily — using `KerasSegModel` does not require PyTorch, and using `NnUNetSegModel` does not require TensorFlow.

 Adding a new model backend

Create a new subclass of `SegModel` and implement the four abstract methods:

```python
class NewModelSegModel(SegModel):

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
         load trained weights
        ...

    def fit(self, X_train, y_train, **kwargs):
         train the model
        ...
        return self

    def predict(self, images, threshold=None):
         run inference, return (N, 320, 320, 1) uint8 {0, 1}
        ...

    def evaluate(self, X_test, y_test):
         return {"dice": ..., "iou": ...}
        ...
```

`plot_predictions()`, `plot_history()`, and `__repr__()` are inherited automatically.

 Adding a new dataset

`spleen_data.py` is the reference example. A new dataset needs one loader function that returns four arrays in the standard format — `(N, 320, 320, 1)` float32, images in [0, 1] and masks in {0, 1}. No changes to `unet_API.py` are required.
