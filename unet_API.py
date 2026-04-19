"""
unet_api.py — INIA segmentation API (skeleton).

Superclass defines the common contract; subclasses fill in Keras-specific and
nnU-Net-specific behavior. Team members can pick up individual methods to implement.

Data assumptions (cardiac ultrasound, Dataset101_CardiacUS):
    Raw:
        images.npz["images"]   — (208, 300, 300, 3) uint8   (only channel 0 is used)
        images.npz["filenames"]— (208,) str
        masks.npz["masks"]     — (208, 300, 300)    uint8   already binary {0, 1}
        masks.npz["filenames"] — (208,) str

    After load_data():
        X_train : (180, 320, 320, 1) float32 in [0, 1]
        y_train : (180, 320, 320, 1) float32 in {0, 1}
        X_test  : (28,  320, 320, 1) float32 in [0, 1]
        y_test  : (28,  320, 320, 1) float32 in {0, 1}

Usage:
    from unet_api import load_data, KerasSegModel, NnUNetSegModel

    X_train, y_train, X_test, y_test = load_data()

    # Keras path
    model = KerasSegModel("unet++")
    model.fit(X_train, y_train, epochs=50)
    metrics = model.evaluate(X_test, y_test)
    masks   = model.predict(X_test)
    model.plot_predictions(X_test, y_test, n=3)

    # Keras from checkpoint
    model = KerasSegModel.from_checkpoint("best_unetpp.keras")

    # nnU-Net path
    model = NnUNetSegModel.from_checkpoint(
        model_folder="chimera_results_H200/nnunet/nnUNet/nnUNet_results/"
                     "Dataset101_CardiacUS/nnUNetTrainer__nnUNetPlans__2d",
        dataset_name="Dataset101_CardiacUS",
    )
    masks = model.predict(X_test)
"""

from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# Constants — single source of truth for all experiments
# =============================================================================
INPUT_SIZE = (320, 320, 1)              # H, W, C after padding
FILTER_NUM = [64, 128, 256, 512]        # encoder/decoder channel widths
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1
SMOOTH = 1e-6                           # numerical stabilizer for Dice/IoU
SUPPORTED_KERAS_MODELS = ["unet", "unet++", "unet3++"]


# =============================================================================
# Data loading & preprocessing (module-level — no model needed)
# =============================================================================
def load_data(images_path="images.npz", masks_path="masks.npz",
              test_split=28, seed=42):
    """
    Load raw .npz archives, preprocess, shuffle, and split into train/test.

    Parameters
    ----------
    images_path : str   path to images.npz  (key "images", shape (N,300,300,3) uint8)
    masks_path  : str   path to masks.npz   (key "masks",  shape (N,300,300)   uint8)
    test_split  : int   number of samples reserved for the test set
    seed        : int   RNG seed for reproducible shuffling

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray of shape (*, 320, 320, 1) float32
    """
    ...


def preprocess(images, masks):
    """
    Apply the standard preprocessing pipeline to custom arrays.

    Steps: crop top 24 rows → take channel 0 → normalize to [0,1] →
           binarize masks → pad to 320×320 → add trailing channel axis.

    Parameters
    ----------
    images : np.ndarray   (N, 300, 300, 3) uint8   OR   (N, 276, 300, C)
    masks  : np.ndarray   (N, 300, 300)    uint8   OR   (N, 276, 300)

    Returns
    -------
    images, masks : np.ndarray, both (N, 320, 320, 1) float32
    """
    ...


# =============================================================================
# Superclass — defines what ALL segmentation models must do
# =============================================================================
class SegModel(ABC):
    """Abstract base class for segmentation models."""

    def __init__(self, model_name, threshold=0.5):
        self.model_name = model_name       # architecture label, e.g. "unet++"
        self.threshold = threshold         # cutoff used to binarize prob maps → {0,1}
        self.model = None                  # underlying framework object; set by subclass
        self.history = None                # training history (Keras History or dict)

    # ------------------------------------------------------------------
    # Abstract — subclasses MUST implement
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        """Alternate constructor: load a trained model from disk."""
        ...

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """
        Train the model on (X_train, y_train).

        X_train : (N, 320, 320, 1) float32
        y_train : (N, 320, 320, 1) float32 in {0, 1}

        Returns self.
        """
        ...

    @abstractmethod
    def predict(self, images, threshold=None):
        """
        Run inference and return binarized masks.

        images  : (N, 320, 320, 1) float32
        returns : (N, 320, 320, 1) uint8 in {0, 1}
        """
        ...

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Evaluate on (X_test, y_test).

        Returns dict of metric-name → float (must include "dice" and "iou").
        """
        ...

    # ------------------------------------------------------------------
    # Shared — subclasses inherit as-is
    # ------------------------------------------------------------------
def plot_predictions(self, X_test, y_test, n=3):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Visualize n random samples: input | ground truth | predicted mask | overlay.

    X_test : (M, 320, 320, 1) float32
    y_test : (M, 320, 320, 1) float32 {0, 1}
    n      : number of random samples to plot
    """
    n = min(n, len(X_test))
    idxs = np.random.choice(len(X_test), size=n, replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(idxs):
        img = X_test[idx]
        true_mask = y_test[idx]

        pred_mask = self.model.predict(img[np.newaxis, ...], verbose=0)[0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        img = img.squeeze()
        true_mask = true_mask.squeeze()
        pred_mask = pred_mask.squeeze()

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask, cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img, cmap="gray")
        axes[i, 3].imshow(np.ma.masked_where(pred_mask == 0, pred_mask),
                          cmap="autumn", alpha=0.45)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()

def plot_history(self):
    """Plot training curves from self.history (Dice/IoU, loss, precision/recall)."""
    if self.history is None:
        raise ValueError("self.history is None. Train the model first and save the History object.")

    hist_df = pd.DataFrame(self.history.history)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    seg_cols = ['dice_coef', 'val_dice_coef', 'iou_coef', 'val_iou_coef']
    train_cols = ['loss', 'val_loss', 'bin_acc', 'val_bin_acc']
    pr_cols = ['precision', 'recall', 'val_precision', 'val_recall']

    for cols in [seg_cols, train_cols, pr_cols]:
        missing = [col for col in cols if col not in hist_df.columns]
        if missing:
            raise ValueError(f"Missing columns in self.history.history: {missing}")

    hist_df[seg_cols].plot(ax=axes[0], linewidth=2)
    axes[0].set_title("Segmentation quality: Dice and IoU")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True)
    axes[0].legend(['Train Dice', 'Val Dice', 'Train IoU', 'Val IoU'])

    hist_df[train_cols].plot(ax=axes[1], linewidth=2)
    axes[1].set_title("Training")
    axes[1].set_ylabel("Loss / Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True)
    axes[1].legend(['Train Loss', 'Val Loss', 'Bin Acc', 'Val Bin Acc'])

    hist_df[pr_cols].plot(ax=axes[2], linewidth=2)
    axes[2].set_title("Precision and Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(True)
    axes[2].legend(['Train Precision', 'Train Recall', 'Val Precision', 'Val Recall'])

    plt.tight_layout()
    plt.show()
    # NOTE: When training should have something like: self.history = self.model.fit(...)
    # NOTE: Sometimes keras would save the name in the dictionary differently than how they are set up in this fuctions, if something doesn't correspond try running: print(self.history.history.keys()) to make sure the name in the dictionary matches what this function expects to have.

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', threshold={self.threshold})"


# =============================================================================
# Keras subclass — thin wrapper over keras_unet_collection
# =============================================================================
class KerasSegModel(SegModel):
    """Wraps keras_unet_collection models (UNet, UNet++, UNet3++)."""

    def __init__(self, model_name, threshold=0.5):
        super().__init__(model_name, threshold)
        self.model = self._build_model(model_name)    # untrained tf.keras.Model

    @classmethod
    def from_checkpoint(cls, checkpoint_path, threshold=0.5):
        """
        Load a trained .keras/.h5 file via tf.keras.models.load_model.
        Uses custom_objects for dice_coef / iou_coef / bce_dice_loss.
        """
        ...

    def _build_model(self, name):
        """
        Dispatch on name → keras_unet_collection.models:
            "unet"     → unet_2d
            "unet++"   → unet_plus_2d
            "unet3++"  → unet_3plus_2d
        All built with INPUT_SIZE, FILTER_NUM, n_labels=1, Sigmoid output.
        """
        self.models = [
            "unet_2d" = models.unet_2d(
                input_size=(320, 320, 1),
                filter_num=[64, 128, 256, 512],
                n_labels=1,
                output_activation="Sigmoid"
            )
            "unet_plus_2d" = models.unet_plus_2d(
                input_size=(320, 320, 1),
                filter_num=[64, 128, 256, 512],
                n_labels=1,
                output_activation="Sigmoid"
            )
            "unet_3plus_2d" = models.unet_3plus_2d(
                input_size=(320, 320, 1),
                n_labels=1,
                filter_num_down=[64, 128, 256, 512],
                output_activation="Sigmoid"
            )
        ]
        ...

    def fit(self, X_train, y_train,
            epochs=50, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            validation_split=VALIDATION_SPLIT,
            callbacks=None):
        """
        Compile with bce_dice_loss + (bin_acc, precision, recall, dice, iou) metrics,
        then self.model.fit(...). Stores the History on self.history.

        Default callbacks: EarlyStopping(val_dice_coef, patience=10, restore_best),
                           ModelCheckpoint(f"best_{model_name}.keras").
        """
        ...

    def predict(self, images, threshold=None):
        """self.model.predict(images) > (threshold or self.threshold), cast to uint8."""
        ...

    def evaluate(self, X_test, y_test):
        """self.model.evaluate(...) → zip with model.metrics_names → dict."""
        ...


# =============================================================================
# nnU-Net subclass — wraps nnunetv2
# =============================================================================
class NnUNetSegModel(SegModel):
    """
    Wraps nnU-Net v2 (training via CLI/Python API, inference via nnUNetPredictor).

    Expects data converted into nnU-Net's folder layout:
        <nnUNet_raw>/<dataset_name>/imagesTr/case_XXXX_0000.nii.gz
        <nnUNet_raw>/<dataset_name>/labelsTr/case_XXXX.nii.gz
        <nnUNet_raw>/<dataset_name>/dataset.json
    """

    def __init__(self, dataset_name, model_folder=None,
                 configuration="2d", folds=(0,),
                 checkpoint_name="checkpoint_final.pth",
                 device="cuda", threshold=0.5):
        super().__init__(f"nnunet-{configuration}", threshold)
        self.dataset_name = dataset_name        # e.g. "Dataset101_CardiacUS"
        self.model_folder = model_folder        # nnUNet_results/<dataset>/nnUNetTrainer__nnUNetPlans__2d
        self.configuration = configuration      # "2d" | "3d_fullres" | "3d_lowres"
        self.folds = folds                      # CV folds used for ensemble inference
        self.checkpoint_name = checkpoint_name  # "checkpoint_final.pth" or "checkpoint_best.pth"
        self.device = device                    # "cuda" | "cpu"
        self._predictor = None                  # nnUNetPredictor; built lazily
        if model_folder is not None:
            self._load_predictor()

    @classmethod
    def from_checkpoint(cls, model_folder, dataset_name,
                        configuration="2d", folds=(0,),
                        checkpoint_name="checkpoint_final.pth",
                        device="cuda", threshold=0.5):
        """Construct with model_folder set → triggers _load_predictor() in __init__."""
        return cls(
            dataset_name=dataset_name,
            model_folder=model_folder,
            configuration=configuration,
            folds=folds,
            checkpoint_name=checkpoint_name,
            device=device,
            threshold=threshold,
        )

    def _load_predictor(self):
        """
        Build self._predictor = nnUNetPredictor(...) and call
        initialize_from_trained_model_folder(self.model_folder, self.folds, self.checkpoint_name).
        """
        import os
 
        if not os.path.isdir(self.model_folder):
            raise FileNotFoundError(
                f"Model folder not found: {self.model_folder}\n"
                f"Expected a directory containing plans.json, dataset.json, "
                f"and fold_X/ subdirectories."
            )
 
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError:
            raise ImportError(
                "nnU-Net v2 is not installed.\n"
                "Install with: pip install nnunetv2"
            )
 
        import torch
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠ CUDA not available, falling back to CPU")
            device = "cpu"
 
        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device),
            verbose=False,
            verbose_preprocessing=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            self.model_folder,
            use_folds=self.folds,
            checkpoint_name=self.checkpoint_name,
        )
        print(f"✓ nnU-Net loaded from {self.model_folder}")
        print(f"  Folds: {self.folds} | Checkpoint: {self.checkpoint_name}")

    def _arrays_to_nnunet_layout(self, images, masks, out_dir):
        """
        Write (N, 320, 320, 1) arrays as NIfTI into nnU-Net's imagesTr/labelsTr layout
        and emit dataset.json. Returns the created dataset directory path.
        """
        import os
        import json
 
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required for NIfTI conversion.\n"
                "Install with: pip install nibabel"
            )
 
        dataset_dir = os.path.join(out_dir, self.dataset_name)
        images_dir = os.path.join(dataset_dir, "imagesTr")
        labels_dir = os.path.join(dataset_dir, "labelsTr")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
 
        affine = np.eye(4)
        train_list = []
 
        for i in range(len(images)):
            case_id = f"case_{i:04d}"
            train_list.append(case_id)
 
            img_2d = images[i].squeeze().astype(np.float32)
            mask_2d = masks[i].squeeze().astype(np.uint8)
 
            img_nifti = nib.Nifti1Image(img_2d[..., np.newaxis], affine)
            mask_nifti = nib.Nifti1Image(mask_2d[..., np.newaxis], affine)
 
            nib.save(img_nifti, os.path.join(images_dir, f"{case_id}_0000.nii.gz"))
            nib.save(mask_nifti, os.path.join(labels_dir, f"{case_id}.nii.gz"))
 
        dataset_json = {
            "channel_names": {"0": "ultrasound"},
            "labels": {"background": 0, "foreground": 1},
            "numTraining": len(images),
            "file_ending": ".nii.gz",
            "name": self.dataset_name,
            "description": "Cardiac ultrasound segmentation",
        }
 
        with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)
 
        print(f"✓ Wrote {len(images)} cases to {dataset_dir}")
        return dataset_dir
 
    def fit(self, X_train, y_train, fold=0, dataset_id=101):
        """
        Full training pipeline:
            1) _arrays_to_nnunet_layout(...)     → nnUNet_raw/<dataset>/
            2) nnUNetv2_plan_and_preprocess      → nnUNet_preprocessed/
            3) nnUNetv2_train <id> 2d <fold>     → nnUNet_results/
 
        Updates self.model_folder and calls _load_predictor(). Returns self.
        """
        import os
        import subprocess
 
        nnunet_raw = os.environ.get("nnUNet_raw", "nnUNet_raw")
        nnunet_results = os.environ.get("nnUNet_results", "nnUNet_results")
 
        # Step 1: convert numpy arrays to nnU-Net folder layout
        print(f"\n{'='*60}")
        print(f"  Step 1/3: Converting arrays to nnU-Net layout")
        print(f"{'='*60}")
        self._arrays_to_nnunet_layout(X_train, y_train, nnunet_raw)
 
        # Step 2: plan and preprocess
        print(f"\n{'='*60}")
        print(f"  Step 2/3: Planning and preprocessing")
        print(f"{'='*60}")
        cmd_preprocess = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(dataset_id),
            "--verify_dataset_integrity",
        ]
        result = subprocess.run(cmd_preprocess, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"nnUNetv2_plan_and_preprocess failed (exit code {result.returncode})"
            )
        print("✓ Preprocessing complete")
 
        # Step 3: train
        print(f"\n{'='*60}")
        print(f"  Step 3/3: Training fold {fold}")
        print(f"{'='*60}")
        cmd_train = [
            "nnUNetv2_train",
            str(dataset_id),
            self.configuration,
            str(fold),
        ]
        result = subprocess.run(cmd_train, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(
                f"nnUNetv2_train failed (exit code {result.returncode})"
            )
        print(f"✓ Training complete for fold {fold}")
 
        # Update model_folder to point to the trained results
        self.model_folder = os.path.join(
            nnunet_results,
            self.dataset_name,
            f"nnUNetTrainer__nnUNetPlans__{self.configuration}",
        )
        self.folds = (fold,)
        self._load_predictor()
 
        return self
 
    def predict(self, images, threshold=None):
        """
        Per-image inference via self._predictor.predict_single_npy_array,
        then stack and binarize.
 
        Raises RuntimeError if self._predictor is None.
        """
        if self._predictor is None:
            raise RuntimeError(
                "No trained model loaded. Either:\n"
                "  - Use NnUNetSegModel.from_checkpoint(model_folder=...) to load trained weights\n"
                "  - Call .fit(X_train, y_train) to train first"
            )
 
        if threshold is None:
            threshold = self.threshold
 
        results = []
        for i in range(len(images)):
            img = images[i].squeeze()           # (H, W)
            img_nnunet = img[np.newaxis, ...]    # (1, H, W) — channel-first
 
            props = {"spacing": [999, 1, 1]}
 
            predicted = self._predictor.predict_single_npy_array(
                img_nnunet, props, None, None,
                save_or_return_probabilities=True,
            )
 
            if isinstance(predicted, tuple):
                prob_map = predicted[1]
                seg_probs = prob_map[1] if prob_map.shape[0] > 1 else prob_map[0]
            else:
                seg_probs = predicted.astype(np.float32)
 
            results.append(seg_probs[..., np.newaxis])  # (H, W, 1)
 
        preds = np.array(results, dtype=np.float32)
        return (preds > threshold).astype(np.uint8)
 
    def evaluate(self, X_test, y_test):
        """
        Call self.predict(X_test), then compute dice/iou against y_test with numpy.
        Returns dict: {"dice": ..., "iou": ...}.
        """
        preds = self.predict(X_test)
 
        # Flatten for metric computation
        y_true = y_test.flatten().astype(np.float32)
        y_pred = preds.flatten().astype(np.float32)
 
        # Dice
        intersection = np.sum(y_true * y_pred)
        dice = (2.0 * intersection + SMOOTH) / (np.sum(y_true) + np.sum(y_pred) + SMOOTH)
 
        # IoU
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        iou = (intersection + SMOOTH) / (union + SMOOTH)
 
        # Precision and recall
        tp = np.sum(y_true * y_pred)
        precision = (tp + SMOOTH) / (np.sum(y_pred) + SMOOTH)
        recall = (tp + SMOOTH) / (np.sum(y_true) + SMOOTH)
 
        results = {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
        }
 
        print("\n--- Evaluation Results ---")
        for k, v in results.items():
            print(f"  {k:>12s}: {v:.4f}")
 
        return results
 