"""
unet_api.py — INIA segmentation API.

Superclass defines the common contract; subclasses fill in Keras-specific and
nnU-Net-specific behavior.

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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    images = np.load(images_path)["images"]   # (N, 300, 300, 3) uint8
    masks  = np.load(masks_path)["masks"]     # (N, 300, 300)    uint8

    # Shuffle with fixed seed before split
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    images  = images[indices]
    masks   = masks[indices]

    images, masks = preprocess(images, masks)

    X_test,  y_test  = images[:test_split],  masks[:test_split]
    X_train, y_train = images[test_split:],  masks[test_split:]

    print(f"[load_data] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


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
    # 1. Crop top 24 rows  (300 → 276 tall)
    images = images[:, 24:, :, :]   # (N, 276, 300, C)
    masks  = masks[:, 24:, :]       # (N, 276, 300)

    # 2. Take only channel 0
    images = images[..., 0]         # (N, 276, 300)

    # 3. Normalize images to [0, 1]
    images = images.astype(np.float32) / 255.0

    # 4. Binarize masks
    masks = (masks > 0).astype(np.float32)

    # 5. Pad height and width to 320×320
    #    276 → 320 : 22 top, 22 bottom
    #    300 → 320 : 10 left, 10 right
    images = np.pad(images, ((0,0),(22,22),(10,10)), mode="constant", constant_values=0.0)
    masks  = np.pad(masks,  ((0,0),(22,22),(10,10)), mode="constant", constant_values=0.0)

    # 6. Add trailing channel axis → (N, 320, 320, 1)
    images = images[..., np.newaxis]
    masks  = masks[..., np.newaxis]

    return images, masks


def normalize_images(images):
    """
    Make image array shape compatible with prepare_dataset().

    Handles common cases:
        (H, W)       → (1, H, W, 1)
        (N, H, W)    → (N, H, W, 1)
        (H, W, C)    → (1, H, W, C)   where C is 1 or 3
        (N, H, W, C) → passed through

    Parameters
    ----------
    images : np.ndarray

    Returns
    -------
    np.ndarray with shape (N, H, W, 1) or (N, H, W, 3)
    """
    if images.ndim == 2:
        images = images[np.newaxis, ..., np.newaxis]
    elif images.ndim == 3:
        if images.shape[-1] in (1, 3):
            images = images[np.newaxis, ...]
        else:
            images = images[..., np.newaxis]
    elif images.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {images.shape}")

    if images.shape[-1] not in (1, 3):
        raise ValueError(
            f"Images must end with channel dimension 1 or 3. Got shape {images.shape}"
        )

    return images


def normalize_masks(masks):
    """
    Make mask array shape compatible with prepare_dataset().

    Handles common cases:
        (H, W)       → (1, H, W, 1)
        (N, H, W)    → (N, H, W, 1)
        (N, H, W, 1) → passed through

    Parameters
    ----------
    masks : np.ndarray

    Returns
    -------
    np.ndarray with shape (N, H, W, 1)
    """
    if masks.ndim == 2:
        masks = masks[np.newaxis, ..., np.newaxis]
    elif masks.ndim == 3:
        if masks.shape[-1] == 1 and masks.shape[0] != 1:
            pass
        else:
            masks = masks[..., np.newaxis]
    elif masks.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    return masks


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

        # Use predict() so this works for any subclass (Keras or nnUNet)
        pred_batch = self.predict(X_test[idxs])

        for i, idx in enumerate(idxs):
            img = X_test[idx].squeeze()
            true_mask = y_test[idx].squeeze()
            pred_mask = pred_batch[i].squeeze()

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
        """Plot training curves from self.history (Dice/IoU, loss, precision/recall).

        NOTE: When training should have something like: self.history = self.model.fit(...)
        NOTE: Sometimes keras saves metric names differently than expected. If a column
              is missing, run: print(self.history.history.keys()) to see the actual names.
        """
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

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', threshold={self.threshold})"


# =============================================================================
# Keras subclass — thin wrapper over keras_unet_collection
# =============================================================================
class KerasSegModel(SegModel):
    """Wraps keras_unet_collection models (UNet, UNet++, UNet3++)."""

    def __init__(self, model_name, threshold=0.5):
        if model_name not in SUPPORTED_KERAS_MODELS:
            raise ValueError(
                f"Unknown architecture '{model_name}'. "
                f"Choose from {SUPPORTED_KERAS_MODELS}."
            )
        super().__init__(model_name, threshold)
        self.model = self._build_model(model_name)   # untrained tf.keras.Model

    # ------------------------------------------------------------------
    # Loss / metric factories (static so from_checkpoint can reuse them)
    # ------------------------------------------------------------------
    @staticmethod
    def _dice_coef(y_true, y_pred):
        import tensorflow as tf
        y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + SMOOTH) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + SMOOTH
        )

    @staticmethod
    def _iou_coef(y_true, y_pred):
        import tensorflow as tf
        y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = (
            tf.keras.backend.sum(y_true_f)
            + tf.keras.backend.sum(y_pred_f)
            - intersection
        )
        return (intersection + SMOOTH) / (union + SMOOTH)

    @staticmethod
    def _bce_dice_loss(y_true, y_pred):
        import tensorflow as tf
        y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        dice_loss = 1.0 - (2.0 * intersection + SMOOTH) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + SMOOTH
        )
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce + dice_loss

    @classmethod
    def _custom_objects(cls):
        return {
            "bce_dice_loss": cls._bce_dice_loss,
            "dice_coef":     cls._dice_coef,
            "iou_coef":      cls._iou_coef,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path, threshold=0.5):
        """
        Load a trained .keras/.h5 file via tf.keras.models.load_model.
        Uses custom_objects for dice_coef / iou_coef / bce_dice_loss.
        """
        import tensorflow as tf
        instance = cls.__new__(cls)
        super(KerasSegModel, instance).__init__(checkpoint_path, threshold)
        instance.model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects=cls._custom_objects(),
        )
        instance.history = None
        return instance

    def _build_model(self, name):
        """
        Dispatch on name → keras_unet_collection.models:
            "unet"     → unet_2d
            "unet++"   → unet_plus_2d
            "unet3++"  → unet_3plus_2d
        All built with INPUT_SIZE, FILTER_NUM, n_labels=1, Sigmoid output.
        """
        from keras_unet_collection import models as kuc

        if name == "unet":
            return kuc.unet_2d(
                input_size=INPUT_SIZE,
                filter_num=FILTER_NUM,
                n_labels=1,
                output_activation="Sigmoid",
                name="unet",
            )
        elif name == "unet++":
            return kuc.unet_plus_2d(
                input_size=INPUT_SIZE,
                filter_num=FILTER_NUM,
                n_labels=1,
                output_activation="Sigmoid",
                name="unetpp",
            )
        elif name == "unet3++":
            return kuc.unet_3plus_2d(
                input_size=INPUT_SIZE,
                filter_num_down=FILTER_NUM,
                n_labels=1,
                output_activation="Sigmoid",
                name="unet3pp",
            )

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
        import tensorflow as tf

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._bce_dice_loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                self._dice_coef,
                self._iou_coef,
            ],
        )

        ckpt_name = f"best_{self.model_name.replace('++', 'pp').replace('+', 'p')}.keras"
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_dice_coef", patience=10,
                    mode="max", restore_best_weights=True, verbose=1,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    ckpt_name, monitor="val_dice_coef",
                    save_best_only=True, mode="max", verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5,
                    min_lr=1e-7, verbose=1,
                ),
            ]

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return self

    def predict(self, images, threshold=None):
        """self.model.predict(images) > (threshold or self.threshold), cast to uint8."""
        t = threshold if threshold is not None else self.threshold
        preds = self.model.predict(images, verbose=0)
        return (preds > t).astype(np.uint8)

    def evaluate(self, X_test, y_test):
        """self.model.evaluate(...) → zip with model.metrics_names → dict."""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        names = self.model.metrics_names
        metrics = dict(zip(names, results))
        # Expose canonical 'dice' and 'iou' keys for cross-framework comparison
        if "dice_coef" in metrics:
            metrics["dice"] = metrics["dice_coef"]
        if "iou_coef" in metrics:
            metrics["iou"] = metrics["iou_coef"]
        return metrics


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
                 device="cuda", threshold=0.5, dataset_id=101):
        super().__init__(f"nnunet-{configuration}", threshold)
        self.dataset_name = dataset_name        # e.g. "Dataset101_CardiacUS"
        self.model_folder = model_folder        # nnUNet_results/<dataset>/nnUNetTrainer__nnUNetPlans__2d
        self.configuration = configuration      # "2d" | "3d_fullres" | "3d_lowres"
        self.folds = folds                      # CV folds used for ensemble inference
        self.checkpoint_name = checkpoint_name  # "checkpoint_final.pth" or "checkpoint_best.pth"
        self.device = device                    # "cuda" | "cpu"
        self.dataset_id = dataset_id            # numeric ID used by nnU-Net CLI commands
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

    # ------------------------------------------------------------------
    # Step 1: Environment setup
    # ------------------------------------------------------------------

    def _check_environment(self):
        import os

        required_vars = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
        missing = [v for v in required_vars if os.environ.get(v) is None]

        if missing:
            raise EnvironmentError(
                f"Missing nnU-Net environment variables: {missing}\n"
                "Call setup_environment(base_dir=...) first."
            )

    def setup_environment(self, base_dir=None):
        """
        Check and set nnU-Net environment variables (nnUNet_raw,
        nnUNet_preprocessed, nnUNet_results). Must be called before training.

        If base_dir is provided, creates the directory structure and sets
        the env vars. If not provided, validates they are already set.

        Parameters
        ----------
        base_dir : str or None
            Root directory for nnU-Net data.

        Returns
        -------
        dict — {"nnUNet_raw": ..., "nnUNet_preprocessed": ..., "nnUNet_results": ...}
        """
        import os

        required_vars = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]

        if base_dir is not None:
            os.makedirs(base_dir, exist_ok=True)
            paths = {}
            for var in required_vars:
                path = os.path.join(base_dir, var)
                os.makedirs(path, exist_ok=True)
                os.environ[var] = path
                paths[var] = path
            print(f"✓ Environment configured at {base_dir}")
            for var, path in paths.items():
                print(f"  {var} = {path}")
            return paths

        paths = {}
        missing = []
        for var in required_vars:
            val = os.environ.get(var)
            if val is None:
                missing.append(var)
            else:
                paths[var] = val

        if missing:
            raise EnvironmentError(
                f"Missing nnU-Net environment variables: {missing}\n"
                f"Either set them manually or call "
                f"setup_environment('/path/to/base_dir') to create them."
            )

        print("✓ Environment variables verified")
        for var, path in paths.items():
            exists = "✓" if os.path.isdir(path) else "✗ NOT FOUND"
            print(f"  {var} = {path}  [{exists}]")

        return paths

    # ------------------------------------------------------------------
    # Step 2-4: Data conversion + folder layout
    # ------------------------------------------------------------------
    def prepare_dataset(self, images, masks, file_format="nifti"):
        """
        Convert numpy arrays into nnU-Net's required folder structure.

        Parameters
        ----------
        images : np.ndarray
            For nifti: shape (N, H, W, 1) float32 grayscale
            For png:   shape (N, H, W, 3) uint8 RGB  OR  (N, H, W, 1) float32
        masks : np.ndarray
            Shape (N, H, W, 1) or (N, H, W) with values {0, 1} or {0, 255}
        file_format : str
            "nifti" or "png"

        Returns
        -------
        str — path to the created dataset directory
        """
        import os
        import json

        if file_format not in ("nifti", "png"):
            raise ValueError(f"file_format must be 'nifti' or 'png', got '{file_format}'")

        nnunet_raw = os.environ.get("nnUNet_raw")
        if nnunet_raw is None:
            raise EnvironmentError("nnUNet_raw is not set. Call setup_environment() first.")

        dataset_dir = os.path.join(nnunet_raw, self.dataset_name)
        images_dir = os.path.join(dataset_dir, "imagesTr")
        labels_dir = os.path.join(dataset_dir, "labelsTr")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Normalize mask shape to (N, H, W)
        masks_2d = masks.squeeze() if masks.ndim == 4 else masks
        if masks_2d.ndim == 2:
            masks_2d = masks_2d[np.newaxis, ...]

        # Validate and normalize mask values
        unique_vals = np.unique(masks_2d)
        if np.array_equal(unique_vals, np.array([0, 255], dtype=unique_vals.dtype)):
            print("  Converting masks from [0, 255] to [0, 1]")
            masks_2d = (masks_2d > 0).astype(np.uint8)
        elif not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"Expected binary masks with values [0,1] or [0,255], "
                f"got unique values: {unique_vals}"
            )
        else:
            masks_2d = masks_2d.astype(np.uint8)

        if file_format == "png":
            self._write_png_cases(images, masks_2d, images_dir, labels_dir)
            file_ending = ".png"
            if images.ndim == 4 and images.shape[-1] == 3:
                channel_names = {"0": "R", "1": "G", "2": "B"}
            else:
                channel_names = {"0": "grayscale"}
        else:
            self._write_nifti_cases(images, masks_2d, images_dir, labels_dir)
            file_ending = ".nii.gz"
            channel_names = {"0": "ultrasound"}

        dataset_json = {
            "channel_names": channel_names,
            "labels": {"background": 0, "foreground": 1},
            "numTraining": len(images),
            "file_ending": file_ending,
            "name": self.dataset_name,
            "description": "Cardiac ultrasound segmentation",
        }

        with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)

        print(f"✓ Wrote {len(images)} cases to {dataset_dir}")
        print(f"  Format: {file_format} ({file_ending})")
        print(f"  imagesTr: {len(images)} files")
        print(f"  labelsTr: {len(images)} files")
        return dataset_dir

    def _write_nifti_cases(self, images, masks_2d, images_dir, labels_dir):
        """Write image/mask pairs as NIfTI .nii.gz files."""
        import os
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required for NIfTI. Install with: pip install nibabel")

        affine = np.eye(4)
        for i in range(len(images)):
            case_id = f"case_{i:04d}"
            img_2d = images[i].squeeze().astype(np.float32)
            img_nifti = nib.Nifti1Image(img_2d[..., np.newaxis], affine)
            mask_nifti = nib.Nifti1Image(masks_2d[i][..., np.newaxis], affine)
            nib.save(img_nifti, os.path.join(images_dir, f"{case_id}_0000.nii.gz"))
            nib.save(mask_nifti, os.path.join(labels_dir, f"{case_id}.nii.gz"))

    def _write_png_cases(self, images, masks_2d, images_dir, labels_dir):
        """Write image/mask pairs as PNG files."""
        import os
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for PNG. Install with: pip install Pillow")

        for i in range(len(images)):
            case_id = f"case_{i:04d}"
            img = images[i]

            if img.ndim == 3 and img.shape[-1] == 3:
                if img.dtype != np.uint8:
                    img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(images_dir, f"{case_id}_0000.png"))
            elif img.ndim == 3 and img.shape[-1] == 1:
                img_2d = img.squeeze()
                if img_2d.dtype != np.uint8:
                    img_2d = np.clip(img_2d * 255 if img_2d.max() <= 1.0 else img_2d, 0, 255).astype(np.uint8)
                Image.fromarray(img_2d, mode="L").save(os.path.join(images_dir, f"{case_id}_0000.png"))
            else:
                raise ValueError(f"Image {i} has unexpected shape {img.shape}. Expected (H,W,3) or (H,W,1).")

            Image.fromarray(masks_2d[i].astype(np.uint8)).save(os.path.join(labels_dir, f"{case_id}.png"))

    # ------------------------------------------------------------------
    # Step 5: Validation
    # ------------------------------------------------------------------
    def validate_dataset(self):
        """
        Check that the dataset directory is correctly structured for nnU-Net.
        Detects file format from dataset.json automatically.

        Returns True if all checks pass, raises ValueError with details if not.
        """
        import os
        import json
        import re

        nnunet_raw = os.environ.get("nnUNet_raw")
        if nnunet_raw is None:
            raise EnvironmentError("nnUNet_raw is not set. Call setup_environment() first.")

        dataset_dir = os.path.join(nnunet_raw, self.dataset_name)
        errors = []

        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}\nCall prepare_dataset() first.")

        json_path = os.path.join(dataset_dir, "dataset.json")
        ds = {}
        if not os.path.isfile(json_path):
            errors.append("dataset.json is missing")
        else:
            with open(json_path) as f:
                ds = json.load(f)
            for field in ["channel_names", "labels", "numTraining", "file_ending"]:
                if field not in ds:
                    errors.append(f"dataset.json missing required field: '{field}'")
            if "labels" in ds:
                if "background" not in ds["labels"]:
                    errors.append("dataset.json labels missing 'background'")
                if ds["labels"].get("background") != 0:
                    errors.append("dataset.json labels['background'] should be 0")

        file_ending = ds.get("file_ending", ".nii.gz")
        images_dir = os.path.join(dataset_dir, "imagesTr")
        labels_dir = os.path.join(dataset_dir, "labelsTr")

        if not os.path.isdir(images_dir):
            errors.append("imagesTr/ directory is missing")
        if not os.path.isdir(labels_dir):
            errors.append("labelsTr/ directory is missing")

        image_files = []
        label_files = []

        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(file_ending)])
            label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(file_ending)])

            if "numTraining" in ds and len(image_files) != ds["numTraining"]:
                errors.append(f"Image count ({len(image_files)}) doesn't match dataset.json numTraining ({ds['numTraining']})")

            ext_escaped = re.escape(file_ending)
            img_pattern = re.compile(rf"^case_\d{{4}}_0000{ext_escaped}$")
            lbl_pattern = re.compile(rf"^case_\d{{4}}{ext_escaped}$")

            bad_images = [f for f in image_files if not img_pattern.match(f)]
            bad_labels = [f for f in label_files if not lbl_pattern.match(f)]

            if bad_images:
                errors.append(f"{len(bad_images)} images don't follow naming convention: {bad_images[:3]}...")
            if bad_labels:
                errors.append(f"{len(bad_labels)} labels don't follow naming convention: {bad_labels[:3]}...")

            suffix_img = f"_0000{file_ending}"
            image_ids = {f.replace(suffix_img, "") for f in image_files}
            label_ids = {f.replace(file_ending, "") for f in label_files}

            if image_ids - label_ids:
                errors.append(f"{len(image_ids - label_ids)} images have no matching label")
            if label_ids - image_ids:
                errors.append(f"{len(label_ids - image_ids)} labels have no matching image")

        if errors:
            error_list = "\n  - ".join(errors)
            raise ValueError(f"Dataset validation failed with {len(errors)} error(s):\n  - {error_list}")

        print(f"✓ Dataset validation passed")
        print(f"  Format: {file_ending}")
        print(f"  {len(image_files)} images, {len(label_files)} labels")
        return True

    # ------------------------------------------------------------------
    # Step 6: Plan and preprocess
    # ------------------------------------------------------------------
    def plan_and_preprocess(self, verify_integrity=True):
        """
        Run nnU-Net's dataset fingerprinting, experiment planning, and preprocessing.

        Surfaces the generated plan (patch sizes, batch size, available configurations)
        to the user after completion.

        Parameters
        ----------
        verify_integrity : bool
            If True, runs nnU-Net's built-in dataset integrity check.

        Returns
        -------
        dict — parsed plans.json, or empty dict if plans file not found
        """
        import os
        import json
        import subprocess

        for var in ["nnUNet_raw", "nnUNet_preprocessed"]:
            if os.environ.get(var) is None:
                raise EnvironmentError(f"{var} is not set. Call setup_environment() first.")

        print(f"\n{'='*60}")
        print(f"  Planning and preprocessing Dataset {self.dataset_id}")
        print(f"{'='*60}")

        cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(self.dataset_id)]
        if verify_integrity:
            cmd.append("--verify_dataset_integrity")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"nnUNetv2_plan_and_preprocess failed (exit code {result.returncode})")

        print("✓ Planning and preprocessing complete")

        plans_path = os.path.join(
            os.environ["nnUNet_preprocessed"], self.dataset_name, "nnUNetPlans.json"
        )
        if os.path.isfile(plans_path):
            with open(plans_path) as f:
                plans = json.load(f)
            configs = list(plans.get("configurations", {}).keys())
            print(f"\n  Available configurations: {configs}")
            if self.configuration in plans.get("configurations", {}):
                config = plans["configurations"][self.configuration]
                print(f"  Selected: {self.configuration}")
                print(f"    Patch size: {config.get('patch_size', 'N/A')}")
                print(f"    Batch size: {config.get('batch_size', 'N/A')}")
            return plans

        print("  ⚠ Could not find plans.json to display summary")
        return {}

    # ------------------------------------------------------------------
    # Step 8: Find best configuration (post-training)
    # ------------------------------------------------------------------
    def find_best_configuration(self):
        """
        Run nnU-Net's automatic configuration comparison after training.
        Should be called after training all desired configurations.

        Returns
        -------
        str — stdout from nnUNetv2_find_best_configuration
        """
        import subprocess

        print(f"\n{'='*60}")
        print(f"  Finding best configuration for Dataset {self.dataset_id}")
        print(f"{'='*60}")

        cmd = ["nnUNetv2_find_best_configuration", str(self.dataset_id)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"nnUNetv2_find_best_configuration failed (exit code {result.returncode})")

        print(result.stdout)
        return result.stdout

    def fit(self, X_train=None, y_train=None, folds="all_cv",
            save_softmax=False, continue_training=False):
        import os
        import subprocess

        if X_train is not None and y_train is not None:
            self.setup_environment()
            self.prepare_dataset(X_train, y_train)
            self.validate_dataset()
            self.plan_and_preprocess()

        nnunet_results = os.environ.get("nnUNet_results")
        if nnunet_results is None:
            raise EnvironmentError("nnUNet_results is not set. Call setup_environment() first.")

        if folds == "all_cv":
            folds_to_train = [0, 1, 2, 3, 4]
        else:
            folds_to_train = [int(folds)]

        for f in folds_to_train:
            print(f"\n{'='*60}")
            print(f"  Training fold {f}")
            print(f"{'='*60}")

            cmd = ["nnUNetv2_train", str(self.dataset_id), self.configuration, str(f)]

            if save_softmax:
                cmd.append("--npz")

            if continue_training:
                cmd.append("--c")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                raise RuntimeError(
                    f"nnUNetv2_train failed on fold {f} (exit code {result.returncode})"
                )

            print(f"✓ Fold {f} complete")

        self.model_folder = os.path.join(
            nnunet_results,
            self.dataset_name,
            f"nnUNetTrainer__nnUNetPlans__{self.configuration}",
        )
        self.folds = tuple(folds_to_train)
        self._load_predictor()

        print(f"\n✓ Training complete — {len(folds_to_train)} fold(s)")
        return self

    def resume_training(self, folds="all_cv", save_softmax=False):
        """Resume a previously interrupted training run."""
        return self.fit(
            X_train=None,
            y_train=None,
            folds=folds,
            save_softmax=save_softmax,
            continue_training=True,
        )

    # ------------------------------------------------------------------
    # Step 9: Inference
    # ------------------------------------------------------------------
    def _load_predictor(self):
        """
        Build self._predictor = nnUNetPredictor(...) and call
        initialize_from_trained_model_folder.
        """
        import os

        self._check_environment()

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

    def predict(self, images, threshold=None):
        """
        Per-image inference via self._predictor.predict_single_npy_array,
        then stack and binarize.

        Raises RuntimeError if self._predictor is None.
        """
        self._check_environment()

        if self._predictor is None:
            raise RuntimeError(
                "No trained model loaded. Either:\n"
                "  - Use NnUNetSegModel.from_checkpoint(model_folder=...) to load trained weights\n"
                "  - Call .fit(X_train, y_train) to train first"
            )

        if threshold is None:
            threshold = self.threshold

        results = []
        n = len(images)
        for i in range(n):
            if n > 5 and (i + 1) % 5 == 0:
                print(f"  Predicting {i + 1}/{n}...", end="\r")

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

        if n > 5:
            print(f"  Predicting {n}/{n}... done")

        preds = np.array(results, dtype=np.float32)
        return (preds > threshold).astype(np.uint8)

    def evaluate(self, X_test, y_test):
        """
        Call self.predict(X_test), then compute dice/iou against y_test with numpy.
        Returns dict: {"dice": ..., "iou": ..., "precision": ..., "recall": ...}.
        """
        preds = self.predict(X_test)

        y_true = y_test.flatten().astype(np.float32)
        y_pred = preds.flatten().astype(np.float32)

        intersection = np.sum(y_true * y_pred)
        dice = (2.0 * intersection + SMOOTH) / (np.sum(y_true) + np.sum(y_pred) + SMOOTH)

        union = np.sum(y_true) + np.sum(y_pred) - intersection
        iou = (intersection + SMOOTH) / (union + SMOOTH)

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
