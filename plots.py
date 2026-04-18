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