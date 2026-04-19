# Team 1 → Team 2 Handoff: UNet Bounding Boxes & Masks

## What's Included

| File | Description | Shape |
|------|-------------|-------|
| `bboxes_for_team2.npy` | Bounding boxes from UNet predictions | `(N, 4)` — `[x_min, y_min, x_max, y_max]` |
| `masks_for_team2.npy` | Full binary predicted masks | `(N, 320, 320, 1)` — uint8 `{0, 1}` |
| `images_test.npy` | Test images (so you can verify alignment) | `(N, 320, 320, 1)` — float32 `[0, 1]` |
| `team1_handoff.py` | Helper script with loaders and visualization | — |

## Quick Start

```python
import numpy as np

# Load bounding boxes
bboxes = np.load("bboxes_for_team2.npy")
print(bboxes.shape)  # (28, 4)
print(bboxes[0])     # [x_min, y_min, x_max, y_max]

# Load predicted masks (if you need the full mask instead of just bbox)
masks = np.load("masks_for_team2.npy")

# Load test images (to verify everything lines up)
images = np.load("images_test.npy")
```

## Using with SAM

```python
# Option 1: Use bbox as SAM box prompt
input_box = np.array([bboxes[0]])  # SAM expects shape (1, 4)

# Option 2: Use center point as SAM point prompt
cx = (bboxes[0][0] + bboxes[0][2]) // 2
cy = (bboxes[0][1] + bboxes[0][3]) // 2
input_point = np.array([[cx, cy]])
input_label = np.array([1])
```

## Using the Helper Script

```python
from team1_handoff import load_bboxes, load_masks, bbox_to_sam_prompt, visualize

bboxes = load_bboxes()
sam_prompt = bbox_to_sam_prompt(bboxes[0])
visualize(idx=0)
```

## Notes

- Image size is **320×320** (grayscale, 1 channel)
- Bounding boxes with `[-1, -1, -1, -1]` mean the model found **no foreground** in that image
- These predictions come from our **UNet++** model (Dice: 0.9459 on test set)
- Dataset: cardiac ultrasound (heart ventricle segmentation)

## Contact

If you need predictions on different images or in a different format, let us know and we'll re-export.
