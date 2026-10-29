"""Shared constants. Training knobs that should stay easy to change live here
and are also exposed as command-line flags where the project promised them.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# COVID is the positive class. Normal is the negative class.
POSITIVE_CLASS = "COVID"
NEGATIVE_CLASS = "Normal"
CLASS_TO_INDEX = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
INDEX_TO_CLASS = {0: NEGATIVE_CLASS, 1: POSITIVE_CLASS}
CLASS_DIRECTORY_NAMES = (POSITIVE_CLASS, NEGATIVE_CLASS)

# Group isolation is more important than hitting these fractions exactly.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL = 0

# Mild training augmentation only. No crops and no flips.
TRAIN_ROTATION_DEGREES = 10
TRAIN_TRANSLATE = 0.05

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_LR_FINETUNE = 1e-4
DEFAULT_EPOCHS = 5
DEFAULT_FINETUNE_EPOCHS = 5
DEFAULT_PATIENCE = 3
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_NUM_WORKERS = 0

SMOKE_TRAIN_PER_CLASS = 16
SMOKE_VAL_PER_CLASS = 8
SMOKE_BATCH_SIZE = 8
SMOKE_EPOCHS = 1

DISCLAIMER = (
    "Educational experiment only. This is not a clinical diagnostic tool. "
    "A model score is not a calibrated probability of infection."
)

if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
    raise RuntimeError("Train, validation, and test ratios must sum to 1.")
