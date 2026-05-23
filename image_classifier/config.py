"""Hyperparameters and paths for training and inference."""

from pathlib import Path

# Project root (parent of image_classifier package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Loader is selected in dataset_source.ACTIVE_DATASET ("cifar10" | "image_directory").
# Used only for image_directory: folder per class, e.g. data/train/cat/
DATA_DIR = PROJECT_ROOT / "data" / "train"
VALIDATION_SPLIT = 0.1
SEED = 42

# Image pipeline
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 64

# Training
EPOCHS = 12
LEARNING_RATE = 3e-4

# Performance toggles (especially helpful on GPU)
ENABLE_GPU_MEMORY_GROWTH = True
ENABLE_MIXED_PRECISION = True
ENABLE_XLA = False

# Artifacts
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
# Keras 3: model.save() must use a .keras or .h5 path (not a bare directory).
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
MODEL_EXPORT_PATH = SAVED_MODEL_DIR / "model.keras"
NORMALIZED_CM_EXCEL_PATH = PROJECT_ROOT / "normalized_confusion_matrix.xlsx"
