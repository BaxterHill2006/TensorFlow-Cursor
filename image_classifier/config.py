"""Hyperparameters and paths for training and inference."""

from pathlib import Path

# Project root (parent of image_classifier package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Loader is selected in dataset_source.ACTIVE_DATASET ("cifar10" | "image_directory").
# Used only for image_directory: folder per class, e.g. data/train/cat/
DATA_DIR = PROJECT_ROOT / "data" / "train"
VALIDATION_SPLIT = 0.2
SEED = 42

# Image pipeline
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Training
EPOCHS = 10
LEARNING_RATE = 1e-4

# Performance toggles (especially helpful on GPU)
ENABLE_GPU_MEMORY_GROWTH = True
ENABLE_MIXED_PRECISION = True
ENABLE_XLA = False

# Artifacts
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_EXPORT_PATH = PROJECT_ROOT / "saved_model"
