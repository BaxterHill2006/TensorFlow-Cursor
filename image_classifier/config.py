"""Hyperparameters and paths for training and inference."""

from pathlib import Path

# Project root (parent of image_classifier package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset: folder per class, e.g. data/train/cat/, data/train/dog/
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

# Artifacts
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_EXPORT_PATH = PROJECT_ROOT / "saved_model"
