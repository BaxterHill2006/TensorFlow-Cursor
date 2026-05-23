"""Load and preprocess datasets; dispatch is driven by dataset_source.ACTIVE_DATASET."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from image_classifier import config
from image_classifier.dataset_source import ACTIVE_DATASET

CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _unpickle_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]
    labels = np.array(batch[b"labels"], dtype=np.int32)
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_cifar10_train(batches_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load training images and labels from cifar-10-batches-py (data_batch_1..5)."""
    root = batches_dir or config.CIFAR10_BATCHES_DIR
    xs, ys = [], []
    for i in range(1, 6):
        x, y = _unpickle_batch(root / f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def load_cifar10_test(test_batch_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load test images and labels from a CIFAR-10 batch pickle (default: config.CIFAR10_TEST_BATCH_PATH)."""
    path = test_batch_path or config.CIFAR10_TEST_BATCH_PATH
    return _unpickle_batch(path)


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def _preprocess_resize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])
    label = tf.cast(tf.reshape(label, []), tf.int32)
    return image, label


def _make_cifar10_datasets():
    x_train, y_train = load_cifar10_train()

    n = len(x_train)
    indices = np.arange(n)
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(indices)

    val_n = max(1, int(n * config.VALIDATION_SPLIT))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_va, y_va = x_train[val_idx], y_train[val_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .map(_preprocess_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000, seed=config.SEED, reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_va, y_va))
        .map(_preprocess_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    num_classes = len(CIFAR10_CLASS_NAMES)
    class_names = list(CIFAR10_CLASS_NAMES)
    return train_ds, val_ds, num_classes, class_names


def _make_image_directory_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes, class_names


def make_datasets():
    """
    Returns (train_ds, val_ds, num_classes, class_names) for the active dataset.
    """
    key = ACTIVE_DATASET.strip().lower()
    if key == "cifar10":
        return _make_cifar10_datasets()
    if key == "image_directory":
        return _make_image_directory_datasets()
    raise ValueError(
        f"Unknown ACTIVE_DATASET {ACTIVE_DATASET!r}; use 'cifar10' or 'image_directory' "
        f"(see image_classifier/dataset_source.py)."
    )


def dataset_for_inference(image_paths):
    """Build a tf.data.Dataset from a list of image file paths (no labels)."""
    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_and_preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [config.IMG_HEIGHT, config.IMG_WIDTH])
        image = image / 255.0
        return image

    return ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(config.BATCH_SIZE)
