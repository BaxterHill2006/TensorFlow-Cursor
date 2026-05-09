"""Load and preprocess image datasets from directory structure."""

from __future__ import annotations

import tensorflow as tf

from image_classifier import config


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def make_datasets():
    """
    Expects: config.DATA_DIR / class_a/*.jpg, config.DATA_DIR / class_b/*.jpg, ...
    Returns: (train_ds, val_ds), class_names
    """
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
