"""CNN model builders for image classification."""

from __future__ import annotations

import tensorflow as tf

from image_classifier import config


def build_transfer_learning_model(num_classes: int, trainable_base: bool = False) -> tf.keras.Model:
    """
    MobileNetV2 feature extractor + global pooling + dense head.
    Set trainable_base=True to fine-tune the backbone (use a lower learning rate).
    """
    shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    base = tf.keras.applications.MobileNetV2(
        input_shape=shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return tf.keras.Model(inputs, outputs, name="mobilenet_classifier")


def build_simple_cnn(num_classes: int) -> tf.keras.Model:
    """Small CNN from scratch (good for tiny datasets or learning)."""
    shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32"),
        ],
        name="simple_cnn",
    )
