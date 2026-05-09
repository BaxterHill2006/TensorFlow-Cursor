"""Train the image classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from image_classifier import config
from image_classifier.dataset import make_datasets
from image_classifier.model import build_simple_cnn, build_transfer_learning_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("transfer", "cnn"),
        default="transfer",
        help="transfer = MobileNetV2 + head; cnn = small CNN from scratch",
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_ds, val_ds, num_classes, class_names = make_datasets()

    if args.architecture == "transfer":
        model = build_transfer_learning_model(num_classes, trainable_base=False)
    else:
        model = build_simple_cnn(num_classes)

    epochs = args.epochs if args.epochs is not None else config.EPOCHS

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.CHECKPOINT_DIR / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[ckpt])

    config.MODEL_EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_EXPORT_PATH)
    names_json = json.dumps(class_names)
    (Path(config.MODEL_EXPORT_PATH) / "class_names.json").write_text(names_json, encoding="utf-8")
    (config.CHECKPOINT_DIR / "class_names.json").write_text(names_json, encoding="utf-8")
    print("Classes (index order):", class_names)


if __name__ == "__main__":
    main()
