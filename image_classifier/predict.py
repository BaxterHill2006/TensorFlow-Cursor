"""Run inference with a saved Keras model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from image_classifier import config
from image_classifier.dataset import dataset_for_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(config.MODEL_EXPORT_PATH))
    parser.add_argument("images", nargs="+", help="One or more image file paths")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    path = Path(args.model)
    names_file = path / "class_names.json" if path.is_dir() else path.parent / "class_names.json"
    if names_file.is_file():
        class_names = json.loads(names_file.read_text(encoding="utf-8"))
    else:
        class_names = [str(i) for i in range(model.output_shape[-1])]

    paths = [str(p) for p in args.images]
    ds = dataset_for_inference(paths)
    probs = model.predict(ds, verbose=0)

    for path, p in zip(paths, probs):
        idx = int(np.argmax(p))
        print(f"{path} -> {class_names[idx]} ({float(p[idx]):.3f})")


if __name__ == "__main__":
    main()
