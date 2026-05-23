"""Evaluate a saved checkpoint on the CIFAR-10 test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import tensorflow as tf

from image_classifier import config
from image_classifier.dataset import CIFAR10_CLASS_NAMES


def load_cifar10_test(
    image_height: int = config.IMG_HEIGHT,
    image_width: int = config.IMG_WIDTH,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load CIFAR-10 test images and labels, resized and normalized."""
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x = tf.image.resize(
        tf.cast(x_test, tf.float32) / 255.0,
        [image_height, image_width],
    )
    y = tf.reshape(tf.cast(y_test, tf.int32), [-1])
    return x, y


def _class_names_for_model(model_path: str, num_classes: int) -> list[str]:
    path = Path(model_path)
    names_file = path / "class_names.json" if path.is_dir() else path.parent / "class_names.json"
    if names_file.is_file():
        return json.loads(names_file.read_text(encoding="utf-8"))
    if num_classes == len(CIFAR10_CLASS_NAMES):
        return list(CIFAR10_CLASS_NAMES)
    return [str(i) for i in range(num_classes)]


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration; NumPy 2.x renamed trapz to trapezoid."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = (y_true == 1).astype(np.int64)
    order = np.argsort(-scores)
    labels = labels[order]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = tps / labels.sum()
    fpr = fps / (len(labels) - labels.sum())
    # Include (0,0) and unique score thresholds
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return _trapz(tpr, fpr)


def _binary_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels = (y_true == 1).astype(np.int64)
    order = np.argsort(-scores)
    labels = labels[order]
    if labels.sum() == 0:
        return float("nan")
    tps = np.cumsum(labels)
    precision = tps / np.arange(1, len(labels) + 1)
    recall = tps / labels.sum()
    precision = np.concatenate([[1.0], precision, [0.0]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    return _trapz(precision, recall)


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    n_classes: int,
) -> list[dict[str, float]]:
    n = len(y_true)
    rows: list[dict[str, float]] = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(n - tp - fn - fp)

        tp_rate = tp / (tp + fn) if (tp + fn) else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp_rate
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom else 0.0

        binary_true = (y_true == c).astype(np.int64)
        roc = _binary_roc_auc(binary_true, y_prob[:, c])
        prc = _binary_pr_auc(binary_true, y_prob[:, c])

        rows.append(
            {
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
                "roc": roc,
                "prc": prc,
            }
        )
    return rows


def _probability_errors(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> dict[str, float]:
    n = len(y_true)
    one_hot = np.zeros((n, n_classes), dtype=np.float64)
    one_hot[np.arange(n), y_true] = 1.0
    diff = y_prob - one_hot
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    mean_actual = one_hot.mean(axis=0)
    rae_denom = float(np.sum(np.abs(one_hot - mean_actual)))
    rrse_denom = float(np.sqrt(np.sum((one_hot - mean_actual) ** 2)))
    rae = float(np.sum(np.abs(diff)) / rae_denom * 100.0) if rae_denom else 0.0
    rrse = float(np.sqrt(np.sum(diff**2)) / rrse_denom * 100.0) if rrse_denom else 0.0
    return {"mae": mae, "rmse": rmse, "rae": rae, "rrse": rrse}


def _column_normalized_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Normalize so each predicted class column sums to 1."""
    col_sums = cm.sum(axis=0, keepdims=True, dtype=np.float64)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    return cm.astype(np.float64) / col_sums


def write_normalized_confusion_matrix_excel(
    cm_norm: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> Path:
    """Write column-normalized confusion matrix to Excel with a 3-color heat map."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = cm_norm.shape[0]
    letters = [chr(ord("a") + i) for i in range(n)]

    wb = Workbook()
    ws = wb.active
    ws.title = "Normalized CM"

    ws.cell(row=1, column=1, value="Actual \\ Predicted")
    for j, letter in enumerate(letters, start=2):
        ws.cell(row=1, column=j, value=letter)
    for i, name in enumerate(class_names):
        row = i + 2
        ws.cell(row=row, column=1, value=f"{letters[i]} = {name}")
        for j in range(n):
            ws.cell(row=row, column=j + 2, value=float(cm_norm[i, j]))

    data_range = f"B2:{get_column_letter(n + 1)}{n + 1}"
    color_scale = ColorScaleRule(
        start_type="num",
        start_value=0,
        start_color="D62828",
        mid_type="num",
        mid_value=0.1,
        mid_color="FFD166",
        end_type="num",
        end_value=1,
        end_color="28D645",
    )
    ws.conditional_formatting.add(data_range, color_scale)

    for col in range(1, n + 2):
        ws.column_dimensions[get_column_letter(col)].width = 14

    try:
        os.remove(path)
    except:
        print(f"Error removing file: {path}")

    wb.save(path)
    return path


def _format_confusion_matrix_lines(
    cm: np.ndarray,
    class_names: list[str],
    letters: list[str],
    *,
    decimal: bool = False,
    decimals: int = 3,
) -> list[str]:
    """Format confusion matrix with aligned column headers."""
    gap = "  "
    if decimal:
        col_width = decimals + 3  # room for "0." and digits, e.g. "0.781"
    else:
        col_width = max(3, len(str(int(cm.max()))), max(len(letter) for letter in letters))

    def matrix_row(cells: list[str | int | float]) -> str:
        parts: list[str] = []
        for cell in cells:
            if isinstance(cell, str):
                parts.append(f"{cell:>{col_width}}")
            elif decimal:
                parts.append(f"{float(cell):>{col_width}.{decimals}f}")
            else:
                parts.append(f"{int(cell):>{col_width}d}")
        return " " + gap.join(parts)

    lines = [matrix_row(letters) + "   <-- classified as"]
    for i, name in enumerate(class_names):
        if decimal:
            row_vals = [float(cm[i, j]) for j in range(cm.shape[0])]
        else:
            row_vals = [int(cm[i, j]) for j in range(cm.shape[0])]
        lines.append(matrix_row(row_vals) + f" |   {letters[i]} = {name}")
    return lines


def _weighted_average(rows: list[dict[str, float]], support: np.ndarray) -> dict[str, float]:
    weights = support / support.sum()
    out: dict[str, float] = {}
    for key in ("tp_rate", "fp_rate", "precision", "recall", "f1", "mcc", "roc", "prc"):
        out[key] = float(np.average([r[key] for r in rows], weights=weights))
    return out


def format_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    loss: float | None = None,
) -> str:
    n_classes = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, n_classes)
    correct = int(np.trace(cm))
    total = int(cm.sum())
    incorrect = total - correct
    accuracy = correct / total if total else 0.0
    kappa = _cohen_kappa(cm)
    prob_err = _probability_errors(y_true, y_prob, n_classes)
    per_class = _per_class_metrics(y_true, y_pred, y_prob, cm, n_classes)
    support = cm.sum(axis=1)
    weighted = _weighted_average(per_class, support)

    letters = [chr(ord("a") + i) for i in range(n_classes)]
    lines: list[str] = []

    if loss is not None:
        lines.append(f"Loss (sparse categorical crossentropy): {loss:.4f}")
    lines.extend(
        [
            "",
            f"Correctly Classified Instances        {correct:5d}               {100 * accuracy:6.2f}   %",
            f"Incorrectly Classified Instances      {incorrect:5d}               {100 * (1 - accuracy):6.2f}   %",
            f"Kappa statistic                          {kappa:.4f}",
            f"Mean absolute error                      {prob_err['mae']:.4f}",
            f"Root mean squared error                  {prob_err['rmse']:.4f}",
            f"Relative absolute error                 {prob_err['rae']:.4f} %",
            f"Root relative squared error             {prob_err['rrse']:.4f} %",
            f"Total Number of Instances            {total:5d}",
            "",
            "=== CLASS DETAILS ===",
            "=== Detailed Accuracy By Class ===",
            "",
            f"{'':17s}TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class",
        ]
    )
    for name, row in zip(class_names, per_class):
        lines.append(
            f"{'':17s}{row['tp_rate']:.3f}    {row['fp_rate']:.3f}    "
            f"{row['precision']:.3f}      {row['recall']:.3f}    "
            f"{row['f1']:.3f}      {row['mcc']:.3f}    "
            f"{row['roc']:.3f}     {row['prc']:.3f}     {name}"
        )
    lines.append(
        f"{'Weighted Avg.':17s}{weighted['tp_rate']:.3f}    {weighted['fp_rate']:.3f}    "
        f"{weighted['precision']:.3f}      {weighted['recall']:.3f}    "
        f"{weighted['f1']:.3f}      {weighted['mcc']:.3f}    "
        f"{weighted['roc']:.3f}     {weighted['prc']:.3f}     "
    )
    lines.extend(["", "=== CONFUSION MATRIX ===", "=== Confusion Matrix ===", ""])
    lines.extend(_format_confusion_matrix_lines(cm, class_names, letters))
    cm_norm = _column_normalized_confusion_matrix(cm)
    lines.extend(
        [
            "",
            "=== NORMALIZED CONFUSION MATRIX (each column sums to 1) ===",
            "",
        ]
    )
    lines.extend(_format_confusion_matrix_lines(cm_norm, class_names, letters, decimal=True))
    lines.extend(
        [
            "",
            f"Accuracy: {100 * accuracy:.2f}",
            f"Kappa: {kappa}",
            f"Weighted F1: {weighted['f1']}",
        ]
    )
    return "\n".join(lines)


def evaluate(
    model_path: str | None = None,
    image_height: int = config.IMG_HEIGHT,
    image_width: int = config.IMG_WIDTH,
    batch_size: int = config.BATCH_SIZE,
    verbose: int = 1,
    excel_path: str | Path | None = config.NORMALIZED_CM_EXCEL_PATH,
) -> dict[str, float | str]:
    """Load model, evaluate on CIFAR-10 test data, print report, return summary metrics."""
    path = model_path or str(config.CHECKPOINT_DIR / "best_model.keras")
    model = tf.keras.models.load_model(path)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    x, y = load_cifar10_test(image_height, image_width)
    y_np = y.numpy()
    metrics = model.evaluate(x, y, verbose=verbose, batch_size=batch_size)
    loss = float(metrics[0])
    keras_accuracy = float(metrics[1])

    y_prob = model.predict(x, verbose=verbose, batch_size=batch_size)
    y_pred = np.argmax(y_prob, axis=1)

    n_classes = y_prob.shape[-1]
    class_names = _class_names_for_model(path, n_classes)
    report = format_evaluation_report(y_np, y_pred, y_prob, class_names, loss=loss)
    print(report)

    cm = _confusion_matrix(y_np, y_pred, n_classes)
    cm_norm = _column_normalized_confusion_matrix(cm)
    excel_out: Path | None = None
    if excel_path is not None:
        excel_out = write_normalized_confusion_matrix_excel(
            cm_norm, class_names, excel_path
        )
        print(f"\nNormalized confusion matrix heat map: {excel_out}")

    result: dict[str, float | str] = {
        "loss": loss,
        "accuracy": keras_accuracy,
        "kappa": _cohen_kappa(cm),
        "weighted_f1": _weighted_average(
            _per_class_metrics(y_np, y_pred, y_prob, cm, n_classes),
            cm.sum(axis=1),
        )["f1"],
        "report": report,
    }
    if excel_out is not None:
        result["excel_path"] = str(excel_out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(config.CHECKPOINT_DIR / "best_model.keras"),
        help="Path to a .keras checkpoint",
    )
    parser.add_argument("--height", type=int, default=config.IMG_HEIGHT)
    parser.add_argument("--width", type=int, default=config.IMG_WIDTH)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Keras progress bars during evaluate/predict",
    )
    parser.add_argument(
        "--excel-out",
        default=str(config.NORMALIZED_CM_EXCEL_PATH),
        help="Path for normalized confusion matrix Excel heat map "
        "(use '' to skip)",
    )
    args = parser.parse_args()
    excel_path = args.excel_out if args.excel_out else None
    evaluate(
        args.model,
        args.height,
        args.width,
        batch_size=args.batch_size,
        verbose=0 if args.quiet else 1,
        excel_path=excel_path,
    )


if __name__ == "__main__":
    main()
