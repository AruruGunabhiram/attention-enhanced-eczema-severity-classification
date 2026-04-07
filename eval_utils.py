import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(model, test_ds, class_names: list) -> dict:
    """Run inference on a dataset and return evaluation metrics.

    Handles both binary (sigmoid, single output unit) and multiclass
    (softmax, one unit per class) model heads.

    Args:
        model: Trained Keras model.
        test_ds: tf.data.Dataset yielding (image_batch, label_batch). Must
            not be shuffled so y_true order is deterministic.
        class_names: Ordered list of class label strings matching the integer
            encoding used during training.

    Returns:
        Dict with keys:
            'y_true'       — list of true integer labels
            'y_pred'       — list of predicted integer labels
            'accuracy'     — float, overall accuracy
            'f1_weighted'  — float, weighted F1 score
            'report'       — str, sklearn classification_report
    """
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())

        if preds.shape[-1] == 1:
            # Binary sigmoid head
            predicted = (preds.squeeze() >= 0.5).astype(int)
            y_pred.extend(predicted.tolist() if predicted.ndim > 0 else [int(predicted)])
        else:
            # Multiclass softmax head
            y_pred.extend(np.argmax(preds, axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "report": report,
    }


def plot_confusion_matrix(
    y_true, y_pred, class_names: list, save_path: str
) -> None:
    """Plot a row-normalized confusion matrix heatmap and save it.

    Values are shown as percentages of each true class (row normalization),
    making it easy to spot per-class recall regardless of class size.

    Args:
        y_true: List or array of true integer labels.
        y_pred: List or array of predicted integer labels.
        class_names: Ordered list of class label strings.
        save_path: File path to save the figure. Parent directory is created
            if it does not exist.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 2), max(4, len(class_names) * 1.6)))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalized)", fontsize=13)
    plt.tight_layout()

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_metrics_table(metrics_dict: dict, save_path: str) -> None:
    """Append a metrics row to a CSV table, creating it if necessary.

    If the file already exists, the new row is appended without rewriting
    the header. If the file does not exist, it is created with headers
    derived from metrics_dict keys.

    Args:
        metrics_dict: Dict of metric names to values, e.g.
            {'stage': 'stage2', 'model': 'cbam', 'accuracy': 0.91, 'f1': 0.90}.
        save_path: Path to the CSV file (e.g. 'logs/results.csv').
    """
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    row = pd.DataFrame([metrics_dict])

    if os.path.exists(save_path):
        row.to_csv(save_path, mode="a", header=False, index=False)
    else:
        row.to_csv(save_path, index=False)
