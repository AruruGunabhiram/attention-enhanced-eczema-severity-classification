"""
Training script for the 3-stage eczema CNN pipeline.

Usage examples:
    python train.py --stage 2 --model mobilenetv2
    python train.py --stage 2 --model efficientnetb0 --batch_size 16
    python train.py --stage 3 --model cbam --epochs 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import build_dataset, get_class_weights, get_label_list
from src.models import build_mobilenetv2, build_efficientnetb0, build_cbam_model

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an eczema classification model."
    )
    parser.add_argument(
        "--stage", type=int, choices=[2, 3], required=True,
        help="Pipeline stage: 2=binary (eczema vs other), 3=severity (3-class).",
    )
    parser.add_argument(
        "--model", choices=["mobilenetv2", "efficientnetb0", "cbam"], required=True,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for train and val datasets (default: 32).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_csv_paths(stage: int) -> tuple[Path, Path]:
    split_dir = Path("data/splits")
    prefix = f"stage{stage}"
    train_csv = split_dir / f"{prefix}_train.csv"
    val_csv   = split_dir / f"{prefix}_val.csv"
    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}\n"
                f"Run scripts/make_splits.py first."
            )
    return train_csv, val_csv


def build_model(model_name: str, num_classes: int) -> tf.keras.Model:
    builders = {
        "mobilenetv2":   build_mobilenetv2,
        "efficientnetb0": build_efficientnetb0,
        "cbam":          build_cbam_model,
    }
    return builders[model_name](num_classes=num_classes)


def compile_model(model: tf.keras.Model, stage: int) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if stage == 2:
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


def make_callbacks(stage: int, model_name: str) -> list:
    tag = f"stage{stage}_{model_name}"
    models_dir = Path("models")
    logs_dir   = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(models_dir / f"{tag}_best.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(logs_dir / f"{tag}_history.csv"),
        append=False,
    )
    return [checkpoint, early_stop, reduce_lr, csv_logger]


def evaluate_and_report(
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    label_list: list[str],
    stage: int,
    model_name: str,
) -> None:
    """Run inference on val set, print accuracy, save classification report."""
    all_labels, all_preds = [], []

    for images, labels in val_dataset:
        preds = model.predict(images, verbose=0)
        if stage == 2:
            # Binary: sigmoid output → threshold at 0.5
            predicted_classes = (preds.squeeze() >= 0.5).astype(int)
        else:
            predicted_classes = np.argmax(preds, axis=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(predicted_classes.tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    accuracy = (all_labels == all_preds).mean()
    print(f"\nFinal validation accuracy: {accuracy:.4f}")

    report = classification_report(
        all_labels, all_preds,
        target_names=label_list,
        digits=4,
    )
    print(report)

    report_path = Path("logs") / f"stage{stage}_{model_name}_report.txt"
    report_path.write_text(
        f"Final validation accuracy: {accuracy:.4f}\n\n{report}"
    )
    print(f"Classification report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Stage {args.stage} | Model: {args.model} | "
          f"Batch: {args.batch_size} | Max epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # --- Data ---
    train_csv, val_csv = get_csv_paths(args.stage)
    label_list = get_label_list(str(train_csv))
    num_classes = len(label_list)
    print(f"Labels ({num_classes}): {label_list}")

    train_ds = build_dataset(
        str(train_csv), label_list,
        batch_size=args.batch_size, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        str(val_csv), label_list,
        batch_size=args.batch_size, augment=False, shuffle=False,
    )

    class_weights = get_class_weights(str(train_csv), label_list)
    print(f"Class weights: {class_weights}")

    # --- Model ---
    model = build_model(args.model, num_classes)
    compile_model(model, args.stage)
    model.summary(line_length=90)

    # --- Train ---
    callbacks = make_callbacks(args.stage, args.model)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # --- Evaluate ---
    evaluate_and_report(model, val_ds, label_list, args.stage, args.model)


if __name__ == "__main__":
    main()
