"""Training script for the eczema CNN pipeline.

Two-phase schedule:
  1. Warmup — backbone frozen, head only.
  2. Fine-tune — unfreeze top `--unfreeze_at` backbone layers (BN stays
     frozen), cosine-decay LR.

Regularization: MixUp (alpha), label smoothing, AdamW weight decay,
strong augmentation. Evaluation uses test-time augmentation (hflip).

Usage:
    python train.py --stage 2 --model cbam_v2s --image_size 300
    python train.py --stage 3 --model cbam_v2s --image_size 300 --epochs 80 \\
        --warmup_epochs 12 --unfreeze_at 60 --mixup 0.3 --focal_gamma 2.0 --balance
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import build_dataset, get_class_weights, get_label_list
from src.losses import focal_loss
from src.models import (
    build_mobilenetv2, build_efficientnetb0, build_efficientnetv2s,
    build_cbam_model, build_cbam_v2s,
    set_trainable_at, get_backbone,
)

MODEL_BUILDERS = {
    "mobilenetv2":    build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
    "efficientnetv2s": build_efficientnetv2s,
    "cbam":           build_cbam_model,
    "cbam_v2s":       build_cbam_v2s,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an eczema classification model.")
    parser.add_argument("--stage", type=int, choices=[2, 3], required=True)
    parser.add_argument("--model", choices=list(MODEL_BUILDERS), required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--unfreeze_at", type=int, default=60)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="MixUp alpha; 0 disables.")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable horizontal-flip TTA at eval.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load best checkpoint and run eval.")
    parser.add_argument("--balance", action="store_true",
                        help="Oversample minority classes to match the majority.")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal loss gamma (stage 3 only). 0 = standard CE. "
                             "Recommended: 2.0 for imbalanced severity classes.")
    return parser.parse_args()


def get_csv_paths(stage: int) -> tuple[Path, Path]:
    split_dir = Path("data/splits")
    prefix = f"stage{stage}"
    train_csv = split_dir / f"{prefix}_train.csv"
    val_csv   = split_dir / f"{prefix}_val.csv"
    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}\nRun scripts/make_splits.py first."
            )
    return train_csv, val_csv


def _stage3_loss(label_smoothing, focal_gamma):
    if focal_gamma > 0:
        return focal_loss(gamma=focal_gamma, label_smoothing=label_smoothing)
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)


def compile_model(model, stage, num_classes, lr, weight_decay, label_smoothing,
                  focal_gamma=0.0):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr, weight_decay=weight_decay,
    )
    if stage == 2:
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    else:
        loss = _stage3_loss(label_smoothing, focal_gamma)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def make_callbacks(stage, model_name, phase, use_plateau):
    tag = f"stage{stage}_{model_name}"
    models_dir = Path("models")
    logs_dir = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    monitor = "val_auc" if stage == 2 else "val_accuracy"
    mode = "max"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / f"{tag}_best.keras"),
            monitor=monitor, save_best_only=True, mode=mode, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=12, restore_best_weights=True,
            mode=mode, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(logs_dir / f"{tag}_{phase}_history.csv"), append=False,
        ),
    ]
    if use_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-7, verbose=1,
        ))
    return callbacks


def _tta_variants(images, use_tta: bool):
    if not use_tta:
        return [images]
    hflip = tf.image.flip_left_right(images)
    vflip = tf.image.flip_up_down(images)
    hvflip = tf.image.flip_up_down(hflip)
    return [images, hflip, vflip, hvflip]


def _predict_tta(model, val_dataset, use_tta: bool):
    all_labels, all_probs = [], []
    for images, labels in val_dataset:
        variants = _tta_variants(images, use_tta)
        probs = np.mean([model.predict(v, verbose=0) for v in variants], axis=0)
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels, all_probs


def _best_threshold(labels, probs):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_and_report(model, val_dataset, label_list, stage, model_name,
                        one_hot_labels, use_tta):
    labels, probs = _predict_tta(model, val_dataset, use_tta)
    if one_hot_labels:
        labels = np.argmax(labels, axis=1).astype(int)
    else:
        labels = labels.astype(int)

    threshold = 0.5
    if stage == 2:
        probs_flat = probs.squeeze()
        threshold, _ = _best_threshold(labels, probs_flat)
        preds = (probs_flat >= threshold).astype(int)
        print(f"Best threshold (val macro F1): {threshold:.3f}")
    else:
        preds = np.argmax(probs, axis=1)

    accuracy = (labels == preds).mean()
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    print(f"\nFinal val accuracy:   {accuracy:.4f}")
    print(f"Final macro F1:       {macro_f1:.4f}")
    print(f"Final weighted F1:    {weighted_f1:.4f}")

    report = classification_report(
        labels, preds, target_names=label_list, digits=4, zero_division=0,
    )
    print(report)
    report_path = Path("logs") / f"stage{stage}_{model_name}_report.txt"
    header = (
        f"Final val accuracy: {accuracy:.4f}\n"
        f"Macro F1:    {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
    )
    if stage == 2:
        header += f"Threshold:   {threshold:.3f}\n"
    report_path.write_text(f"{header}\n{report}")
    print(f"Classification report saved to {report_path}")


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Stage {args.stage} | Model: {args.model} | "
          f"Image: {args.image_size} | Batch: {args.batch_size}")
    print(f"  MixUp: {args.mixup} | LS: {args.label_smoothing} | "
          f"TTA: {not args.no_tta}")
    print(f"  Warmup: {args.warmup_epochs} ep @ {args.warmup_lr}")
    print(f"  Fine-tune: {args.epochs - args.warmup_epochs} ep @ "
          f"{args.finetune_lr}, unfreeze_at={args.unfreeze_at}")
    print(f"{'='*60}\n")

    train_csv, val_csv = get_csv_paths(args.stage)
    label_list = get_label_list(str(train_csv))
    num_classes = len(label_list)
    one_hot = num_classes > 2
    print(f"Labels ({num_classes}): {label_list}")

    img = (args.image_size, args.image_size)
    train_ds = build_dataset(
        str(train_csv), label_list, image_size=img,
        batch_size=args.batch_size, augment=True, shuffle=True,
        one_hot=one_hot, mixup_alpha=args.mixup, balance=args.balance,
    )
    val_ds = build_dataset(
        str(val_csv), label_list, image_size=img,
        batch_size=args.batch_size, augment=False, shuffle=False,
        one_hot=one_hot, mixup_alpha=0.0,
    )

    class_weights = None if (args.balance or args.mixup > 0) else \
        get_class_weights(str(train_csv), label_list)
    if class_weights:
        print(f"Class weights: {class_weights}")
    elif args.balance:
        print("Class weights disabled (balance oversampling active).")
    else:
        print("Class weights disabled (mixup active).")

    if args.eval_only:
        from src.models import CUSTOM_OBJECTS
        ckpt = Path("models") / f"stage{args.stage}_{args.model}_best.keras"
        print(f"Loading checkpoint: {ckpt}")
        model = tf.keras.models.load_model(
            str(ckpt), custom_objects=CUSTOM_OBJECTS, compile=False,
        )
        evaluate_and_report(model, val_ds, label_list, args.stage, args.model,
                            one_hot, use_tta=not args.no_tta)
        return

    builder = MODEL_BUILDERS[args.model]
    model = builder(num_classes=num_classes,
                    input_shape=(args.image_size, args.image_size, 3),
                    dropout=args.dropout)
    compile_model(model, args.stage, num_classes, args.warmup_lr,
                  args.weight_decay, args.label_smoothing, args.focal_gamma)
    model.summary(line_length=100)

    print("\n--- Phase 1: warmup (backbone frozen) ---")
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.warmup_epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(args.stage, args.model, "warmup", use_plateau=True),
    )

    if args.unfreeze_at > 0 and args.epochs > args.warmup_epochs:
        print(f"\n--- Phase 2: fine-tune (unfreeze top {args.unfreeze_at}) ---")
        backbone = get_backbone(model)
        set_trainable_at(backbone, args.unfreeze_at)

        n_train = len(pd.read_csv(train_csv))
        steps_per_epoch = max(1, (n_train + args.batch_size - 1) // args.batch_size)
        total_steps = (args.epochs - args.warmup_epochs) * steps_per_epoch
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.finetune_lr,
            decay_steps=total_steps,
            alpha=0.01,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=schedule, weight_decay=args.weight_decay,
        )
        if args.stage == 2:
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
            metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                       tf.keras.metrics.AUC(name="auc")]
        else:
            loss = _stage3_loss(args.label_smoothing, args.focal_gamma)
            metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=args.warmup_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(args.stage, args.model, "finetune", use_plateau=False),
        )

    evaluate_and_report(model, val_ds, label_list, args.stage, args.model,
                        one_hot, use_tta=not args.no_tta)


if __name__ == "__main__":
    main()
