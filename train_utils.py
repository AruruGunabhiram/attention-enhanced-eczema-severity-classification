import os

import matplotlib.pyplot as plt
import tensorflow as tf


def get_callbacks(model_save_path: str, log_dir: str) -> list:
    """Build a standard set of training callbacks.

    Creates log_dir if it does not already exist.

    Args:
        model_save_path: File path where the best model checkpoint is saved
            (e.g. 'models/stage2_cbam_best.h5').
        log_dir: Directory for the CSVLogger output. Created if missing.

    Returns:
        List of [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger].
    """
    os.makedirs(log_dir, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=5,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, "training_log.csv"),
            append=False,
        ),
    ]


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    num_classes: int = 2,
) -> None:
    """Compile a Keras model in place with the appropriate loss and metrics.

    Binary (num_classes == 2):
        loss: binary_crossentropy
        metrics: accuracy, AUC

    Multiclass (num_classes > 2):
        loss: sparse_categorical_crossentropy
        metrics: accuracy

    Args:
        model: Keras model to compile.
        learning_rate: Learning rate for the Adam optimizer.
        num_classes: Number of output classes. Controls loss selection.
    """
    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )


def plot_history(history: tf.keras.callbacks.History, save_path: str) -> None:
    """Plot train/val accuracy and loss curves and save the figure.

    Creates the parent directory of save_path if it does not exist.

    Args:
        history: History object returned by model.fit().
        save_path: File path to save the figure (e.g. 'logs/stage2_curves.png').
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
