"""
TensorFlow data pipeline for eczema classification.

Provides functions to build tf.data.Dataset pipelines from CSV split files,
compute class weights for imbalanced datasets, and extract label lists.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def get_label_list(csv_path: str) -> list[str]:
    """Return a sorted list of unique labels found in the CSV.

    Args:
        csv_path: Path to a CSV file with a "label" column.

    Returns:
        Sorted list of unique label strings, e.g. ["eczema", "other_skin"].
    """
    df = pd.read_csv(csv_path)
    return sorted(df["label"].dropna().unique().tolist())


def get_class_weights(csv_path: str, label_list: list[str]) -> dict[int, float]:
    """Compute balanced class weights for use in model.fit().

    Uses sklearn's 'balanced' strategy: weight = n_samples / (n_classes * n_i).

    Args:
        csv_path: Path to a CSV file with a "label" column.
        label_list: Ordered list of label strings (defines integer encoding).
            Index in this list becomes the integer class label.

    Returns:
        Dict mapping integer class index to float weight,
        e.g. {0: 1.2, 1: 0.8}.
    """
    df = pd.read_csv(csv_path)
    label_to_int = {label: i for i, label in enumerate(label_list)}
    y = df["label"].map(label_to_int).values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_list)),
        y=y,
    )
    return {i: float(w) for i, w in enumerate(weights)}


def build_dataset(
    csv_path: str,
    label_list: list[str],
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data.Dataset from a CSV split file.

    Each row in the CSV must have a "filepath" column (absolute or relative
    path to an image) and a "label" column (string class name).

    Preprocessing applied to every sample:
      - JPEG/PNG decode via tf.io
      - Resize to image_size
      - Normalize pixels to [0, 1]

    Augmentation (when augment=True):
      - Random horizontal flip
      - Random brightness delta ±0.1
      - Random contrast range [0.9, 1.1]
      - Random rotation ±~5.7° (0.1 radians) via tf.keras.layers.RandomRotation

    Args:
        csv_path: Path to a CSV file with "filepath" and "label" columns.
        label_list: Ordered list of label strings used to encode labels as
            integers. Must be the same list across train/val/test to ensure
            consistent encoding.
        image_size: (height, width) to resize images to. Defaults to (224, 224).
        batch_size: Number of samples per batch. Defaults to 32.
        augment: Whether to apply data augmentation. Defaults to False.
        shuffle: Whether to shuffle the dataset. Defaults to True.

    Returns:
        A tf.data.Dataset yielding (image_tensor, label_tensor) batches where:
          - image_tensor shape: (batch_size, H, W, 3), dtype float32, range [0,1]
          - label_tensor shape: (batch_size,), dtype int32
    """
    df = pd.read_csv(csv_path)
    filepaths = df["filepath"].tolist()
    label_to_int = {label: i for i, label in enumerate(label_list)}
    labels = df["label"].map(label_to_int).astype(int).tolist()

    # Build augmentation layer once (stateless, applied inside map)
    rotation_layer = tf.keras.layers.RandomRotation(
        factor=0.1 / (2 * np.pi),  # convert radians to fraction of 2π
        fill_mode="reflect",
    )

    def load_and_preprocess(filepath: tf.Tensor, label: tf.Tensor):
        raw = tf.io.read_file(filepath)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def apply_augmentation(image: tf.Tensor, label: tf.Tensor):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = rotation_layer(image, training=True)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=42)

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
