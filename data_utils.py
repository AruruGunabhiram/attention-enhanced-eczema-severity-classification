import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Read a split CSV and print a brief summary.

    Args:
        csv_path: Path to a CSV file with "filepath" and "label" columns.

    Returns:
        DataFrame with at least "filepath" and "label" columns.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: shape={df.shape}")
    print(df["label"].value_counts().to_string())
    return df


def build_dataset(
    df: pd.DataFrame,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
) -> tuple[tf.data.Dataset, list[str]]:
    """Build a batched, prefetched tf.data.Dataset from a DataFrame.

    Labels are encoded as integers in alphabetical order of their string
    values, so the mapping is deterministic regardless of the order rows
    appear in the CSV.

    Args:
        df: DataFrame with "filepath" and "label" columns.
        image_size: (height, width) to resize images to.
        batch_size: Number of samples per batch.
        augment: If True, applies random flip, brightness, and contrast jitter.
        shuffle: If True, shuffles with a fixed seed before batching.

    Returns:
        dataset: tf.data.Dataset yielding (image, label) batches.
            image shape: (batch, H, W, 3), float32 in [0, 1].
            label shape: (batch,), int32.
        label_list: Sorted list of unique label strings. Index in this list
            is the integer class label used in the dataset.
    """
    label_list = sorted(df["label"].unique().tolist())
    label_to_int = {lbl: i for i, lbl in enumerate(label_list)}

    filepaths = df["filepath"].tolist()
    labels = df["label"].map(label_to_int).astype(int).tolist()

    def load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def apply_augmentation(img: tf.Tensor, label: tf.Tensor):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.1, upper=0.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=42)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, label_list


def get_class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Compute balanced class weights for model.fit(class_weight=...).

    Uses sklearn's 'balanced' strategy: weight = n_samples / (n_classes * n_i).
    Label-to-integer mapping uses the same alphabetical order as build_dataset.

    Args:
        df: DataFrame with a "label" column.

    Returns:
        Dict mapping integer class index to float weight, e.g. {0: 1.2, 1: 0.8}.
    """
    label_list = sorted(df["label"].unique().tolist())
    label_to_int = {lbl: i for i, lbl in enumerate(label_list)}
    y = df["label"].map(label_to_int).values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_list)),
        y=y,
    )
    return {i: float(w) for i, w in enumerate(weights)}
