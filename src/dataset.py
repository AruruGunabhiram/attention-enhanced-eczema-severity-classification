"""tf.data pipeline for eczema classification.

Images are yielded as float32 in [0, 255]; preprocess_input lives inside
the model. Optional MixUp is applied post-batch when mixup_alpha > 0.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def get_label_list(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return sorted(df["label"].dropna().unique().tolist())


def get_class_weights(csv_path: str, label_list: list[str]) -> dict[int, float]:
    df = pd.read_csv(csv_path)
    label_to_int = {label: i for i, label in enumerate(label_list)}
    y = df["label"].map(label_to_int).values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_list)),
        y=y,
    )
    return {i: float(w) for i, w in enumerate(weights)}


def _mixup_batch(images, labels, alpha: float):
    batch_size = tf.shape(images)[0]
    a = tf.random.gamma([batch_size], alpha)
    b = tf.random.gamma([batch_size], alpha)
    lam = a / (a + b + 1e-7)

    idx = tf.random.shuffle(tf.range(batch_size))
    images_s = tf.gather(images, idx)
    labels_s = tf.gather(labels, idx)

    lam_img = tf.reshape(lam, (batch_size, 1, 1, 1))
    mixed_images = lam_img * images + (1.0 - lam_img) * images_s

    lam_lbl = tf.reshape(lam, (batch_size,) + (1,) * (len(labels.shape) - 1))
    labels = tf.cast(labels, tf.float32)
    labels_s = tf.cast(labels_s, tf.float32)
    mixed_labels = lam_lbl * labels + (1.0 - lam_lbl) * labels_s
    return mixed_images, mixed_labels


def build_dataset(
    csv_path: str,
    label_list: list[str],
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    one_hot: bool = False,
    mixup_alpha: float = 0.0,
    balance: bool = False,
) -> tf.data.Dataset:
    df = pd.read_csv(csv_path)
    if balance:
        max_n = df["label"].value_counts().max()
        df = (
            df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(n=max_n, replace=True, random_state=42))
              .reset_index(drop=True)
        )
    filepaths = df["filepath"].tolist()
    label_to_int = {label: i for i, label in enumerate(label_list)}
    labels = df["label"].map(label_to_int).astype(int).tolist()
    num_classes = len(label_list)

    aug_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.15, fill_mode="reflect"),
    ])

    def load_and_preprocess(filepath, label):
        raw = tf.io.read_file(filepath)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, label

    def apply_augmentation(image, label):
        image = aug_pipeline(image, training=True)
        image = tf.image.random_brightness(image, max_delta=0.15 * 255.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
        image = tf.image.random_hue(image, max_delta=0.04)
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    def to_one_hot(image, label):
        return image, tf.one_hot(label, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=42,
                                  reshuffle_each_iteration=True)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    if one_hot:
        dataset = dataset.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    if augment and mixup_alpha > 0.0:
        dataset = dataset.map(
            lambda x, y: _mixup_batch(x, y, mixup_alpha),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)
