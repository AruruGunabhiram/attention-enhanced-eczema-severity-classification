import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Load a split CSV. Expects columns: filepath, label"""
    df = pd.read_csv(csv_path)
    assert 'filepath' in df.columns and 'label' in df.columns
    return df

def build_dataset(
    df: pd.DataFrame,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Returns a batched, prefetched tf.data.Dataset.
    Labels are integer-encoded from df['label'].
    Augmentation applies only when augment=True (training only).
    """
    filepaths = df['filepath'].values
    labels = pd.Categorical(df['label']).codes  # int encode

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)  # decode_jpeg silently drops PNGs; decode_image handles JPEG/PNG/BMP/GIF
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32)  # EfficientNetB0 applies its own preprocess_input internally; dividing by 255 causes double-preprocessing and class collapse
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_class_weights(df: pd.DataFrame, label_col: str = 'label') -> dict:
    """Compute class weights to handle imbalance. Returns {int: float}."""
    from sklearn.utils.class_weight import compute_class_weight
    labels = pd.Categorical(df[label_col]).codes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(enumerate(weights))

def show_sample_grid(df: pd.DataFrame, n: int = 9):
    """Display n random images with labels. For sanity checks only."""
    sample = df.sample(n)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        img = plt.imread(row['filepath'])
        ax.imshow(img)
        ax.set_title(row['label'])
        ax.axis('off')
    plt.tight_layout()
    plt.show()