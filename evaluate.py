"""
Grad-CAM evaluation and visualization for eczema classification models.

Usage:
    python evaluate.py \\
        --model_path models/stage2_mobilenetv2_best.h5 \\
        --image_path data/raw/dermnet/train/Atopic\\ Dermatitis\\ Photos/img.jpg \\
        --label_list eczema,other_skin

The output image is saved to outputs/gradcam_{filename}.png.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------------
# Core Grad-CAM
# ---------------------------------------------------------------------------

def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the name of the last Conv2D layer in the model.

    Traverses model.layers in reverse; if a layer is itself a Model (e.g. a
    frozen base network), its internal layers are searched recursively so that
    the last convolutional layer inside a sub-model is discovered.

    Args:
        model: A compiled or loaded Keras model.

    Returns:
        The name of the last Conv2D layer found.

    Raises:
        ValueError: If no Conv2D layer exists anywhere in the model.
    """
    def _last_conv_in_layers(layers):
        for layer in reversed(layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if isinstance(layer, tf.keras.Model):
                result = _last_conv_in_layers(layer.layers)
                if result is not None:
                    return result
        return None

    name = _last_conv_in_layers(model.layers)
    if name is None:
        raise ValueError("No Conv2D layer found in the model.")
    return name


def grad_cam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int,
    last_conv_layer_name: str,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a given class and conv layer.

    Builds a sub-model that outputs both the target conv layer's activations
    and the final predictions, then uses GradientTape to compute gradients of
    the chosen class score w.r.t. those activations.

    Args:
        model: Keras model to explain.
        img_array: Preprocessed image, shape (1, H, W, 3), values in [0, 1].
        class_index: Integer class index to explain.
        last_conv_layer_name: Name of the Conv2D layer to hook into.

    Returns:
        Heatmap as a float32 ndarray of shape (H_orig, W_orig) with values in
        [0, 1], resized to match the input spatial dimensions.
    """
    # Sub-model: inputs → (conv_layer_output, final_output)
    conv_layer = model.get_layer(last_conv_layer_name)
    # Handle layers nested inside a sub-model
    if conv_layer is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    conv_layer = layer.get_layer(last_conv_layer_name)
                    break
                except ValueError:
                    continue

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        # Support both sigmoid (binary) and softmax (multiclass) heads
        if predictions.shape[-1] == 1:
            class_score = predictions[:, 0]
        else:
            class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)          # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (C,)

    conv_outputs = conv_outputs[0]                             # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]     # (h, w, 1)
    heatmap = tf.squeeze(heatmap).numpy()                      # (h, w)
    heatmap = np.maximum(heatmap, 0)                           # ReLU
    heatmap = heatmap / (heatmap.max() + 1e-8)                 # normalize to [0,1]

    # Resize heatmap to original input spatial size
    orig_h, orig_w = img_array.shape[1], img_array.shape[2]
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (orig_w, orig_h), resample=Image.BILINEAR
    )
    return np.array(heatmap_img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_grad_cam(
    model: tf.keras.Model,
    img_path: str,
    label_list: list[str],
    image_size: tuple[int, int] = (224, 224),
) -> None:
    """Load an image, run inference, overlay Grad-CAM, and save the result.

    Saves the overlay to outputs/gradcam_{original_filename}.png.

    Args:
        model: Loaded Keras model.
        img_path: Path to the input image file.
        label_list: Ordered list of class label strings (same order used during
            training so integer indices map to the correct names).
        image_size: (height, width) the model was trained on. Defaults to
            (224, 224).
    """
    img_path = Path(img_path)

    # --- Load and preprocess ---
    pil_img = Image.open(img_path).convert("RGB")
    original = np.array(pil_img.resize((image_size[1], image_size[0])))  # (H, W, 3) uint8

    img_array = np.array(
        pil_img.resize((image_size[1], image_size[0])), dtype=np.float32
    ) / 255.0
    img_array = np.expand_dims(img_array, 0)  # (1, H, W, 3)

    # --- Inference ---
    predictions = model.predict(img_array, verbose=0)
    if predictions.shape[-1] == 1:
        # Binary sigmoid
        confidence = float(predictions[0, 0])
        class_index = int(confidence >= 0.5)
        confidence = confidence if class_index == 1 else 1.0 - confidence
    else:
        class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0, class_index])

    predicted_label = label_list[class_index] if class_index < len(label_list) else str(class_index)
    print(f"Predicted class : {predicted_label} (index {class_index})")
    print(f"Confidence      : {confidence:.4f}")

    # --- Grad-CAM ---
    last_conv = find_last_conv_layer(model)
    print(f"Last Conv2D layer: {last_conv}")

    heatmap = grad_cam(model, img_array, class_index, last_conv)

    # --- Overlay ---
    colormap = plt.get_cmap("jet")
    colored = np.uint8(colormap(heatmap)[:, :, :3] * 255)   # (H, W, 3) RGB jet
    overlay = np.uint8(0.4 * colored + 0.6 * original)       # alpha blend

    # --- Plot and save ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\n{predicted_label} ({confidence:.2%})")
    axes[2].axis("off")

    plt.tight_layout()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / f"gradcam_{img_path.stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Grad-CAM visualization to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for an eczema model."
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to a saved Keras model (.h5 or SavedModel directory).",
    )
    parser.add_argument(
        "--image_path", required=True,
        help="Path to the input image to visualize.",
    )
    parser.add_argument(
        "--label_list", required=True,
        help="Comma-separated class labels in training order, "
             "e.g. 'eczema,other_skin' or 'mild,moderate,severe'.",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[224, 224],
        metavar=("HEIGHT", "WIDTH"),
        help="Image size the model was trained on (default: 224 224).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    label_list = [l.strip() for l in args.label_list.split(",") if l.strip()]
    image_size = tuple(args.image_size)

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    visualize_grad_cam(
        model=model,
        img_path=args.image_path,
        label_list=label_list,
        image_size=image_size,
    )
