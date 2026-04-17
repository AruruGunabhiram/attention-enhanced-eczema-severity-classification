import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from PIL import Image

def make_gradcam_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str) -> np.ndarray:
    """img_array shape: (1, H, W, 3), float32 in [0, 255]"""
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)
    return heatmap

def overlay_gradcam(img_path: str, heatmap: np.ndarray, alpha: float = 0.4):
    img = np.array(Image.open(img_path).resize((224, 224)))
    heatmap_resized = np.uint8(255 * heatmap)
    colormap = cm.get_cmap('jet')
    colored = colormap(np.arange(256))[:, :3]
    colored_heatmap = colored[heatmap_resized]
    colored_heatmap = np.uint8(colored_heatmap * 255)
    colored_heatmap_resized = np.array(
        Image.fromarray(colored_heatmap).resize((img.shape[1], img.shape[0]))
    )
    superimposed = np.uint8(alpha * colored_heatmap_resized + (1 - alpha) * img)
    return superimposed

def run_gradcam_batch(model, df_sample, last_conv_layer_name: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for _, row in df_sample.iterrows():
        img = tf.image.resize(
            tf.cast(tf.image.decode_jpeg(tf.io.read_file(row['filepath'])), tf.float32),
            (224, 224)
        )[tf.newaxis, ...]
        heatmap = make_gradcam_heatmap(model, img, last_conv_layer_name)
        overlay = overlay_gradcam(row['filepath'], heatmap)
        fname = os.path.basename(row['filepath'])
        plt.imsave(os.path.join(save_dir, f'gradcam_{fname}'), overlay)