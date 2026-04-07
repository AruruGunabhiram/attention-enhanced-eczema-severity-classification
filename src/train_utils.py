import tensorflow as tf
import matplotlib.pyplot as plt
import os

def get_callbacks(model_save_path: str, log_dir: str, patience: int = 5):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, save_best_only=True, monitor='val_loss', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True, monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

def compile_model(model, learning_rate=1e-3, num_classes=2):
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=metrics
    )

def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='train')
    axes[0].plot(history.history['val_accuracy'], label='val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='train')
    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].set_title('Loss')
    axes[1].legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()