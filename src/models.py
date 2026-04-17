import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import mobilenet_v2

L2 = 5e-4


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No backbone sub-model found.")


def _head(x, num_classes: int, dropout: float):
    x = layers.Dropout(dropout)(x)
    if num_classes == 2:
        return layers.Dense(
            1, activation="sigmoid",
            kernel_regularizer=regularizers.l2(L2), name="head",
        )(x)
    return layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=regularizers.l2(L2), name="head",
    )(x)


def set_trainable_at(base: tf.keras.Model, trainable_at: int) -> None:
    """Unfreeze the top `trainable_at` layers; keep BN layers frozen."""
    if trainable_at <= 0:
        base.trainable = False
        return
    base.trainable = True
    for layer in base.layers[:-trainable_at]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def _mobilenet_preprocess(x):
    return mobilenet_v2.preprocess_input(x)


def _spatial_mean(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)


def _spatial_max(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)


def cbam_block(tensor, ratio: int = 16):
    channels = tensor.shape[-1]
    avg_p = layers.GlobalAveragePooling2D()(tensor)
    max_p = layers.GlobalMaxPooling2D()(tensor)
    avg_p = layers.Reshape((1, 1, channels))(avg_p)
    max_p = layers.Reshape((1, 1, channels))(max_p)
    shared_1 = layers.Dense(channels // ratio, activation="relu", use_bias=False)
    shared_2 = layers.Dense(channels, use_bias=False)
    ch_att = layers.Add()([shared_2(shared_1(avg_p)), shared_2(shared_1(max_p))])
    ch_att = layers.Activation("sigmoid")(ch_att)
    x = layers.Multiply()([tensor, ch_att])

    avg_s = layers.Lambda(_spatial_mean, name="cbam_spatial_mean")(x)
    max_s = layers.Lambda(_spatial_max,  name="cbam_spatial_max")(x)
    concat = layers.Concatenate(axis=-1)([avg_s, max_s])
    sp_att = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                           use_bias=False, name="cbam_spatial_conv")(concat)
    return layers.Multiply()([x, sp_att])


def _build(backbone_fn, num_classes, input_shape, dropout, trainable_at,
           use_cbam, preprocess_fn=None):
    base = backbone_fn(
        input_shape=input_shape, include_top=False, weights="imagenet",
        pooling=None if use_cbam else "avg",
    )
    set_trainable_at(base, trainable_at)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    if preprocess_fn is not None:
        x = layers.Lambda(preprocess_fn, name="preprocess")(x)
    x = base(x, training=False)
    if use_cbam:
        x = cbam_block(x)
        x = layers.GlobalAveragePooling2D()(x)
    outputs = _head(x, num_classes, dropout)
    return Model(inputs, outputs)


def build_mobilenetv2(num_classes, input_shape=(224, 224, 3),
                     dropout=0.4, trainable_at=0):
    return _build(tf.keras.applications.MobileNetV2, num_classes,
                  input_shape, dropout, trainable_at,
                  use_cbam=False, preprocess_fn=_mobilenet_preprocess)


def build_efficientnetb0(num_classes, input_shape=(224, 224, 3),
                        dropout=0.4, trainable_at=0):
    return _build(tf.keras.applications.EfficientNetB0, num_classes,
                  input_shape, dropout, trainable_at, use_cbam=False)


def build_efficientnetv2s(num_classes, input_shape=(300, 300, 3),
                         dropout=0.4, trainable_at=0):
    return _build(tf.keras.applications.EfficientNetV2S, num_classes,
                  input_shape, dropout, trainable_at, use_cbam=False)


def build_cbam_model(num_classes, input_shape=(224, 224, 3),
                     dropout=0.4, trainable_at=0):
    return _build(tf.keras.applications.EfficientNetB0, num_classes,
                  input_shape, dropout, trainable_at, use_cbam=True)


def build_cbam_v2s(num_classes, input_shape=(300, 300, 3),
                   dropout=0.4, trainable_at=0):
    return _build(tf.keras.applications.EfficientNetV2S, num_classes,
                  input_shape, dropout, trainable_at, use_cbam=True)


CUSTOM_OBJECTS = {
    "_mobilenet_preprocess": _mobilenet_preprocess,
    "_spatial_mean": _spatial_mean,
    "_spatial_max": _spatial_max,
}
