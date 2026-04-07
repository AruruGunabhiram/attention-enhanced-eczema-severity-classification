import tensorflow as tf
from tensorflow.keras import layers, Model

def build_mobilenetv2(num_classes: int, input_shape=(224,224,3), dropout=0.3) -> Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False,
        weights='imagenet', pooling='avg'
    )
    base.trainable = False  # frozen for initial training
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def build_efficientnetb0(num_classes: int, input_shape=(224,224,3), dropout=0.3) -> Model:
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False,
        weights='imagenet', pooling='avg'
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def cbam_block(tensor, ratio=8):
    """Channel + Spatial attention (CBAM)."""
    # Channel attention
    channels = tensor.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(tensor)
    max_pool = layers.GlobalMaxPooling2D()(tensor)
    shared_dense1 = layers.Dense(channels // ratio, activation='relu')
    shared_dense2 = layers.Dense(channels, activation='sigmoid')
    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))
    channel_att = layers.Multiply()([tensor, layers.Add()([avg_out, max_out])[:, None, None, :]])

    # Spatial attention
    avg_s = tf.reduce_mean(channel_att, axis=-1, keepdims=True)
    max_s = tf.reduce_max(channel_att, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_s, max_s])
    spatial_att = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([channel_att, spatial_att])

def build_cbam_model(num_classes: int, input_shape=(224,224,3), dropout=0.3) -> Model:
    base = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)     # shape: (batch, H, W, C)
    x = cbam_block(x)                    # attention on feature maps
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)