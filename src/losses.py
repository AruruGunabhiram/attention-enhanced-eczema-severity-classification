import tensorflow as tf


def focal_loss(gamma: float = 2.0, label_smoothing: float = 0.0):
    """
    Categorical focal loss: down-weights easy examples so the model focuses
    on the hard (minority) class. Combines with optional label smoothing.

    gamma=0 reduces to standard cross-entropy.
    gamma=2 is the value used in the original RetinaNet paper (Lin et al. 2017).
    """
    def loss_fn(y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        return focal_weight * ce

    loss_fn.__name__ = f"focal_loss_g{gamma}"
    return loss_fn
