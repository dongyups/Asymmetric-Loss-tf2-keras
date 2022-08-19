"""Implements asymmetric loss."""

from typing import Optional
import tensorflow as tf
from typeguard import typechecked

@tf.keras.utils.register_keras_serializable(package="Addons")
class SigmoidAsymmetricLoss(tf.keras.losses.Loss):
    """
    Implements the asymmetric loss function.
    (https://arxiv.org/pdf/2009.14119.pdf).
    This is the first one among the three versions of the loss: 
    AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel
    (https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)
    Usage with `tf.keras` API:
    >>> model = tf.keras.Model()
    >>> model.compile(optimizer='sgd', loss=SigmoidAsymmetricLoss())
    """
    @typechecked
    def __init__(
        self,
        from_logits: bool = False,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        epsilon: float = 1e-8,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def call(self, y_true, y_pred):
        # Set Data Types
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Calculating Probabilities
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        xs_pos = y_pred
        xs_neg = 1.0 - y_pred

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = xs_neg + self.clip
            xs_neg = tf.clip_by_value(xs_neg, clip_value_min=tf.reduce_min(xs_neg), clip_value_max=1.0)

        # Basic CE calculation
        xs_pos = tf.clip_by_value(xs_pos, clip_value_min=self.epsilon, clip_value_max=tf.reduce_max(xs_pos))
        xs_neg = tf.clip_by_value(xs_neg, clip_value_min=self.epsilon, clip_value_max=tf.reduce_max(xs_neg))
        los_pos = y_true * tf.math.log(xs_pos)
        los_neg = (1.0 - y_true) * tf.math.log(xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y_true
            pt1 = xs_neg * (1.0 - y_true) # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_true + self.gamma_neg * (1.0 - y_true)
            one_sided_w = tf.math.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -tf.math.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
            "gamma_neg": self.gamma_neg,
            "gamma_pos": self.gamma_pos,
            "clip": self.clip,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}