# reference: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
# This is the first one among the three versions of the loss: AsymmetricLoss, AsymmetricLossOptimized, ASLSingleLabel
# Since this code is written in nested functions, not class, just copy and paste it before you apply loss function
# ex) model.compile(loss=AsymmetricLoss(...), ...)


import tensorflow as tf

def AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    """
    This loss function is for multiple labels so sigmoid would be used.
    x: y_pred (0 ~~ 1 for each label)
    y: y_true (0 or 1 for each label)
    """
    def multi_label_asymmetric_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Calculating Probabilities
        xs_pos = y_pred
        xs_neg = 1.0 - y_pred
        # Asymmetric Clipping
        if clip is not None and clip > 0:
            xs_neg = xs_neg + clip
            xs_neg = tf.clip_by_value(xs_neg, clip_value_min=tf.reduce_min(xs_neg), clip_value_max=1.0)
        # Basic CE calculation
        xs_pos = tf.clip_by_value(xs_pos, clip_value_min=eps, clip_value_max=tf.reduce_max(xs_pos))
        xs_neg = tf.clip_by_value(xs_neg, clip_value_min=eps, clip_value_max=tf.reduce_max(xs_neg))
        los_pos = y_true * tf.math.log(xs_pos)
        los_neg = (1.0 - y_true) * tf.math.log(xs_neg)
        loss = los_pos + los_neg
        # Asymmetric Focusing
        if gamma_neg > 0 or gamma_pos > 0:
            pt0 = xs_pos * y_true
            pt1 = xs_neg * (1.0 - y_true) # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = gamma_pos * y_true + gamma_neg * (1.0 - y_true)
            one_sided_w = tf.math.pow(1.0 - pt, one_sided_gamma)
            loss *= one_sided_w
        return -tf.math.reduce_sum(loss, axis=-1)
    return multi_label_asymmetric_loss_fixed
