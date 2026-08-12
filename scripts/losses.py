# dgnet_tf/losses.py

import tensorflow as tf


def structure_loss(
    prediction,
    target
):

    # ---------------------------------------
    # Weighted region
    # ---------------------------------------

    avg = tf.nn.avg_pool2d(
        target,
        ksize=31,
        strides=1,
        padding="SAME"
    )

    weit = (
        1.0
        +
        5.0 * tf.abs(
            avg - target
        )
    )

    # ---------------------------------------
    # Weighted BCE
    # ---------------------------------------

    bce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target,
        logits=prediction
    )

    bce = (
        tf.reduce_sum(
            weit * bce,
            axis=[1, 2, 3]
        )
        /
        (
            tf.reduce_sum(
                weit,
                axis=[1, 2, 3]
            )
            + 1e-8
        )
    )

    # ---------------------------------------
    # Weighted IoU
    # ---------------------------------------

    pred = tf.sigmoid(
        prediction
    )

    inter = tf.reduce_sum(
        pred * target * weit,
        axis=[1, 2, 3]
    )

    union = tf.reduce_sum(
        (pred + target) * weit,
        axis=[1, 2, 3]
    )

    wiou = 1.0 - (
        (inter + 1.0)
        /
        (
            union
            - inter
            + 1.0
        )
    )

    return tf.reduce_mean(
        bce + wiou
    )


def gradient_loss(
    prediction,
    target
):

    return tf.reduce_mean(
        tf.square(
            prediction - target
        )
    )


def total_loss(
    prediction,
    target,
    gradient_prediction,
    gradient_target
):

    seg = structure_loss(
        prediction,
        target
    )

    grad = gradient_loss(
        gradient_prediction,
        gradient_target
    )

    total = seg + grad

    return (
        total,
        seg,
        grad
    )