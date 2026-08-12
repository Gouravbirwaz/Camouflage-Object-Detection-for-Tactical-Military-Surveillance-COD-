# dgnet_tf/train.py

import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataset import ArmyDataset
from model import DGNetS
from losses import total_loss


# =========================================================
# Configuration
# =========================================================

DATASET_ROOT = (
    r"C:\Users\Acer\Desktop\COD_PROJECT"
    r"\data\raw\army_dataset"
    r"\dataset-splitM"
)

IMAGE_SIZE = 352

BATCH_SIZE = 8

EPOCHS = 100

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "./checkpoints"

LOG_DIR = "./logs"


# =========================================================
# Metrics
# =========================================================

def calculate_iou(
    prediction,
    target
):

    prediction = tf.cast(
        prediction > 0.5,
        tf.float32
    )

    target = tf.cast(
        target > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        prediction * target,
        axis=[1, 2, 3]
    )

    union = tf.reduce_sum(
        prediction + target
        - prediction * target,
        axis=[1, 2, 3]
    )

    return tf.reduce_mean(
        (
            intersection + 1e-7
        )
        /
        (
            union + 1e-7
        )
    )


def calculate_dice(
    prediction,
    target
):

    prediction = tf.cast(
        prediction > 0.5,
        tf.float32
    )

    target = tf.cast(
        target > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        prediction * target,
        axis=[1, 2, 3]
    )

    denominator = (
        tf.reduce_sum(
            prediction,
            axis=[1, 2, 3]
        )
        +
        tf.reduce_sum(
            target,
            axis=[1, 2, 3]
        )
    )

    return tf.reduce_mean(
        (
            2.0 * intersection + 1e-7
        )
        /
        (
            denominator + 1e-7
        )
    )


# =========================================================
# Training
# =========================================================

def train_epoch(
    model,
    dataset,
    optimizer
):

    total = 0.0
    segmentation = 0.0
    gradient = 0.0

    steps = 0

    progress = tqdm(
        dataset,
        desc="Training"
    )

    for images, targets in progress:

        with tf.GradientTape() as tape:

            outputs = model(
                images,
                training=True
            )

            loss, seg_loss, grad_loss = (
                total_loss(
                    outputs["mask"],
                    targets["mask"],
                    outputs["gradient"],
                    targets["gradient"]
                )
            )

        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(
            gradients,
            0.5
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        total += float(
            loss.numpy()
        )

        segmentation += float(
            seg_loss.numpy()
        )

        gradient += float(
            grad_loss.numpy()
        )

        steps += 1

        progress.set_postfix(
            loss=f"{loss.numpy():.4f}"
        )

    return (
        total / steps,
        segmentation / steps,
        gradient / steps
    )


# =========================================================
# Validation
# =========================================================

def validate(
    model,
    dataset
):

    total_loss_value = 0.0

    total_iou = 0.0

    total_dice = 0.0

    steps = 0

    for images, targets in tqdm(
        dataset,
        desc="Validation"
    ):

        outputs = model(
            images,
            training=False
        )

        loss, _, _ = total_loss(
            outputs["mask"],
            targets["mask"],
            outputs["gradient"],
            targets["gradient"]
        )

        prediction = tf.sigmoid(
            outputs["mask"]
        )

        iou = calculate_iou(
            prediction,
            targets["mask"]
        )

        dice = calculate_dice(
            prediction,
            targets["mask"]
        )

        total_loss_value += float(
            loss.numpy()
        )

        total_iou += float(
            iou.numpy()
        )

        total_dice += float(
            dice.numpy()
        )

        steps += 1

    return (
        total_loss_value / steps,
        total_iou / steps,
        total_dice / steps
    )


# =========================================================
# Save prediction
# =========================================================

def save_prediction(
    prediction,
    epoch,
    filename
):

    prediction = tf.sigmoid(
        prediction
    )

    prediction = prediction[0, :, :, 0]

    prediction = (
        prediction.numpy() * 255
    ).astype(np.uint8)

    os.makedirs(
        "./predictions",
        exist_ok=True
    )

    tf.keras.utils.save_img(
        f"./predictions/"
        f"epoch_{epoch}_{filename}.png",
        prediction[:, :, None],
        scale=False
    )


# =========================================================
# Main
# =========================================================

def main():

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    # ---------------------------------------
    # GPU
    # ---------------------------------------

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        f"GPUs available: {len(gpus)}"
    )

    # ---------------------------------------
    # Dataset
    # ---------------------------------------

    train_data = ArmyDataset(
        DATASET_ROOT,
        split="Training",
        validation=False,
        validation_ratio=0.1
    )

    val_data = ArmyDataset(
        DATASET_ROOT,
        split="Training",
        validation=True,
        validation_ratio=0.1
    )

    train_ds = train_data.get_dataset(
        batch_size=BATCH_SIZE,
        training=True
    )

    val_ds = val_data.get_dataset(
        batch_size=BATCH_SIZE,
        training=False
    )

    # ---------------------------------------
    # Model
    # ---------------------------------------

    model = DGNetS(
        input_size=IMAGE_SIZE
    )

    # Build model
    dummy = tf.zeros(
        [
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ]
    )

    outputs = model(
        dummy,
        training=False
    )

    print(
        "Mask output:",
        outputs["mask"].shape
    )

    print(
        "Gradient output:",
        outputs["gradient"].shape
    )

    model.summary()

    # ---------------------------------------
    # Optimizer
    # ---------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    # ---------------------------------------
    # TensorBoard
    # ---------------------------------------

    writer = tf.summary.create_file_writer(
        LOG_DIR
    )

    # ---------------------------------------
    # Best score
    # ---------------------------------------

    best_iou = -1.0

    # ---------------------------------------
    # Training
    # ---------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            "\n"
            f"==============================\n"
            f"Epoch {epoch}/{EPOCHS}\n"
            f"=============================="
        )

        train_loss, seg_loss, grad_loss = (
            train_epoch(
                model,
                train_ds,
                optimizer
            )
        )

        val_loss, val_iou, val_dice = (
            validate(
                model,
                val_ds
            )
        )

        print(
            f"\nTrain Loss      : {train_loss:.5f}"
        )

        print(
            f"Train Seg Loss  : {seg_loss:.5f}"
        )

        print(
            f"Train Grad Loss : {grad_loss:.5f}"
        )

        print(
            f"Val Loss        : {val_loss:.5f}"
        )

        print(
            f"Val IoU         : {val_iou:.5f}"
        )

        print(
            f"Val Dice        : {val_dice:.5f}"
        )

        # -----------------------------------
        # TensorBoard
        # -----------------------------------

        with writer.as_default():

            tf.summary.scalar(
                "train/loss",
                train_loss,
                step=epoch
            )

            tf.summary.scalar(
                "train/segmentation_loss",
                seg_loss,
                step=epoch
            )

            tf.summary.scalar(
                "train/gradient_loss",
                grad_loss,
                step=epoch
            )

            tf.summary.scalar(
                "validation/loss",
                val_loss,
                step=epoch
            )

            tf.summary.scalar(
                "validation/iou",
                val_iou,
                step=epoch
            )

            tf.summary.scalar(
                "validation/dice",
                val_dice,
                step=epoch
            )

        # -----------------------------------
        # Save best
        # -----------------------------------

        if val_iou > best_iou:

            best_iou = val_iou

            print(
                "\n*** New best model ***"
            )

            model.save(
                os.path.join(
                    CHECKPOINT_DIR,
                    "best_dgnet_s.keras"
                )
            )

        # -----------------------------------
        # Periodic checkpoint
        # -----------------------------------

        if epoch % 10 == 0:

            model.save(
                os.path.join(
                    CHECKPOINT_DIR,
                    f"dgnet_s_epoch_{epoch}.keras"
                )
            )


if __name__ == "__main__":

    main()