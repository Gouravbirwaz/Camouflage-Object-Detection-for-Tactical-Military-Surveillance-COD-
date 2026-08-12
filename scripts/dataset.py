# dgnet_tf/dataset.py

from pathlib import Path
import random

import tensorflow as tf


IMAGE_SIZE = 352


class ArmyDataset:

    def __init__(
        self,
        root,
        split="Training",
        validation=False,
        validation_ratio=0.1,
        seed=42
    ):

        self.root = Path(root)
        self.split = split
        self.validation = validation

        image_dir = self.root / split / "images"
        mask_dir = self.root / split / "GT"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        if not mask_dir.exists():
            raise FileNotFoundError(
                f"GT directory not found: {mask_dir}"
            )

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff"
        }

        images = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        masks = {}

        for p in mask_dir.iterdir():

            if p.suffix.lower() in image_extensions:
                masks[p.stem] = p

        pairs = []

        for image in images:

            if image.stem not in masks:

                print(
                    f"WARNING: no GT found for {image.name}"
                )

                continue

            pairs.append(
                (
                    str(image),
                    str(masks[image.stem])
                )
            )

        if len(pairs) == 0:
            raise RuntimeError(
                f"No image/GT pairs found in {image_dir}"
            )

        # ---------------------------------------
        # Deterministic split
        # ---------------------------------------

        rng = random.Random(seed)

        rng.shuffle(pairs)

        validation_count = int(
            len(pairs) * validation_ratio
        )

        if split == "Training":

            if validation:

                self.pairs = pairs[
                    :validation_count
                ]

            else:

                self.pairs = pairs[
                    validation_count:
                ]

        else:

            self.pairs = pairs

        print(
            f"{split} "
            f"{'VALIDATION' if validation else 'TRAIN'}: "
            f"{len(self.pairs)} samples"
        )


    def _load_image(self, image_path):

        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image = tf.image.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            method="bilinear"
        )

        image = tf.cast(
            image,
            tf.float32
        )

        # Keep [0,255] because Keras EfficientNet
        # contains its own preprocessing layer.
        return image


    def _load_mask(self, mask_path):

        mask = tf.io.read_file(
            mask_path
        )

        mask = tf.image.decode_image(
            mask,
            channels=1,
            expand_animations=False
        )

        mask = tf.image.resize(
            mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            method="nearest"
        )

        mask = tf.cast(
            mask,
            tf.float32
        )

        mask = mask / 255.0

        mask = tf.where(
            mask > 0.5,
            1.0,
            0.0
        )

        return mask


    def _gradient_target(self, mask):

        # Sobel edge extraction from GT mask.
        #
        # This provides the auxiliary gradient target
        # required by the DGNet training idea.

        edges = tf.image.sobel_edges(mask)

        gx = edges[..., 0]
        gy = edges[..., 1]

        magnitude = tf.sqrt(
            gx * gx +
            gy * gy +
            1e-8
        )

        max_value = tf.reduce_max(
            magnitude
        )

        magnitude = tf.where(
            max_value > 0,
            magnitude / max_value,
            magnitude
        )

        return magnitude


    def _augment(
        self,
        image,
        mask
    ):

        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:

            image = tf.image.flip_left_right(
                image
            )

            mask = tf.image.flip_left_right(
                mask
            )

        # Random brightness
        image = tf.image.random_brightness(
            image,
            max_delta=20.0
        )

        # Random contrast
        image = tf.image.random_contrast(
            image,
            lower=0.8,
            upper=1.2
        )

        image = tf.clip_by_value(
            image,
            0.0,
            255.0
        )

        return image, mask


    def _load(
        self,
        image_path,
        mask_path,
        augment=False
    ):

        image = self._load_image(
            image_path
        )

        mask = self._load_mask(
            mask_path
        )

        if augment:

            image, mask = self._augment(
                image,
                mask
            )

        gradient = self._gradient_target(
            mask
        )

        return (
            image,
            {
                "mask": mask,
                "gradient": gradient
            }
        )


    def get_dataset(
        self,
        batch_size=8,
        training=False
    ):

        image_paths = [
            p[0] for p in self.pairs
        ]

        mask_paths = [
            p[1] for p in self.pairs
        ]

        ds = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                mask_paths
            )
        )

        if training:

            ds = ds.shuffle(
                buffer_size=len(self.pairs),
                reshuffle_each_iteration=True
            )

        ds = ds.map(
            lambda x, y:
                self._load(
                    x,
                    y,
                    augment=training
                ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        ds = ds.batch(
            batch_size,
            drop_remainder=False
        )

        ds = ds.prefetch(
            tf.data.AUTOTUNE
        )

        return ds