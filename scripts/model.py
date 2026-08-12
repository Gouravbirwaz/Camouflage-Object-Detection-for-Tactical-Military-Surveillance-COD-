# dgnet_tf/model.py

import tensorflow as tf
from tensorflow.keras import layers


# =========================================================
# Conv + BatchNorm + ReLU
# =========================================================

class ConvBR(layers.Layer):

    def __init__(
        self,
        filters,
        kernel_size,
        stride=1,
        groups=1,
        name=None
    ):

        super().__init__(name=name)

        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding="same",
            groups=groups,
            use_bias=False
        )

        self.bn = layers.BatchNormalization()

        self.relu = layers.ReLU()


    def call(
        self,
        x,
        training=False
    ):

        x = self.conv(x)

        x = self.bn(
            x,
            training=training
        )

        x = self.relu(x)

        return x


# =========================================================
# Dimensional Reduction
# =========================================================

class DimensionalReduction(layers.Layer):

    def __init__(
        self,
        channels=32,
        name=None
    ):

        super().__init__(name=name)

        self.reduce1 = ConvBR(
            channels,
            3
        )

        self.reduce2 = ConvBR(
            channels,
            3
        )


    def call(
        self,
        x,
        training=False
    ):

        x = self.reduce1(
            x,
            training=training
        )

        x = self.reduce2(
            x,
            training=training
        )

        return x


# =========================================================
# Texture Encoder
# =========================================================

class TextureEncoder(
    layers.Layer
):

    def __init__(
        self,
        name=None
    ):

        super().__init__(name=name)

        self.conv1 = ConvBR(
            64,
            7,
            stride=2
        )

        self.conv2 = ConvBR(
            64,
            3,
            stride=2
        )

        self.conv3 = ConvBR(
            32,
            3,
            stride=2
        )

        self.conv_out = ConvBR(
            1,
            1
        )


    def call(
        self,
        x,
        training=False
    ):

        x = self.conv1(
            x,
            training=training
        )

        x = self.conv2(
            x,
            training=training
        )

        xg = self.conv3(
            x,
            training=training
        )

        pg = self.conv_out(
            xg,
            training=training
        )

        return xg, pg


# =========================================================
# Soft Grouping Strategy
# =========================================================

class SoftGroupingStrategy(
    layers.Layer
):

    def __init__(
        self,
        channels=32,
        groups=8,
        name=None
    ):

        super().__init__(name=name)

        self.conv = layers.Conv2D(
            channels,
            1,
            padding="same",
            groups=groups,
            use_bias=False
        )


    def call(self, x):

        return self.conv(x)


# =========================================================
# Gradient-Induced Transition
# =========================================================

class GradientInducedTransition(
    layers.Layer
):

    def __init__(
        self,
        channels=32,
        M=(8, 8, 8),
        N=(8, 16, 32),
        name=None
    ):

        super().__init__(name=name)

        self.channels = channels
        self.M = M
        self.N = N

        self.sgs3 = SoftGroupingStrategy(
            channels=channels,
            groups=N[0]
        )

        self.sgs4 = SoftGroupingStrategy(
            channels=channels,
            groups=N[1]
        )

        self.sgs5 = SoftGroupingStrategy(
            channels=channels,
            groups=N[2]
        )


    def _group_features(
        self,
        xr,
        xg,
        M
    ):

        # DGNet's grouping idea:
        #
        # xr = context feature
        # xg = gradient feature
        #
        # Split channels into M groups and
        # interleave context/gradient groups.

        xr_groups = tf.split(
            xr,
            M,
            axis=-1
        )

        xg_groups = tf.split(
            xg,
            M,
            axis=-1
        )

        pieces = []

        for i in range(M):

            pieces.append(
                xr_groups[i]
            )

            pieces.append(
                xg_groups[i]
            )

        return tf.concat(
            pieces,
            axis=-1
        )


    def _resize(
        self,
        x,
        reference
    ):

        size = tf.shape(
            reference
        )[1:3]

        return tf.image.resize(
            x,
            size,
            method="bilinear"
        )


    def _apply_sgs(
        self,
        xr,
        xg,
        M,
        sgs,
        training=False
    ):

        xg = self._resize(
            xg,
            xr
        )

        q = self._group_features(
            xr,
            xg,
            M
        )

        z = sgs(q)

        return xr + z


    def call(
        self,
        xr3,
        xr4,
        xr5,
        xg,
        training=False
    ):

        zt3 = self._apply_sgs(
            xr3,
            xg,
            self.M[0],
            self.sgs3,
            training
        )

        zt4 = self._apply_sgs(
            xr4,
            xg,
            self.M[1],
            self.sgs4,
            training
        )

        zt5 = self._apply_sgs(
            xr5,
            xg,
            self.M[2],
            self.sgs5,
            training
        )

        return (
            zt3,
            zt4,
            zt5
        )


# =========================================================
# Neighbor Connection Decoder
# =========================================================

class NeighborConnectionDecoder(
    layers.Layer
):

    def __init__(
        self,
        channels=32,
        name=None
    ):

        super().__init__(name=name)

        self.up1 = ConvBR(
            channels,
            3
        )

        self.up2 = ConvBR(
            channels,
            3
        )

        self.up3 = ConvBR(
            channels,
            3
        )

        self.up4 = ConvBR(
            channels,
            3
        )

        self.up5 = ConvBR(
            channels * 2,
            3
        )

        self.concat2 = ConvBR(
            channels * 2,
            3
        )

        self.concat3 = ConvBR(
            channels * 3,
            3
        )

        self.conv4 = ConvBR(
            channels * 3,
            3
        )

        self.conv5 = layers.Conv2D(
            1,
            1,
            padding="same"
        )


    def resize_like(
        self,
        x,
        ref
    ):

        return tf.image.resize(
            x,
            tf.shape(ref)[1:3],
            method="bilinear"
        )


    def call(
        self,
        zt5,
        zt4,
        zt3,
        training=False
    ):

        # --------------------------------------
        # Stage 5
        # --------------------------------------

        zt5_1 = zt5

        # --------------------------------------
        # Stage 4
        # --------------------------------------

        up5 = self.resize_like(
            zt5,
            zt4
        )

        zt4_1 = self.up1(
            up5,
            training=training
        )

        zt4_1 = zt4_1 * zt4

        # --------------------------------------
        # Stage 3
        # --------------------------------------

        up41 = self.resize_like(
            zt4_1,
            zt3
        )

        up4 = self.resize_like(
            zt4,
            zt3
        )

        a = self.up2(
            up41,
            training=training
        )

        b = self.up3(
            up4,
            training=training
        )

        zt3_1 = a * b * zt3

        # --------------------------------------
        # Fusion 4
        # --------------------------------------

        up51 = self.resize_like(
            zt5_1,
            zt4_1
        )

        up51 = self.up4(
            up51,
            training=training
        )

        zt4_2 = tf.concat(
            [
                zt4_1,
                up51
            ],
            axis=-1
        )

        zt4_2 = self.concat2(
            zt4_2,
            training=training
        )

        # --------------------------------------
        # Fusion 3
        # --------------------------------------

        up42 = self.resize_like(
            zt4_2,
            zt3_1
        )

        up42 = self.up5(
            up42,
            training=training
        )

        zt3_2 = tf.concat(
            [
                zt3_1,
                up42
            ],
            axis=-1
        )

        zt3_2 = self.concat3(
            zt3_2,
            training=training
        )

        # --------------------------------------
        # Final prediction
        # --------------------------------------

        pc = self.conv4(
            zt3_2,
            training=training
        )

        pc = self.conv5(
            pc
        )

        return pc


# =========================================================
# DGNet-S
# =========================================================

class DGNetS(
    tf.keras.Model
):

    def __init__(
        self,
        input_size=352
    ):

        super().__init__()

        self.input_size = input_size

        # --------------------------------------
        # EfficientNet-B1 context encoder
        # --------------------------------------

        backbone = (
            tf.keras.applications.EfficientNetB1(
                include_top=False,
                weights="imagenet",
                input_shape=(
                    input_size,
                    input_size,
                    3
                )
            )
        )

        # We dynamically locate the feature maps
        # at 1/8, 1/16 and 1/32 resolution.

        self.context_encoder = self._make_encoder(
            backbone,
            input_size
        )

        # --------------------------------------
        # Texture encoder
        # --------------------------------------

        self.texture_encoder = TextureEncoder()

        # --------------------------------------
        # Feature reductions
        # --------------------------------------

        self.dr3 = DimensionalReduction(
            32
        )

        self.dr4 = DimensionalReduction(
            32
        )

        self.dr5 = DimensionalReduction(
            32
        )

        # --------------------------------------
        # Gradient induced transition
        # --------------------------------------

        self.git = GradientInducedTransition(
            channels=32,
            M=(8, 8, 8),
            N=(8, 16, 32)
        )

        # --------------------------------------
        # Neighbor connection decoder
        # --------------------------------------

        self.ncd = NeighborConnectionDecoder(
            channels=32
        )


    def _make_encoder(
        self,
        backbone,
        input_size
    ):

        target_sizes = [
            input_size // 8,
            input_size // 16,
            input_size // 32
        ]

        selected = {}

        # Search from later to earlier layers.
        # Prefer feature maps with the required
        # spatial resolutions.

        for layer in reversed(
            backbone.layers
        ):

            try:

                shape = layer.output.shape

                if len(shape) != 4:
                    continue

                h = shape[1]
                w = shape[2]

                if h is None or w is None:
                    continue

                for size in target_sizes:

                    if (
                        h == size
                        and
                        w == size
                        and
                        size not in selected
                    ):

                        selected[size] = layer.output

            except Exception:

                continue

        missing = [
            x for x in target_sizes
            if x not in selected
        ]

        if missing:

            raise RuntimeError(
                "Could not locate EfficientNet-B1 "
                f"feature maps for sizes: {missing}"
            )

        outputs = [
            selected[
                target_sizes[0]
            ],
            selected[
                target_sizes[1]
            ],
            selected[
                target_sizes[2]
            ]
        ]

        return tf.keras.Model(
            inputs=backbone.input,
            outputs=outputs,
            name="EfficientNetB1Context"
        )


    def call(
        self,
        x,
        training=False
    ):

        # --------------------------------------
        # Context encoder
        # --------------------------------------

        x3, x4, x5 = (
            self.context_encoder(
                x,
                training=training
            )
        )

        xr3 = self.dr3(
            x3,
            training=training
        )

        xr4 = self.dr4(
            x4,
            training=training
        )

        xr5 = self.dr5(
            x5,
            training=training
        )

        # --------------------------------------
        # Texture / gradient encoder
        # --------------------------------------

        xg, pg = (
            self.texture_encoder(
                x,
                training=training
            )
        )

        # --------------------------------------
        # Gradient-induced transition
        # --------------------------------------

        zt3, zt4, zt5 = self.git(
            xr3,
            xr4,
            xr5,
            xg,
            training=training
        )

        # --------------------------------------
        # Neighbor connection decoder
        # --------------------------------------

        pc = self.ncd(
            zt5,
            zt4,
            zt3,
            training=training
        )

        # --------------------------------------
        # Final resolution
        # --------------------------------------

        pc = tf.image.resize(
            pc,
            [self.input_size, self.input_size],
            method="bilinear"
        )

        pg = tf.image.resize(
            pg,
            [self.input_size, self.input_size],
            method="bilinear"
        )

        return {
            "mask": pc,
            "gradient": pg
        }