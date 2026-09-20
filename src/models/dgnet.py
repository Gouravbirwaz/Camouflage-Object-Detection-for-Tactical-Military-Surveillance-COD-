"""
Official DGNet: Deep Gradient Network for Realtime Camouflaged Object Detection (COD)
Pure TensorFlow 2.x / Keras Implementation
Direct translation of official reference: https://github.com/GewelsJI/DGNet (MIR 2023)

Architecture Overview:
1. Context Encoder: Multi-scale feature extractor (MobileNetV3Large, EfficientNetB1/B4, ResNet50)
   extracting features at strides 8, 16, 32 (x3, x4, x5).
2. Dimensional Reduction: 2-layer ConvBR per scale reducing x3, x4, x5 to fixed channel width (xr3, xr4, xr5).
3. Texture Encoder: 3-stage convolutional encoder taking raw RGB image x, producing
   gradient features xg (stride 8, 32 channels) and predicted gradient pg (stride 8, 1 channel).
4. Gradient-Induced Transition (GIT): Feature interleaving + Soft Grouping Strategy (SGS)
   with grouped 1x1 convolutions (N=[4, 8, 16]) and attention residual learning (zt3, zt4, zt5).
5. Neighbor Connection Decoder (NCD): Cross-scale neighbor fusion with multiplicative and
   concatenation interactions generating high-resolution, boundary-preserving camouflage masks.
6. Multi-Task Structure Loss: Weighted BCE + Weighted IoU with 31x31 adaptive edge weighting (weit)
   plus auxiliary MSE loss on object gradient maps.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses


class ConvBR(layers.Layer):
    """
    Standard Convolution + Batch Normalization + ReLU block with He Normal initialization.
    Equivalent to PyTorch ConvBR in official DGNet.
    """

    def __init__(
        self,
        out_channel: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "same",
        dilation_rate: int = 1,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate

        self.conv = layers.Conv2D(
            filters=out_channel,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name}_conv" if name else None
        )
        self.bn = layers.BatchNormalization(name=f"{name}_bn" if name else None)
        self.relu = layers.ReLU(name=f"{name}_relu" if name else None)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM):
    Sequential Channel Attention + Spatial Attention for effective feature focus.
    """

    def __init__(self, channels: int, reduction: int = 16, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        reduced_ch = max(channels // reduction, 8)
        self.fc1 = layers.Dense(reduced_ch, activation="relu", use_bias=False, name=f"{name}_fc1" if name else None)
        self.fc2 = layers.Dense(channels, use_bias=False, name=f"{name}_fc2" if name else None)
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid", use_bias=False, name=f"{name}_spconv" if name else None)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Channel Attention (Global Avg & Max Pooling)
        avg_p = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_p = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        ch_attn = tf.nn.sigmoid(self.fc2(self.fc1(avg_p)) + self.fc2(self.fc1(max_p)))
        x = x * ch_attn

        # Spatial Attention
        avg_s = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_s = tf.reduce_max(x, axis=-1, keepdims=True)
        sp_concat = tf.concat([avg_s, max_s], axis=-1)
        sp_attn = self.spatial_conv(sp_concat)
        x = x * sp_attn
        return x


class DimensionalReduction(layers.Layer):
    """
    Dimensional Reduction module in official DGNet enhanced with CBAM Attention:
    Passes extracted backbone features through two 3x3 ConvBR blocks followed by CBAM attention.
    Includes SpatialDropout2D for regularization against overfitting on small datasets.
    """

    def __init__(self, out_channel: int, use_cbam: bool = True, dropout_rate: float = 0.1, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduce1 = ConvBR(out_channel, kernel_size=3, padding="same", name=f"{name}_br1" if name else None)
        self.reduce2 = ConvBR(out_channel, kernel_size=3, padding="same", name=f"{name}_br2" if name else None)
        self.cbam = CBAM(out_channel, name=f"{name}_cbam" if name else None) if use_cbam else None
        # SpatialDropout2D drops entire feature maps (channels) rather than individual elements,
        # which is more effective for convolutional features to prevent overfitting
        self.spatial_dropout = layers.SpatialDropout2D(rate=dropout_rate, name=f"{name}_sdrop" if name else None) if dropout_rate > 0 else None

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.reduce1(x, training=training)
        x = self.reduce2(x, training=training)
        if self.cbam is not None:
            x = self.cbam(x)
        if self.spatial_dropout is not None:
            x = self.spatial_dropout(x, training=training)
        return x


class SoftGroupingStrategy(layers.Layer):
    """
    Soft Grouping Strategy (SGS) from official DGNet:
    Applies 3 parallel 1x1 grouped convolutions with group sizes N=[4, 8, 16] and sums their outputs.
    """

    def __init__(self, in_channel: int, out_channel: int, N: List[int] = [4, 8, 16], name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.g_conv1 = layers.Conv2D(out_channel, kernel_size=1, groups=N[0], use_bias=False, kernel_initializer="he_normal")
        self.g_conv2 = layers.Conv2D(out_channel, kernel_size=1, groups=N[1], use_bias=False, kernel_initializer="he_normal")
        self.g_conv3 = layers.Conv2D(out_channel, kernel_size=1, groups=N[2], use_bias=False, kernel_initializer="he_normal")

    def call(self, q: tf.Tensor) -> tf.Tensor:
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)


class GradientInducedTransition(layers.Layer):
    """
    Gradient-Induced Transition (GIT) module from official DGNet:
    Transmits gradient cues from the Texture Encoder into multi-scale context features.
    Uses grouped feature interleaving followed by Soft Grouping Strategy (SGS) and residual addition.
    """

    def __init__(
        self,
        channel: int = 64,
        M: List[int] = [8, 8, 8],
        N: List[int] = [4, 8, 16],
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.channel = channel
        self.M = M
        self.N = N

        self.sgs3 = SoftGroupingStrategy(in_channel=channel + 32, out_channel=channel, N=N, name=f"{name}_sgs3" if name else None)
        self.sgs4 = SoftGroupingStrategy(in_channel=channel + 32, out_channel=channel, N=N, name=f"{name}_sgs4" if name else None)
        self.sgs5 = SoftGroupingStrategy(in_channel=channel + 32, out_channel=channel, N=N, name=f"{name}_sgs5" if name else None)

    def _gradient_induced_feature_grouping(self, xr: tf.Tensor, xg: tf.Tensor, M: int) -> tf.Tensor:
        """
        Interleaves feature channels between context features (xr) and gradient features (xg).
        Splits both into M channel chunks and interleaves them along the channel axis.
        """
        if M not in [1, 2, 4, 8, 16, 32]:
            raise ValueError(f"Invalid Group Number M: {M}. Must be one of [1, 2, 4, 8, 16, 32].")

        if M == 1:
            return tf.concat([xr, xg], axis=-1)

        xr_g = tf.split(xr, num_or_size_splits=M, axis=-1)
        xg_g = tf.split(xg, num_or_size_splits=M, axis=-1)
        interleaved = []
        for i in range(M):
            interleaved.extend([xr_g[i], xg_g[i]])

        return tf.concat(interleaved, axis=-1)

    def call(self, xr3: tf.Tensor, xr4: tf.Tensor, xr5: tf.Tensor, xg: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Spatial dimensions of context features
        h4, w4 = tf.shape(xr4)[1], tf.shape(xr4)[2]
        h5, w5 = tf.shape(xr5)[1], tf.shape(xr5)[2]

        # Downsample gradient feature xg to match xr4 (H/16) and xr5 (H/32)
        xg_down2 = tf.image.resize(xg, (h4, w4), method="bilinear")
        xg_down4 = tf.image.resize(xg, (h5, w5), method="bilinear")

        # Transmit gradient cues into context embeddings
        q3 = self._gradient_induced_feature_grouping(xr3, xg, M=self.M[0])
        q4 = self._gradient_induced_feature_grouping(xr4, xg_down2, M=self.M[1])
        q5 = self._gradient_induced_feature_grouping(xr5, xg_down4, M=self.M[2])

        # Attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5


class NeighborConnectionDecoder(layers.Layer):
    """
    Neighbor Connection Decoder (NCD) from official DGNet with Deep Supervision:
    Fuses multi-scale features hierarchically from coarse to fine (zt5 -> zt4 -> zt3)
    using multiplicative neighbor connections and cross-scale concatenation.
    Produces high-resolution, boundary-faithful camouflage predictions.
    Deep supervision outputs (pc4, pc5) are generated for multi-scale loss supervision.
    """

    def __init__(self, channel: int = 64, deep_supervision: bool = True, dropout_rate: float = 0.15, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channel = channel
        self.deep_supervision = deep_supervision

        self.conv_upsample1 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_u1" if name else None)
        self.conv_upsample2 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_u2" if name else None)
        self.conv_upsample3 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_u3" if name else None)
        self.conv_upsample4 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_u4" if name else None)
        self.conv_upsample5 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_u5" if name else None)

        # Official DGNet: conv_concat takes concatenated 2*channel (128) and projects back to channel (64)
        self.conv_concat2 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_c2" if name else None)
        self.conv_concat3 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_c3" if name else None)
        self.conv4 = ConvBR(channel, kernel_size=3, padding="same", name=f"{name}_c4" if name else None)
        # Dropout before final prediction heads to prevent decoder feature co-adaptation
        self.decoder_dropout = layers.Dropout(rate=dropout_rate, name=f"{name}_drop" if name else None) if dropout_rate > 0 else None
        self.conv5 = layers.Conv2D(1, kernel_size=1, use_bias=True, kernel_initializer="he_normal", name=f"{name}_c5" if name else None)

        if self.deep_supervision:
            self.aux_conv4 = layers.Conv2D(1, kernel_size=1, use_bias=True, kernel_initializer="he_normal", name=f"{name}_aux4" if name else None)
            self.aux_conv5 = layers.Conv2D(1, kernel_size=1, use_bias=True, kernel_initializer="he_normal", name=f"{name}_aux5" if name else None)

    def _resize_to(self, x: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        h, w = tf.shape(target)[1], tf.shape(target)[2]
        return tf.image.resize(x, (h, w), method="bilinear")

    def call(self, zt5: tf.Tensor, zt4: tf.Tensor, zt3: tf.Tensor, training: Optional[bool] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(self._resize_to(zt5, zt4), training=training) * zt4
        zt3_1 = (
            self.conv_upsample2(self._resize_to(zt4_1, zt3), training=training) *
            self.conv_upsample3(self._resize_to(zt4, zt3), training=training) *
            zt3
        )

        zt4_2 = tf.concat([zt4_1, self.conv_upsample4(self._resize_to(zt5_1, zt4), training=training)], axis=-1)
        zt4_2 = self.conv_concat2(zt4_2, training=training)

        zt3_2 = tf.concat([zt3_1, self.conv_upsample5(self._resize_to(zt4_2, zt3), training=training)], axis=-1)
        zt3_2 = self.conv_concat3(zt3_2, training=training)

        pc = self.conv4(zt3_2, training=training)
        if self.decoder_dropout is not None:
            pc = self.decoder_dropout(pc, training=training)
        pc = self.conv5(pc)

        if self.deep_supervision:
            pc4 = self.aux_conv4(zt4_2)
            pc5 = self.aux_conv5(zt5_1)
            return pc, pc4, pc5

        return pc


class TextureEncoder(layers.Layer):
    """
    Texture Encoder from official DGNet:
    Processes the raw 3-channel RGB image through 3 strided convolutional stages (total stride 8).
    Outputs:
      xg: Texture gradient feature map at stride 8 with 32 channels.
      pg: Gradient prediction map at stride 8 with 1 channel.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = ConvBR(64, kernel_size=7, stride=2, padding="same", name=f"{name}_c1" if name else None)
        self.conv2 = ConvBR(64, kernel_size=3, stride=2, padding="same", name=f"{name}_c2" if name else None)
        self.conv3 = ConvBR(32, kernel_size=3, stride=2, padding="same", name=f"{name}_c3" if name else None)
        self.conv_out = layers.Conv2D(1, kernel_size=1, strides=1, padding="same", use_bias=True, kernel_initializer="he_normal", name=f"{name}_out" if name else None)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        feat = self.conv1(x, training=training)
        feat = self.conv2(feat, training=training)
        xg = self.conv3(feat, training=training)
        pg = self.conv_out(xg)
        return xg, pg


class ContextEncoder(layers.Layer):
    """
    Context Encoder: Multi-scale semantic feature extractor across strides 8, 16, 32.
    Supports MobileNetV3Large, EfficientNetB1 (official DGNet-S), EfficientNetB4 (official DGNet), and ResNet50.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        input_shape: Tuple[int, int, int] = (384, 384, 3),
        pretrained: bool = True,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.backbone_name = backbone_name.lower().replace("-", "_")
        weights = "imagenet" if pretrained else None

        if "resnet50" in self.backbone_name:
            base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
            x3_name, x4_name, x5_name = "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"
            self.in_channels = [512, 1024, 2048]
        elif "efficientnet_b1" in self.backbone_name:
            base = tf.keras.applications.EfficientNetB1(include_top=False, weights=weights, input_shape=input_shape)
            x3_name, x4_name, x5_name = "block3c_add", "block5d_add", "block7b_add"
            self.in_channels = [40, 112, 320]
        elif "efficientnet_b4" in self.backbone_name:
            base = tf.keras.applications.EfficientNetB4(include_top=False, weights=weights, input_shape=input_shape)
            x3_name, x4_name, x5_name = "block3d_add", "block5f_add", "block7b_add"
            self.in_channels = [56, 160, 448]
        else:  # mobilenet_v3_large default
            base = tf.keras.applications.MobileNetV3Large(include_top=False, weights=weights, input_shape=input_shape)
            layer_map = {l.name: l for l in base.layers}
            x3_candidates = ["expanded_conv_5_add", "expanded_conv_3_project_bn", "expanded_conv_3/Add", "expanded_conv_3_add"]
            x4_candidates = ["expanded_conv_11_add", "expanded_conv_11/Add", "expanded_conv_11_project_bn"]
            x5_candidates = ["expanded_conv_14_add", "expanded_conv_14_project_bn", "expanded_conv_14/Add", "expanded_conv_13_add", "conv_1"]

            def _find_layer(candidates):
                for c in candidates:
                    if c in layer_map:
                        return c
                raise ValueError(f"Could not find valid layer among {candidates} in MobileNetV3Large.")

            x3_name = _find_layer(x3_candidates)
            x4_name = _find_layer(x4_candidates)
            x5_name = _find_layer(x5_candidates)
            self.in_channels = [
                base.get_layer(x3_name).output.shape[-1],
                base.get_layer(x4_name).output.shape[-1],
                base.get_layer(x5_name).output.shape[-1]
            ]

        self.extractor = models.Model(
            inputs=base.input,
            outputs=[
                base.get_layer(x3_name).output,
                base.get_layer(x4_name).output,
                base.get_layer(x5_name).output
            ],
            name=f"{self.backbone_name}_multi_scale"
        )

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.extractor(x, training=training)


class DGNet(models.Model):
    """
    Official Deep Gradient Network (DGNet) in TensorFlow 2.x / Keras.
    Direct implementation of the MIR 2023 paper and GewelsJI/DGNet repository.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        input_shape: Tuple[int, int, int] = (384, 384, 3),
        channel: int = 64,
        M: List[int] = [8, 8, 8],
        N: List[int] = [4, 8, 16],
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.channel = channel
        self.gradient_kernel = kwargs.get("gradient_kernel", "sobel")
        self.num_classes = kwargs.get("num_classes", 1)

        # 1. Context Encoder
        self.context_encoder = ContextEncoder(
            backbone_name=backbone_name,
            input_shape=input_shape,
            pretrained=pretrained,
            name="context_encoder"
        )

        # 2. Texture Encoder
        self.texture_encoder = TextureEncoder(name="texture_encoder")

        # 3. Dimensional Reduction for Context Features (x3, x4, x5 -> xr3, xr4, xr5)
        self.dr3 = DimensionalReduction(out_channel=channel, name="dr3")
        self.dr4 = DimensionalReduction(out_channel=channel, name="dr4")
        self.dr5 = DimensionalReduction(out_channel=channel, name="dr5")

        # 4. Gradient-Induced Transition (GIT)
        self.git = GradientInducedTransition(channel=channel, M=M, N=N, name="git")

        # 5. Neighbor Connection Decoder (NCD)
        self.ncd = NeighborConnectionDecoder(channel=channel, name="ncd")

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        target_h = tf.shape(inputs)[1]
        target_w = tf.shape(inputs)[2]

        # 1. Context Path: extract multi-scale features at strides 8, 16, 32
        x3, x4, x5 = self.context_encoder(inputs, training=training)
        xr3 = self.dr3(x3, training=training)
        xr4 = self.dr4(x4, training=training)
        xr5 = self.dr5(x5, training=training)

        # 2. Spatial / Texture Path: extract gradient features and edge predictions
        xg, pg = self.texture_encoder(inputs, training=training)

        # 3. Gradient-Induced Transition (GIT)
        zt3, zt4, zt5 = self.git(xr3, xr4, xr5, xg, training=training)

        # 4. Neighbor Connection Decoder (NCD)
        ncd_out = self.ncd(zt5, zt4, zt3, training=training)
        if isinstance(ncd_out, tuple):
            pc, pc4, pc5 = ncd_out
        else:
            pc = ncd_out
            pc4 = pc5 = None

        # 5. Upsample predictions to full resolution
        mask_logits = tf.image.resize(pc, (target_h, target_w), method="bilinear")
        grad_logits = tf.image.resize(pg, (target_h, target_w), method="bilinear")

        if training or return_dict:
            res_dict = {
                "mask_pred": mask_logits,
                "grad_pred": grad_logits
            }
            if pc4 is not None:
                res_dict["mask_pred_aux4"] = tf.image.resize(pc4, (target_h, target_w), method="bilinear")
            if pc5 is not None:
                res_dict["mask_pred_aux5"] = tf.image.resize(pc5, (target_h, target_w), method="bilinear")
            return res_dict

        return tf.nn.sigmoid(mask_logits)


class DGNetLoss(losses.Loss):
    """
    Official DGNet Multi-Task Structure Loss + Novel Deep Supervision & Boundary IoU Loss:
    Total_Loss = structure_loss(mask_logits, gt_mask)
                 + 0.5 * structure_loss(mask_aux4, gt_mask)
                 + 0.25 * structure_loss(mask_aux5, gt_mask)
                 + 0.5 * boundary_loss(mask_logits, gt_mask)
                 + lambda_grad * MSE(grad_pred, grad_gt)
    """

    def __init__(self, lambda_grad_loss: float = 1.0, label_smoothing: float = 0.02, name: str = "dgnet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_grad_loss = lambda_grad_loss
        self.label_smoothing = label_smoothing
        # 3x3 Laplacian kernel for edge detection
        self.laplacian_kernel = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=tf.float32)
        self.laplacian_kernel = tf.reshape(self.laplacian_kernel, [3, 3, 1, 1])

    def structure_loss(self, y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
        """
        Structure loss matching official DGNet (F3Net AAAI 2020).
        Label smoothing applied to prevent overconfident predictions on small datasets.
        """
        y_true_f = tf.cast(y_true, tf.float32)
        # Apply label smoothing: hard 0/1 labels become soft 0.02/0.98
        if self.label_smoothing > 0:
            y_true_f = y_true_f * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0

        # Adaptive spatial weight map (kernel 31x31, stride 1, padding same)
        avg_pool = tf.nn.avg_pool2d(y_true_f, ksize=[1, 31, 31, 1], strides=[1, 1, 1, 1], padding="SAME")
        weit = 1.0 + 5.0 * tf.abs(avg_pool - y_true_f)

        # Weighted BCE loss with logits
        bce_raw = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_f, logits=y_pred_logits)
        wbce = tf.reduce_sum(weit * bce_raw, axis=[1, 2, 3]) / (tf.reduce_sum(weit, axis=[1, 2, 3]) + 1e-8)

        # Weighted IoU loss
        pred_sig = tf.nn.sigmoid(y_pred_logits)
        inter = tf.reduce_sum((pred_sig * y_true_f) * weit, axis=[1, 2, 3])
        union = tf.reduce_sum((pred_sig + y_true_f) * weit, axis=[1, 2, 3])
        wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

        return tf.reduce_mean(wbce + wiou)

    def boundary_loss(self, y_true: tf.Tensor, y_pred_logits: tf.Tensor) -> tf.Tensor:
        """Boundary IoU loss to sharpen object boundaries."""
        y_true_f = tf.cast(y_true, tf.float32)
        pred_sig = tf.nn.sigmoid(y_pred_logits)

        gt_edges = tf.abs(tf.nn.conv2d(y_true_f, self.laplacian_kernel, strides=[1, 1, 1, 1], padding="SAME"))
        gt_edges = tf.clip_by_value(gt_edges, 0.0, 1.0)

        pred_edges = tf.abs(tf.nn.conv2d(pred_sig, self.laplacian_kernel, strides=[1, 1, 1, 1], padding="SAME"))
        pred_edges = tf.clip_by_value(pred_edges, 0.0, 1.0)

        inter = tf.reduce_sum(gt_edges * pred_edges, axis=[1, 2, 3])
        union = tf.reduce_sum(gt_edges + pred_edges - gt_edges * pred_edges, axis=[1, 2, 3])
        boundary_iou = 1.0 - (inter + 1.0) / (union + 1.0)
        return tf.reduce_mean(boundary_iou)

    def call(self, y_true: tf.Tensor, y_pred_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        mask_logits = y_pred_dict["mask_pred"]
        grad_logits = y_pred_dict["grad_pred"]

        seg_loss = self.structure_loss(y_true, mask_logits)

        # Auxiliary deep supervision losses
        if "mask_pred_aux4" in y_pred_dict:
            seg_loss += 0.5 * self.structure_loss(y_true, y_pred_dict["mask_pred_aux4"])
        if "mask_pred_aux5" in y_pred_dict:
            seg_loss += 0.25 * self.structure_loss(y_true, y_pred_dict["mask_pred_aux5"])

        # Boundary refinement loss
        b_loss = self.boundary_loss(y_true, mask_logits)
        seg_loss += 0.5 * b_loss

        # Auxiliary gradient loss (MSE)
        grad_gt_target = y_pred_dict.get("gt_grad", y_pred_dict.get("grad_gt", None))
        if grad_gt_target is not None:
            gt_grad = tf.cast(grad_gt_target, tf.float32)
            grad_pred_val = tf.nn.sigmoid(grad_logits)
            grad_loss = tf.reduce_mean(tf.square(grad_pred_val - gt_grad))
            total_loss = seg_loss + self.lambda_grad_loss * grad_loss
        else:
            total_loss = seg_loss

        return total_loss


def build_dgnet_model(
    backbone_name: str = "mobilenet_v3_large",
    input_shape: Tuple[int, int, int] = (384, 384, 3),
    channel: int = 64,
    gradient_kernel: Optional[str] = None,
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs
) -> DGNet:
    """Factory helper function to instantiate official DGNet model."""
    return DGNet(
        backbone_name=backbone_name,
        input_shape=input_shape,
        channel=channel,
        pretrained=pretrained,
        **kwargs
    )
