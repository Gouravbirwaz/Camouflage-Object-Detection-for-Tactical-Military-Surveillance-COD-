"""
Comprehensive Camouflaged Object Detection (COD) Metrics Module
TensorFlow 2.x compatible, batch-aware computation.

Computes all standard COD evaluation metrics:
- Pixel Accuracy, Mean Absolute Error (MAE)
- IoU (Jaccard), Dice Coefficient
- F-measure (Fβ, β²=0.3), Precision, Recall, Specificity
- False Positive Rate (FPR), False Negative Rate (FNR)
"""

import tensorflow as tf
from typing import Dict


def compute_confusion_components(
    y_true: tf.Tensor,
    y_pred_prob: tf.Tensor,
    threshold: float = 0.5
) -> Dict[str, tf.Tensor]:
    """
    Computes TP, TN, FP, FN counts from ground truth masks and predicted probability maps.

    Args:
        y_true: Ground truth binary masks [B, H, W, 1], values in {0, 1}.
        y_pred_prob: Predicted probability maps [B, H, W, 1], values in [0, 1].
        threshold: Binarization threshold for predictions.

    Returns:
        Dict with scalar tensors: 'tp', 'tn', 'fp', 'fn'.
    """
    y_pred_bin = tf.cast(y_pred_prob >= threshold, tf.float32)
    y_true_f = tf.cast(y_true, tf.float32)

    tp = tf.reduce_sum(y_pred_bin * y_true_f)
    tn = tf.reduce_sum((1.0 - y_pred_bin) * (1.0 - y_true_f))
    fp = tf.reduce_sum(y_pred_bin * (1.0 - y_true_f))
    fn = tf.reduce_sum((1.0 - y_pred_bin) * y_true_f)

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def pixel_accuracy(tp: tf.Tensor, tn: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor) -> tf.Tensor:
    """Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
    total = tp + tn + fp + fn
    return tf.where(total > 0, (tp + tn) / total, 0.0)


def mean_absolute_error(y_true: tf.Tensor, y_pred_prob: tf.Tensor) -> tf.Tensor:
    """Mean Absolute Error (MAE) = mean(|pred - gt|) over all pixels."""
    return tf.reduce_mean(tf.abs(y_pred_prob - tf.cast(y_true, tf.float32)))


def precision_score(tp: tf.Tensor, fp: tf.Tensor) -> tf.Tensor:
    """Precision = TP / (TP + FP)"""
    denom = tp + fp
    return tf.where(denom > 0, tp / denom, 0.0)


def recall_score(tp: tf.Tensor, fn: tf.Tensor) -> tf.Tensor:
    """Recall (Sensitivity) = TP / (TP + FN)"""
    denom = tp + fn
    return tf.where(denom > 0, tp / denom, 0.0)


def specificity_score(tn: tf.Tensor, fp: tf.Tensor) -> tf.Tensor:
    """Specificity = TN / (TN + FP)"""
    denom = tn + fp
    return tf.where(denom > 0, tn / denom, 0.0)


def false_positive_rate(fp: tf.Tensor, tn: tf.Tensor) -> tf.Tensor:
    """False Positive Rate = FP / (FP + TN)"""
    denom = fp + tn
    return tf.where(denom > 0, fp / denom, 0.0)


def false_negative_rate(fn: tf.Tensor, tp: tf.Tensor) -> tf.Tensor:
    """False Negative Rate = FN / (FN + TP)"""
    denom = fn + tp
    return tf.where(denom > 0, fn / denom, 0.0)


def iou_score(tp: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor) -> tf.Tensor:
    """Intersection over Union (Jaccard) = TP / (TP + FP + FN)"""
    denom = tp + fp + fn
    return tf.where(denom > 0, tp / denom, 0.0)


def dice_score(tp: tf.Tensor, fp: tf.Tensor, fn: tf.Tensor) -> tf.Tensor:
    """Dice Coefficient = 2*TP / (2*TP + FP + FN)"""
    denom = 2.0 * tp + fp + fn
    return tf.where(denom > 0, 2.0 * tp / denom, 0.0)


def f_measure(precision: tf.Tensor, recall: tf.Tensor, beta_sq: float = 0.3) -> tf.Tensor:
    """
    F-measure (Fβ) = (1 + β²) * Precision * Recall / (β² * Precision + Recall)
    Standard COD uses β² = 0.3 to weight precision more than recall.
    """
    denom = beta_sq * precision + recall
    return tf.where(denom > 0, (1.0 + beta_sq) * precision * recall / denom, 0.0)


def compute_all_metrics(
    y_true: tf.Tensor,
    y_pred_prob: tf.Tensor,
    threshold: float = 0.5,
    beta_sq: float = 0.3
) -> Dict[str, float]:
    """
    Computes all COD metrics in a single pass.

    Args:
        y_true: Ground truth binary masks [B, H, W, 1].
        y_pred_prob: Predicted probability maps [B, H, W, 1] (after sigmoid).
        threshold: Binarization threshold.
        beta_sq: Beta-squared for F-measure (0.3 is standard for COD).

    Returns:
        Dict of metric_name -> float value.
    """
    cc = compute_confusion_components(y_true, y_pred_prob, threshold)
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]

    prec = precision_score(tp, fp)
    rec = recall_score(tp, fn)

    return {
        "pixel_accuracy": float(pixel_accuracy(tp, tn, fp, fn)),
        "mae": float(mean_absolute_error(y_true, y_pred_prob)),
        "iou": float(iou_score(tp, fp, fn)),
        "dice": float(dice_score(tp, fp, fn)),
        "f_measure": float(f_measure(prec, rec, beta_sq)),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity_score(tn, fp)),
        "fpr": float(false_positive_rate(fp, tn)),
        "fnr": float(false_negative_rate(fn, tp)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


class CODMetricsAccumulator:
    """
    Accumulates confusion matrix components across batches for epoch-level metric computation.
    Use reset() at epoch start, update() per batch, and result() at epoch end.
    """

    def __init__(self, threshold: float = 0.5, beta_sq: float = 0.3):
        self.threshold = threshold
        self.beta_sq = beta_sq
        self.reset()

    def reset(self):
        """Reset all accumulators at the start of each epoch."""
        self.tp_sum = 0.0
        self.tn_sum = 0.0
        self.fp_sum = 0.0
        self.fn_sum = 0.0
        self.mae_sum = 0.0
        self.pixel_count = 0
        self.sample_count = 0

    def update(self, y_true: tf.Tensor, y_pred_prob: tf.Tensor):
        """
        Accumulate metrics for a single batch.

        Args:
            y_true: Ground truth binary masks [B, H, W, 1].
            y_pred_prob: Predicted probability maps [B, H, W, 1] (after sigmoid).
        """
        cc = compute_confusion_components(y_true, y_pred_prob, self.threshold)
        self.tp_sum += float(cc["tp"])
        self.tn_sum += float(cc["tn"])
        self.fp_sum += float(cc["fp"])
        self.fn_sum += float(cc["fn"])

        # MAE accumulated as total absolute error
        batch_size = tf.shape(y_true)[0]
        h = tf.shape(y_true)[1]
        w = tf.shape(y_true)[2]
        num_pixels = int(batch_size) * int(h) * int(w)
        abs_err = tf.reduce_sum(tf.abs(y_pred_prob - tf.cast(y_true, tf.float32)))
        self.mae_sum += float(abs_err)
        self.pixel_count += num_pixels
        self.sample_count += int(batch_size)

    def result(self) -> Dict[str, float]:
        """Compute all epoch-level metrics from accumulated values."""
        tp = self.tp_sum
        tn = self.tn_sum
        fp = self.fp_sum
        fn = self.fn_sum

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        mae = self.mae_sum / self.pixel_count if self.pixel_count > 0 else 0.0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0

        beta_sq = self.beta_sq
        fm_denom = beta_sq * prec + rec
        fm = (1.0 + beta_sq) * prec * rec / fm_denom if fm_denom > 0 else 0.0

        return {
            "pixel_accuracy": round(acc, 6),
            "mae": round(mae, 6),
            "iou": round(iou, 6),
            "dice": round(dice, 6),
            "f_measure": round(fm, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "specificity": round(spec, 6),
            "fpr": round(fpr, 6),
            "fnr": round(fnr, 6),
            "tp_total": round(tp, 1),
            "tn_total": round(tn, 1),
            "fp_total": round(fp, 1),
            "fn_total": round(fn, 1),
            "sample_count": self.sample_count,
        }
