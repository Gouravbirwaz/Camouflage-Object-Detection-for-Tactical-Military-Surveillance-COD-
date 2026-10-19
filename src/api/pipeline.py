"""
Self-contained, lightweight image preprocessing and mask postprocessing pipeline.
Zero dependency on training modules, DVC, MLflow, or heavy framework code.
"""

from typing import Tuple
import cv2
import numpy as np

# Exact ImageNet statistics required by the trained DGNet model
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def edge_preprocess_frame(
    frame_bgr: np.ndarray,
    target_size: Tuple[int, int] = (384, 384)
) -> np.ndarray:
    """
    Preprocess BGR OpenCV frame for DGNet inference:
    1. BGR -> RGB
    2. Resize to model dimensions (384, 384)
    3. Normalize [0, 1] and apply ImageNet mean/std
    4. Expand batch dimension -> [1, 384, 384, 3] float32
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size[1], target_size[0])).astype(np.float32)
    normalized = (resized / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.expand_dims(normalized, axis=0).astype(np.float32)


def process_prediction_mask(
    pred_prob: np.ndarray,
    threshold: float = 0.50,
    min_contour_area: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess raw DGNet sigmoid probabilities into a clean binary camouflage mask.
    - Gate 1: If max confidence < threshold, early-exit with zero mask (prevents false alarms).
    - Gate 2: Absolute probability threshold (prob >= threshold).
    - Gate 3: Filter small noise blobs smaller than min_contour_area.
    """
    is_3d = (pred_prob.ndim == 3)
    prob_2d = pred_prob[:, :, 0] if is_3d else pred_prob.copy()

    # Gate 1: Early-exit when peak confidence is below threshold
    if float(prob_2d.max()) < threshold:
        zero = np.zeros_like(prob_2d, dtype=np.float32)
        if is_3d:
            return zero[:, :, np.newaxis], zero[:, :, np.newaxis]
        return zero, zero

    # Gate 2: Binary mask from threshold
    binary_2d = (prob_2d >= threshold).astype(np.uint8)

    # Gate 3: Filter texture noise blobs
    if binary_2d.any():
        contours, _ = cv2.findContours(binary_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(binary_2d)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_contour_area:
                cv2.drawContours(cleaned, [cnt], -1, 1, thickness=-1)
        binary_2d = cleaned

    binary_float = binary_2d.astype(np.float32)
    if is_3d:
        return prob_2d[:, :, np.newaxis], binary_float[:, :, np.newaxis]
    return prob_2d, binary_float
