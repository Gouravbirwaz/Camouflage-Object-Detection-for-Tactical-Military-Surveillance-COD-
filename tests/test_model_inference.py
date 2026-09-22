#!/usr/bin/env python3
"""
DGNet Multi-Model Inference & Visual Camouflage Detection Test Suite
Loads trained model checkpoints/formats (Keras weights, TFLite, ONNX), runs inference on sample RGB images,
and saves predicted camouflage masks, color heatmaps, contour overlays, side-by-side multi-model comparisons,
and an interactive HTML visual report.

Usage:
    python tests/test_model_inference.py
    python tests/test_model_inference.py --all-models
    python tests/test_model_inference.py --image data/raw/dataset-splitM/Testing/Images/image515.jpg --all-models
    python tests/test_model_inference.py --threshold 0.35 --output-dir tests/unit/output
"""

import argparse
import html
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Try importing cv2 for color heatmap and contour rendering
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dgnet import build_dgnet_model
from src.utils.config import load_project_config
from src.utils.metrics import compute_all_metrics


def load_and_preprocess_image(image_path: str, image_size: tuple = (384, 384)) -> tuple:
    """
    Loads an RGB image, resizes it, and normalizes to ImageNet stats.

    Returns:
        (input_tensor, original_image)
        - input_tensor: [1, H, W, 3] float32 tensor
        - original_image: original-resolution uint8 numpy array for visualization
    """
    raw_bytes = tf.io.read_file(image_path)
    original = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    original_np = original.numpy()

    resized = tf.image.resize(original, image_size)
    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    normalized = (tf.cast(resized, tf.float32) / 255.0 - imagenet_mean) / imagenet_std
    input_tensor = tf.expand_dims(normalized, axis=0)  # [1, H, W, 3]

    return input_tensor, original_np


def load_ground_truth(gt_path: str, image_size: tuple = (384, 384)):
    """Loads ground truth mask if available. Returns None if not found."""
    if not os.path.exists(gt_path):
        return None
    raw_bytes = tf.io.read_file(gt_path)
    mask = tf.image.decode_image(raw_bytes, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, image_size, method="nearest")
    mask = tf.cast(mask >= 128, tf.float32)
    return mask


def process_prediction_mask(
    pred_prob: np.ndarray,
    threshold: float = 0.5,
    sharpen: bool = True,
    adaptive_threshold: bool = True,
    auto_invert: bool = False
) -> tuple:
    """
    Processes raw prediction probability array into a scaled probability map and binary mask.

    Returns:
        (scaled_prob, binary_mask)
    """
    is_3d = (pred_prob.ndim == 3)
    pred_prob_2d = pred_prob[:, :, 0] if is_3d else pred_prob

    p_min, p_max = float(pred_prob_2d.min()), float(pred_prob_2d.max())

    # Sharpening via Min-Max Scaling to full dynamic range [0, 1]
    if sharpen and (p_max - p_min) > 1e-4:
        scaled_prob_2d = (pred_prob_2d - p_min) / (p_max - p_min)
    else:
        scaled_prob_2d = pred_prob_2d.copy()

    # Inversion handling if background is bright
    initial_cov = np.mean(scaled_prob_2d >= threshold)
    if auto_invert and initial_cov > 0.5:
        scaled_prob_2d = 1.0 - scaled_prob_2d

    # Adaptive or fixed thresholding
    if adaptive_threshold and p_max < threshold and (p_max - p_min) > 1e-4:
        effective_thresh = 0.5
    else:
        effective_thresh = threshold

    binary_mask_2d = (scaled_prob_2d >= effective_thresh).astype(np.float32)

    if is_3d:
        scaled_prob = scaled_prob_2d[:, :, np.newaxis]
        binary_mask = binary_mask_2d[:, :, np.newaxis]
    else:
        scaled_prob = scaled_prob_2d
        binary_mask = binary_mask_2d

    return scaled_prob, binary_mask


def save_mask_png(mask_array: np.ndarray, output_path: str):
    """Saves a [H, W] or [H, W, 1] float mask as a grayscale PNG (0-255)."""
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_uint8 = (np.clip(mask_array, 0.0, 1.0) * 255).astype(np.uint8)
    encoded = tf.image.encode_png(tf.expand_dims(mask_uint8, -1))
    tf.io.write_file(output_path, encoded)


def save_color_heatmap_png(prob_array: np.ndarray, output_path: str, sharpen: bool = True):
    """
    Renders probability prediction array into a vibrant color heatmap (JET colormap)
    to visually highlight soft camouflage confidence regions.
    """
    if prob_array.ndim == 3:
        prob_array = prob_array[:, :, 0]

    p_min, p_max = float(prob_array.min()), float(prob_array.max())
    if sharpen and (p_max - p_min) > 1e-4:
        prob_scaled = (prob_array - p_min) / (p_max - p_min)
    else:
        prob_scaled = prob_array

    prob_clipped = np.clip(prob_scaled, 0.0, 1.0)
    prob_uint8 = (prob_clipped * 255).astype(np.uint8)

    if HAS_CV2:
        heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Fallback RGB pseudo-color
        r = prob_uint8
        g = (np.sin(prob_clipped * np.pi) * 255).astype(np.uint8)
        b = (255 - prob_uint8)
        heatmap_rgb = np.stack([r, g, b], axis=-1)

    encoded = tf.image.encode_png(heatmap_rgb)
    tf.io.write_file(output_path, encoded)


def save_contour_overlay_image(original: np.ndarray, pred_mask: np.ndarray, output_path: str,
                              image_size: tuple = (384, 384)):
    """
    Overlays vibrant camouflage detection boundary contours (neon green/red) on the RGB image.
    """
    orig_resized = tf.image.resize(original, image_size).numpy().astype(np.uint8)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]

    mask_bool = pred_mask > 0.5
    overlay = orig_resized.copy()

    if HAS_CV2:
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw semi-transparent fill
        fill = overlay.copy()
        fill[mask_bool] = (fill[mask_bool] * 0.6 + np.array([0, 225, 120]) * 0.4).astype(np.uint8)
        cv2.drawContours(fill, contours, -1, (255, 30, 30), 2)  # Red outline
        overlay = fill
    else:
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 128, 0, 255)

    encoded = tf.image.encode_png(overlay)
    tf.io.write_file(output_path, encoded)


def save_comparison_image(original: np.ndarray, pred_mask: np.ndarray, output_path: str,
                          gt_mask: np.ndarray = None, image_size: tuple = (384, 384),
                          model_name: str = "PREDICTION", **kwargs):
    """
    Creates and saves a clean, uncolorized black-and-white comparison:
    [ GROUND TRUTH | MODEL: {model_name} ]
    NO colormap, NO heatmap, NO contour overlay, NO color tints. Pure grayscale binary (0/255).
    """
    h, w = image_size

    # Prepare Predicted binary mask [H, W] uint8
    if hasattr(pred_mask, "numpy"):
        pred_mask = pred_mask.numpy()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]
    pred_u8 = (pred_mask > 0.5 if pred_mask.max() <= 1.0 else pred_mask >= 128).astype(np.uint8) * 255
    if pred_u8.shape[:2] != (h, w) and HAS_CV2:
        pred_u8 = cv2.resize(pred_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    # Prepare GT mask [H, W] uint8
    if gt_mask is not None:
        if hasattr(gt_mask, "numpy"):
            gt_mask = gt_mask.numpy()
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_u8 = (gt_mask > 0.5 if gt_mask.max() <= 1.0 else gt_mask >= 128).astype(np.uint8) * 255
        if gt_u8.shape[:2] != (h, w) and HAS_CV2:
            gt_u8 = cv2.resize(gt_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        gt_u8 = np.zeros((h, w), dtype=np.uint8)

    # 4-pixel vertical divider
    sep_width = 4
    separator = np.full((h, sep_width), 128, dtype=np.uint8)

    body = np.hstack([gt_u8, separator, pred_u8])

    # Header bar with clear labels
    header_h = 36
    header = np.zeros((header_h, body.shape[1]), dtype=np.uint8)
    if HAS_CV2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        cv2.putText(header, "GROUND TRUTH", (15, 24), font, font_scale, 255, thickness, cv2.LINE_AA)
        label_text = f"MODEL: {model_name}"
        right_x = w + sep_width + 15
        cv2.putText(header, label_text, (right_x, 24), font, font_scale, 255, thickness, cv2.LINE_AA)

    h_div = np.full((2, body.shape[1]), 128, dtype=np.uint8)
    combined = np.vstack([header, h_div, body])

    encoded = tf.image.encode_png(tf.expand_dims(combined, -1))
    tf.io.write_file(output_path, encoded)


def save_multi_model_comparison(original: np.ndarray, gt_mask: np.ndarray,
                                model_results: list, output_path: str,
                                image_size: tuple = (384, 384)):
    """
    Creates a consolidated side-by-side comparison grid placing output masks of ALL models together:
    [ GROUND TRUTH | MODEL 1 | MODEL 2 | ... ]
    Pure black-and-white (grayscale binary). NO colormap, NO heatmap, NO overlay.
    """
    h, w = image_size
    panels = []
    headers = []

    # 1. Ground Truth
    if gt_mask is not None:
        gt_np = gt_mask.numpy() if hasattr(gt_mask, "numpy") else gt_mask
        if gt_np.ndim == 3:
            gt_np = gt_np[:, :, 0]
        gt_u8 = (gt_np > 0.5 if gt_np.max() <= 1.0 else gt_np >= 128).astype(np.uint8) * 255
    else:
        gt_u8 = np.zeros((h, w), dtype=np.uint8)

    if gt_u8.shape[:2] != (h, w) and HAS_CV2:
        gt_u8 = cv2.resize(gt_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    panels.append(gt_u8)
    headers.append("GROUND TRUTH")

    # 2. Each Model's Binary Prediction Mask
    for res in model_results:
        p_mask = res["binary_mask"]
        if hasattr(p_mask, "numpy"):
            p_mask = p_mask.numpy()
        if p_mask.ndim == 3:
            p_mask = p_mask[:, :, 0]
        p_u8 = (p_mask > 0.5 if p_mask.max() <= 1.0 else p_mask >= 128).astype(np.uint8) * 255
        if p_u8.shape[:2] != (h, w) and HAS_CV2:
            p_u8 = cv2.resize(p_u8, (w, h), interpolation=cv2.INTER_NEAREST)

        panels.append(p_u8)
        tag = res.get("tag", res.get("name", "Model"))
        headers.append(tag)

    sep_width = 4
    separator = np.full((h, sep_width), 128, dtype=np.uint8)

    interleaved_panels = []
    for idx, p in enumerate(panels):
        interleaved_panels.append(p)
        if idx < len(panels) - 1:
            interleaved_panels.append(separator)

    body = np.hstack(interleaved_panels)

    header_h = 36
    header = np.zeros((header_h, body.shape[1]), dtype=np.uint8)
    if HAS_CV2:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        current_x = 0
        for idx, title in enumerate(headers):
            text_x = current_x + 10
            cv2.putText(header, str(title), (text_x, 24), font, font_scale, 255, thickness, cv2.LINE_AA)
            current_x += w + sep_width

    h_div = np.full((2, body.shape[1]), 128, dtype=np.uint8)
    combined = np.vstack([header, h_div, body])

    encoded = tf.image.encode_png(tf.expand_dims(combined, -1))
    tf.io.write_file(output_path, encoded)



def find_default_sample_images() -> list:
    """Finds candidate test images from the dataset."""
    candidates = [
        # PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images/image515.jpg",
        PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images/image519.jpg",
        PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images/image501.jpg",
        PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images/image1000.jpg",
    ]
    found = [str(c) for c in candidates if c.exists()]
    if found:
        return found

    test_img_dir = PROJECT_ROOT / "data/raw/dataset-splitM/Testing/Images"
    if test_img_dir.exists():
        for f in sorted(test_img_dir.glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                found.append(str(f))
                if len(found) >= 5:
                    break
    return found


def find_ground_truth_for_image(image_path: str) -> str:
    """Attempts to find matching ground truth mask for an image."""
    img_path = Path(image_path)
    stem = img_path.stem

    gt_candidates = [
        img_path.parent.parent / "GT" / f"{stem}.png",
        img_path.parent.parent / "Masks" / f"{stem}.png",
        img_path.parent.parent / "GT" / f"{stem}.jpg",
    ]
    for gt in gt_candidates:
        if gt.exists():
            return str(gt)
    return None


def discover_all_models(models_dir: Path = None) -> list:
    """
    Crawls models/ directory to discover all checkpoints and exported models:
    - .weights.h5 / .keras / .h5 Keras checkpoints
    - .tflite TFLite models
    - .onnx ONNX models
    """
    target_dir = models_dir or (PROJECT_ROOT / "models")
    model_entries = []

    if not target_dir.exists():
        return model_entries

    valid_exts = {".weights.h5", ".keras", ".h5", ".tflite", ".onnx"}

    for path in sorted(target_dir.rglob("*")):
        if path.is_file():
            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if any(name_lower.endswith(ext) for ext in valid_exts):
                if path.stat().st_size == 0:
                    continue

                rel_path = path.relative_to(target_dir)
                parent_folder = path.parent.name
                stem = path.stem.replace(".weights", "")

                # Infer backbone from filename or path
                backbone = "mobilenet_v3_large"
                if "resnet50" in name_lower:
                    backbone = "resnet50"
                elif "efficientnet_b1" in name_lower:
                    backbone = "efficientnet_b1"
                elif "efficientnet_b4" in name_lower:
                    backbone = "efficientnet_b4"
                elif "efficientnet_b0" in name_lower:
                    backbone = "efficientnet_b0"

                model_type = "keras"
                if name_lower.endswith(".tflite"):
                    model_type = "tflite"
                elif name_lower.endswith(".onnx"):
                    model_type = "onnx"

                tag = stem

                model_entries.append({
                    "path": str(path.resolve()),
                    "name": path.name,
                    "tag": tag,
                    "rel_path": str(rel_path),
                    "parent_folder": parent_folder,
                    "model_type": model_type,
                    "backbone": backbone,
                    "size_mb": path.stat().st_size / (1024 * 1024)
                })

    return model_entries


class ModelRunner:
    """Wrapper to instantiate and run inference across different model formats."""

    def __init__(self, model_info: dict, img_size: tuple = (384, 384)):
        self.info = model_info
        self.img_size = img_size
        self.model_type = model_info["model_type"]
        self.path = model_info["path"]
        self.backbone = model_info["backbone"]
        self.loaded_model = None
        self._init_model()

    def _init_model(self):
        if self.model_type == "keras":
            self.loaded_model = build_dgnet_model(
                backbone_name=self.backbone,
                input_shape=(self.img_size[0], self.img_size[1], 3),
                gradient_kernel="sobel",
                pretrained=False
            )
            dummy = tf.zeros((1, self.img_size[0], self.img_size[1], 3))
            _ = self.loaded_model(dummy, training=False)
            try:
                if self.path.endswith(".keras"):
                    self.loaded_model = tf.keras.models.load_model(self.path, compile=False)
                else:
                    self.loaded_model.load_weights(self.path)
            except Exception as e:
                print(f"       Notice: Partial weight load for {self.info['name']} ({e})")
        elif self.model_type == "tflite":
            interpreter = tf.lite.Interpreter(model_path=self.path)
            interpreter.allocate_tensors()
            self.loaded_model = interpreter
        elif self.model_type == "onnx":
            try:
                import onnxruntime as ort
                self.loaded_model = ort.InferenceSession(self.path)
            except ImportError:
                print(f"       Notice: onnxruntime not installed, skipping ONNX model {self.info['name']}")
                self.loaded_model = None

    def predict(self, input_tensor: tf.Tensor) -> np.ndarray:
        """Executes forward pass and returns [H, W, 1] sigmoid numpy array."""
        if self.model_type == "keras":
            pred = self.loaded_model(input_tensor, training=False)
            return pred.numpy()[0]
        elif self.model_type == "tflite":
            interpreter = self.loaded_model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            inp_np = input_tensor.numpy()
            interpreter.set_tensor(input_details[0]['index'], inp_np)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index'])
            if pred.ndim == 4:
                pred = pred[0]
            return pred
        elif self.model_type == "onnx":
            if self.loaded_model is None:
                return np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32)
            input_name = self.loaded_model.get_inputs()[0].name
            inp_np = input_tensor.numpy()
            outputs = self.loaded_model.run(None, {input_name: inp_np})
            pred = outputs[0]
            if pred.ndim == 4:
                pred = pred[0]
            return pred
        return np.zeros((self.img_size[0], self.img_size[1], 1), dtype=np.float32)


def generate_html_visual_report(image_stem: str, model_results: list, output_dir: Path, gt_path: str = None):
    """
    Generates an HTML report visualizing all camouflage detection masks,
    probability heatmaps, and metrics across all models side-by-side.
    """
    report_path = output_dir / f"{image_stem}_visual_report.html"

    cards_html = ""
    for res in model_results:
        tag = res["tag"]
        name = res["name"]
        folder = res["parent_folder"]
        metrics = res.get("metrics", {})

        metrics_html = ""
        if metrics:
            metrics_html = f"""
            <div class="metrics-grid">
                <div class="metric"><span class="lbl">IoU</span><span class="val">{metrics.get('iou', 0):.4f}</span></div>
                <div class="metric"><span class="lbl">Dice</span><span class="val">{metrics.get('dice', 0):.4f}</span></div>
                <div class="metric"><span class="lbl">MAE</span><span class="val">{metrics.get('mae', 0):.4f}</span></div>
                <div class="metric"><span class="lbl">Pixel Acc</span><span class="val">{metrics.get('pixel_accuracy', 0)*100:.1f}%</span></div>
            </div>
            """

        cards_html += f"""
        <div class="model-card">
            <h3>{html.escape(name)} <span class="badge">{html.escape(folder)}</span></h3>
            <p class="meta">Backbone: <strong>{html.escape(res['backbone'])}</strong> | Type: <strong>{html.escape(res['model_type'])}</strong> | Inference: <strong>{res['inference_time_ms']:.1f} ms</strong></p>
            <div class="image-trio">
                <div>
                    <img src="{image_stem}_gt.png" alt="Ground Truth">
                    <p class="img-lbl">Ground Truth (GT)</p>
                </div>
                <div>
                    <img src="{image_stem}_{tag}_pred.png" alt="Binary Mask">
                    <p class="img-lbl">Predicted Mask ({html.escape(tag)})</p>
                </div>
                <div>
                    <img src="{image_stem}_{tag}_comparison.png" alt="Comparison">
                    <p class="img-lbl">Comparison [GT | {html.escape(tag)}]</p>
                </div>
            </div>
            {metrics_html}
        </div>
        """

    gt_html = f"<p><strong>Ground Truth:</strong> {html.escape(gt_path)}</p>" if gt_path else "<p><strong>Ground Truth:</strong> Not provided</p>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DGNet Camouflage Detection Visual Model Report - {image_stem}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; font-size: 24px; margin-bottom: 8px; }}
        h2 {{ color: #94a3b8; font-size: 16px; margin-top: 0; margin-bottom: 24px; font-weight: normal; }}
        .header-card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 24px; }}
        .multi-grid {{ text-align: center; margin: 24px 0; }}
        .multi-grid img {{ max-width: 100%; border-radius: 8px; border: 2px solid #38bdf8; shadow: 0 4px 12px rgba(0,0,0,0.5); }}
        .model-card {{ background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 20px; margin-bottom: 24px; }}
        .model-card h3 {{ margin: 0 0 8px 0; color: #f1f5f9; display: flex; align-items: center; justify-content: space-between; }}
        .badge {{ background: #0284c7; color: #fff; font-size: 11px; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; }}
        .meta {{ color: #94a3b8; font-size: 13px; margin: 0 0 16px 0; }}
        .image-trio {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 16px; }}
        .image-trio img {{ width: 100%; border-radius: 6px; border: 1px solid #475569; background: #020617; }}
        .img-lbl {{ text-align: center; font-size: 12px; color: #cbd5e1; margin-top: 6px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; }}
        .metric {{ text-align: center; }}
        .lbl {{ display: block; font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
        .val {{ font-size: 16px; font-weight: bold; color: #38bdf8; }}
    </style>
</head>
<body>
    <div class="header-card">
        <h1>Tactical Camouflage Detection Visual Comparison</h1>
        <h2>Image Stem: <strong>{image_stem}</strong> | Models Evaluated: <strong>{len(model_results)}</strong></h2>
        {gt_html}
    </div>

    <div class="multi-grid">
        <h3 style="color: #cbd5e1;">Multi-Model Side-by-Side Comparison Grid</h3>
        <p style="color: #94a3b8; font-size: 13px;">[ Original RGB | Ground Truth (if avail) | Models Output Masks ... ]</p>
        <img src="{image_stem}_all_models_comparison.png" alt="Multi-Model Comparison">
    </div>

    {cards_html}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  Generated HTML Visual Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="DGNet Multi-Model Visual Inference Test Suite")
    parser.add_argument("--image", type=str, default=None, help="Path to input RGB image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model weights")
    parser.add_argument("--model-tag", "--model-name", type=str, default=None, help="Custom model tag or name for output filenames")
    parser.add_argument("--all-models", action="store_true", help="Auto-discover and test all models in models/ directory")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone architecture")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    parser.add_argument("--invert", action="store_true", help="Invert predicted mask")
    parser.add_argument("--auto-invert", action="store_true", default=False, help="Auto-detect reversed polarity")
    parser.add_argument("--sharpen", action="store_true", default=True, help="Apply Min-Max probability scaling for sharp GT-like binary prediction masks")
    parser.add_argument("--no-sharpen", dest="sharpen", action="store_false", help="Disable Min-Max probability scaling")
    parser.add_argument("--adaptive-threshold", action="store_true", default=True, help="Automatically scale threshold to peak prediction dynamic range")
    parser.add_argument("--no-adaptive-threshold", dest="adaptive_threshold", action="store_false", help="Disable adaptive thresholding")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for masks")
    args = parser.parse_args()

    cfg = load_project_config()
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    img_size = tuple(data_cfg.get("image_size", [384, 384]))
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "tests" / "unit" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine images to test
    sample_images = [args.image] if args.image else find_default_sample_images()
    if not sample_images:
        print("[ERROR] No input images found.")
        sys.exit(1)

    # Discover models
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        ckpt_stem = ckpt_path.name.replace(".weights.h5", "").replace(".keras", "").replace(".h5", "").replace(".tflite", "").replace(".onnx", "")
        tag = args.model_tag or ckpt_stem
        models_to_test = [{
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "tag": tag,
            "parent_folder": ckpt_path.parent.name,
            "model_type": "keras" if not ckpt_path.name.endswith((".tflite", ".onnx")) else ("tflite" if ckpt_path.name.endswith(".tflite") else "onnx"),
            "backbone": args.backbone or "mobilenet_v3_large"
        }]
    else:
        discovered = discover_all_models()
        if args.all_models or len(discovered) > 1:
            models_to_test = discovered
        elif discovered:
            models_to_test = [discovered[0]]
        else:
            print("[ERROR] No models discovered in models/ directory.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("      DGNet Multi-Model Visual Camouflage Inference Test Suite")
    print("=" * 70)
    print(f"  Target Image(s):  {len(sample_images)} image(s)")
    print(f"  Models to Test:   {len(models_to_test)} model(s)")
    for m in models_to_test:
        print(f"    • [{m['model_type'].upper()}] {m['name']} ({m['parent_folder']})")
    print(f"  Image Size:       {img_size}")
    print(f"  Threshold:        {args.threshold} (Adaptive Sharpening: {args.sharpen})")
    print(f"  Output Dir:       {output_dir.resolve()}")
    print("-" * 70 + "\n")

    for img_path_str in sample_images:
        img_path = Path(img_path_str).resolve()
        image_stem = img_path.stem
        print(f"\n[Processing Image] {img_path.name}...")

        input_tensor, original_image = load_and_preprocess_image(str(img_path), image_size=img_size)
        gt_path = find_ground_truth_for_image(str(img_path))
        gt_mask = load_ground_truth(gt_path, image_size=img_size) if gt_path else None

        model_results = []

        for m_idx, m_info in enumerate(models_to_test, 1):
            print(f"  [{m_idx}/{len(models_to_test)}] Running model: {m_info['name']} ({m_info['parent_folder']})...", flush=True)

            try:
                runner = ModelRunner(m_info, img_size=img_size)
                start_time = time.perf_counter()
                pred_prob_np = runner.predict(input_tensor)
                inference_time = (time.perf_counter() - start_time) * 1000.0

                p_min, p_max = float(pred_prob_np.min()), float(pred_prob_np.max())

                # Sharpening via Min-Max Scaling to full dynamic range [0, 1]
                if args.sharpen and (p_max - p_min) > 1e-4:
                    scaled_prob = (pred_prob_np - p_min) / (p_max - p_min)
                else:
                    scaled_prob = pred_prob_np.copy()

                # Inversion handling
                initial_cov = np.mean(scaled_prob >= args.threshold)
                if args.invert or (args.auto_invert and initial_cov > 0.5):
                    scaled_prob = 1.0 - scaled_prob

                # Adaptive or fixed thresholding
                if args.adaptive_threshold and p_max < args.threshold and (p_max - p_min) > 1e-4:
                    effective_thresh = 0.5
                else:
                    effective_thresh = args.threshold

                pred_binary = (scaled_prob >= effective_thresh).astype(np.float32)

                tag = m_info["tag"]

                # 1. Save clean Ground Truth mask (pure binary 0/255)
                if gt_mask is not None:
                    gt_out = output_dir / f"{image_stem}_gt.png"
                    if not gt_out.exists():
                        save_mask_png(gt_mask, str(gt_out))

                # 2. Save clean binary Prediction mask named with model tag
                mask_path = output_dir / f"{image_stem}_{tag}_pred.png"
                save_mask_png(pred_binary, str(mask_path))
                # Compatibility symlink/alias
                save_mask_png(pred_binary, str(output_dir / f"{image_stem}_{tag}_pred_mask.png"))

                # 3. Save clean Black-and-White comparison [GROUND TRUTH | MODEL: {tag}]
                comp_path = output_dir / f"{image_stem}_{tag}_comparison.png"
                gt_for_vis = gt_mask.numpy() if gt_mask is not None else None
                save_comparison_image(original_image, pred_binary, str(comp_path),
                                      gt_mask=gt_for_vis, image_size=img_size, model_name=tag)

                # Compute metrics if GT exists
                metrics = {}
                if gt_mask is not None:
                    pred_prob_tensor = tf.convert_to_tensor(pred_prob_np[np.newaxis, ...], dtype=tf.float32)
                    metrics = compute_all_metrics(tf.expand_dims(gt_mask, 0), pred_prob_tensor, threshold=args.threshold)

                res_dict = {
                    "info": m_info,
                    "name": m_info["name"],
                    "tag": tag,
                    "parent_folder": m_info["parent_folder"],
                    "model_type": m_info["model_type"],
                    "backbone": m_info["backbone"],
                    "threshold": args.threshold,
                    "inference_time_ms": inference_time,
                    "pred_prob": pred_prob_np,
                    "binary_mask": pred_binary,
                    "metrics": metrics
                }
                model_results.append(res_dict)

                print(f"      -> Inference: {inference_time:.1f}ms | Coverage: {np.mean(pred_binary)*100:.2f}% | Range: [{pred_prob_np.min():.3f}, {pred_prob_np.max():.3f}]")

            except Exception as err:
                print(f"      [ERROR] Could not evaluate model {m_info['name']}: {err}")

        # Save consolidated multi-model comparison image
        if model_results:
            multi_comp_path = output_dir / f"{image_stem}_all_models_comparison.png"
            save_multi_model_comparison(original_image, gt_mask, model_results, str(multi_comp_path), image_size=img_size)
            print(f"\n  Saved Multi-Model Comparison Panel: {multi_comp_path}")

            # Also save default fallback filenames for backwards compatibility
            first_res = model_results[0]
            save_mask_png(first_res["binary_mask"], str(output_dir / f"{image_stem}_pred.png"))
            save_comparison_image(original_image, first_res["binary_mask"], str(output_dir / f"{image_stem}_comparison.png"),
                                  gt_mask=(gt_mask.numpy() if gt_mask is not None else None), image_size=img_size, model_name=first_res["tag"])

            # Generate HTML Visual Report
            generate_html_visual_report(image_stem, model_results, output_dir, gt_path=gt_path)

    print("\n" + "=" * 70)
    print("  Visual Camouflage Detection Test Complete!")
    print(f"  Inspect visual output masks and HTML reports in: {output_dir.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
