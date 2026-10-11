"""
Multi-Model Inference and Camouflage Target Prediction Engine.
Supports running inference across Keras weights (.h5, .keras), TensorFlow Lite (.tflite),
and ONNX models (.onnx) with automatic preprocessing and postprocessing.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.utils.config import get_project_root, load_project_config
from src.utils.metrics import compute_all_metrics
from src.models.dgnet import build_dgnet_model
from src.visualization.visualizer import (
    save_mask_png,
    save_comparison_image,
    save_multi_model_comparison,
    generate_html_visual_report,
)

PROJECT_ROOT = get_project_root()


def load_and_preprocess_image(image_path: Union[str, Path], image_size: Tuple[int, int] = (384, 384)) -> Tuple[tf.Tensor, np.ndarray]:
    """
    Loads an RGB image, resizes it, and normalizes using ImageNet mean/std.

    Returns:
        (input_tensor, original_image)
        - input_tensor: [1, H, W, 3] float32 tensor
        - original_image: original-resolution uint8 numpy array for visualization
    """
    raw_bytes = tf.io.read_file(str(image_path))
    original = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    original_np = original.numpy()

    resized = tf.image.resize(original, image_size)
    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    normalized = (tf.cast(resized, tf.float32) / 255.0 - imagenet_mean) / imagenet_std
    input_tensor = tf.expand_dims(normalized, axis=0)

    return input_tensor, original_np


def load_ground_truth(gt_path: Union[str, Path], image_size: Tuple[int, int] = (384, 384)) -> Optional[tf.Tensor]:
    """Loads ground truth mask if available. Returns None if not found."""
    gt_p = Path(gt_path)
    if not gt_p.exists():
        return None
    raw_bytes = tf.io.read_file(str(gt_p))
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes raw prediction probability array into a scaled probability map and binary mask.

    Returns:
        (scaled_prob, binary_mask)
    """
    is_3d = (pred_prob.ndim == 3)
    pred_prob_2d = pred_prob[:, :, 0] if is_3d else pred_prob

    p_min, p_max = float(pred_prob_2d.min()), float(pred_prob_2d.max())

    if sharpen and (p_max - p_min) > 1e-4:
        scaled_prob_2d = (pred_prob_2d - p_min) / (p_max - p_min)
    else:
        scaled_prob_2d = pred_prob_2d.copy()

    initial_cov = np.mean(scaled_prob_2d >= threshold)
    if auto_invert and initial_cov > 0.5:
        scaled_prob_2d = 1.0 - scaled_prob_2d

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


def find_default_sample_images() -> List[str]:
    """Finds candidate test images from the dataset."""
    candidates = [
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


def find_ground_truth_for_image(image_path: Union[str, Path]) -> Optional[str]:
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


def discover_all_models(models_dir: Optional[Path] = None) -> List[Dict]:
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

    def __init__(self, model_info: Dict, img_size: Tuple[int, int] = (384, 384)):
        self.info = model_info
        self.img_size = img_size
        self.model_type = model_info["model_type"]
        self.path = model_info["path"]
        self.backbone = model_info.get("backbone", "mobilenet_v3_large")
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


def run_inference_pipeline(
    image: Optional[str] = None,
    checkpoint: Optional[str] = None,
    model_tag: Optional[str] = None,
    all_models: bool = False,
    backbone: Optional[str] = None,
    threshold: float = 0.5,
    invert: bool = False,
    auto_invert: bool = False,
    sharpen: bool = True,
    adaptive_threshold: bool = True,
    output_dir: Optional[str] = None,
):
    """Executes multi-model inference pipeline and generates reports and visual masks."""
    cfg = load_project_config()
    data_cfg = cfg.get("data", cfg.get("dataset", {}))
    img_size = tuple(data_cfg.get("image_size", [384, 384]))
    out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "tests" / "unit" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = [image] if image else find_default_sample_images()
    if not sample_images:
        print("[ERROR] No input images found.")
        return

    if checkpoint:
        ckpt_path = Path(checkpoint).resolve()
        ckpt_stem = ckpt_path.name.replace(".weights.h5", "").replace(".keras", "").replace(".h5", "").replace(".tflite", "").replace(".onnx", "")
        tag = model_tag or ckpt_stem
        models_to_test = [{
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "tag": tag,
            "parent_folder": ckpt_path.parent.name,
            "model_type": "keras" if not ckpt_path.name.endswith((".tflite", ".onnx")) else ("tflite" if ckpt_path.name.endswith(".tflite") else "onnx"),
            "backbone": backbone or "mobilenet_v3_large"
        }]
    else:
        discovered = discover_all_models()
        if all_models or len(discovered) > 1:
            models_to_test = discovered
        elif discovered:
            models_to_test = [discovered[0]]
        else:
            print("[ERROR] No models discovered in models/ directory.")
            return

    print("\n" + "=" * 70)
    print("      DGNet Multi-Model Visual Camouflage Inference Engine")
    print("=" * 70)
    print(f"  Target Image(s):  {len(sample_images)} image(s)")
    print(f"  Models to Test:   {len(models_to_test)} model(s)")
    for m in models_to_test:
        print(f"    • [{m['model_type'].upper()}] {m['name']} ({m['parent_folder']})")
    print(f"  Image Size:       {img_size}")
    print(f"  Threshold:        {threshold} (Adaptive Sharpening: {sharpen})")
    print(f"  Output Dir:       {out_dir.resolve()}")
    print("-" * 70 + "\n")

    for img_path_str in sample_images:
        img_path = Path(img_path_str).resolve()
        image_stem = img_path.stem
        print(f"\n[Processing Image] {img_path.name}...")

        input_tensor, original_image = load_and_preprocess_image(str(img_path), image_size=img_size)
        gt_path = find_ground_truth_for_image(str(img_path))
        gt_mask = load_ground_truth(gt_path, image_size=img_size) if gt_path else None

        if gt_mask is not None:
            save_mask_png(gt_mask.numpy(), str(out_dir / f"{image_stem}_gt.png"))

        model_results = []

        for m_idx, m_info in enumerate(models_to_test, 1):
            print(f"  [{m_idx}/{len(models_to_test)}] Running model: {m_info['name']} ({m_info['parent_folder']})...", flush=True)

            try:
                runner = ModelRunner(m_info, img_size=img_size)
                start_time = time.perf_counter()
                pred_prob_np = runner.predict(input_tensor)
                inference_time = (time.perf_counter() - start_time) * 1000.0

                scaled_prob, binary_mask = process_prediction_mask(
                    pred_prob_np,
                    threshold=threshold,
                    sharpen=sharpen,
                    adaptive_threshold=adaptive_threshold,
                    auto_invert=auto_invert or invert
                )

                tag = m_info["tag"]
                save_mask_png(binary_mask, str(out_dir / f"{image_stem}_{tag}_pred.png"))

                save_comparison_image(
                    original=original_image,
                    pred_mask=binary_mask,
                    output_path=str(out_dir / f"{image_stem}_{tag}_comparison.png"),
                    gt_mask=gt_mask,
                    image_size=img_size,
                    model_name=tag
                )

                metrics = {}
                if gt_mask is not None:
                    metrics = compute_all_metrics(gt_mask, tf.convert_to_tensor(binary_mask, dtype=tf.float32), threshold=0.5)

                model_results.append({
                    **m_info,
                    "binary_mask": binary_mask,
                    "inference_time_ms": inference_time,
                    "metrics": metrics
                })
            except Exception as e:
                print(f"       Failed inference on {m_info['name']}: {e}")

        if model_results:
            save_multi_model_comparison(
                original=original_image,
                gt_mask=gt_mask,
                model_results=model_results,
                output_path=str(out_dir / f"{image_stem}_all_models_comparison.png"),
                image_size=img_size
            )
            generate_html_visual_report(image_stem, model_results, out_dir, gt_path=gt_path)

    print(f"\n[Completed] Visual output masks saved to: {out_dir.resolve()}\n")
