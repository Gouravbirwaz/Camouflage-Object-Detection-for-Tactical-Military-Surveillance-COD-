"""
Visual Reporting and Mask Rendering Utilities for COD.
Provides tools to generate black-and-white comparison masks, color heatmaps,
contour overlays, multi-model side-by-side grids, and standalone HTML reports.
"""

import html
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def save_mask_png(mask_array: np.ndarray, output_path: Union[str, Path]):
    """Saves a [H, W] or [H, W, 1] float mask as a grayscale PNG (0-255)."""
    out_p = str(output_path)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_uint8 = (np.clip(mask_array, 0.0, 1.0) * 255).astype(np.uint8)
    encoded = tf.image.encode_png(tf.expand_dims(mask_uint8, -1))
    tf.io.write_file(out_p, encoded)


def save_color_heatmap_png(prob_array: np.ndarray, output_path: Union[str, Path], sharpen: bool = True):
    """
    Renders probability prediction array into a vibrant color heatmap (JET colormap)
    to visually highlight soft camouflage confidence regions.
    """
    out_p = str(output_path)
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
        r = prob_uint8
        g = (np.sin(prob_clipped * np.pi) * 255).astype(np.uint8)
        b = (255 - prob_uint8)
        heatmap_rgb = np.stack([r, g, b], axis=-1)

    encoded = tf.image.encode_png(heatmap_rgb)
    tf.io.write_file(out_p, encoded)


def save_contour_overlay_image(
    original: np.ndarray,
    pred_mask: np.ndarray,
    output_path: Union[str, Path],
    image_size: Tuple[int, int] = (384, 384)
):
    """Overlays vibrant camouflage detection boundary contours on the RGB image."""
    out_p = str(output_path)
    orig_resized = tf.image.resize(original, image_size).numpy().astype(np.uint8)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]

    mask_bool = pred_mask > 0.5
    overlay = orig_resized.copy()

    if HAS_CV2:
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fill = overlay.copy()
        fill[mask_bool] = (fill[mask_bool] * 0.6 + np.array([0, 225, 120]) * 0.4).astype(np.uint8)
        cv2.drawContours(fill, contours, -1, (255, 30, 30), 2)
        overlay = fill
    else:
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 128, 0, 255)

    encoded = tf.image.encode_png(overlay)
    tf.io.write_file(out_p, encoded)


def save_comparison_image(
    original: np.ndarray,
    pred_mask: np.ndarray,
    output_path: Union[str, Path],
    gt_mask: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (384, 384),
    model_name: str = "PREDICTION",
    **kwargs
):
    """
    Creates and saves a clean, uncolorized black-and-white comparison:
    [ GROUND TRUTH | MODEL: {model_name} ]
    Pure grayscale binary (0/255).
    """
    out_p = str(output_path)
    h, w = image_size

    if hasattr(pred_mask, "numpy"):
        pred_mask = pred_mask.numpy()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[:, :, 0]
    pred_u8 = (pred_mask > 0.5 if pred_mask.max() <= 1.0 else pred_mask >= 128).astype(np.uint8) * 255
    if pred_u8.shape[:2] != (h, w) and HAS_CV2:
        pred_u8 = cv2.resize(pred_u8, (w, h), interpolation=cv2.INTER_NEAREST)

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

    sep_width = 4
    separator = np.full((h, sep_width), 128, dtype=np.uint8)
    body = np.hstack([gt_u8, separator, pred_u8])

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
    tf.io.write_file(out_p, encoded)


def save_multi_model_comparison(
    original: np.ndarray,
    gt_mask: Optional[np.ndarray],
    model_results: list,
    output_path: Union[str, Path],
    image_size: Tuple[int, int] = (384, 384)
):
    """
    Creates a consolidated side-by-side comparison grid placing output masks of ALL models together:
    [ GROUND TRUTH | MODEL 1 | MODEL 2 | ... ]
    Pure black-and-white (grayscale binary).
    """
    out_p = str(output_path)
    h, w = image_size
    panels = []
    headers = []

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
    tf.io.write_file(out_p, encoded)


def generate_html_visual_report(
    image_stem: str,
    model_results: list,
    output_dir: Union[str, Path],
    gt_path: Optional[str] = None
) -> Path:
    """Generates an HTML report visualizing all camouflage detection masks and metrics side-by-side."""
    out_dir = Path(output_dir)
    report_path = out_dir / f"{image_stem}_visual_report.html"

    cards_html = ""
    for res in model_results:
        tag = res["tag"]
        name = res["name"]
        folder = res.get("parent_folder", "checkpoints")
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
            <p class="meta">Backbone: <strong>{html.escape(res.get('backbone', 'unknown'))}</strong> | Type: <strong>{html.escape(res.get('model_type', 'keras'))}</strong> | Inference: <strong>{res.get('inference_time_ms', 0):.1f} ms</strong></p>
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
        .multi-grid img {{ max-width: 100%; border-radius: 8px; border: 2px solid #38bdf8; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
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
        <p style="color: #94a3b8; font-size: 13px;">[ Ground Truth | Model Predictions ... ]</p>
        <img src="{image_stem}_all_models_comparison.png" alt="Multi-Model Comparison">
    </div>

    <h2>Individual Model Output Inspections</h2>
    {cards_html}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path
