#!/usr/bin/env python3
"""
Edge Inference Script for DGNet Camouflage Object Detection (COD)
==================================================================
Deploy this script on your edge device (NVIDIA Jetson, Raspberry Pi, etc.)
to run the TFLite model on a live camera feed or on image files.

ALL known false-positive bugs are fixed here:
  1. BGR -> RGB conversion   (OpenCV reads BGR; model trained on RGB)
  2. ImageNet normalization  (matches src/data/dataset.py training pipeline exactly)
  3. NO min-max stretching   (old code inflated weak noise to 1.0, causing false alarms)
  4. Gate 1: Frame-level confidence gate  (low-confidence frames return zero mask)
  5. Gate 2: Absolute threshold on raw sigmoid probabilities
  6. Gate 3: Contour-area filter  (rejects small noise blobs from ordinary edges)

Usage
-----
  # Run on a live camera:
  python tests/edge/edge_inference.py --tflite models/quantized/dgnet_fp32.tflite

  # Run on image files:
  python tests/edge/edge_inference.py --tflite models/quantized/dgnet_fp32.tflite --images data/raw/dataset-splitM/Testing/Images

  # Tune threshold (0.50-0.65) and min blob size if needed:
  python tests/edge/edge_inference.py --tflite models/quantized/dgnet_fp32.tflite --threshold 0.60 --min-area 500
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Dependency check
# ──────────────────────────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python not installed. Run:  pip install opencv-python-headless")
    sys.exit(1)

try:
    import tensorflow as tf
    _interpreter_class = tf.lite.Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as _interpreter_class  # type: ignore
    except ImportError:
        print("[ERROR] Neither tensorflow nor tflite-runtime is installed.")
        print("  On Jetson:  pip install tflite-runtime")
        print("  Or:         pip install tensorflow")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — MUST match src/data/dataset.py (training pipeline)
# ──────────────────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = (384, 384)   # (height, width) — match your trained model


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — THE most common source of false positives
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame: np.ndarray, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Convert a raw camera BGR frame into a model-ready float32 batch tensor.

    Steps (must match dataset.py training pipeline exactly):
      1. BGR -> RGB  (cv2.VideoCapture gives BGR; model trained on RGB)
      2. Resize to model input resolution
      3. Normalize to [0, 1]
      4. Subtract ImageNet mean and divide by ImageNet std
      5. Add batch dimension -> (1, H, W, 3)
    """
    # Step 1: BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 2: Resize
    resized = cv2.resize(rgb, (image_size[1], image_size[0]))  # cv2 takes (W, H)

    # Step 3: [0, 255] → [0.0, 1.0]
    f32 = resized.astype(np.float32) / 255.0

    # Step 4: ImageNet normalization
    normalized = (f32 - _IMAGENET_MEAN) / _IMAGENET_STD

    # Step 5: Batch dim
    return np.expand_dims(normalized, axis=0)


def preprocess_image_file(image_path: str, image_size: Tuple[int, int] = IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image file and return (input_batch, original_bgr_for_display).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return preprocess_frame(img, image_size), img


# ──────────────────────────────────────────────────────────────────────────────
# Postprocessing — THE bug that caused random objects to be flagged
# ──────────────────────────────────────────────────────────────────────────────
def postprocess_prediction(
    raw_pred: np.ndarray,
    threshold: float = 0.50,
    min_contour_area: int = 300,
) -> Tuple[np.ndarray, bool]:
    """
    Convert raw model sigmoid output into a clean binary detection mask.

    Args:
        raw_pred:         Output from the TFLite interpreter. Shape can be
                          (1, H, W, 1), (H, W, 1), or (H, W).
                          Values are already in [0, 1] (sigmoid applied).
        threshold:        Minimum sigmoid confidence to label a pixel as
                          'camouflaged'. Start at 0.50; raise to 0.60-0.65
                          if you still see false positives.
        min_contour_area: Minimum blob area in pixels to keep. Smaller blobs
                          are noise from ordinary object edges and are removed.
                          Default 300 px at 384×384 res. Raise to 500-1000 to
                          be more strict.

    Returns:
        (binary_mask_uint8, detected):
            binary_mask_uint8 — shape (H, W) uint8, 0 or 255, ready for cv2.
            detected          — True if any camouflaged region was found.

    WHY NO MIN-MAX SCALING:
        The old code did:
            scaled = (pred - pred.min()) / (pred.max() - pred.min())
        This is the root cause of all false positives. When pointing the camera
        at a normal chair/wall with no camouflage, the model correctly outputs
        max~0.02. Min-max scaling stretches 0.02 → 1.0, which then exceeds the
        threshold → false alarm every single frame.

        We use raw sigmoid values directly. If the model says 0.02, it means 2%
        confidence. It will never exceed a threshold of 0.50.
    """
    # --- Squeeze to (H, W) ---
    prob = raw_pred.squeeze()

    # --- Gate 1: Frame-level confidence gate ----------------------------------
    # If the model's peak confidence anywhere in the frame is below the
    # threshold, there is NO camouflaged object. Return immediately.
    if float(prob.max()) < threshold:
        h, w = prob.shape
        return np.zeros((h, w), dtype=np.uint8), False

    # --- Gate 2: Pixel-level absolute threshold -------------------------------
    # Use raw sigmoid probabilities — no stretching.
    binary = (prob >= threshold).astype(np.uint8)  # 0 or 1

    # --- Gate 3: Contour-area noise filter ------------------------------------
    # Normal objects (chairs, walls, people in non-camo clothing) produce sharp
    # edges which can occasionally exceed the threshold. Reject small blobs.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            cv2.drawContours(cleaned, [cnt], -1, 1, thickness=-1)

    detected = bool(cleaned.any())
    return (cleaned * 255).astype(np.uint8), detected


# ──────────────────────────────────────────────────────────────────────────────
# TFLite Interpreter wrapper
# ──────────────────────────────────────────────────────────────────────────────
class DGNetEdgeRunner:
    """Loads a DGNet TFLite model and runs inference."""

    def __init__(self, tflite_path: str, num_threads: int = 4):
        print(f"[DGNet] Loading TFLite model: {tflite_path}")
        if not Path(tflite_path).exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

        self.interpreter = _interpreter_class(
            model_path=tflite_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        in_shape = self.input_details[0]['shape']   # e.g. [1, 384, 384, 3]
        out_shape = self.output_details[0]['shape']
        print(f"[DGNet] Input shape:  {list(in_shape)}")
        print(f"[DGNet] Output shape: {list(out_shape)}")
        print(f"[DGNet] Model ready.\n")

    def run(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a preprocessed batch.

        Args:
            input_batch: float32 numpy array, shape (1, H, W, 3),
                         ImageNet-normalized (use preprocess_frame()).

        Returns:
            Raw sigmoid probability map, shape (1, H, W, 1) or (H, W, 1).
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_batch)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Some TFLite exports return the output under a named key dict
        if isinstance(output, dict):
            key = "camouflage_mask" if "camouflage_mask" in output else list(output.keys())[0]
            output = output[key]

        return output


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────
def overlay_mask_on_frame(frame_bgr: np.ndarray, mask_uint8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a red detection overlay onto the BGR frame where the mask is active.
    Also draws a bounding box and label around each detected region.
    """
    result = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

    # Red overlay on detected pixels
    overlay = result.copy()
    overlay[mask_resized > 0] = [0, 0, 200]   # BGR red
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    # Green contour lines around detected blobs
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Bounding box + label for each blob
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = "CAMOUFLAGED TARGET"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x, y - lh - 8), (x + lw, y), (0, 200, 0), -1)
        cv2.putText(result, label, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return result


def draw_status_bar(frame: np.ndarray, detected: bool, fps: float, threshold: float) -> np.ndarray:
    """Draws a HUD status bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 36
    bar = frame.copy()
    bar_color = (0, 60, 0) if not detected else (0, 0, 140)
    cv2.rectangle(bar, (0, 0), (w, bar_h), bar_color, -1)

    status_text = "STATUS: CLEAR" if not detected else "!! CAMOUFLAGE DETECTED !!"
    status_color = (0, 255, 80) if not detected else (0, 100, 255)
    cv2.putText(bar, status_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f}  |  Thresh: {threshold:.2f}"
    cv2.putText(bar, fps_text, (w - 260, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


# ──────────────────────────────────────────────────────────────────────────────
# Live camera mode
# ──────────────────────────────────────────────────────────────────────────────
def run_live_camera(
    runner: DGNetEdgeRunner,
    camera_id: int = 0,
    threshold: float = 0.50,
    min_contour_area: int = 300,
    display: bool = True,
    save_detections: bool = False,
    output_dir: Optional[str] = None,
):
    print(f"[Camera] Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    save_dir = Path(output_dir or "tests/edge/detections")
    if save_detections:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Camera] Saving detections to: {save_dir.resolve()}")

    print(f"[Camera] Running. Press 'q' to quit, 's' to save a snapshot.\n")

    fps = 0.0
    frame_idx = 0
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Failed to read frame.")
            break

        # ── Inference ──────────────────────────────────────────────────────
        t0 = time.perf_counter()

        input_batch = preprocess_frame(frame, IMAGE_SIZE)
        raw_pred    = runner.run(input_batch)
        mask, detected = postprocess_prediction(raw_pred, threshold, min_contour_area)

        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0
        fps = 0.9 * fps + 0.1 * (1.0 / max(t1 - t_prev, 1e-6))
        t_prev = t1

        # ── Visualization ──────────────────────────────────────────────────
        if display or save_detections:
            vis = overlay_mask_on_frame(frame, mask)
            status_bar = draw_status_bar(vis, detected, fps, threshold)
            # Combine status bar + frame
            display_frame = np.vstack([status_bar, vis])

        if detected:
            print(f"[Frame {frame_idx:05d}] CAMOUFLAGED TARGET DETECTED  "
                  f"| infer={infer_ms:.1f}ms  fps={fps:.1f}")

            if save_detections:
                fname = save_dir / f"detection_{frame_idx:05d}.jpg"
                cv2.imwrite(str(fname), display_frame if (display or save_detections) else frame)

        if display:
            cv2.imshow("DGNet Edge — Camouflage Detection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snap = save_dir / f"snapshot_{frame_idx:05d}.jpg"
                save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap), display_frame)
                print(f"[Camera] Snapshot saved: {snap}")

        frame_idx += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()
    print(f"\n[Camera] Done. Processed {frame_idx} frames.")


# ──────────────────────────────────────────────────────────────────────────────
# Image-file mode
# ──────────────────────────────────────────────────────────────────────────────
def run_on_images(
    runner: DGNetEdgeRunner,
    images_path: str,
    threshold: float = 0.50,
    min_contour_area: int = 300,
    output_dir: Optional[str] = None,
    display: bool = False,
):
    path = Path(images_path)
    image_files = []
    if path.is_file():
        image_files = [path]
    elif path.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(sorted(path.glob(ext)))
    else:
        print(f"[ERROR] Not a valid file or directory: {images_path}")
        sys.exit(1)

    if not image_files:
        print(f"[ERROR] No images found at: {images_path}")
        sys.exit(1)

    out_dir = Path(output_dir or "tests/edge/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Images] Processing {len(image_files)} image(s). Output -> {out_dir.resolve()}\n")

    n_detected = 0
    total_ms   = 0.0

    for img_file in image_files:
        try:
            input_batch, original_bgr = preprocess_image_file(str(img_file), IMAGE_SIZE)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        t0 = time.perf_counter()
        raw_pred = runner.run(input_batch)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += infer_ms

        mask, detected = postprocess_prediction(raw_pred, threshold, min_contour_area)

        status = "DETECTED" if detected else "clear"
        print(f"  {img_file.name:<40s}  {status:<10s}  {infer_ms:.1f} ms")

        if detected:
            n_detected += 1

        # Save overlay
        vis = overlay_mask_on_frame(original_bgr, mask)
        out_path = out_dir / f"{img_file.stem}_result.jpg"
        cv2.imwrite(str(out_path), vis)

        # Save raw binary mask
        mask_path = out_dir / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        if display:
            cv2.imshow(f"DGNet — {img_file.name}", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    if display:
        cv2.destroyAllWindows()

    avg_ms = total_ms / max(len(image_files), 1)
    print(f"\n[Images] Done. Detections: {n_detected}/{len(image_files)} images.")
    print(f"[Images] Average inference time: {avg_ms:.1f} ms  ({1000/max(avg_ms,1):.1f} FPS)\n")
    print(f"[Images] Results saved to: {out_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    # Resolve project root so the script can be run from any working directory
    project_root = Path(__file__).resolve().parent.parent.parent
    default_tflite = str(project_root / "models" / "quantized" / "dgnet_benchmark_best_fp32.tflite")

    parser = argparse.ArgumentParser(
        description="DGNet Edge Inference — Camouflage Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model
    parser.add_argument(
        "--tflite",
        type=str,
        default=default_tflite,
        help="Path to .tflite model (default: models/quantized/dgnet_benchmark_best_fp32.tflite)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads for TFLite runtime (default: 4)"
    )

    # Input
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="ID",
        help="Camera device index (e.g. 0 for /dev/video0). Default if no --images given."
    )
    mode.add_argument(
        "--images",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a single image file or a directory of images."
    )

    # Detection parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Sigmoid confidence threshold (0.0–1.0). Raise to 0.60-0.65 to reduce false positives. (default: 0.50)"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=300,
        dest="min_contour_area",
        help="Minimum detection blob area in pixels. Raise to reduce small false-positive blobs. (default: 300)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output images/detections."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow window (useful on headless devices)."
    )
    parser.add_argument(
        "--save-detections",
        action="store_true",
        help="Save annotated frames when a detection occurs (camera mode)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    runner = DGNetEdgeRunner(args.tflite, num_threads=args.threads)

    display = not args.no_display

    if args.images is not None:
        run_on_images(
            runner,
            images_path=args.images,
            threshold=args.threshold,
            min_contour_area=args.min_contour_area,
            output_dir=args.output_dir,
            display=display,
        )
    else:
        cam_id = args.camera if args.camera is not None else 0
        run_live_camera(
            runner,
            camera_id=cam_id,
            threshold=args.threshold,
            min_contour_area=args.min_contour_area,
            display=display,
            save_detections=args.save_detections,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
