"""
Core Edge AI Inference Runtime for Camouflaged Object Detection.
Manages continuous camera acquisition, TFLite tensor execution, mask postprocessing,
tactical visualization, and dynamic user-controlled confidence thresholding.
"""

import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.metrics import (
    CAMERA_CONNECTED,
    CONFIDENCE_THRESHOLD,
    DETECTION_ACTIVE,
    DETECTION_CONFIDENCE,
    E2E_DURATION,
    FEED_MODE,
    FRAMES_PROCESSED,
    INFERENCE_DURATION,
    INFERENCE_ERRORS,
    MASK_PIXEL_COVERAGE,
    MODEL_READY,
    OBJECTS_DETECTED,
    fps_monitor,
    update_uptime,
)
from src.api.pipeline import edge_preprocess_frame, process_prediction_mask


class EdgeInferenceRuntime:
    """
    Primary edge inference engine.
    Runs continuous model inference loop in a background thread and maintains
    the latest frame and telemetry in bounded memory.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_source: Optional[Union[int, str]] = None,
        target_size: Tuple[int, int] = (384, 384),
        threshold: float = 0.50,
        min_contour_area: int = 300,
        target_fps: float = 15.0
    ):
        self.model_path = model_path or os.getenv(
            "MODEL_PATH",
            "models/quantized/proposed_mobilenet_v3_fp16.tflite"
        )
        self.camera_source = camera_source or os.getenv("CAMERA_SOURCE", "0")
        self.target_size = target_size
        self.threshold = float(os.getenv("CONFIDENCE_THRESHOLD", str(threshold)))
        self.min_contour_area = int(os.getenv("MIN_CONTOUR_AREA", str(min_contour_area)))
        self.target_fps = float(os.getenv("TARGET_FPS", str(target_fps)))
        self.num_threads = int(os.getenv("NUM_THREADS", str(min(4, os.cpu_count() or 4))))

        p = Path(self.model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        self.resolved_model_path = p.resolve()

        # Model state
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_type = "tflite"
        self.is_model_ready = False

        # Camera & Threading state
        self.cap = None
        self.is_camera_connected = False
        self.last_camera_error = ""
        source_str = str(self.camera_source).strip().lower()
        self._is_ingest_mode = source_str in {"ingest", "bridge", "none"}
        self._last_external_frame_time = 0.0
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        # Initialize Gauge
        CONFIDENCE_THRESHOLD.set(round(self.threshold, 2))

        # Bounded frame buffer
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self.simulation_active = os.getenv("AUTO_SIMULATE", "false").lower() in {"true", "1", "yes"}
        self._sample_img: Optional[np.ndarray] = None
        self._latest_telemetry: Dict = {
            "status": "initializing",
            "detected": False,
            "confidence_max": 0.0,
            "confidence_mean": 0.0,
            "object_pixels": 0,
            "inference_latency_ms": 0.0,
            "fps": 0.0,
            "camera_connected": False,
            "model_ready": False,
            "threshold": self.threshold,
            "min_contour_area": self.min_contour_area,
            "timestamp": time.time(),
        }

    def set_threshold(self, threshold: float):
        """Dynamically updates the confidence threshold from the user interface."""
        val = max(0.01, min(0.99, float(threshold)))
        self.threshold = val
        CONFIDENCE_THRESHOLD.set(round(val, 2))
        print(f"[Runtime Config] Confidence threshold dynamically updated to: {val:0.2f}")

    def set_min_contour_area(self, area: int):
        """Dynamically updates minimum contour area noise filter from the user interface."""
        val = max(0, int(area))
        self.min_contour_area = val
        print(f"[Runtime Config] Min contour area updated to: {val} px")

    def start(self):
        """Loads model, initializes camera negotiation, and starts continuous inference loop."""
        self._load_model()
        self._open_camera()
        self._worker_thread = threading.Thread(
            target=self._continuous_inference_loop,
            name="EdgeInferenceWorker",
            daemon=True
        )
        self._worker_thread.start()

    def stop(self):
        """Signals thread to stop and cleans up resources."""
        print("[Runtime] Stopping edge inference runtime...")
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self._close_camera()
        MODEL_READY.set(0)
        CAMERA_CONNECTED.set(0)
        print("[Runtime] Runtime stopped and resources released.")

    def _load_model(self):
        """Loads and validates the TFLite interpreter and allocates tensors."""
        if not self.resolved_model_path.exists():
            MODEL_READY.set(0)
            raise FileNotFoundError(f"Model file not found at: {self.resolved_model_path}")

        print(f"[Runtime] Loading TFLite model from {self.resolved_model_path.name}...")

        Interpreter = None
        for mod_name in ["ai_edge_litert.interpreter", "tflite_runtime.interpreter"]:
            try:
                mod = __import__(mod_name, fromlist=["Interpreter"])
                Interpreter = getattr(mod, "Interpreter")
                print(f"[Runtime] Initialized interpreter using: {mod_name}")
                break
            except (ImportError, AttributeError):
                continue

        if Interpreter is None:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                print("[Runtime] Initialized interpreter using: tensorflow.lite")
            except (ImportError, AttributeError):
                pass

        if Interpreter is None:
            MODEL_READY.set(0)
            raise RuntimeError("No compatible TFLite runtime found (ai_edge_litert, tflite_runtime, or tensorflow.lite).")

        try:
            interpreter = Interpreter(
                model_path=str(self.resolved_model_path),
                num_threads=self.num_threads
            )
        except TypeError:
            interpreter = Interpreter(model_path=str(self.resolved_model_path))
        print(f"[Runtime] Initialized TFLite interpreter with {self.num_threads} worker threads.")

        interpreter.allocate_tensors()
        self.interpreter = interpreter
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        in_shape = self.input_details[0]["shape"]
        print(f"[Runtime] Input details: shape={in_shape}, dtype={self.input_details[0]['dtype']}")
        print(f"[Runtime] Output details: shape={self.output_details[0]['shape']}, dtype={self.output_details[0]['dtype']}")

        # Warm-up pass
        dummy = np.zeros(in_shape, dtype=np.float32)
        self._run_forward_pass(dummy)

        self.is_model_ready = True
        MODEL_READY.set(1)
        print("[Runtime] Model validation and warm-up successful. Engine is READY.")

    def _run_forward_pass(self, input_tensor: np.ndarray) -> np.ndarray:
        """Executes thread-safe raw forward pass through TFLite interpreter."""
        with self._inference_lock:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()
            pred = self.interpreter.get_tensor(self.output_details[0]["index"])
        if pred.ndim == 4:
            return pred[0]
        return pred

    def _open_camera(self) -> bool:
        """Attempts to open the configured camera source with multi-backend negotiation."""
        source = str(self.camera_source).strip()

        if source.lower() in {"ingest", "bridge", "none"}:
            self._is_ingest_mode = True
            self.is_camera_connected = True
            CAMERA_CONNECTED.set(1)
            print("[Runtime] Configured in INGEST/BRIDGE mode. Awaiting external frame feed at /api/feed.")
            return True

        # Resolve device index (/dev/video0 -> 0, or "0" -> 0)
        idx = None
        if source.isdigit():
            idx = int(source)
        elif source.startswith("/dev/video") and source[10:].isdigit():
            idx = int(source[10:])

        system = platform.system().lower()
        cap = None

        if idx is not None:
            if "linux" in system:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            elif "windows" in system:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)
        else:
            # RTSP, HTTP, or video file stream
            cap = cv2.VideoCapture(source)

        if cap is not None and cap.isOpened():
            # Configure frame negotiation for low-latency edge streaming
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap
            self.is_camera_connected = True
            self.last_camera_error = ""
            CAMERA_CONNECTED.set(1)
            print(f"[Runtime] Camera hardware '{source}' successfully opened.")
            return True

        self.is_camera_connected = False
        self.last_camera_error = f"Hardware camera '{source}' not accessible inside this container. Ingest endpoint (/api/feed) is active and awaiting frames."
        self._is_ingest_mode = True  # Seamless fallback to ingest mode
        CAMERA_CONNECTED.set(0)
        print(f"[Runtime Notice] {self.last_camera_error}")
        return False

    def _close_camera(self):
        """Safely releases the camera handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_camera_connected = False
        CAMERA_CONNECTED.set(0)

    def _continuous_inference_loop(self):
        """Continuous background inference loop decoupled from HTTP requests."""
        reconnect_delay = 1.0

        while not self._stop_event.is_set():
            # Check if external feed is active
            has_live_feed = (time.time() - self._last_external_frame_time) < 2.5

            # If ingest mode is active:
            if self._is_ingest_mode:
                if has_live_feed:
                    FEED_MODE.set(1)  # Live external feed active
                    time.sleep(0.02)
                    continue
                else:
                    # Autonomous simulation active so Prometheus & Grafana always have live telemetry!
                    FEED_MODE.set(2)
                    CAMERA_CONNECTED.set(1)
                    if self._sample_img is None:
                        for p in [PROJECT_ROOT / "test_data.jpeg", Path("/app/test_data.jpeg"), Path("test_data.jpeg")]:
                            if p.exists():
                                self._sample_img = cv2.imread(str(p))
                                break
                        if self._sample_img is None:
                            self._sample_img = np.zeros((384, 384, 3), dtype=np.uint8)
                            cv2.circle(self._sample_img, (192, 192), 60, (120, 150, 130), -1)

                    loop_start = time.perf_counter()
                    self._process_single_frame(self._sample_img)
                    elapsed = time.perf_counter() - loop_start
                    delay = (1.0 / min(self.target_fps, 10.0)) - elapsed
                    if delay > 0:
                        time.sleep(delay)
                    continue

            # Internal simulation loop for zero-dependency container test streaming
            if self.simulation_active:
                FEED_MODE.set(2)
                if self._sample_img is None:
                    for p in [PROJECT_ROOT / "test_data.jpeg", Path("/app/test_data.jpeg"), Path("test_data.jpeg")]:
                        if p.exists():
                            self._sample_img = cv2.imread(str(p))
                            break
                    if self._sample_img is None:
                        self._sample_img = np.zeros((384, 384, 3), dtype=np.uint8)
                        cv2.circle(self._sample_img, (192, 192), 60, (120, 150, 130), -1)
                if self._sample_img is not None:
                    loop_start = time.perf_counter()
                    self._process_single_frame(self._sample_img)
                    elapsed = time.perf_counter() - loop_start
                    delay = (1.0 / self.target_fps) - elapsed
                    if delay > 0:
                        time.sleep(delay)
                    continue

            if not self.is_camera_connected and not self._is_ingest_mode:
                success = self._open_camera()
                if not success:
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 5.0)
                    continue
                else:
                    reconnect_delay = 1.0

            loop_start = time.perf_counter()
            ret, frame = False, None
            if self.cap is not None and self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                except Exception as exc:
                    self.last_camera_error = f"Camera read error: {exc}"
                    ret = False

            if not ret or frame is None:
                self.is_camera_connected = False
                self.last_camera_error = "Failed to grab frame from camera source."
                CAMERA_CONNECTED.set(0)
                self._close_camera()
                time.sleep(1.0)
                continue

            FEED_MODE.set(1)
            self._process_single_frame(frame)

            elapsed = time.perf_counter() - loop_start
            target_delay = (1.0 / self.target_fps) - elapsed
            if target_delay > 0:
                time.sleep(target_delay)

    def toggle_simulation(self) -> bool:
        """Toggles container-internal sample frame loop without external bridges."""
        self.simulation_active = not self.simulation_active
        if self.simulation_active:
            self.is_camera_connected = True
            CAMERA_CONNECTED.set(1)
        return self.simulation_active

    def feed_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        External ingestion entry point.
        Allows host webcam bridges to push frames directly into the runtime.
        """
        self._last_external_frame_time = time.time()
        self.simulation_active = False  # Real frame received, pause simulation
        self.is_camera_connected = True
        FEED_MODE.set(1)
        CAMERA_CONNECTED.set(1)
        return self._process_single_frame(frame_bgr)

    def _process_single_frame(self, frame_bgr: np.ndarray) -> Dict:
        """Preprocesses frame, executes forward pass, postprocesses, and updates buffers."""
        t_e2e_start = time.perf_counter()
        try:
            orig_h, orig_w = frame_bgr.shape[:2]
            target_h, target_w = self.target_size

            # 1. Preprocess
            input_tensor = edge_preprocess_frame(frame_bgr, (target_h, target_w))

            # 2. Pure Inference forward pass
            t0 = time.perf_counter()
            raw_pred = self._run_forward_pass(input_tensor)
            inf_duration = time.perf_counter() - t0

            INFERENCE_DURATION.observe(inf_duration)
            inf_ms = round(inf_duration * 1000.0, 2)

            # 3. Postprocess using current dynamic threshold
            prob_map, binary_mask = process_prediction_mask(
                raw_pred,
                threshold=self.threshold,
                min_contour_area=self.min_contour_area
            )

            prob_2d = prob_map[:, :, 0] if prob_map.ndim == 3 else prob_map
            mask_2d = binary_mask[:, :, 0] if binary_mask.ndim == 3 else binary_mask

            max_conf = float(np.max(prob_2d))
            mean_conf = float(np.mean(prob_2d))
            obj_pixels = int(np.count_nonzero(mask_2d > 0))
            detected = bool(obj_pixels > 0)

            # Update Telemetry & Detection Metrics
            DETECTION_ACTIVE.set(1.0 if detected else 0.0)
            DETECTION_CONFIDENCE.set(round(max_conf, 4))
            MASK_PIXEL_COVERAGE.set(obj_pixels)
            CONFIDENCE_THRESHOLD.set(round(self.threshold, 2))
            if detected:
                OBJECTS_DETECTED.inc()

            # 4. Render visualization overlay with active threshold
            annotated = self._render_tactical_hud(
                frame_bgr, mask_2d, detected, max_conf, inf_ms, obj_pixels
            )

            # 5. Pre-encode JPEG for fast streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, jpeg_buf = cv2.imencode(".jpg", annotated, encode_param)
            jpeg_bytes = jpeg_buf.tobytes() if success else None

            # 6. Update thread-safe bounded buffers
            fps = fps_monitor.record_frame()
            update_uptime()
            e2e_sec = time.perf_counter() - t_e2e_start
            E2E_DURATION.observe(e2e_sec)
            FRAMES_PROCESSED.labels(model_type=self.model_type, status="success").inc()

            telemetry = {
                "status": "ready",
                "detected": detected,
                "confidence_max": round(max_conf, 4),
                "confidence_mean": round(mean_conf, 4),
                "object_pixels": obj_pixels,
                "inference_latency_ms": inf_ms,
                "fps": round(fps, 1),
                "camera_connected": self.is_camera_connected,
                "model_ready": self.is_model_ready,
                "threshold": round(self.threshold, 2),
                "min_contour_area": self.min_contour_area,
                "input_dimensions": [orig_h, orig_w],
                "model_dimensions": [target_h, target_w],
                "timestamp": time.time(),
            }

            with self._frame_lock:
                self._latest_raw_frame = frame_bgr
                self._latest_annotated_frame = annotated
                self._latest_jpeg = jpeg_bytes
                self._latest_telemetry = telemetry

            return telemetry

        except Exception as exc:
            INFERENCE_ERRORS.labels(error_type=type(exc).__name__).inc()
            return {"status": "error", "error": str(exc)}

    def _render_tactical_hud(
        self,
        frame: np.ndarray,
        mask_2d: np.ndarray,
        detected: bool,
        max_conf: float,
        inf_ms: float,
        obj_pixels: int
    ) -> np.ndarray:
        """Applies camouflage mask overlay and tactical military HUD telemetry."""
        out = frame.copy()
        h, w = out.shape[:2]

        if detected and mask_2d.any():
            if mask_2d.shape[:2] != (h, w):
                mask_full = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_full = mask_2d

            mask_bool = mask_full > 0.5
            colored = np.zeros_like(out)
            colored[:, :] = (0, 69, 255)  # Orange-red (BGR)
            out[mask_bool] = cv2.addWeighted(out[mask_bool], 0.55, colored[mask_bool], 0.45, 0)

            mask_u8 = (mask_full * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

        bar_h = 42
        sub_img = out[0:bar_h, 0:w]
        dark = np.zeros_like(sub_img)
        out[0:bar_h, 0:w] = cv2.addWeighted(sub_img, 0.4, dark, 0.6, 0)

        status_color = (0, 0, 255) if detected else (0, 255, 128)
        status_str = "[TARGET ACQUIRED]" if detected else "[SCANNING / CLEAR]"
        cv2.putText(out, status_str, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2, cv2.LINE_AA)

        fps_val = self._latest_telemetry.get("fps", 0.0)
        telemetry_str = f"INF: {inf_ms:0.1f}ms | FPS: {fps_val:0.1f} | THRESH: {self.threshold*100:0.0f}% | PIX: {obj_pixels}"
        cv2.putText(out, telemetry_str, (215, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)

        blen = 16
        cv2.line(out, (8, 50), (8 + blen, 50), status_color, 2)
        cv2.line(out, (8, 50), (8, 50 + blen), status_color, 2)
        cv2.line(out, (w - 8, 50), (w - 8 - blen, 50), status_color, 2)
        cv2.line(out, (w - 8, 50), (w - 8, 50 + blen), status_color, 2)

        return out

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the latest annotated JPEG frame bytes (thread-safe)."""
        with self._frame_lock:
            return self._latest_jpeg

    def get_telemetry(self) -> Dict:
        """Returns the latest telemetry dictionary (thread-safe)."""
        with self._frame_lock:
            return dict(self._latest_telemetry)
