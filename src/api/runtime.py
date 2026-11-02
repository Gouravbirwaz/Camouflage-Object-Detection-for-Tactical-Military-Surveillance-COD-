"""
Core Edge AI Inference Runtime for Camouflaged Object Detection.
Manages continuous camera acquisition, TFLite tensor execution, mask postprocessing,
tactical visualization, and dynamic user-controlled confidence thresholding.
"""

import base64
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


class InternalCameraGrabber:
    """
    Continuously grabs frames in a background thread to prevent OpenCV driver
    buffer latency and camera I/O blocking.
    """
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.running = True
        self.has_frame = False
        self.latest_frame: Optional[np.ndarray] = None
        self.consecutive_errors = 0
        self.thread = threading.Thread(target=self._worker, name="InternalCameraGrabber", daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest_frame = frame
                        self.has_frame = True
                    self.consecutive_errors = 0
                else:
                    self.consecutive_errors += 1
                    time.sleep(0.01)
            except Exception:
                self.consecutive_errors += 1
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.has_frame or self.latest_frame is None:
                return False, None
            return True, self.latest_frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


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
        target_fps: float = 25.0
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
        self.num_threads = int(os.getenv("NUM_THREADS", "2"))

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
        self.grabber: Optional[InternalCameraGrabber] = None
        self.is_camera_connected = False
        self.last_camera_error = ""
        source_str = str(self.camera_source).strip().lower()
        self._is_ingest_mode = source_str in {"ingest", "bridge", "none"}
        self._last_external_frame_time = 0.0
        self._stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._infer_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._incoming_lock = threading.Lock()
        self._infer_data_lock = threading.Lock()
        self._latest_incoming_frame: Optional[np.ndarray] = None
        self._latest_infer_frame: Optional[np.ndarray] = None
        self._is_inferring: bool = False

        # Detection cache for asynchronous HUD overlay
        self._latest_mask_2d: Optional[np.ndarray] = None
        self._latest_detected: bool = False
        self._latest_max_conf: float = 0.0
        self._latest_mean_conf: float = 0.0
        self._latest_obj_pixels: int = 0
        self._latest_inf_ms: float = 0.0
        self._jpeg_id: int = 0

        # Initialize Gauge
        CONFIDENCE_THRESHOLD.set(round(self.threshold, 2))

        # Bounded frame buffer
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self.simulation_active = os.getenv("AUTO_SIMULATE", "false").lower() in {"true", "1", "yes"}
        self._sample_img: Optional[np.ndarray] = None
        self._sample_images: List[Tuple[str, np.ndarray]] = []
        self._sample_idx: int = 0
        self._last_sample_switch: float = 0.0
        self._sample_switch_interval: float = 4.0
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
        """Loads model, initializes camera negotiation, and starts streaming & inference threads."""
        self._load_model()
        self._open_camera()
        self._stop_event.clear()
        self._infer_event.clear()
        self._stream_thread = threading.Thread(
            target=self._video_stream_loop,
            name="VideoStreamWorker",
            daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self._continuous_inference_loop,
            name="InferenceWorker",
            daemon=True
        )
        self._stream_thread.start()
        self._inference_thread.start()

    def stop(self):
        """Signals threads to stop and cleans up resources."""
        print("[Runtime] Stopping edge inference runtime...")
        self._stop_event.set()
        self._infer_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
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

        try:
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
        except Exception as exc:
            print(f"[Runtime Warning] Camera hardware negotiation failed: {exc}")
            cap = None

        if cap is not None and cap.isOpened():
            # Configure frame negotiation for low-latency edge streaming
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap
            self.grabber = InternalCameraGrabber(cap)
            self.is_camera_connected = True
            self.last_camera_error = ""
            CAMERA_CONNECTED.set(1)
            print(f"[Runtime] Camera hardware '{source}' successfully opened with threaded grabber.")
            return True

        self.is_camera_connected = False
        self.last_camera_error = (
            f"Hardware camera '{source}' not accessible inside this environment. "
            "Autonomous zero-dependency fallback active; ingest endpoint (/api/feed) is ready."
        )
        CAMERA_CONNECTED.set(0)
        print(f"[Runtime Notice] {self.last_camera_error}")
        return False

    def _close_camera(self):
        """Safely releases the camera handle and background grabber thread."""
        if self.grabber is not None:
            try:
                self.grabber.stop()
            except Exception:
                pass
            self.grabber = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_camera_connected = False
        CAMERA_CONNECTED.set(0)

    def _trigger_inference(self, frame: np.ndarray):
        """Hands off a frame to the asynchronous inference worker if idle."""
        if not self._is_inferring and self.is_model_ready:
            with self._incoming_lock:
                self._latest_infer_frame = frame
            self._infer_event.set()

    def _video_stream_loop(self):
        """High-speed video rendering & streaming loop targeting 20-30 FPS."""
        frame_interval = 1.0 / max(5.0, min(60.0, self.target_fps))
        last_reconnect_attempt = time.time()

        while not self._stop_event.is_set():
            t_loop_start = time.perf_counter()
            now = time.time()
            has_live_feed = (now - self._last_external_frame_time) < 2.5
            current_frame: Optional[np.ndarray] = None

            # Priority 1: Ingested frame from external camera (WebSocket / HTTP)
            if has_live_feed:
                FEED_MODE.set(1)
                CAMERA_CONNECTED.set(1)
                with self._incoming_lock:
                    if self._latest_incoming_frame is not None:
                        current_frame = self._latest_incoming_frame
                if current_frame is None:
                    with self._frame_lock:
                        current_frame = self._latest_raw_frame

            # Priority 2: Direct hardware camera / RTSP stream if not explicitly 'ingest' mode
            elif not self._is_ingest_mode:
                if not self.is_camera_connected or self.grabber is None or self.grabber.consecutive_errors > 15:
                    if (now - last_reconnect_attempt) > 3.0:
                        last_reconnect_attempt = now
                        self._close_camera()
                        self._open_camera()

                if self.is_camera_connected and self.grabber is not None:
                    ret, grab_frame = self.grabber.read()
                    if ret and grab_frame is not None:
                        FEED_MODE.set(1)
                        CAMERA_CONNECTED.set(1)
                        current_frame = grab_frame

            # Priority 3: Autonomous simulation fallback (cycles through multiple edge cases)
            if current_frame is None:
                FEED_MODE.set(2)
                CAMERA_CONNECTED.set(1 if (self.is_camera_connected or self.simulation_active) else 0)
                self._load_sample_images()

                if (now - self._last_sample_switch) > self._sample_switch_interval and len(self._sample_images) > 1:
                    self._sample_idx = (self._sample_idx + 1) % len(self._sample_images)
                    self._sample_img = self._sample_images[self._sample_idx][1]
                    self._last_sample_switch = now

                current_frame = self._sample_img if self._sample_img is not None else (self._sample_images[0][1] if self._sample_images else None)
                if current_frame is None:
                    current_frame = np.zeros((384, 384, 3), dtype=np.uint8)
                    cv2.putText(current_frame, "Awaiting Camera / Feed...", (20, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Hand off frame to inference engine if ready
            self._trigger_inference(current_frame)

            # Retrieve active detection state
            with self._infer_data_lock:
                mask_2d = self._latest_mask_2d
                detected = self._latest_detected
                max_conf = self._latest_max_conf
                mean_conf = self._latest_mean_conf
                obj_pixels = self._latest_obj_pixels
                inf_ms = self._latest_inf_ms

            # Render tactical HUD overlay on current frame
            annotated = self._render_tactical_hud(
                current_frame, mask_2d, detected, max_conf, inf_ms, obj_pixels
            )

            # Fast JPEG encode for streaming
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]
            success, jpeg_buf = cv2.imencode(".jpg", annotated, encode_param)
            jpeg_bytes = jpeg_buf.tobytes() if success else None

            # Record FPS
            fps = fps_monitor.record_frame()
            update_uptime()

            orig_h, orig_w = current_frame.shape[:2]
            target_h, target_w = self.target_size
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
                self._latest_raw_frame = current_frame
                self._latest_annotated_frame = annotated
                self._latest_jpeg = jpeg_bytes
                self._jpeg_id += 1
                self._latest_telemetry = telemetry

            elapsed = time.perf_counter() - t_loop_start
            delay = frame_interval - elapsed
            if delay > 0:
                time.sleep(delay)

    def _continuous_inference_loop(self):
        """Asynchronous worker that continuously executes model inference on newest frames."""
        while not self._stop_event.is_set():
            self._infer_event.wait(timeout=0.04)
            self._infer_event.clear()

            if self._stop_event.is_set():
                break

            frame_to_process = None
            with self._incoming_lock:
                if self._latest_infer_frame is not None:
                    frame_to_process = self._latest_infer_frame
                    self._latest_infer_frame = None

            if frame_to_process is None:
                continue

            self._is_inferring = True
            try:
                t0 = time.perf_counter()
                target_h, target_w = self.target_size

                # 1. Preprocess
                input_tensor = edge_preprocess_frame(frame_to_process, (target_h, target_w))

                # 2. Forward pass
                t_inf = time.perf_counter()
                raw_pred = self._run_forward_pass(input_tensor)
                inf_duration = time.perf_counter() - t_inf
                inf_ms = round(inf_duration * 1000.0, 2)
                INFERENCE_DURATION.observe(inf_duration)

                # 3. Postprocess
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

                # Update Prometheus Gauges
                DETECTION_ACTIVE.set(1.0 if detected else 0.0)
                DETECTION_CONFIDENCE.set(round(max_conf, 4))
                MASK_PIXEL_COVERAGE.set(obj_pixels)
                CONFIDENCE_THRESHOLD.set(round(self.threshold, 2))
                if detected:
                    OBJECTS_DETECTED.inc()

                e2e_sec = time.perf_counter() - t0
                E2E_DURATION.observe(e2e_sec)
                FRAMES_PROCESSED.labels(model_type=self.model_type, status="success").inc()

                with self._infer_data_lock:
                    self._latest_mask_2d = mask_2d
                    self._latest_detected = detected
                    self._latest_max_conf = max_conf
                    self._latest_mean_conf = mean_conf
                    self._latest_obj_pixels = obj_pixels
                    self._latest_inf_ms = inf_ms

            except Exception as exc:
                INFERENCE_ERRORS.labels(error_type=type(exc).__name__).inc()
            finally:
                self._is_inferring = False

    def _load_sample_images(self):
        """Loads and caches multiple real edge cases and tactical test frames."""
        if self._sample_images:
            return

        candidates = [
            ("Standard Benchmark", PROJECT_ROOT / "test_data.jpeg"),
            ("Mesh & Foliage Occlusion", PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_mesh_foliage_3.jpg"),
            ("Heavy Fog & Lighting", PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_fog_lighting_1.jpg"),
            ("Rain & Dusk Low Visibility", PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_rain_dusk_4.jpg"),
            ("Thermal Sensor Noise", PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_thermal_noise_2.jpg"),
        ]

        loaded = []
        for name, p in candidates:
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    loaded.append((name, img))

        if not loaded:
            dummy = np.zeros((384, 384, 3), dtype=np.uint8)
            cv2.circle(dummy, (192, 192), 60, (120, 150, 130), -1)
            loaded.append(("Procedural Target", dummy))

        self._sample_images = loaded
        if self._sample_img is None and loaded:
            self._sample_img = loaded[0][1]

    def next_edge_case(self) -> str:
        """Manually advances simulation to the next edge-case frame."""
        self._load_sample_images()
        if not self._sample_images:
            return "No edge cases"
        self._sample_idx = (self._sample_idx + 1) % len(self._sample_images)
        self._sample_img = self._sample_images[self._sample_idx][1]
        self._last_sample_switch = time.time()
        return self._sample_images[self._sample_idx][0]

    def set_edge_case_by_name(self, name: str) -> bool:
        """Switches simulation or live feed to a specific named edge case."""
        self._load_sample_images()
        for idx, (cname, img) in enumerate(self._sample_images):
            if cname.lower() == name.lower() or name.lower() in cname.lower():
                self._sample_idx = idx
                self._sample_img = img
                self._last_sample_switch = time.time()
                return True
        return False

    def add_custom_sample(self, name: str, frame_bgr: np.ndarray):
        """Adds a user-uploaded image into the edge cases sample bank and switches to it."""
        self._load_sample_images()
        self._sample_images.append((name, frame_bgr))
        self._sample_idx = len(self._sample_images) - 1
        self._sample_img = frame_bgr
        self._last_sample_switch = time.time()

    def toggle_simulation(self) -> bool:
        """Toggles container-internal sample frame loop without external bridges."""
        self.simulation_active = not self.simulation_active
        if self.simulation_active:
            self.is_camera_connected = True
            CAMERA_CONNECTED.set(1)
        return self.simulation_active

    def feed_frame_bytes(self, jpeg_bytes: bytes) -> bool:
        """
        Fast external ingestion entry point for raw image bytes (WebSocket or HTTP).
        Avoids blocking the FastAPI event loop.
        """
        self._last_external_frame_time = time.time()
        self.simulation_active = False
        self.is_camera_connected = True
        FEED_MODE.set(1)
        CAMERA_CONNECTED.set(1)
        try:
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                with self._incoming_lock:
                    self._latest_incoming_frame = frame_bgr
                self._trigger_inference(frame_bgr)
                return True
        except Exception:
            pass
        return False

    def feed_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        External ingestion entry point.
        Asynchronously buffers frame so the sender never blocks on inference.
        """
        self._last_external_frame_time = time.time()
        self.simulation_active = False  # Real frame received, pause simulation
        self.is_camera_connected = True
        FEED_MODE.set(1)
        CAMERA_CONNECTED.set(1)
        with self._incoming_lock:
            self._latest_incoming_frame = frame_bgr
        self._trigger_inference(frame_bgr)
        return self.get_telemetry()

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict:
        """Synchronous frame prediction for /predict endpoint."""
        return self._process_single_frame(frame_bgr)

    def analyze_image(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
        min_contour_area: Optional[int] = None
    ) -> Dict:
        """
        Runs comprehensive edge inference analysis on an image file or test frame.
        Generates tactical HUD overlay, binary segmentation mask, and telemetry.
        """
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]
        target_h, target_w = self.target_size
        eff_thresh = float(threshold) if threshold is not None else self.threshold
        eff_min_area = int(min_contour_area) if min_contour_area is not None else self.min_contour_area

        # 1. Preprocess
        input_tensor = edge_preprocess_frame(frame_bgr, (target_h, target_w))

        # 2. Forward pass
        t_inf_start = time.perf_counter()
        raw_pred = self._run_forward_pass(input_tensor)
        inf_duration = time.perf_counter() - t_inf_start
        inf_ms = round(inf_duration * 1000.0, 2)

        # 3. Postprocess with chosen threshold and min area
        prob_map, binary_mask = process_prediction_mask(
            raw_pred,
            threshold=eff_thresh,
            min_contour_area=eff_min_area
        )

        prob_2d = prob_map[:, :, 0] if prob_map.ndim == 3 else prob_map
        mask_2d = binary_mask[:, :, 0] if binary_mask.ndim == 3 else binary_mask

        max_conf = float(np.max(prob_2d))
        mean_conf = float(np.mean(prob_2d))
        obj_pixels = int(np.count_nonzero(mask_2d > 0))
        detected = bool(obj_pixels > 0)

        # 4. Render Tactical HUD
        annotated = self._render_tactical_hud(
            frame_bgr, mask_2d, detected, max_conf, inf_ms, obj_pixels
        )

        # 5. Render Visual Segmentation Mask
        if mask_2d.shape[:2] != (orig_h, orig_w):
            mask_resized = cv2.resize((mask_2d * 255).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = (mask_2d * 255).astype(np.uint8)

        mask_vis = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_vis, contours, -1, (0, 255, 128), 2)

        # 6. Encode Base64 strings for direct client-side rendering
        _, orig_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        _, anno_buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        _, mask_buf = cv2.imencode(".png", mask_vis)

        orig_b64 = base64.b64encode(orig_buf).decode("utf-8")
        anno_b64 = base64.b64encode(anno_buf).decode("utf-8")
        mask_b64 = base64.b64encode(mask_buf).decode("utf-8")

        return {
            "status": "success",
            "detected": detected,
            "confidence_max": round(max_conf, 4),
            "confidence_mean": round(mean_conf, 4),
            "object_pixels": obj_pixels,
            "inference_latency_ms": inf_ms,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "threshold": round(eff_thresh, 2),
            "min_contour_area": eff_min_area,
            "input_dimensions": [orig_h, orig_w],
            "original_base64": orig_b64,
            "annotated_base64": anno_b64,
            "mask_base64": mask_b64,
        }

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

    def get_latest_jpeg_with_id(self) -> Tuple[Optional[bytes], int]:
        """Returns the latest annotated JPEG frame bytes and sequential ID."""
        with self._frame_lock:
            return self._latest_jpeg, self._jpeg_id

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the latest annotated JPEG frame bytes (thread-safe)."""
        with self._frame_lock:
            return self._latest_jpeg

    def get_telemetry(self) -> Dict:
        """Returns the latest telemetry dictionary (thread-safe)."""
        with self._frame_lock:
            return dict(self._latest_telemetry)
