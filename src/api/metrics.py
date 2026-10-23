"""
Prometheus Metrics Instrumentation for Edge AI Inference Service.
Provides robust, continuous metrics collection for Prometheus & Grafana.
Includes graceful fallback if prometheus_client is not installed on bare-metal host.
"""

import time
from collections import deque
from threading import Lock
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _DummyMetric:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
        def observe(self, amount): pass
        def set(self, val): pass
        def labels(self, *args, **kwargs): return self

    Counter = Histogram = Gauge = _DummyMetric
    def generate_latest(registry=None):
        return b"# prometheus_client not installed on baremetal host\n"

# 1. Total frames processed counter
FRAMES_PROCESSED = Counter(
    "edge_frames_processed_total",
    "Total number of camera frames successfully processed by the edge model",
    ["model_type", "status"]
)

# 2. Total HTTP inference requests counter
INFERENCE_REQUESTS = Counter(
    "edge_inference_requests_total",
    "Total HTTP requests received by the service",
    ["endpoint", "http_status"]
)

# 3. Total inference errors counter
INFERENCE_ERRORS = Counter(
    "edge_inference_errors_total",
    "Total inference failures and processing errors encountered",
    ["error_type"]
)

# 4. Pure model forward-pass execution latency histogram (seconds)
INFERENCE_DURATION = Histogram(
    "edge_inference_duration_seconds",
    "Time spent exclusively inside the neural network forward pass (seconds)",
    buckets=(0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.35, 0.5, 1.0, 2.0)
)

# 5. End-to-end edge pipeline duration histogram (seconds)
E2E_DURATION = Histogram(
    "edge_e2e_pipeline_duration_seconds",
    "Total end-to-end edge pipeline duration including preprocess, forward pass, and postprocess (seconds)",
    buckets=(0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.35, 0.5, 1.0, 2.0)
)

# 6. Real-time model processing throughput gauge (FPS)
PROCESSING_FPS = Gauge(
    "edge_processing_fps",
    "Instantaneous model processing throughput in frames per second (computed over rolling window)"
)

# 7. Camera connectivity state gauge (1=connected, 0=disconnected)
CAMERA_CONNECTED = Gauge(
    "edge_camera_connected",
    "Camera acquisition status (1 = connected and acquiring, 0 = disconnected or error)"
)

# 8. Model readiness state gauge (1=ready, 0=not ready)
MODEL_READY = Gauge(
    "edge_model_ready",
    "Edge model readiness state (1 = loaded and ready, 0 = not ready)"
)

# 9. Process uptime in seconds
PROCESS_UPTIME = Gauge(
    "edge_process_uptime_seconds",
    "Inference application process uptime in seconds"
)

# 10. Real-time detection active alert gauge (1 = target acquired, 0 = no target)
DETECTION_ACTIVE = Gauge(
    "edge_detection_active",
    "Binary target acquisition alert flag (1 = camouflaged target detected, 0 = background clear)"
)

# 11. Maximum camouflage prediction confidence in latest frame (0.0 - 1.0)
DETECTION_CONFIDENCE = Gauge(
    "edge_detection_confidence",
    "Maximum prediction confidence score of detected camouflaged object in latest frame (0.0 to 1.0)"
)

# 12. Cumulative total of target detection events
OBJECTS_DETECTED = Counter(
    "edge_objects_detected_total",
    "Total number of frames in which a camouflaged object exceeded the confidence threshold"
)

# 13. Pixel area covered by detected camouflage mask
MASK_PIXEL_COVERAGE = Gauge(
    "edge_mask_pixel_coverage",
    "Total number of non-zero segmented pixels belonging to detected camouflaged targets"
)

# 14. Active user-configured confidence threshold
CONFIDENCE_THRESHOLD = Gauge(
    "edge_confidence_threshold",
    "Active dynamic confidence threshold setting applied to model predictions"
)

# 15. Feed mode (1 = hardware/ingest camera, 2 = autonomous edge simulation)
FEED_MODE = Gauge(
    "edge_feed_mode",
    "Operational feed mode (1 = live webcam/bridge feed, 2 = autonomous edge simulation)"
)


class FPSMonitor:
    """Thread-safe sliding-window FPS estimator for model processing throughput."""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.lock = Lock()
        self.last_record_time = time.time()

    def record_frame(self) -> float:
        now = time.time()
        with self.lock:
            self.last_record_time = now
            self.timestamps.append(now)
            if len(self.timestamps) >= 2:
                elapsed = self.timestamps[-1] - self.timestamps[0]
                if elapsed > 0:
                    fps = (len(self.timestamps) - 1) / elapsed
                    PROCESSING_FPS.set(round(fps, 2))
                    return fps
            PROCESSING_FPS.set(0.0)
            return 0.0

    def check_idle(self, timeout_sec: float = 3.0):
        """Resets FPS gauge to 0 if no frames arrived within timeout."""
        with self.lock:
            if time.time() - self.last_record_time > timeout_sec:
                PROCESSING_FPS.set(0.0)


fps_monitor = FPSMonitor()
START_TIME = time.time()


def update_uptime():
    """Updates the process uptime gauge."""
    PROCESS_UPTIME.set(round(time.time() - START_TIME, 1))
