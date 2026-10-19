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

# 5. Real-time model processing throughput gauge (FPS)
PROCESSING_FPS = Gauge(
    "edge_processing_fps",
    "Instantaneous model processing throughput in frames per second (computed over rolling window)"
)

# 6. Camera connectivity state gauge (1=connected, 0=disconnected)
CAMERA_CONNECTED = Gauge(
    "edge_camera_connected",
    "Camera acquisition status (1 = connected and acquiring, 0 = disconnected or error)"
)

# 7. Model readiness state gauge (1=ready, 0=not ready)
MODEL_READY = Gauge(
    "edge_model_ready",
    "Edge model readiness state (1 = loaded and ready, 0 = not ready)"
)

# 8. Process uptime in seconds
PROCESS_UPTIME = Gauge(
    "edge_process_uptime_seconds",
    "Inference application process uptime in seconds"
)


class FPSMonitor:
    """Thread-safe sliding-window FPS estimator for model processing throughput."""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.lock = Lock()

    def record_frame(self) -> float:
        now = time.time()
        with self.lock:
            self.timestamps.append(now)
            if len(self.timestamps) >= 2:
                elapsed = self.timestamps[-1] - self.timestamps[0]
                if elapsed > 0:
                    fps = (len(self.timestamps) - 1) / elapsed
                    PROCESSING_FPS.set(round(fps, 2))
                    return fps
            PROCESSING_FPS.set(0.0)
            return 0.0


fps_monitor = FPSMonitor()
START_TIME = time.time()


def update_uptime():
    """Updates the process uptime gauge."""
    PROCESS_UPTIME.set(round(time.time() - START_TIME, 1))
