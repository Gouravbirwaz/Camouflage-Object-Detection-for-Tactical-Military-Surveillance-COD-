# syntax=docker/dockerfile:1
# Minimal, Portable Edge AI Inference Container for DGNet Camouflaged Object Detection
# Multi-arch support: linux/amd64 (PC simulation) and linux/arm64 (NVIDIA Jetson / ARM devices)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/proposed_mobilenet_v3_fp16.tflite \
    CAMERA_SOURCE=0 \
    INPUT_HEIGHT=384 \
    INPUT_WIDTH=384 \
    CONFIDENCE_THRESHOLD=0.50 \
    MIN_CONTOUR_AREA=300 \
    TARGET_FPS=15.0 \
    NUM_THREADS=4 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal OS runtime libraries required for OpenCV and Linux V4L2 camera capture
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libv4l-0 \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lean runtime dependencies only (FastAPI, uvicorn, prometheus-client, TFLite/TF)
COPY requirements/edge_requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Create non-root system user with video group permissions for /dev/video* hardware access
RUN groupadd -g 10001 edgegroup && \
    useradd -u 10001 -g edgegroup -G video -s /bin/bash -m edgeuser

# Copy ONLY the minimal inference runtime, single TFLite model, and lightweight test image
COPY src/api/ /app/src/api/
COPY models/quantized/proposed_mobilenet_v3_fp16.tflite /app/models/proposed_mobilenet_v3_fp16.tflite
COPY test_data.jpeg /app/test_data.jpeg

RUN chown -R edgeuser:edgegroup /app

USER edgeuser

EXPOSE 8000

# Container liveness check
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single primary application entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
