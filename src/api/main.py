"""
FastAPI Command & Control (C2) and Operational Web Interface for Edge AI Inference.
Exposes freeze-free live frame rendering, dynamic UI-controlled threshold adjustment,
Prometheus metrics, health probes, and external camera frame ingestion endpoints.
"""

import base64
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.metrics import (
    CONTENT_TYPE_LATEST,
    INFERENCE_ERRORS,
    INFERENCE_REQUESTS,
    generate_latest,
    update_uptime,
)
from src.api.runtime import EdgeInferenceRuntime

runtime: Optional[EdgeInferenceRuntime] = None


class ConfigUpdate(BaseModel):
    threshold: Optional[float] = None
    min_contour_area: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes EdgeInferenceRuntime on startup and terminates on shutdown."""
    global runtime
    print("[Startup] Initializing Edge AI Inference Service...")
    runtime = EdgeInferenceRuntime()
    try:
        runtime.start()
    except Exception as exc:
        print(f"[Startup ERROR] Failed to start runtime: {exc}")
        INFERENCE_ERRORS.labels(error_type="runtime_init_failure").inc()

    yield

    print("[Shutdown] Releasing edge runtime resources...")
    if runtime:
        runtime.stop()


app = FastAPI(
    title="DGNet Edge AI Tactical Surveillance",
    description="Minimal, Portable Edge AI Inference Runtime for Camouflaged Object Detection",
    version="2.0.0",
    lifespan=lifespan
)


# Minimal Tactical Web Interface (Fast, Freeze-Free Canvas/Image update + Live Controls)
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DGNet Edge AI Tactical C2</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #131926;
      --border: #222d42;
      --text: #c5d1e0;
      --text-bright: #ffffff;
      --alert-red: #ff3333;
      --alert-green: #00e676;
      --cyan: #00b0ff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background: var(--bg); color: var(--text); padding: 1.25rem; font-size: 14px; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 0.8rem; margin-bottom: 1.2rem; }
    h1 { font-size: 1.2rem; color: var(--text-bright); display: flex; align-items: center; gap: 0.5rem; }
    .badge { padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: bold; background: #0070f3; color: white; }
    .grid { display: grid; grid-template-columns: 2.2fr 1fr; gap: 1.2rem; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; }
    .video-container { position: relative; width: 100%; border-radius: 4px; overflow: hidden; background: #000; min-height: 380px; display: flex; align-items: center; justify-content: center; }
    .video-container img { width: 100%; height: auto; display: block; object-fit: contain; }
    .status-pill { display: inline-block; padding: 0.3rem 0.7rem; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .status-green { background: rgba(0, 230, 118, 0.15); color: var(--alert-green); border: 1px solid var(--alert-green); }
    .status-red { background: rgba(255, 51, 51, 0.15); color: var(--alert-red); border: 1px solid var(--alert-red); }
    .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin: 0.8rem 0; }
    .metric-box { background: #090d16; border: 1px solid var(--border); border-radius: 4px; padding: 0.6rem; }
    .metric-label { font-size: 0.7rem; color: #7f8c9b; text-transform: uppercase; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.15rem; font-weight: bold; color: var(--text-bright); }
    
    /* Dynamic Controls */
    .control-group { margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--border); }
    .control-label { font-size: 0.8rem; font-weight: bold; color: var(--text-bright); display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
    input[type=range] { width: 100%; height: 6px; background: #222d42; border-radius: 3px; outline: none; cursor: pointer; accent-color: var(--cyan); }
    .links-bar { margin-top: 1.2rem; padding-top: 0.8rem; border-top: 1px solid var(--border); display: flex; gap: 1rem; font-size: 0.8rem; }
    .links-bar a { color: var(--cyan); text-decoration: none; }
    .links-bar a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <h1>DGNet Tactical Edge C2 <span class="badge">Simulated Jetson Nano</span></h1>
    <div id="target-status" class="status-pill status-green">SCANNING / CLEAR</div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="video-container">
        <!-- Freeze-free dynamic frame rendered via non-blocking asynchronous pipeline -->
        <img id="live-stream" src="/api/frame" alt="Live Tactical Feed">
      </div>
    </div>

    <div class="card">
      <h2 style="font-size: 0.95rem; color: var(--text-bright);">Active Surveillance Telemetry</h2>
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">Inference Latency</div>
          <div class="metric-value" id="val-inf">-- ms</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Throughput</div>
          <div class="metric-value" id="val-fps">-- FPS</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Max Confidence</div>
          <div class="metric-value" id="val-conf">-- %</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Target Area</div>
          <div class="metric-value" id="val-pixels">0 px</div>
        </div>
      </div>

      <!-- User-Controlled Dynamic Threshold (Not Hardcoded!) -->
      <div class="control-group">
        <div class="control-label">
          <span>Confidence Threshold:</span>
          <span id="thresh-val" style="color: var(--cyan);">50%</span>
        </div>
        <input type="range" id="thresh-slider" min="5" max="95" value="50" step="1" oninput="onThresholdChange(this.value)">
        <p style="font-size: 0.72rem; color: #7f8c9b; margin-top: 0.3rem;">
          Pixels with confidence below this threshold are ignored to eliminate false positives.
        </p>
      </div>

      <div class="control-group">
        <div class="control-label">
          <span>Noise Filter (Min Area):</span>
          <span id="area-val" style="color: var(--cyan);">300 px</span>
        </div>
        <input type="range" id="area-slider" min="50" max="1500" value="300" step="50" oninput="onAreaChange(this.value)">
      </div>

      <!-- Action Buttons: Zero Host Installs Needed! -->
      <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem;">
        <button id="btn-webcam" onclick="toggleBrowserCamera()" style="width: 100%; padding: 0.65rem; background: #238636; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 0.85rem; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.5rem; transition: background 0.2s;">
          📹 Stream Browser Webcam (0 Installs)
        </button>
        <button id="btn-sim" onclick="toggleInternalSim()" style="width: 100%; padding: 0.65rem; background: #1f6feb; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 0.85rem; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.5rem; transition: background 0.2s;">
          ⚡ Run In-Container Test Stream
        </button>
      </div>

      <div style="margin-top: 0.8rem; font-size: 0.75rem; color: #7f8c9b;">
        <span>Camera Source: </span>
        <strong id="val-cam" style="color: #00e676;">Connecting...</strong>
        <p style="font-size: 0.70rem; margin-top: 0.4rem; color: #8b949e;">
          <strong>No separate scripts needed:</strong> Click <em>Stream Browser Webcam</em> to stream your webcam directly, or click <em>Run In-Container Test Stream</em> to simulate continuous feed.
        </p>
      </div>
    </div>
  </div>

  <div class="links-bar">
    <span>Endpoints:</span>
    <a href="/metrics" target="_blank">/metrics (Prometheus)</a>
    <a href="/health" target="_blank">/health (Liveness)</a>
    <a href="/ready" target="_blank">/ready (Readiness)</a>
    <a href="/api/status" target="_blank">/api/status (JSON)</a>
  </div>

  <script>
    // 1. Freeze-Free Live Video Rendering
    const streamImg = document.getElementById('live-stream');
    let frameLoading = false;

    function renderNextFrame() {
      if (frameLoading) return;
      frameLoading = true;
      const nextImg = new Image();
      nextImg.onload = () => {
        streamImg.src = nextImg.src;
        frameLoading = false;
        requestAnimationFrame(renderNextFrame);
      };
      nextImg.onerror = () => {
        frameLoading = false;
        setTimeout(renderNextFrame, 150);
      };
      nextImg.src = '/api/frame?t=' + performance.now();
    }
    renderNextFrame();

    // 2. Real-time User-Controlled Threshold (Dynamic, No Hardcoding)
    let updateTimeout = null;
    function onThresholdChange(val) {
      document.getElementById('thresh-val').innerText = val + '%';
      clearTimeout(updateTimeout);
      updateTimeout = setTimeout(() => {
        fetch('/api/config?threshold=' + (val / 100.0), { method: 'POST' });
      }, 50);
    }

    function onAreaChange(val) {
      document.getElementById('area-val').innerText = val + ' px';
      clearTimeout(updateTimeout);
      updateTimeout = setTimeout(() => {
        fetch('/api/config?min_contour_area=' + val, { method: 'POST' });
      }, 50);
    }

    // 3. Telemetry Updates
    async function updateTelemetry() {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const d = await res.json();
          document.getElementById('val-inf').innerText = (d.inference_latency_ms || 0) + ' ms';
          document.getElementById('val-fps').innerText = (d.fps || 0) + ' FPS';
          document.getElementById('val-conf').innerText = ((d.confidence_max || 0) * 100).toFixed(1) + ' %';
          document.getElementById('val-pixels').innerText = (d.object_pixels || 0) + ' px';

          const statusElem = document.getElementById('target-status');
          if (d.detected) {
            statusElem.innerText = '[TARGET ACQUIRED]';
            statusElem.className = 'status-pill status-red';
          } else {
            statusElem.innerText = '[SCANNING / CLEAR]';
            statusElem.className = 'status-pill status-green';
          }

          document.getElementById('val-cam').innerText = d.camera_connected ? 'Active / Ingesting' : 'Waiting for Feed';
          document.getElementById('val-cam').style.color = d.camera_connected ? '#00e676' : '#ff3333';
        }
      } catch (e) {}
    }
    setInterval(updateTelemetry, 400);

    // 4. In-Browser Webcam Streaming (Direct to K8s Pod without Terminal Script)
    let localStream = null;
    let streamActive = false;
    let isSending = false;
    const hiddenVideo = document.createElement('video');
    hiddenVideo.autoplay = true;
    hiddenVideo.playsInline = true;
    hiddenVideo.muted = true;
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = 640;
    hiddenCanvas.height = 480;
    const ctx = hiddenCanvas.getContext('2d');

    async function toggleBrowserCamera() {
      const btn = document.getElementById('btn-webcam');
      if (!streamActive) {
        try {
          localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
          });
          hiddenVideo.srcObject = localStream;
          await hiddenVideo.play();
          streamActive = true;
          btn.innerText = '⏹ Stop Streaming Webcam';
          btn.style.background = '#da3633';
          sendFramesLoop();
        } catch (err) {
          alert('Could not access webcam: ' + err.message + '\\nPlease grant camera permissions or use python scripts/camera_bridge.py.');
        }
      } else {
        streamActive = false;
        if (localStream) {
          localStream.getTracks().forEach(track => track.stop());
          localStream = null;
        }
        btn.innerText = '📹 Stream Host Webcam Directly';
        btn.style.background = '#238636';
      }
    }

    function sendFramesLoop() {
      if (!streamActive) return;
      if (!isSending) {
        isSending = true;
        ctx.drawImage(hiddenVideo, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        hiddenCanvas.toBlob(async (blob) => {
          if (blob && streamActive) {
            try {
              await fetch('/api/feed', {
                method: 'POST',
                body: blob,
                headers: { 'Content-Type': 'application/octet-stream' }
              });
            } catch (e) {}
          }
          isSending = false;
          if (streamActive) setTimeout(sendFramesLoop, 65); // ~15 FPS
        }, 'image/jpeg', 0.80);
      } else {
        setTimeout(sendFramesLoop, 20);
      }
    }
    // 5. In-Container Direct Simulation (Zero Installs Anywhere!)
    let simActive = false;
    async function toggleInternalSim() {
      const btn = document.getElementById('btn-sim');
      try {
        const res = await fetch('/api/simulate', { method: 'POST' });
        const d = await res.json();
        simActive = d.simulation_active;
        if (simActive) {
          btn.innerText = '⏹ Stop In-Container Test Stream';
          btn.style.background = '#da3633';
        } else {
          btn.innerText = '⚡ Run In-Container Test Stream';
          btn.style.background = '#1f6feb';
        }
      } catch (e) {
        alert('Could not toggle simulation: ' + e.message);
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Lightweight operational dashboard."""
    INFERENCE_REQUESTS.labels(endpoint="/", http_status="200").inc()
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/api/frame")
def get_frame():
    """
    Returns the latest pre-encoded JPEG frame directly with no-cache headers.
    Enables smooth, freeze-free asynchronous rendering in the browser.
    """
    INFERENCE_REQUESTS.labels(endpoint="/api/frame", http_status="200").inc()
    if runtime is not None:
        jpeg = runtime.get_latest_jpeg()
        if jpeg is not None:
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )

    # Clean fallback frame if awaiting camera
    placeholder = np.zeros((384, 384, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Awaiting Camera Feed...", (40, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    _, p_buf = cv2.imencode(".jpg", placeholder)
    return Response(content=p_buf.tobytes(), media_type="image/jpeg")


@app.post("/api/config")
def update_config(
    threshold: Optional[float] = Query(None),
    min_contour_area: Optional[int] = Query(None),
    body: Optional[ConfigUpdate] = None
):
    """
    Dynamically updates model confidence threshold and noise filter on the fly.
    Prompted directly by the user via the UI without hardcoding.
    """
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")

    t_val = threshold if threshold is not None else (body.threshold if body else None)
    a_val = min_contour_area if min_contour_area is not None else (body.min_contour_area if body else None)

    if t_val is not None:
        runtime.set_threshold(t_val)
    if a_val is not None:
        runtime.set_min_contour_area(a_val)

    return {
        "status": "success",
        "threshold": runtime.threshold,
        "min_contour_area": runtime.min_contour_area
    }


@app.post("/api/simulate")
def toggle_simulation():
    """Toggles container-internal sample frame loop without any external bridges."""
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    active = runtime.toggle_simulation()
    INFERENCE_REQUESTS.labels(endpoint="/api/simulate", http_status="200").inc()
    return {"status": "success", "simulation_active": active}


def generate_mjpeg_stream():
    """Generator yielding MJPEG multipart frames."""
    while True:
        if runtime is not None:
            jpeg = runtime.get_latest_jpeg()
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
        time.sleep(0.04)


@app.get("/stream")
@app.get("/video_feed")
def video_feed():
    """MJPEG live camera and tactical camouflage segmentation stream."""
    INFERENCE_REQUESTS.labels(endpoint="/stream", http_status="200").inc()
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Liveness probe: verifies process is alive and responsive."""
    INFERENCE_REQUESTS.labels(endpoint="/health", http_status="200").inc()
    return {"status": "healthy", "service": "edge-ai-inference"}


@app.get("/ready")
def readiness():
    """Readiness probe: verifies model is loaded and ready to accept traffic."""
    if runtime is None or not runtime.is_model_ready:
        INFERENCE_REQUESTS.labels(endpoint="/ready", http_status="503").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed initialization"
        )
    INFERENCE_REQUESTS.labels(endpoint="/ready", http_status="200").inc()
    shape = [int(x) for x in runtime.input_details[0]["shape"]] if (runtime and runtime.input_details) else [1, 384, 384, 3]
    return {
        "status": "ready",
        "model_type": runtime.model_type,
        "model_path": str(runtime.resolved_model_path.name),
        "input_shape": shape,
        "camera_connected": runtime.is_camera_connected
    }


@app.get("/metrics")
def metrics():
    """Prometheus scraping endpoint."""
    update_uptime()
    INFERENCE_REQUESTS.labels(endpoint="/metrics", http_status="200").inc()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/api/status")
def status_endpoint():
    """Telemetry status JSON endpoint for UI polling and automation."""
    if runtime is None:
        return {"status": "uninitialized"}
    return runtime.get_telemetry()


@app.post("/api/feed")
async def ingest_frame(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Ingest endpoint for host webcam bridge or external video sources.
    Processes the frame immediately and updates the bounded buffer.
    """
    if runtime is None or not runtime.is_model_ready:
        raise HTTPException(status_code=503, detail="Model runtime not ready")

    if file is not None:
        img_bytes = await file.read()
    else:
        img_bytes = await request.body()

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty frame payload")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    telemetry = runtime.feed_frame(frame_bgr)
    INFERENCE_REQUESTS.labels(endpoint="/api/feed", http_status="200").inc()
    return telemetry


@app.post("/predict")
async def predict_endpoint(request: Request, file: Optional[UploadFile] = File(None)):
    """Single-frame prediction endpoint returning mask PNG and latency breakdown."""
    if runtime is None or not runtime.is_model_ready:
        raise HTTPException(status_code=503, detail="Model runtime not ready")

    if file is not None:
        img_bytes = await file.read()
    else:
        img_bytes = await request.body()

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty frame payload")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    telemetry = runtime.feed_frame(frame_bgr)

    jpeg = runtime.get_latest_jpeg()
    mask_b64 = base64.b64encode(jpeg).decode("utf-8") if jpeg else ""

    INFERENCE_REQUESTS.labels(endpoint="/predict", http_status="200").inc()
    return JSONResponse(
        status_code=200,
        content={
            **telemetry,
            "annotated_jpeg_base64": mask_b64
        }
    )
