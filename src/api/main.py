"""
FastAPI Command & Control (C2) and Operational Web Interface for Edge AI Inference.
Exposes freeze-free live frame rendering, dynamic UI-controlled threshold adjustment,
Prometheus metrics, health probes, external camera frame ingestion, and multi-edge-case
image upload and evaluation endpoints.
"""

import base64
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect, status
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

# Preset Edge Cases Library for One-Click Operational Testing
PRESET_EDGE_CASES: List[Dict[str, Any]] = [
    {
        "id": "standard_benchmark",
        "name": "Standard Benchmark",
        "category": "Natural Camouflage",
        "description": "Baseline camouflaged subject on natural substrate.",
        "path": PROJECT_ROOT / "test_data.jpeg",
    },
    {
        "id": "mesh_foliage",
        "name": "Mesh & Foliage Occlusion",
        "category": "Dense Cover",
        "description": "Camouflaged subject obscured behind tactical mesh and vegetation.",
        "path": PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_mesh_foliage_3.jpg",
    },
    {
        "id": "fog_lighting",
        "name": "Heavy Fog & Low Light",
        "category": "Atmospheric Degradation",
        "description": "Severe contrast attenuation and scattering due to atmospheric fog.",
        "path": PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_fog_lighting_1.jpg",
    },
    {
        "id": "rain_dusk",
        "name": "Rain & Dusk Low Visibility",
        "category": "Adverse Weather",
        "description": "Dusk ambient lighting with simulated precipitation streaks.",
        "path": PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_rain_dusk_4.jpg",
    },
    {
        "id": "thermal_noise",
        "name": "Sensor Thermal Noise",
        "category": "Sensor Artifact",
        "description": "High-gain CMOS/thermal sensor noise testing false alarm rejection.",
        "path": PROJECT_ROOT / "data/augmented/military_tactical/Training/images/image1009_mil_thermal_noise_2.jpg",
    },
]


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
    version="2.1.0",
    lifespan=lifespan
)


# Tactical Command & Control Web Interface (Dual-Mode: Live Video Stream & Multi-Edge-Case Upload)
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
      --card-hover: #1b2438;
      --border: #222d42;
      --text: #c5d1e0;
      --text-bright: #ffffff;
      --alert-red: #ff3333;
      --alert-green: #00e676;
      --cyan: #00b0ff;
      --yellow: #ffd600;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background: var(--bg); color: var(--text); padding: 1.25rem; font-size: 14px; min-height: 100vh; }
    
    /* Header & Navigation Tabs */
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; gap: 0.8rem; }
    h1 { font-size: 1.25rem; color: var(--text-bright); display: flex; align-items: center; gap: 0.6rem; }
    .badge { padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: bold; background: #0070f3; color: white; }
    
    .nav-tabs { display: flex; gap: 0.5rem; background: #0e1422; padding: 4px; border-radius: 6px; border: 1px solid var(--border); }
    .nav-tab { padding: 0.45rem 1rem; border-radius: 4px; border: none; background: transparent; color: var(--text); font-weight: bold; font-size: 0.85rem; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 0.4rem; }
    .nav-tab:hover { color: var(--text-bright); background: rgba(255,255,255,0.05); }
    .nav-tab.active { background: var(--cyan); color: #000; box-shadow: 0 0 10px rgba(0, 176, 255, 0.4); }

    .status-pill { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 4px; font-size: 0.82rem; font-weight: bold; letter-spacing: 0.5px; }
    .status-green { background: rgba(0, 230, 118, 0.15); color: var(--alert-green); border: 1px solid var(--alert-green); }
    .status-red { background: rgba(255, 51, 51, 0.20); color: var(--alert-red); border: 1px solid var(--alert-red); box-shadow: 0 0 10px rgba(255, 51, 51, 0.3); }

    /* Layout Grids */
    .view-panel { display: none; }
    .view-panel.active { display: block; }
    .grid { display: grid; grid-template-columns: 2.2fr 1fr; gap: 1.2rem; }
    @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 1.1rem; }
    
    /* Live Video Container */
    .video-container { position: relative; width: 100%; border-radius: 4px; overflow: hidden; background: #000; min-height: 400px; display: flex; align-items: center; justify-content: center; }
    .video-container img { width: 100%; height: auto; display: block; object-fit: contain; }
    
    /* Metrics */
    .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin: 0.8rem 0; }
    .metric-box { background: #090d16; border: 1px solid var(--border); border-radius: 4px; padding: 0.65rem; }
    .metric-label { font-size: 0.7rem; color: #7f8c9b; text-transform: uppercase; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.2rem; font-weight: bold; color: var(--text-bright); }
    
    /* Dynamic Controls */
    .control-group { margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--border); }
    .control-label { font-size: 0.8rem; font-weight: bold; color: var(--text-bright); display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
    input[type=range] { width: 100%; height: 6px; background: #222d42; border-radius: 3px; outline: none; cursor: pointer; accent-color: var(--cyan); }
    
    .btn { padding: 0.65rem 1rem; border: none; border-radius: 4px; font-weight: bold; font-size: 0.85rem; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 0.45rem; transition: background 0.2s, transform 0.1s; text-decoration: none; }
    .btn:active { transform: scale(0.98); }
    .btn-green { background: #238636; color: white; }
    .btn-green:hover { background: #2ea043; }
    .btn-blue { background: #1f6feb; color: white; }
    .btn-blue:hover { background: #388bfd; }
    .btn-cyan { background: var(--cyan); color: #000; }
    .btn-cyan:hover { background: #33c2ff; }
    .btn-red { background: #da3633; color: white; }
    .btn-red:hover { background: #f85149; }
    .btn-secondary { background: #21262d; color: var(--text); border: 1px solid var(--border); }
    .btn-secondary:hover { background: #30363d; color: var(--text-bright); }

    /* Upload & Edge Cases Styles */
    .upload-zone { border: 2px dashed var(--border); border-radius: 8px; padding: 2rem 1.5rem; text-align: center; background: #0c121e; cursor: pointer; transition: all 0.2s; position: relative; margin-bottom: 1.2rem; }
    .upload-zone:hover, .upload-zone.dragover { border-color: var(--cyan); background: #111a2d; }
    .upload-zone h3 { font-size: 1.1rem; color: var(--text-bright); margin-bottom: 0.4rem; }
    .upload-zone p { font-size: 0.82rem; color: #7f8c9b; }

    .preset-bar { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.2rem; align-items: center; }
    .preset-chip { padding: 0.4rem 0.8rem; background: #162032; border: 1px solid var(--border); border-radius: 20px; font-size: 0.78rem; color: var(--text); cursor: pointer; display: flex; align-items: center; gap: 0.4rem; transition: all 0.2s; }
    .preset-chip:hover { border-color: var(--cyan); color: var(--text-bright); background: #1f2d45; }
    .preset-chip.active { border-color: var(--cyan); background: rgba(0, 176, 255, 0.15); color: var(--cyan); font-weight: bold; }

    /* Edge Cases Results Gallery */
    .edge-cases-scroll { display: flex; gap: 0.8rem; overflow-x: auto; padding-bottom: 0.6rem; margin-bottom: 1.2rem; }
    .edge-cases-scroll::-webkit-scrollbar { height: 6px; }
    .edge-cases-scroll::-webkit-scrollbar-thumb { background: #222d42; border-radius: 3px; }
    .edge-card { min-width: 200px; max-width: 200px; background: #0e1422; border: 1px solid var(--border); border-radius: 6px; padding: 0.6rem; cursor: pointer; transition: all 0.2s; flex-shrink: 0; }
    .edge-card:hover { border-color: var(--cyan); transform: translateY(-2px); }
    .edge-card.selected { border-color: var(--cyan); box-shadow: 0 0 10px rgba(0, 176, 255, 0.3); background: #151e33; }
    .edge-card img { width: 100%; height: 110px; object-fit: cover; border-radius: 4px; margin-bottom: 0.4rem; display: block; background: #000; }
    .edge-card-title { font-size: 0.78rem; font-weight: bold; color: var(--text-bright); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .edge-card-metrics { font-size: 0.70rem; color: #7f8c9b; margin-top: 0.3rem; display: flex; justify-content: space-between; }

    /* 3-Way Inspector */
    .inspector-container { margin-top: 1rem; }
    .inspector-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 0.8rem; padding-bottom: 0.6rem; border-bottom: 1px solid var(--border); }
    .inspector-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    @media (max-width: 860px) { .inspector-grid { grid-template-columns: 1fr; } }
    .inspector-panel { background: #090d16; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    .inspector-title { padding: 0.5rem 0.8rem; font-size: 0.75rem; font-weight: bold; background: #111726; color: var(--text-bright); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }
    .inspector-panel img { width: 100%; height: 280px; object-fit: contain; display: block; background: #000; }

    .links-bar { margin-top: 1.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); display: flex; gap: 1.2rem; font-size: 0.8rem; flex-wrap: wrap; }
    .links-bar a { color: var(--cyan); text-decoration: none; }
    .links-bar a:hover { text-decoration: underline; }
  </style>
</head>
<body>

  <!-- Top Header with Dual-Mode Navigation -->
  <header>
    <div>
      <h1>DGNet Tactical Edge C2 <span class="badge">Simulated Jetson Nano</span></h1>
      <span style="font-size: 0.75rem; color: #7f8c9b;">Edge Camouflage Object Detection & Surveillance Engine</span>
    </div>

    <!-- Mode Selector Tabs -->
    <div class="nav-tabs">
      <button id="tab-btn-live" class="nav-tab active" onclick="switchTab('live')">
        📹 Live Video Stream
      </button>
      <button id="tab-btn-upload" class="nav-tab" onclick="switchTab('upload')">
        🖼️ Upload & Multiple Edge Cases
      </button>
    </div>

    <div id="target-status" class="status-pill status-green">[SCANNING / CLEAR]</div>
  </header>

  <!-- ========================================== -->
  <!-- TAB 1: LIVE VIDEO STREAM VIEW -->
  <!-- ========================================== -->
  <div id="view-live" class="view-panel active">
    <div class="grid">
      <div class="card">
        <div class="video-container">
          <!-- Native, zero-overhead MJPEG video stream -->
          <img id="live-stream" src="/stream" alt="Live Tactical Feed">
        </div>
        <div style="margin-top: 0.6rem; display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 0.75rem; color: #7f8c9b;">Continuous Tactical Camouflage Inference Stream</span>
          <button class="btn btn-secondary" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="cycleNextEdgeCase()">
            ⏭ Next Simulation Edge Case
          </button>
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

        <!-- Dynamic User-Controlled Thresholds -->
        <div class="control-group">
          <div class="control-label">
            <span>Confidence Threshold:</span>
            <span id="thresh-val" style="color: var(--cyan);">50%</span>
          </div>
          <input type="range" id="thresh-slider" min="5" max="95" value="50" step="1" oninput="onThresholdChange(this.value)">
          <p style="font-size: 0.72rem; color: #7f8c9b; margin-top: 0.3rem;">
            Pixels with confidence below threshold are suppressed to prevent false alarms.
          </p>
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Noise Filter (Min Area):</span>
            <span id="area-val" style="color: var(--cyan);">300 px</span>
          </div>
          <input type="range" id="area-slider" min="50" max="1500" value="300" step="50" oninput="onAreaChange(this.value)">
        </div>

        <!-- Live Camera Controls -->
        <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem;">
          <button id="btn-webcam" class="btn btn-green" onclick="toggleBrowserCamera()">
            📹 Stream Browser Webcam (0 Installs)
          </button>
          <button id="btn-sim" class="btn btn-blue" onclick="toggleInternalSim()">
            ⚡ Run In-Container Test Stream
          </button>
          <button class="btn btn-secondary" onclick="switchTab('upload')">
            🖼️ Upload Custom Images & Edge Cases &rarr;
          </button>
        </div>

        <div style="margin-top: 0.8rem; font-size: 0.75rem; color: #7f8c9b;">
          <span>Feed Status: </span>
          <strong id="val-cam" style="color: #00e676;">Active</strong>
        </div>
      </div>
    </div>
  </div>


  <!-- ========================================== -->
  <!-- TAB 2: UPLOAD IMAGE & MULTIPLE EDGE CASES -->
  <!-- ========================================== -->
  <div id="view-upload" class="view-panel">
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1rem;">
        <div>
          <h2 style="font-size: 1.05rem; color: var(--text-bright);">Edge Cases & Image Upload Suite</h2>
          <p style="font-size: 0.78rem; color: #7f8c9b; margin-top: 0.2rem;">
            Upload single or multiple images simultaneously to test false positive rejection, foliage occlusion, rain/dusk visibility, and sensor noise.
          </p>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
          <span style="font-size: 0.78rem; color: #7f8c9b;">Eval Threshold:</span>
          <span id="upload-thresh-val" style="color: var(--cyan); font-weight: bold; font-size: 0.85rem;">50%</span>
          <input type="range" id="upload-thresh-slider" min="5" max="95" value="50" step="1" style="width: 120px;" oninput="onUploadThresholdChange(this.value)">
        </div>
      </div>

      <!-- Drag & Drop Upload Zone -->
      <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <input type="file" id="file-input" accept="image/*" multiple style="display: none;" onchange="handleFileSelect(this.files)">
        <h3>📁 Drag & Drop Images or Click to Browse</h3>
        <p>Supports single or multiple images (JPEG, PNG, WebP). Test edge cases at once.</p>
        <div style="margin-top: 0.8rem;">
          <button class="btn btn-cyan" style="display: inline-flex;" onclick="event.stopPropagation(); document.getElementById('file-input').click();">
            Upload Image(s)
          </button>
        </div>
      </div>

      <!-- Preset Edge Cases Bar -->
      <div style="margin-bottom: 0.4rem; font-size: 0.8rem; font-weight: bold; color: var(--text-bright);">
        Preset Military & Environmental Edge Cases:
      </div>
      <div class="preset-bar">
        <button class="preset-chip" onclick="runPresetEdgeCase('mesh_foliage')">🌲 Mesh Foliage Occlusion</button>
        <button class="preset-chip" onclick="runPresetEdgeCase('fog_lighting')">🌫️ Heavy Fog & Low Contrast</button>
        <button class="preset-chip" onclick="runPresetEdgeCase('rain_dusk')">🌧️ Rain & Dusk Visibility</button>
        <button class="preset-chip" onclick="runPresetEdgeCase('thermal_noise')">🎯 Sensor Thermal Noise</button>
        <button class="preset-chip" onclick="runPresetEdgeCase('standard_benchmark')">🦎 Standard Benchmark</button>
        <button class="btn btn-blue" style="padding: 0.35rem 0.8rem; font-size: 0.75rem;" onclick="runAllPresets()">
          ⚡ Run All 5 Presets
        </button>
      </div>

      <!-- Edge Cases Results Gallery -->
      <div id="results-section" style="margin-top: 1rem; display: none;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
          <h3 style="font-size: 0.9rem; color: var(--text-bright);">
            Tested Edge Cases (<span id="results-count">0</span>):
          </h3>
          <span style="font-size: 0.75rem; color: #7f8c9b;">Click any card to inspect full 3-way telemetry</span>
        </div>

        <div class="edge-cases-scroll" id="edge-cases-list">
          <!-- Dynamically populated cards -->
        </div>

        <!-- 3-Way Inspector for Selected Edge Case -->
        <div class="inspector-container" id="inspector" style="display: none;">
          <div class="inspector-header">
            <div>
              <h3 id="inspect-title" style="font-size: 1rem; color: var(--text-bright);">Selected Edge Case</h3>
              <div id="inspect-meta" style="font-size: 0.76rem; color: #7f8c9b; margin-top: 0.2rem;">--</div>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
              <span id="inspect-badge" class="status-pill status-green">[CLEAR]</span>
              <button class="btn btn-green" onclick="sendSelectedToLiveStream()">
                🚀 Push to Live Video Stream
              </button>
            </div>
          </div>

          <!-- 3-Way Side-by-Side Visual Grid -->
          <div class="inspector-grid">
            <div class="inspector-panel">
              <div class="inspector-title">
                <span>1. Original Input Image</span>
                <span id="inspect-dims">--</span>
              </div>
              <img id="inspect-orig" src="" alt="Original">
            </div>

            <div class="inspector-panel">
              <div class="inspector-title">
                <span>2. Tactical HUD Overlay</span>
                <span style="color: var(--alert-red);" id="inspect-conf">--</span>
              </div>
              <img id="inspect-anno" src="" alt="HUD Overlay">
            </div>

            <div class="inspector-panel">
              <div class="inspector-title">
                <span>3. Camouflage Mask & Contours</span>
                <span style="color: var(--alert-green);" id="inspect-pixels">-- px</span>
              </div>
              <img id="inspect-mask" src="" alt="Binary Mask">
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="links-bar">
    <span>C2 Endpoints:</span>
    <a href="/metrics" target="_blank">/metrics (Prometheus)</a>
    <a href="/health" target="_blank">/health (Liveness)</a>
    <a href="/ready" target="_blank">/ready (Readiness)</a>
    <a href="/api/status" target="_blank">/api/status (JSON)</a>
    <a href="/api/edge_cases" target="_blank">/api/edge_cases (Library)</a>
  </div>

  <script>
    // Tab Switching
    function switchTab(mode) {
      document.getElementById('tab-btn-live').className = 'nav-tab' + (mode === 'live' ? ' active' : '');
      document.getElementById('tab-btn-upload').className = 'nav-tab' + (mode === 'upload' ? ' active' : '');
      document.getElementById('view-live').className = 'view-panel' + (mode === 'live' ? ' active' : '');
      document.getElementById('view-upload').className = 'view-panel' + (mode === 'upload' ? ' active' : '');
    }

    // 1. Live Stream Reconnect
    const streamImg = document.getElementById('live-stream');
    streamImg.onerror = () => {
      setTimeout(() => { streamImg.src = '/stream?t=' + performance.now(); }, 1000);
    };

    // 2. Real-time User-Controlled Threshold for Live Stream
    let updateTimeout = null;
    function onThresholdChange(val) {
      document.getElementById('thresh-val').innerText = val + '%';
      document.getElementById('upload-thresh-val').innerText = val + '%';
      document.getElementById('upload-thresh-slider').value = val;
      clearTimeout(updateTimeout);
      updateTimeout = setTimeout(() => {
        fetch('/api/config?threshold=' + (val / 100.0), { method: 'POST' });
      }, 50);
    }

    function onUploadThresholdChange(val) {
      document.getElementById('upload-thresh-val').innerText = val + '%';
      document.getElementById('thresh-val').innerText = val + '%';
      document.getElementById('thresh-slider').value = val;
      clearTimeout(updateTimeout);
      updateTimeout = setTimeout(() => {
        fetch('/api/config?threshold=' + (val / 100.0), { method: 'POST' });
        // Re-evaluate currently inspected case if open
        if (currentSelectedResult) {
          reEvaluateCurrent(val / 100.0);
        }
      }, 60);
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

          document.getElementById('val-cam').innerText = d.camera_connected ? (wsConnected ? 'Active (⚡ High-Speed Direct Stream)' : 'Active / Ingesting') : 'Standby';
          document.getElementById('val-cam').style.color = d.camera_connected ? '#00e676' : '#ff3333';
        }
      } catch (e) {}
    }
    setInterval(updateTelemetry, 450);

    // 4. In-Browser Webcam Streaming (High-Speed Direct WebSocket & Adaptive HTTP Fallback)
    let localStream = null;
    let streamActive = false;
    let isSending = false;
    let lastSendTime = 0;
    const TARGET_INTERVAL_MS = 38; // ~25 FPS target
    let wsFeed = null;
    let wsConnected = false;

    function initWebSocket() {
      try {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsFeed = new WebSocket(`${proto}//${window.location.host}/ws/feed`);
        wsFeed.binaryType = 'arraybuffer';
        wsFeed.onopen = () => {
          wsConnected = true;
          console.log('[Webcam] High-speed direct WebSocket feed connected.');
        };
        wsFeed.onclose = () => {
          wsConnected = false;
          setTimeout(initWebSocket, 2000);
        };
        wsFeed.onerror = () => {
          wsConnected = false;
        };
      } catch (e) {}
    }
    initWebSocket();

    const hiddenVideo = document.createElement('video');
    hiddenVideo.autoplay = true;
    hiddenVideo.playsInline = true;
    hiddenVideo.muted = true;
    const hiddenCanvas = document.createElement('canvas');
    hiddenCanvas.width = 480;
    hiddenCanvas.height = 360;
    const ctx = hiddenCanvas.getContext('2d', { alpha: false });

    async function toggleBrowserCamera() {
      const btn = document.getElementById('btn-webcam');
      if (!streamActive) {
        try {
          localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }
          });
          hiddenVideo.srcObject = localStream;
          await hiddenVideo.play();
          streamActive = true;
          btn.innerText = '⏹ Stop Streaming Webcam';
          btn.className = 'btn btn-red';
          sendFramesLoop();
        } catch (err) {
          alert('Could not access webcam: ' + err.message);
        }
      } else {
        streamActive = false;
        if (localStream) {
          localStream.getTracks().forEach(track => track.stop());
          localStream = null;
        }
        btn.innerText = '📹 Stream Browser Webcam (0 Installs)';
        btn.className = 'btn btn-green';
      }
    }

    function sendFramesLoop() {
      if (!streamActive) return;

      const now = performance.now();
      if (!isSending && (now - lastSendTime) >= TARGET_INTERVAL_MS) {
        isSending = true;
        lastSendTime = now;
        ctx.drawImage(hiddenVideo, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        hiddenCanvas.toBlob(async (blob) => {
          if (!blob || !streamActive) {
            isSending = false;
            requestAnimationFrame(sendFramesLoop);
            return;
          }

          if (wsConnected && wsFeed && wsFeed.readyState === WebSocket.OPEN) {
            wsFeed.send(blob);
            isSending = false;
            requestAnimationFrame(sendFramesLoop);
          } else {
            try {
              await fetch('/api/feed', {
                method: 'POST',
                body: blob,
                headers: { 'Content-Type': 'application/octet-stream' }
              });
            } catch (e) {}
            isSending = false;
            requestAnimationFrame(sendFramesLoop);
          }
        }, 'image/jpeg', 0.65);
      } else {
        requestAnimationFrame(sendFramesLoop);
      }
    }

    // 5. In-Container Simulation & Edge Case Cycling
    let simActive = false;
    async function toggleInternalSim() {
      const btn = document.getElementById('btn-sim');
      try {
        const res = await fetch('/api/simulate', { method: 'POST' });
        const d = await res.json();
        simActive = d.simulation_active;
        if (simActive) {
          btn.innerText = '⏹ Stop In-Container Test Stream';
          btn.className = 'btn btn-red';
        } else {
          btn.innerText = '⚡ Run In-Container Test Stream';
          btn.className = 'btn btn-blue';
        }
      } catch (e) {
        alert('Could not toggle simulation: ' + e.message);
      }
    }

    async function cycleNextEdgeCase() {
      try {
        const res = await fetch('/api/simulation/next', { method: 'POST' });
        const d = await res.json();
        // Force refresh frame
        streamImg.src = '/stream?t=' + performance.now();
      } catch (e) {}
    }

    // ==========================================
    // MULTI-EDGE-CASE & IMAGE UPLOAD LOGIC
    // ==========================================
    let evaluatedResults = [];
    let currentSelectedResult = null;

    // Drag and Drop listeners
    const dropZone = document.getElementById('drop-zone');
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
    });
    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length) {
        handleFileSelect(dt.files);
      }
    });

    async function handleFileSelect(files) {
      if (!files || !files.length) return;
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      const thresh = document.getElementById('upload-thresh-slider').value / 100.0;
      
      try {
        const res = await fetch('/api/upload?threshold=' + thresh, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.status === 'success' && data.results) {
          data.results.forEach(r => evaluatedResults.unshift(r));
          renderResultsGallery();
          if (data.results.length > 0) {
            selectResult(0);
          }
        }
      } catch (err) {
        alert('Failed to upload and analyze image(s): ' + err.message);
      }
    }

    async function runPresetEdgeCase(caseId) {
      const thresh = document.getElementById('upload-thresh-slider').value / 100.0;
      try {
        const res = await fetch(`/api/edge_cases/run?case_id=${caseId}&threshold=${thresh}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success' && data.results) {
          data.results.forEach(r => evaluatedResults.unshift(r));
          renderResultsGallery();
          selectResult(0);
        }
      } catch (err) {
        alert('Failed to run preset edge case: ' + err.message);
      }
    }

    async function runAllPresets() {
      const thresh = document.getElementById('upload-thresh-slider').value / 100.0;
      try {
        const res = await fetch(`/api/edge_cases/run?threshold=${thresh}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success' && data.results) {
          // Prepend all results
          for (let i = data.results.length - 1; i >= 0; i--) {
            evaluatedResults.unshift(data.results[i]);
          }
          renderResultsGallery();
          selectResult(0);
        }
      } catch (err) {
        alert('Failed to run presets: ' + err.message);
      }
    }

    async function reEvaluateCurrent(newThresh) {
      if (!currentSelectedResult) return;
      // If result is preset, re-run with new threshold
      if (currentSelectedResult.preset_id) {
        try {
          const res = await fetch(`/api/edge_cases/run?case_id=${currentSelectedResult.preset_id}&threshold=${newThresh}`, { method: 'POST' });
          const data = await res.json();
          if (data.status === 'success' && data.results && data.results.length) {
            const updated = data.results[0];
            const idx = evaluatedResults.indexOf(currentSelectedResult);
            if (idx !== -1) evaluatedResults[idx] = updated;
            currentSelectedResult = updated;
            renderResultsGallery();
            displayInspector(updated);
          }
        } catch (e) {}
      }
    }

    function renderResultsGallery() {
      const container = document.getElementById('edge-cases-list');
      const section = document.getElementById('results-section');
      document.getElementById('results-count').innerText = evaluatedResults.length;
      section.style.display = evaluatedResults.length ? 'block' : 'none';
      container.innerHTML = '';

      evaluatedResults.forEach((res, idx) => {
        const card = document.createElement('div');
        const isSel = (currentSelectedResult === res);
        card.className = 'edge-card' + (isSel ? ' selected' : '');
        card.onclick = () => selectResult(idx);

        const badgeClass = res.detected ? 'color: var(--alert-red);' : 'color: var(--alert-green);';
        const badgeText = res.detected ? 'ACQUIRED' : 'CLEAR';

        card.innerHTML = `
          <img src="data:image/jpeg;base64,${res.annotated_base64}" alt="${res.filename}">
          <div class="edge-card-title">${res.filename}</div>
          <div class="edge-card-metrics">
            <span style="${badgeClass} font-weight:bold;">[${badgeText}]</span>
            <span>${(res.confidence_max * 100).toFixed(1)}%</span>
          </div>
          <div class="edge-card-metrics" style="margin-top: 2px;">
            <span>${res.inference_latency_ms} ms</span>
            <span>${res.object_pixels} px</span>
          </div>
        `;
        container.appendChild(card);
      });
    }

    function selectResult(index) {
      if (index >= 0 && index < evaluatedResults.length) {
        currentSelectedResult = evaluatedResults[index];
        renderResultsGallery();
        displayInspector(currentSelectedResult);
      }
    }

    function displayInspector(res) {
      const inspector = document.getElementById('inspector');
      inspector.style.display = 'block';

      document.getElementById('inspect-title').innerText = res.filename;
      document.getElementById('inspect-meta').innerText = 
        `Inference: ${res.inference_latency_ms} ms | Total: ${res.total_latency_ms || res.inference_latency_ms} ms | Max Conf: ${(res.confidence_max * 100).toFixed(1)}% | Mean: ${(res.confidence_mean * 100).toFixed(1)}% | Thresh: ${(res.threshold * 100).toFixed(0)}%`;

      const badge = document.getElementById('inspect-badge');
      if (res.detected) {
        badge.innerText = '[TARGET ACQUIRED]';
        badge.className = 'status-pill status-red';
      } else {
        badge.innerText = '[SCANNING / CLEAR]';
        badge.className = 'status-pill status-green';
      }

      document.getElementById('inspect-dims').innerText = `${res.input_dimensions[1]}x${res.input_dimensions[0]}`;
      document.getElementById('inspect-conf').innerText = `${(res.confidence_max * 100).toFixed(1)}% Conf`;
      document.getElementById('inspect-pixels').innerText = `${res.object_pixels} px`;

      document.getElementById('inspect-orig').src = 'data:image/jpeg;base64,' + res.original_base64;
      document.getElementById('inspect-anno').src = 'data:image/jpeg;base64,' + res.annotated_base64;
      document.getElementById('inspect-mask').src = 'data:image/png;base64,' + res.mask_base64;
    }

    async function sendSelectedToLiveStream() {
      if (!currentSelectedResult) return;
      try {
        const bodyPayload = {
          image_base64: currentSelectedResult.original_base64,
          name: currentSelectedResult.filename
        };
        const res = await fetch('/api/feed_image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(bodyPayload)
        });
        if (res.ok) {
          switchTab('live');
          streamImg.src = '/stream?t=' + performance.now();
        }
      } catch (err) {
        alert('Could not push image to live stream: ' + err.message);
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Tactical command & control operational dashboard."""
    INFERENCE_REQUESTS.labels(endpoint="/", http_status="200").inc()
    return HTMLResponse(content=HTML_DASHBOARD)


@app.get("/api/edge_cases")
def list_edge_cases():
    """Returns catalog of preset edge-case benchmarks for one-click testing."""
    return [
        {
            "id": c["id"],
            "name": c["name"],
            "category": c["category"],
            "description": c["description"],
            "available": c["path"].exists()
        }
        for c in PRESET_EDGE_CASES
    ]


@app.post("/api/edge_cases/run")
def run_edge_cases(
    case_id: Optional[str] = Query(None),
    threshold: Optional[float] = Query(None),
    min_contour_area: Optional[int] = Query(None)
):
    """
    Runs inference on a specific preset edge case or all preset edge cases.
    Returns full telemetry, original image, tactical HUD overlay, and segmentation mask.
    """
    if runtime is None or not runtime.is_model_ready:
        raise HTTPException(status_code=503, detail="Model runtime not ready")

    cases_to_run = [c for c in PRESET_EDGE_CASES if c["id"] == case_id] if case_id else PRESET_EDGE_CASES
    if not cases_to_run:
        raise HTTPException(status_code=404, detail=f"Edge case '{case_id}' not found")

    results = []
    for c in cases_to_run:
        p = c["path"]
        if not p.exists():
            continue
        frame_bgr = cv2.imread(str(p))
        if frame_bgr is None:
            continue

        res = runtime.analyze_image(
            frame_bgr,
            threshold=threshold,
            min_contour_area=min_contour_area
        )
        res["filename"] = c["name"]
        res["preset_id"] = c["id"]
        res["category"] = c["category"]
        results.append(res)

    INFERENCE_REQUESTS.labels(endpoint="/api/edge_cases/run", http_status="200").inc()
    return {"status": "success", "results": results}


@app.post("/api/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    threshold: Optional[float] = Query(None),
    min_contour_area: Optional[int] = Query(None)
):
    """
    Upload single or multiple edge-case images for tactical camouflage detection.
    Evaluates each image and returns telemetry, tactical HUD overlay, and binary masks.
    """
    if runtime is None or not runtime.is_model_ready:
        raise HTTPException(status_code=503, detail="Model runtime not ready")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = []
    for f in files:
        img_bytes = await f.read()
        if not img_bytes:
            continue
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        res = runtime.analyze_image(
            frame_bgr,
            threshold=threshold,
            min_contour_area=min_contour_area
        )
        res["filename"] = f.filename
        res["preset_id"] = None
        results.append(res)

        # Cache custom image into simulation rotation
        runtime.add_custom_sample(f.filename or "Custom Upload", frame_bgr)

    INFERENCE_REQUESTS.labels(endpoint="/api/upload", http_status="200").inc()
    return {"status": "success", "count": len(results), "results": results}


class FeedImagePayload(BaseModel):
    image_base64: str
    name: Optional[str] = "Uploaded Frame"


@app.post("/api/feed_image")
async def feed_image_to_live_stream(payload: FeedImagePayload):
    """
    Feeds an uploaded image or inspected edge case directly into the live video stream.
    """
    if runtime is None or not runtime.is_model_ready:
        raise HTTPException(status_code=503, detail="Model runtime not ready")

    try:
        raw_bytes = base64.b64decode(payload.image_base64)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")
        
        runtime.add_custom_sample(payload.name or "Uploaded Frame", frame_bgr)
        telemetry = runtime.feed_frame(frame_bgr)
        return {"status": "success", "telemetry": telemetry}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image feed: {exc}")


@app.post("/api/simulation/next")
def simulation_next():
    """Advances the continuous simulation loop to the next edge case."""
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    name = runtime.next_edge_case()
    return {"status": "success", "current_edge_case": name}


@app.get("/api/frame")
def get_frame():
    """Returns latest pre-encoded JPEG frame directly with no-cache headers."""
    INFERENCE_REQUESTS.labels(endpoint="/api/frame", http_status="200").inc()
    if runtime is not None:
        jpeg = runtime.get_latest_jpeg()
        if jpeg is not None:
            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )

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
    """Dynamically updates model confidence threshold and noise filter on the fly."""
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
    """Generator yielding MJPEG multipart frames with low-latency polling."""
    last_jpeg_id = None
    while True:
        if runtime is not None:
            jpeg, jpeg_id = runtime.get_latest_jpeg_with_id()
            if jpeg is not None and jpeg_id != last_jpeg_id:
                last_jpeg_id = jpeg_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
        time.sleep(0.01)


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


@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """
    High-speed binary WebSocket ingestion for camera streaming.
    Streams raw JPEG frames directly into runtime buffer with zero HTTP overhead.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            if runtime and data:
                runtime.feed_frame_bytes(data)
                INFERENCE_REQUESTS.labels(endpoint="/ws/feed", http_status="200").inc()
    except (WebSocketDisconnect, Exception):
        pass


@app.post("/api/feed")
async def ingest_frame(request: Request, file: Optional[UploadFile] = File(None)):
    """
    Ingest endpoint for host webcam bridge or external video sources.
    Processes the frame immediately into the runtime buffer.
    """
    if runtime is None or not runtime.is_model_ready:
        raise HTTPException(status_code=503, detail="Model runtime not ready")

    if file is not None:
        img_bytes = await file.read()
    else:
        img_bytes = await request.body()

    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty frame payload")

    success = runtime.feed_frame_bytes(img_bytes)
    if not success:
        raise HTTPException(status_code=400, detail="Could not decode image")

    INFERENCE_REQUESTS.labels(endpoint="/api/feed", http_status="200").inc()
    return {
        "status": "ok",
        "detected": runtime._latest_detected,
        "inference_latency_ms": runtime._latest_inf_ms,
        "fps": runtime._latest_telemetry.get("fps", 0.0)
    }


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

    telemetry = runtime.predict_frame(frame_bgr)

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
