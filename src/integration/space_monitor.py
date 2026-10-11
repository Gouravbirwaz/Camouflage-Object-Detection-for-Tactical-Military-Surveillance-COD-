"""
Lightweight HTTP Status Monitor for Hugging Face Space (Port 7860).
Provides a live web dashboard showing training progress, GPU status, and metrics,
while satisfying HF Space port health check requirements.
"""

import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from typing import Optional

PORT = int(os.getenv("PORT", 7860))
STATUS_FILE = Path("/tmp/training_status.txt")
LOG_OUTPUT_FILE = Path("/tmp/training_output.log")
METRIC_CANDIDATES = [
    Path("/data/metrics/train_metrics.json"),
    Path("metrics/train_metrics.json"),
    Path("/app/metrics/train_metrics.json"),
]


class StatusHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        status_text = "Initializing training worker..."
        if STATUS_FILE.exists():
            try:
                status_text = STATUS_FILE.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        live_logs = ""
        if LOG_OUTPUT_FILE.exists():
            try:
                lines = LOG_OUTPUT_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
                live_logs = "\n".join(lines[-25:])
            except Exception:
                pass

        metrics_html = "<p style='color:#8b949e;'>No epoch metrics recorded yet.</p>"
        for mf in METRIC_CANDIDATES:
            if mf.exists():
                try:
                    with open(mf, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    epochs = metrics.get("epochs", [])
                    if epochs:
                        last_ep = epochs[-1]
                        metrics_html = (
                            f"<table style='width:100%; border-collapse:collapse;'>"
                            f"<tr><th style='text-align:left; padding:6px; border-bottom:1px solid #30363d;'>Epoch</th>"
                            f"<th style='text-align:left; padding:6px; border-bottom:1px solid #30363d;'>Train Loss</th>"
                            f"<th style='text-align:left; padding:6px; border-bottom:1px solid #30363d;'>Val Loss</th>"
                            f"<th style='text-align:left; padding:6px; border-bottom:1px solid #30363d;'>Val IoU</th></tr>"
                            f"<tr><td style='padding:6px;'>{last_ep.get('epoch', '-')}</td>"
                            f"<td style='padding:6px;'>{last_ep.get('train_loss', 0):.4f}</td>"
                            f"<td style='padding:6px;'>{last_ep.get('val_loss', 0):.4f}</td>"
                            f"<td style='padding:6px; color:#3fb950;'><b>{last_ep.get('val_iou', 0):.4f}</b></td></tr>"
                            f"</table>"
                        )
                        break
                except Exception:
                    pass

        model_version = os.getenv("MODEL_VERSION", "v2.0_official_dgnet")
        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DGNet GPU Training Worker</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 30px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; margin-top: 0; font-size: 22px; }}
        .badge {{ background: #238636; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        pre {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; overflow-x: auto; color: #79c0ff; font-size: 12px; max-height: 350px; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
<div class="card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1>🎯 DGNet Camouflage Object Detection - GPU Training</h1>
        <span class="badge">Active</span>
    </div>
    <p><b>Model Version:</b> <code style="color:#f0883e;">{model_version}</code> | <b>Target:</b> Hugging Face Dedicated GPU</p>
    <hr style="border:0; border-top:1px solid #30363d; margin:15px 0;" />
    <h3>Live Training Output</h3>
    <p><b>Status:</b> {status_text}</p>
    <pre>{live_logs or status_text}</pre>
    <h3>Latest Epoch Metrics</h3>
    {metrics_html}
    <hr style="border:0; border-top:1px solid #30363d; margin:15px 0;" />
    <p style="font-size:12px; color:#8b949e;">Page refreshes automatically every 5 seconds. Checkpoints are synchronized to HF Hub automatically upon epoch completion.</p>
</div>
</body>
</html>
"""
        self.wfile.write(html_page.encode("utf-8"))

    def log_message(self, format, *args):
        # Silence per-request access logs
        pass


def run_monitor(port: int = PORT):
    """Starts the monitor HTTP server."""
    with socketserver.TCPServer(("", port), StatusHandler) as httpd:
        print(f"[Space Monitor] Server listening on port {port}...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Space Monitor] Shutting down...")


def main():
    run_monitor(PORT)


if __name__ == "__main__":
    main()
