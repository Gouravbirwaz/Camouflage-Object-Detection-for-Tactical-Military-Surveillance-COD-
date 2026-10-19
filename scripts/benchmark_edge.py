#!/usr/bin/env python3
"""
Edge AI Benchmark and Endpoint Validation Script for DGNet.
Measures latency (p50, p90, p95, p99), throughput (FPS), and validates /metrics.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="DGNet Edge AI Benchmark Suite")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("EDGE_API_URL", "http://localhost:8000/predict"),
        help="Inference API endpoint (default: http://localhost:8000/predict)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Measured iterations (default: 50)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_data.jpeg",
        help="Test image file (default: test_data.jpeg)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("      DGNet Edge AI Latency & Throughput Benchmark")
    print("=" * 70)
    print(f"  Target Endpoint:     {args.endpoint}")
    print(f"  Warm-up Cycles:      {args.warmup}")
    print(f"  Measured Inferences: {args.iterations}")

    base_url = args.endpoint.rsplit("/", 1)[0]
    session = requests.Session()

    # 1. Health & Readiness check
    try:
        r_ready = session.get(f"{base_url}/ready", timeout=5.0)
        if r_ready.status_code != 200:
            print(f"[ERROR] Service not ready: HTTP {r_ready.status_code}")
            sys.exit(1)
        info = r_ready.json()
        print(f"  Model Engine:        {info.get('model_type')} ({info.get('model_path')})")
        print(f"  Input Shape:         {info.get('input_shape')}")
    except Exception as exc:
        print(f"[ERROR] Could not connect to {base_url}/ready: {exc}")
        sys.exit(1)

    # 2. Load test image
    img_p = Path(args.image)
    if not img_p.is_absolute():
        img_p = PROJECT_ROOT / img_p

    if not img_p.exists():
        print(f"[ERROR] Test image not found at {img_p}")
        sys.exit(1)

    with open(img_p, "rb") as f:
        img_bytes = f.read()

    # 3. Warm-up
    print(f"\n[1/2] Running {args.warmup} warm-up cycles...")
    for _ in range(args.warmup):
        session.post(args.endpoint, data=img_bytes, headers={"Content-Type": "application/octet-stream"})

    # 4. Measured inferences
    print(f"[2/2] Running {args.iterations} measured inferences...")
    inf_latencies = []
    e2e_latencies = []

    t_start = time.perf_counter()
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        resp = session.post(args.endpoint, data=img_bytes, headers={"Content-Type": "application/octet-stream"})
        t_e2e = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            d = resp.json()
            inf_latencies.append(d.get("inference_latency_ms", 0.0))
            e2e_latencies.append(t_e2e)

    total_time = time.perf_counter() - t_start
    throughput = len(inf_latencies) / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 70)
    print("                     BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Completed:           {len(inf_latencies)}/{args.iterations} inferences")
    print(f"  Throughput:          {throughput:0.2f} FPS")
    print("-" * 70)
    print(f"  Inference (Pure):    p50: {np.percentile(inf_latencies, 50):0.1f}ms | "
          f"p95: {np.percentile(inf_latencies, 95):0.1f}ms | p99: {np.percentile(inf_latencies, 99):0.1f}ms")
    print(f"  End-to-End (HTTP):   p50: {np.percentile(e2e_latencies, 50):0.1f}ms | "
          f"p95: {np.percentile(e2e_latencies, 95):0.1f}ms | p99: {np.percentile(e2e_latencies, 99):0.1f}ms")
    print("=" * 70)

    # Validate metrics
    r_metrics = session.get(f"{base_url}/metrics")
    if r_metrics.status_code == 200:
        lines = [l for l in r_metrics.text.splitlines() if l.startswith("edge_") and not l.startswith("#")]
        print(f"\n[Validation] Prometheus /metrics active! Exported metrics count: {len(lines)}")


if __name__ == "__main__":
    main()
