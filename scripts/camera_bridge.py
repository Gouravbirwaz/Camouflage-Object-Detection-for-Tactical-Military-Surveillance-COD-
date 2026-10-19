#!/usr/bin/env python3
"""
Host Webcam Bridge for Edge AI Docker & Kubernetes (Kind) Environments.
Captures live frames from the development host webcam (Windows, macOS, Linux)
and streams them to the Edge AI container at /api/feed.
"""

import argparse
import os
import platform
import sys
import time
from typing import Optional

import cv2
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Host Webcam Bridge for Edge AI Container")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("EDGE_FEED_URL", "http://localhost:8000/api/feed"),
        help="Target container ingest URL (default: http://localhost:8000/api/feed)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=int(os.getenv("CAMERA_INDEX", "0")),
        help="Host webcam index (default: 0)"
    )
    parser.add_argument(
        "--video-source",
        type=str,
        default=os.getenv("VIDEO_SOURCE", ""),
        help="Optional path to video file or image for testing without physical webcam"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture frame width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture frame height (default: 480)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target streaming FPS (default: 15.0)"
    )
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """Opens webcam using optimal platform-specific backend."""
    sys_name = platform.system().lower()
    cap = None

    if "windows" in sys_name:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    elif "linux" in sys_name:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return None


def main():
    args = parse_args()
    print("=" * 70)
    print("      DGNet Edge AI Host Camera Bridge")
    print("=" * 70)
    print(f"  Target Ingest Endpoint: {args.endpoint}")
    print(f"  Target Streaming Rate:  {args.fps} FPS")

    is_file = bool(args.video_source and os.path.exists(args.video_source))
    if is_file:
        print(f"  Source: Video/Image '{args.video_source}'")
        cap = cv2.VideoCapture(args.video_source)
    else:
        print(f"  Source: Webcam Device Index {args.camera_index}")
        cap = open_camera(args.camera_index, args.width, args.height)

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open camera (index {args.camera_index}) or file '{args.video_source}'.")
        print("[TIP] For testing without physical camera, pass --video-source test_data.jpeg")
        sys.exit(1)

    print("\n[Bridge] Streaming frames to container. Press Ctrl+C to stop.\n")
    session = requests.Session()
    delay = 1.0 / args.fps
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()

            if not ret or frame is None:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue

            success, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            try:
                resp = session.post(
                    args.endpoint,
                    data=jpeg_buf.tobytes(),
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=1.5
                )
                if resp.status_code == 200:
                    frame_count += 1
                    data = resp.json()
                    if frame_count % 30 == 0:
                        elapsed = time.time() - t_start
                        fps = frame_count / elapsed
                        status_str = "ACQUIRED" if data.get("detected") else "CLEAR"
                        print(
                            f"[Stream] Sent: {frame_count} frames ({fps:0.1f} FPS) | "
                            f"Inf: {data.get('inference_latency_ms', 0):0.1f}ms | Target: [{status_str}]"
                        )
                else:
                    print(f"[Bridge Warning] Endpoint returned HTTP {resp.status_code}")
            except requests.exceptions.RequestException as exc:
                if frame_count % 15 == 0:
                    print(f"[Bridge Notice] Service unavailable ({exc}), retrying...")
                time.sleep(0.5)

            elapsed = time.perf_counter() - t0
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user.")
    finally:
        cap.release()
        print("[Bridge] Camera released.")


if __name__ == "__main__":
    main()
