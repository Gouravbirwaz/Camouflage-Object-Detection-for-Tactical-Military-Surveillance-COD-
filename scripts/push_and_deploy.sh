#!/usr/bin/env bash
# ==============================================================================
# Single-Command Edge AI Build, Push to Docker Hub, and Deploy to Kubernetes
# ==============================================================================
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: ./scripts/push_and_deploy.sh <dockerhub-username>"
    echo "Example: ./scripts/push_and_deploy.sh myuser"
    exit 1
fi

DOCKER_USER="$1"
IMAGE_TAG="${DOCKER_USER}/edge-ai-inference:latest"

echo "======================================================================"
echo "    DGNet Edge AI: Build -> Push to Docker Hub -> Deploy to K8s"
echo "======================================================================"

# 1. Build minimal inference image
echo "[1/4] Building minimal Docker image (${IMAGE_TAG})..."
docker build -t "${IMAGE_TAG}" -f Dockerfile .

# 2. Push to Docker Hub
echo "[2/4] Pushing image to Docker Hub..."
docker push "${IMAGE_TAG}"

# 3. Apply Kubernetes manifests with updated image
echo "[3/4] Deploying to Kubernetes cluster..."
kubectl apply -f k8s/namespace.yaml

# Patch image dynamically without modifying repository files permanently
kubectl apply -f k8s/service.yaml
if kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1; then
    kubectl apply -f k8s/servicemonitor.yaml
fi

# Apply deployment and set image to Docker Hub repository
kubectl apply -f k8s/deployment.yaml
kubectl set image deployment/edge-ai-inference inference="${IMAGE_TAG}" -n edge-ai

# 4. Wait for rollout
echo "[4/4] Waiting for pod rollout..."
kubectl rollout status deployment/edge-ai-inference -n edge-ai --timeout=120s

echo ""
echo "======================================================================"
echo "    DEPLOYMENT READY! To use with your host webcam:"
echo "======================================================================"
echo "  1. Forward port:"
echo "     kubectl port-forward -n edge-ai svc/edge-ai-inference 8000:8000"
echo ""
echo "  2. Start host webcam bridge:"
echo "     python scripts/camera_bridge.py"
echo ""
echo "  3. Open Tactical UI in browser:"
echo "     http://localhost:8000"
echo "======================================================================"
