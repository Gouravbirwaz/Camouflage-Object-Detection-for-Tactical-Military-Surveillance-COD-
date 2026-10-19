# ==============================================================================
# Single-Command Edge AI Build, Push to Docker Hub, and Deploy to Kubernetes (PowerShell)
# ==============================================================================
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$DockerUser
)

$ErrorActionPreference = "Stop"
$ImageTag = "${DockerUser}/edge-ai-inference:latest"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "    DGNet Edge AI: Build -> Push to Docker Hub -> Deploy to K8s" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Build minimal Docker image
Write-Host "[1/4] Building minimal Docker image ($ImageTag)..." -ForegroundColor Yellow
docker build -t $ImageTag -f Dockerfile .

# 2. Push to Docker Hub
Write-Host "[2/4] Pushing image to Docker Hub..." -ForegroundColor Yellow
docker push $ImageTag

# 3. Apply Kubernetes manifests
Write-Host "[3/4] Deploying to Kubernetes cluster..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/service.yaml

$hasSM = kubectl get crd servicemonitors.monitoring.coreos.com 2>$null
if ($hasSM) {
    kubectl apply -f k8s/servicemonitor.yaml
}

kubectl apply -f k8s/deployment.yaml
kubectl set image deployment/edge-ai-inference inference=$ImageTag -n edge-ai

# 4. Wait for rollout
Write-Host "[4/4] Waiting for pod rollout..." -ForegroundColor Yellow
kubectl rollout status deployment/edge-ai-inference -n edge-ai --timeout=120s

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "    DEPLOYMENT READY! To use with your host webcam:" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  1. Forward port:" -ForegroundColor White
Write-Host "     kubectl port-forward -n edge-ai svc/edge-ai-inference 8000:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Start host webcam bridge:" -ForegroundColor White
Write-Host "     python scripts/camera_bridge.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Open Tactical UI in browser:" -ForegroundColor White
Write-Host "     http://localhost:8000" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Green
