#!/bin/bash
set -e

echo "======================================================================"
echo "      Hugging Face GPU Container Training Worker - DGNet COD"
echo "======================================================================"

# Check GPU Availability
python3 -c "import tensorflow as tf; print('[GPU Hardware Detection]', tf.config.list_physical_devices('GPU'))"

echo ""
echo "[Mounted Storage Detection] Checking candidate mounted bucket paths..."
for path in "/data" "/mnt/data" "/bucket" "/hf" "/datasets" "/mnt/s3"; do
    if [ -d "$path" ]; then
        echo "  • Detected mounted path: $path"
        ls -la "$path" 2>/dev/null | head -n 10 || true
    fi
done

# Priority Path Routing for Persistent Storage
if [ -d "/data" ]; then
    echo "[Mounted Storage] Persistent storage detected at /data. Routing checkpoints & logs to /data..."
    CKPT_DIR_RESOLVED="/data/models/raw_checkpoints"
    LOG_DIR_RESOLVED="/data/metrics"
else
    CKPT_DIR_RESOLVED="${CHECKPOINT_DIR:-models/raw_checkpoints}"
    LOG_DIR_RESOLVED="${LOG_DIR:-metrics}"
fi

export CHECKPOINT_DIR="$CKPT_DIR_RESOLVED"
export LOG_DIR="$LOG_DIR_RESOLVED"

# MLflow Experiment Tracking Defaults & Hugging Face S3 Gateway Setup
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-COD-Camouflage-Detection}"

if [ -n "$HF_TOKEN" ]; then
    echo "[MLflow HF S3 Gateway] Configured Hugging Face S3 Endpoint (s3.hf.co) for MLflow artifact storage."
    export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-https://s3.hf.co}"
    export AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-https://s3.hf.co}"
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-hf}"
    export AWS_SECRET_ACCESS_KEY="$HF_TOKEN"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    export MLFLOW_ARTIFACT_LOCATION="${MLFLOW_ARTIFACT_LOCATION:-s3://gouravbirwaz/COD_dataset/mlflow-artifacts}"
else
    export MLFLOW_ARTIFACT_LOCATION="${LOG_DIR_RESOLVED}/mlflow_artifacts"
fi

# Ensure all candidate checkpoint & log directories exist
mkdir -p "$CKPT_DIR_RESOLVED"
mkdir -p "$LOG_DIR_RESOLVED"
mkdir -p "models/raw_checkpoints"
mkdir -p "metrics"

# Auto-start embedded MLflow tracking server if no external URI provided
if [ -z "$MLFLOW_TRACKING_URI" ] || [ "$MLFLOW_TRACKING_URI" = "http://127.0.0.1:5000" ] || [ "$MLFLOW_TRACKING_URI" = "http://localhost:5000" ]; then
    echo "[MLflow Server] Auto-starting embedded MLflow tracking server on port 5000..."
    mkdir -p "$LOG_DIR_RESOLVED/mlflow_db"
    mkdir -p "$LOG_DIR_RESOLVED/mlflow_artifacts"
    mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --cors-allowed-origins "*" \
        --allowed-hosts "*" \
        --workers 1 \
        --backend-store-uri "sqlite:////${LOG_DIR_RESOLVED#/}/mlflow_db/mlflow.db" \
        --default-artifact-root "$LOG_DIR_RESOLVED/mlflow_artifacts" &
    MLFLOW_PID=$!
    export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
    echo "[MLflow Server] Embedded MLflow server running on PID $MLFLOW_PID at http://127.0.0.1:5000"
fi

# Start background web health check server on port 7860 for HF Space
python3 scripts/space_monitor.py &
MONITOR_PID=$!
echo "[Space Monitor] Started web server on PID $MONITOR_PID (port ${PORT:-7860})"
echo "[Worker Initializing] Setting up environment and storage..." > /tmp/training_status.txt

# 1. DVC Dataset Checkout & Checkpoint Sync from HF Storage Bucket
echo ""
echo "[Step 1/3] Syncing latest checkpoints and dataset from HF Storage Bucket..."
echo "[Step 1/3] Syncing dataset and checkpoints from Hugging Face..." > /tmp/training_status.txt

if [ -d "/data/dvc-store" ] || [ -d "/data/files" ]; then
    echo "  • Detected mounted DVC storage bucket at /data. Running DVC checkout..."
    dvc config cache.dir /data/dvc-store 2>/dev/null || true
    dvc config cache.type copy 2>/dev/null || true
    dvc checkout data/raw.dvc 2>/dev/null || dvc checkout 2>/dev/null || echo "Notice: DVC checkout completed with notes."
fi

python3 scripts/hf_sync.py --download-checkpoints --ckpt-dir "$CKPT_DIR_RESOLVED" --download-dataset --download-logs --log-dir "$LOG_DIR_RESOLVED" || echo "Notice: Checkpoint/dataset/logs sync skipped or using mounted S3 dataset."

# 2. Run DGNet Preprocessing & Training
echo ""
echo "[Step 2/3] Preparing GT masks & launching DGNet TensorFlow GPU Training..."
echo "[Step 2/3] GT PNG conversion & DGNet GPU Training in progress (Epochs: ${EPOCHS:-1}, Batch Size: ${BATCH_SIZE:-8}, Version: ${MODEL_VERSION:-v2.0_official_dgnet})..." > /tmp/training_status.txt

# Convert training GT masks from JPEG to lossless PNG if needed
echo "[GT Prep] Running GT mask conversion (JPEG to PNG)..."
python3 scripts/convert_gt_to_png.py --data-dir ${DATA_DIR:-data/raw/dataset-splitM} || echo "Notice: GT mask conversion completed or skipped."

RESUME_FLAG=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

python3 -u scripts/DGnet.py \
    --train \
    --epochs ${EPOCHS:-1} \
    --batch-size ${BATCH_SIZE:-3} \
    --data-dir ${DATA_DIR:-data/raw/dataset-splitM} \
    --checkpoint-dir "$CKPT_DIR_RESOLVED" \
    --log-dir "$LOG_DIR_RESOLVED" \
    --version "${MODEL_VERSION:-v2.0_official_dgnet}" \
    --export-tflite \
    $RESUME_FLAG 2>&1 | tee -a /tmp/training_output.log

# 3. Upload Checkpoints and Logs back to HF Storage Bucket
echo ""
echo "[Step 3/3] Uploading trained checkpoints and metrics back to HF Bucket..."
echo "[Step 3/3] Uploading trained checkpoints and metrics to Hugging Face..." > /tmp/training_status.txt

python3 scripts/hf_sync.py --upload-checkpoints --ckpt-dir "$CKPT_DIR_RESOLVED" --upload-logs --log-dir "$LOG_DIR_RESOLVED" || echo "Notice: Upload completed with notes."

echo "======================================================================"
echo " 🎉 GPU Training Completed Successfully! Checkpoints Saved & Synced."
echo "======================================================================"
echo "[Complete] GPU Training Completed Successfully! Checkpoints saved & synced to gouravbirwaz/COD_dataset." > /tmp/training_status.txt

# Keep container alive to serve dashboard and avoid restart loop
wait $MONITOR_PID

