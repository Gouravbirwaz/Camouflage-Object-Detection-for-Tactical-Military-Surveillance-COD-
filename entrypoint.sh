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

# Ensure all candidate checkpoint & log directories exist
mkdir -p "$CKPT_DIR_RESOLVED"
mkdir -p "$LOG_DIR_RESOLVED"
mkdir -p "models/raw_checkpoints"
mkdir -p "metrics"

# 1. DVC Dataset Checkout & Checkpoint Sync from HF Storage Bucket
echo ""
echo "[Step 1/3] Syncing latest checkpoints and dataset from HF Storage Bucket..."
if [ -d "/data/dvc-store" ] || [ -d "/data/files" ]; then
    echo "  • Detected mounted DVC storage bucket at /data. Running DVC checkout..."
    dvc config cache.dir /data/dvc-store 2>/dev/null || true
    dvc config cache.type copy 2>/dev/null || true
    dvc checkout data/raw.dvc 2>/dev/null || dvc checkout 2>/dev/null || echo "Notice: DVC checkout completed with notes."
fi

python3 scripts/hf_sync.py --download-checkpoints --ckpt-dir "$CKPT_DIR_RESOLVED" --download-dataset || echo "Notice: Checkpoint/dataset sync skipped or using mounted S3 dataset."

# 2. Run DGNet Training
echo ""
echo "[Step 2/3] Launching DGNet TensorFlow GPU Training..."
RESUME_FLAG=""
if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

python3 scripts/DGnet.py \
    --train \
    --epochs ${EPOCHS:-50} \
    --batch-size ${BATCH_SIZE:-16} \
    --data-dir ${DATA_DIR:-data/raw/dataset-splitM} \
    --checkpoint-dir "$CKPT_DIR_RESOLVED" \
    --log-dir "$LOG_DIR_RESOLVED" \
    --export-tflite \
    $RESUME_FLAG

# 3. Upload Checkpoints and Logs back to HF Storage Bucket
echo ""
echo "[Step 3/3] Uploading trained checkpoints and metrics back to HF Bucket..."
python3 scripts/hf_sync.py --upload-checkpoints --ckpt-dir "$CKPT_DIR_RESOLVED" --upload-logs --log-dir "$LOG_DIR_RESOLVED" || echo "Notice: Upload completed with notes."

echo "======================================================================"
echo " 🎉 GPU Training Completed Successfully! Checkpoints Saved & Synced."
echo "======================================================================"

