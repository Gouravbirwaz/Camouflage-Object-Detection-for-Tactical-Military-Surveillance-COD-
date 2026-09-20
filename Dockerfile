# Production Dockerfile for Hugging Face GPU Space (NVIDIA T4 / A10G / A100)
FROM tensorflow/tensorflow:2.16.1-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source files and dataset
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY params.yaml .
COPY dvc.yaml .
COPY entrypoint.sh .

# Set permissions & sanitize Windows CRLF line endings
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=data/raw/dataset-splitM
ENV MODEL_DIR=models
ENV CHECKPOINT_DIR=models/raw_checkpoints
ENV LOG_DIR=metrics
ENV EPOCHS=200
ENV BATCH_SIZE=8
ENV RESUME=false
ENV MODEL_VERSION=v2.0_official_dgnet

# MLflow Experiment Tracking & S3 Artifact Defaults
ENV MLFLOW_TRACKING_URI=http://3.84.17.133:5000
ENV MLFLOW_EXPERIMENT_NAME=COD-Camouflage-Detection
ENV MLFLOW_S3_ENDPOINT_URL=https://s3.hf.co/gouravbirwaz
ENV AWS_ENDPOINT_URL_S3=https://s3.hf.co/gouravbirwaz
ENV MLFLOW_ARTIFACT_LOCATION=s3://COD_dataset/dvc-store/mlflow-artifacts
ENV HF_S3_URI=s3://COD_dataset/dvc-store
ENV AWS_DEFAULT_REGION=us-east-1

# Expose port for HF Space health checks
EXPOSE 7860

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
