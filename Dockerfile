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
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source files and dataset
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/
COPY params.yaml .
COPY dvc.yaml .
COPY entrypoint.sh .

# Set permissions
RUN chmod +x entrypoint.sh

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=data/raw/dataset-splitM
ENV MODEL_DIR=models
ENV CHECKPOINT_DIR=models/raw_checkpoints
ENV LOG_DIR=metrics
ENV EPOCHS=50
ENV BATCH_SIZE=16
ENV RESUME=true

# Expose port for HF Space health checks
EXPOSE 7860

# Run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
