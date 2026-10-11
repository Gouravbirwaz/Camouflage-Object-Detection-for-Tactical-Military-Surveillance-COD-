---
title: COD DGNet GPU Training
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Edge-Optimized ML Framework for Realtime Camouflaged Target Detection

A professional, production-grade Machine Learning directory structure designed for **Realtime Camouflaged Target Detection in Tactical Military Surveillance Systems** deployed on **NVIDIA Jetson** edge devices.

---

## Directory Architecture

```text
COD_PROJECT/
├── .github/
│   └── workflows/                 # CI/CD pipeline directory
├── configs/                       # Modular YAML configuration files
│   ├── data/                      # Dataset configurations (COD10K, CAMO, NC4K, synthetic)
│   ├── model/                     # Model architecture configs (DGNet-S, DGNet-ResNet)
│   ├── optimization/              # Pruning, INT8 Quantization & Distillation configs
│   └── deployment/                # Edge Jetson TensorRT & ONNX deployment configs
├── data/                          # DVC-tracked dataset directory (Git ignored)
│   ├── raw/                       # Original raw datasets
│   ├── interim/                   # Cleaned and augmented intermediate data
│   ├── processed/                 # Standardized tensors / preprocessed datasets
│   └── synthetic/                 # Synthetic military terrain generations
├── docker/                        # Docker container configurations for Jetson / Edge deployment
├── docs/                          # Architecture & technical specifications documentation
│   └── architecture/              # Network diagrams, dataflow & MAVLink protocol specs
├── experiments/                   # Experiment tracking, logs, and outputs
├── metrics/                       # Evaluation metrics (S-measure, E-measure, MAE, FPS)
├── models/                        # Model artifacts & exported binaries directory
│   ├── raw_checkpoints/           # Trained weights (.weights.h5, .keras, .pt)
│   ├── pruned/                    # Structurally pruned weights
│   ├── quantized/                 # INT8 quantized PyTorch / ONNX / TFLite models
│   ├── onnx/                      # Exported ONNX graph files
│   └── tensorrt/                  # Optimized TensorRT .engine files for NVIDIA Jetson
├── notebooks/                     # Exploratory analysis & prototyping notebooks
├── requirements/                  # Requirement specifications
├── scripts/                       # Lightweight CLI wrappers (backward-compatible entrypoints)
├── src/                           # Modular Python architecture
│   ├── api/                       # FastAPI Command & Control (C2) service
│   ├── data/                      # Streaming tf.data pipeline, GT mask converter, augmentation
│   ├── inference/                 # Multi-model inference engine (Keras/TFLite/ONNX) & test evaluator
│   ├── integration/               # Hugging Face Hub sync, remote GPU job launchers, deployment, monitor
│   ├── models/                    # DGNet architecture (Context, Texture, GIT, NCD, Multi-Task loss)
│   ├── optimization/              # High-fidelity TFLite converter (FP32 lossless, FP16, INT8)
│   ├── tracking/                  # MLflow experiment tracking & artifact logging
│   ├── training/                  # DGNet training loop, learning rate schedules, benchmark runner
│   ├── utils/                     # Config manager, GPU environment detection, COD metrics
│   └── visualization/             # Mask rendering, color heatmaps, contour overlays, HTML reports
├── tests/                         # Comprehensive Pytest test suite
├── .dvcignore                     # DVC ignore rules
├── .gitignore                     # Production Git ignore configuration
├── .env.example                   # Environment configuration template
├── dvc.yaml                       # DVC Pipeline definition template
├── params.yaml                    # DVC Hyperparameters configuration file
└── requirements.txt               # Main requirements specification
```

---

## Hugging Face + DVC + GPU Training

### Workflow Architecture

```text
Local development
    ↓
DVC tracks dataset
    ↓
DVC push
    ↓
Hugging Face Storage Bucket (hf://buckets/gouravbirwaz/COD_dataset)
    ↓
Hugging Face GPU Job
    ↓
A100 training
    ↓
checkpoints/logs saved to HF bucket
    ↓
resume/download model
```

### Hugging Face Storage Bucket Structure
Remote storage bucket: `hf://buckets/gouravbirwaz/COD_dataset`

```text
COD_dataset/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── models/
│   ├── pretrained/
│   └── raw_checkpoints/
├── checkpoints/
└── logs/
```

### 1. Credentials Configuration
Never commit credentials to Git. Create a `.env` file from `.env.example`:

```bash
# Copy template files
cp .env.example .env
cp .dvc/config.local.example .dvc/config.local
```

Fill in your Hugging Face credentials in `.env` and `.dvc/config.local`:
- `HF_TOKEN`: Hugging Face User Access Token (with Write permissions).
- `AWS_ACCESS_KEY_ID`: Hugging Face S3 Access Key ID (for DVC remote gateway).
- `AWS_SECRET_ACCESS_KEY`: Hugging Face S3 Secret Access Key.

---

### 2. DVC Commands (Dataset Versioning)

Configure default remote:
```bash
dvc remote default hf-bucket
```

Track datasets with DVC:
```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track dataset with DVC via HF Storage Bucket"
```

Push/Pull data to Hugging Face S3 Gateway:
```bash
dvc push
dvc pull
```

---

### 3. Local Training & Testing (PowerShell / Windows)

Test training locally with configurable paths:
```powershell
python scripts/DGnet.py --train --epochs 2 --batch-size 4
```

Test model summary and GPU detection:
```powershell
python scripts/DGnet.py --summary
```

Export quantized TFLite model:
```powershell
python scripts/DGnet.py --export-tflite --output models/quantized/dgnet_mobilenet_v3.tflite
```

---

### 4. Hugging Face CLI & Storage Sync Commands

Authenticate with Hugging Face:
```bash
hf auth login
```

Upload dataset to Hugging Face Storage Bucket:
```bash
python scripts/hf_sync.py --upload-dataset --data-dir data/raw
```

Upload model checkpoints and logs:
```bash
python scripts/hf_sync.py --upload-checkpoints
python scripts/hf_sync.py --upload-logs
```

Download latest model checkpoints from bucket:
```bash
python scripts/hf_sync.py --download-checkpoints
```

---

### 5. Launch NVIDIA A100 GPU Training Job

Generate GPU job command or execute via `hf`:
```bash
python scripts/launch_hf_job.py --flavor a100-large --epochs 50 --batch-size 16
```

Or execute directly with HF CLI:
```bash
hf job run \
  --flavor a100-large \
  --env HF_TOKEN=$HF_TOKEN \
  --env HF_REPO_ID=gouravbirwaz/COD_dataset \
  --command "pip install -r requirements.txt && python scripts/hf_sync.py --download-checkpoints && python scripts/DGnet.py --train --epochs 50 --batch-size 16 && python scripts/hf_sync.py --upload-checkpoints --upload-logs"
```

---

### 6. Resuming Training from Checkpoint

To resume training from the latest saved checkpoint in `models/raw_checkpoints` or HF bucket:

Local PowerShell:
```powershell
python scripts/DGnet.py --train --epochs 50 --batch-size 8 --resume
```

Using environment variable:
```powershell
$env:RESUME="true"; python scripts/DGnet.py --train --epochs 50
```

On Hugging Face GPU Job:
```bash
python scripts/launch_hf_job.py --flavor a100-large --epochs 50 --resume
```

---

### 7. Downloading Final Model Artifacts

Download trained model checkpoints (`dgnet_best.weights.h5` and `dgnet_best_model.keras`) locally:
```bash
python scripts/hf_sync.py --download-checkpoints
```

Evaluate the downloaded model:
```powershell
python scripts/DGnet.py --evaluate
```

---

## MLflow Experiment Tracking

MLflow acts as the centralized experiment tracking and observability layer for DGNet training runs across local workstations and remote Hugging Face GPU jobs.

### Architecture Overview

```text
Hugging Face GPU Worker / Local GPU
         │
         ├── [TensorFlow Training Loop]
         │        │
         │        ├── (Epoch Metrics & Hyperparameters) ──► MLflow Tracking Server (PostgreSQL)
         │        │                                                │
         │        └── (Config Artifacts & Run Tags)   ─────────────┘
         │
         └── [DVC & S3 Storage Gateway]
                  │
                  └── (Raw Datasets & Heavy Checkpoints) ──► Hugging Face S3 Bucket (hf-bucket)
```

> [!NOTE]
> **Separation of Concerns:**
> - **MLflow**: Tracks experiment runs, hyperparameters, per-epoch metrics, git commit SHA, hardware metadata, config artifacts, and references/URIs to checkpoints.
> - **DVC & S3**: Remains the primary source of truth for large dataset versions (`data/raw`) and heavy binary checkpoint files (`models/raw_checkpoints/*.weights.h5`). Checkpoints are **not** duplicated into MLflow.

### 1. Install Dependencies

Ensure MLflow is installed from the project requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set your remote MLflow tracking URI and optional credentials. The tracking URI is **strictly dynamic** and read from environment variables (no hard-coded IPs or secrets):

**Windows PowerShell:**
```powershell
$env:MLFLOW_TRACKING_URI="https://YOUR-MLFLOW-SERVER"
$env:MLFLOW_EXPERIMENT_NAME="COD-Camouflage-Detection"  # Optional, defaults to COD-Camouflage-Detection

# If authentication is required:
$env:MLFLOW_TRACKING_USERNAME="your-username"
$env:MLFLOW_TRACKING_PASSWORD="your-password"
# Or token authentication:
# $env:MLFLOW_TRACKING_TOKEN="your-token"
```

**Linux / macOS Bash:**
```bash
export MLFLOW_TRACKING_URI="https://YOUR-MLFLOW-SERVER"
export MLFLOW_EXPERIMENT_NAME="COD-Camouflage-Detection"
export MLFLOW_TRACKING_USERNAME="your-username"
export MLFLOW_TRACKING_PASSWORD="your-password"
```

> [!TIP]
> **Offline / Graceful Degradation:** If `MLFLOW_TRACKING_URI` is unset or the server is unreachable, training continues seamlessly in offline no-op mode without crashing or blocking.

### 3. Launch Training with Tracking

Start training as normal; MLflow hooks run automatically:
```powershell
python scripts/DGnet.py --train --epochs 50 --batch-size 8
```

To customize the experiment name or disable tracking via CLI:
```powershell
# Custom experiment name:
python scripts/DGnet.py --train --experiment-name "COD-ResNet50-Ablation"

# Disable tracking for quick local debugging:
python scripts/DGnet.py --train --no-mlflow
```

### 4. Remote Hugging Face GPU Jobs with MLflow

When launching remote Hugging Face A100/T4 jobs, `scripts/launch_hf_job.py` automatically forwards `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, and credentials to the container:
```powershell
$env:MLFLOW_TRACKING_URI="https://YOUR-MLFLOW-SERVER"
python scripts/launch_hf_job.py --flavor a100-large --epochs 50 --batch-size 16
```

### 5. Tracked Parameters & Metrics

* **Tracked Parameters:**
  - `model_name` & `model_architecture` (`DGNet`)
  - `model_version` (e.g. `v1.0`)
  - `context_backbone` (e.g. `mobilenet_v3_large`, `efficientnet-b0`)
  - `gradient_kernel` (`sobel`, `scharr`, `laplacian`)
  - `image_size` (`[384, 384]`)
  - `batch_size` & `epochs`
  - `learning_rate` & `learning_rate_scheduler` (`WarmupCosineDecay`)
  - `optimizer` (`Adam` with `clipnorm=0.5`)
  - `loss_function` (`DGNetLoss` - StructureLoss + MSE) & `lambda_grad_loss`
  - `random_seed` (`42`)
  - `dataset_name`, `dataset_path`, `train_samples`, `val_samples`
  - `augmentation_*` (horizontal flip, vertical flip, rotation 90, color jitter, gaussian noise)
  - `training_environment` (`huggingface` or local)
* **Epoch-by-Epoch Metrics:**
  - `train_total_loss`, `val_total_loss`
  - `val_iou`, `val_dice`, `val_precision`, `val_recall`, `val_f_measure`
  - `val_mae`, `val_pixel_accuracy`, `val_fpr`, `val_fnr`
* **Tags:**
  - `project=COD`, `task=camouflage_detection`, `framework=tensorflow`
  - `git_commit` (SHA auto-detected)
  - `os_platform`, `os_release`, `python_version`, `tensorflow_version`, `gpu_count`, `gpu_device_names`
* **Artifacts:**
  - `params.yaml`, `configs/data/data.yaml`, `configs/model/dgnet.yaml`, `configs/optimization/train.yaml`, `configs/deployment/huggingface.yaml`
  - `metrics/train_metrics.json`
* **Checkpoint Metadata:**
  - `checkpoint_local_path`, `checkpoint_filename`, `s3_checkpoint_uri`, `dvc_remote_store`, `best_epoch`, `best_val_loss`, `best_val_iou`, `best_val_dice`

---

## MLOps & Version Control Workflow

1. **Data & Model Versioning**: Data (`data/`) and model weights (`models/`) are managed via **DVC** and **Hugging Face Storage Bucket**, keeping the Git repository lightweight.
2. **Experiment Tracking**: **MLflow** tracks experiment runs, parameters, metrics, and references to DVC/S3 checkpoint artifacts.
3. **Edge Hardware Target**: Designed for **NVIDIA Jetson** utilizing **TensorRT (INT8)** inference engines and **TFLite**.
4. **UAV Integration**: Integration layer prepared for **MAVLink protocol** communication with autonomous flight controllers.

