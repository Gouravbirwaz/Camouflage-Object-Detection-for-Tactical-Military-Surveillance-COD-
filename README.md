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
├── scripts/                       # Execution scripts (DGnet.py, hf_sync.py, launch_hf_job.py)
├── src/                           # Core Python source package (`cod_framework`)
│   ├── api/                       # FastAPI Command & Control (C2) service
│   ├── data/                      # Data loaders & DVC pipelines
│   ├── inference/                 # High-performance TensorRT edge inference engine
│   ├── integration/               # MAVLink UAV flight controller communication handler
│   ├── models/                    # DGNet architecture (Context Branch, Texture Branch, PDC Decoder)
│   ├── optimization/              # Pruning, INT8 Quantization, Distillation modules
│   ├── utils/                     # Loss functions (BCE + Gradient Loss) & metrics
│   └── visualization/             # Video stream rendering, mask overlays & bounding boxes
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

## MLOps & Version Control Workflow

1. **Data & Model Versioning**: Data (`data/`) and model weights (`models/`) are managed via **DVC** and **Hugging Face Storage Bucket**, keeping the Git repository lightweight.
2. **Edge Hardware Target**: Designed for **NVIDIA Jetson** utilizing **TensorRT (INT8)** inference engines and **TFLite**.
3. **UAV Integration**: Integration layer prepared for **MAVLink protocol** communication with autonomous flight controllers.
