# Edge-Optimized ML Framework for Realtime Camouflaged Target Detection

A professional, production-grade Machine Learning directory structure designed for **Realtime Camouflaged Target Detection in Tactical Military Surveillance Systems** deployed on **NVIDIA Jetson** edge devices.

---

## Directory Architecture

```
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
│   ├── raw_checkpoints/           # Trained PyTorch .pt weights
│   ├── pruned/                    # Structurally pruned weights
│   ├── quantized/                 # INT8 quantized PyTorch / ONNX models
│   ├── onnx/                      # Exported ONNX graph files
│   └── tensorrt/                  # Optimized TensorRT .engine files for NVIDIA Jetson
├── notebooks/                     # Exploratory analysis & prototyping notebooks
├── requirements/                  # Modular requirement specifications
├── scripts/                       # Execution scripts (data prep, train, prune, quantize, export, build_tensorrt, run_inference)
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
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── edge/                      # Jetson hardware compatibility tests
├── .dvcignore                     # DVC ignore rules
├── .gitignore                     # Production Git ignore configuration
├── dvc.yaml                       # DVC Pipeline definition template
├── params.yaml                    # DVC Hyperparameters configuration file
└── requirements.txt               # Main requirements wrapper
```

---

## MLOps & Version Control Workflow

1. **Data & Model Versioning**: Data (`data/`) and model weights (`models/`) are managed via **DVC**, keeping the Git repository lightweight.
2. **Edge Hardware Target**: Designed for **NVIDIA Jetson** utilizing **TensorRT (INT8)** inference engines.
3. **UAV Integration**: Integration layer prepared for **MAVLink protocol** communication with autonomous flight controllers.
