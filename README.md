# Camouflaged Object Detection (COD) Framework

An ML framework for Realtime Camouflaged Target Detection based on Deep Gradient Networks (DGNet).

## Project Overview
This project implements a decoupled learning strategy for camouflaged object detection:
- **Context Branch (Searcher)**: Captures global semantic features.
- **Texture Branch (Identifier)**: Identifies fine-grained gradient-induced transitions (GIT).
- **Partial Decoder Component (PDC)**: Aggregates multi-level features to generate binary segmentation masks.

## Folder Structure
- `configs/`: YAML configuration files for dataset, model, and training setup.
- `data/`: Raw and processed dataset files (COD10K, CAMO, NC4K).
- `notebooks/`: Jupyter notebooks for data analysis & prototyping.
- `src/`: Modular Python codebase containing dataset modules, DGNet architecture, losses, and metrics.
- `scripts/`: Execution entry points for training, evaluation, and inference.
- `tests/`: Unit test suite.
- `checkpoints/`: Directory for storing model weights.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install package in editable mode:
   ```bash
   pip install -e .
   ```
3. Run training:
   ```bash
   python scripts/train.py --config configs/train.yaml
   ```
