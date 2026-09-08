"""
Central Configuration Manager for COD_PROJECT.
Parses modular YAML configuration files inside `configs/` and applies runtime environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Resolve project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot notation access."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                return ConfigDict(val)
            return val
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely loads a YAML configuration file."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    return {}


def load_project_config(config_dir: str = None) -> ConfigDict:
    """
    Parses structured YAML files in configs/ directory and merges them into a unified ConfigDict.
    
    Structure:
    - configs/data/data.yaml -> config.data
    - configs/model/dgnet.yaml -> config.model
    - configs/optimization/train.yaml -> config.train
    - configs/deployment/huggingface.yaml -> config.huggingface
    
    Applies runtime environment variable overrides for paths and secrets.
    """
    base_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "configs"

    data_cfg = load_yaml(base_dir / "data" / "data.yaml")
    model_cfg = load_yaml(base_dir / "model" / "dgnet.yaml")
    train_cfg = load_yaml(base_dir / "optimization" / "train.yaml")
    hf_cfg = load_yaml(base_dir / "deployment" / "huggingface.yaml")

    if os.getenv("MODEL_VERSION"):
        train_cfg["model_version"] = os.getenv("MODEL_VERSION")
        model_cfg["version"] = os.getenv("MODEL_VERSION")
    if os.getenv("DATA_DIR"):
        data_cfg["raw_data_dir"] = os.getenv("DATA_DIR")
    if os.getenv("CHECKPOINT_DIR"):
        train_cfg["checkpoint_dir"] = os.getenv("CHECKPOINT_DIR")
    if os.getenv("LOG_DIR"):
        train_cfg["log_dir"] = os.getenv("LOG_DIR")
    if os.getenv("MODEL_DIR"):
        train_cfg["model_dir"] = os.getenv("MODEL_DIR")
    if os.getenv("RESUME"):
        train_cfg["resume"] = os.getenv("RESUME").lower() in ("true", "1", "t", "yes")

    if os.getenv("HF_REPO_ID"):
        hf_cfg["repo_id"] = os.getenv("HF_REPO_ID")
    if os.getenv("HF_SPACE_ID"):
        hf_cfg["space_id"] = os.getenv("HF_SPACE_ID")
    if os.getenv("HF_TOKEN"):
        hf_cfg["token"] = os.getenv("HF_TOKEN")

    unified = {
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg,
        "huggingface": hf_cfg,
    }

    return ConfigDict(unified)
