"""
MLflow Experiment Tracking Utility for COD_PROJECT.
Pure, lightweight wrapper for MLflow tracking that integrates seamlessly
with the existing TensorFlow / DVC / S3 architecture.

Design:
- Configured strictly via environment variables (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME).
- Never hard-codes IPs, domains, or credentials.
- Graceful degradation: If MLFLOW_TRACKING_URI is not set or MLflow is unreachable,
  it logs a notice and operates in no-op mode without breaking training.
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_git_commit_sha() -> Optional[str]:
    """Safely retrieves the current Git commit SHA without throwing errors."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3
        ).decode("utf-8").strip()
        return commit if commit else None
    except Exception:
        return None


def get_hardware_info() -> Dict[str, str]:
    """Retrieves hardware, OS, and TensorFlow environment metadata for MLflow tags."""
    info = {
        "os_platform": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
    }
    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_count"] = str(len(gpus))
        if gpus:
            info["gpu_device_names"] = ", ".join([gpu.name for gpu in gpus])
    except Exception:
        pass
    return info


class MLflowTracker:
    """
    Centralized MLflow tracking helper for DGNet training runs.

    Usage:
        tracker = MLflowTracker(experiment_name="COD-Camouflage-Detection")
        tracker.start_run(run_name="DGNet-S-v1.0")
        tracker.log_params({"batch_size": 16, "lr": 0.0001})
        tracker.log_metrics({"train_loss": 0.25, "val_iou": 0.65}, step=1)
        tracker.log_artifact("params.yaml")
        tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        enabled: Optional[bool] = None,
        log_dir: Optional[str] = None,
        *args,
        **kwargs
    ):
        # Backward compatibility: older notebooks/scripts pass log_dir=...
        if log_dir is not None:
            os.environ.setdefault("LOG_DIR", str(log_dir))

        # Configure Hugging Face S3 Gateway (s3.hf.co) for boto3 / MLflow artifact storage
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            hf_token = hf_token.strip()
            s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://s3.hf.co")
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
            os.environ.setdefault("AWS_ENDPOINT_URL_S3", s3_endpoint)
            os.environ.setdefault("AWS_ACCESS_KEY_ID", "hf")
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", hf_token)
            os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

        # Priority: explicit arg > env var > default
        self.remote_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = (
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or "COD-Camouflage-Detection"
        )

        # Hugging Face S3 Gateway vs Local Fallback for artifacts
        local_artifacts_path = (PROJECT_ROOT / "metrics" / "mlflow_db" / "artifacts").resolve()
        local_artifacts_path.mkdir(parents=True, exist_ok=True)
        default_artifact_loc = "s3://gouravbirwaz/COD_dataset/mlflow-artifacts" if hf_token else local_artifacts_path.as_uri()

        self.artifact_location = (
            artifact_location
            or os.getenv("MLFLOW_ARTIFACT_LOCATION")
            or default_artifact_loc
        )

        # Respect explicit enabled=False; otherwise ALWAYS enable if mlflow is installed
        if enabled is not None:
            self.enabled = enabled and MLFLOW_AVAILABLE
        else:
            self.enabled = MLFLOW_AVAILABLE

        self.active_run = None

        if not MLFLOW_AVAILABLE:
            print("[MLflow] Notice: 'mlflow' package is not installed. Tracking disabled.")
            return

        # Build persistent local SQLite DB path (travels with the repo across platforms)
        log_dir_env = os.getenv("LOG_DIR", "metrics")
        resolved_log_dir = _resolve_path(log_dir_env)
        for cand in [Path("/data/metrics"), Path("/app/metrics"), PROJECT_ROOT / "metrics"]:
            if cand.exists() and cand.is_dir():
                resolved_log_dir = cand
                break
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        db_dir = resolved_log_dir / "mlflow_db"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / "mlflow.db"

        # Check and auto-migrate legacy alembic schema version if present
        if db_file.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_file))
                tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                if 'alembic_version' in tables:
                    ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()
                    if ver and ver[0] == 'b7e2c1a4d9f3':
                        conn.execute("UPDATE alembic_version SET version_num = '1b5f0d9ad7c1'")
                        conn.commit()
                conn.close()
            except Exception:
                pass

        # Absolute SQLite URI formatted for Linux and Windows compatibility
        posix_path = db_file.resolve().as_posix()
        if posix_path.startswith("/"):
            direct_sqlite_uri = f"sqlite:////{posix_path.lstrip('/')}"
        else:
            direct_sqlite_uri = f"sqlite:///{posix_path}"

        # Determine final tracking URI: try remote HTTP first, always fall back to local SQLite
        self.tracking_uri = direct_sqlite_uri  # Default: always persist locally
        if self.remote_tracking_uri and self.remote_tracking_uri.startswith(("http://", "https://")):
            if not any(lh in self.remote_tracking_uri for lh in ("127.0.0.1", "localhost", "0.0.0.0")):
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(self.remote_tracking_uri)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                try:
                    sock = socket.create_connection((host, port), timeout=2.0)
                    sock.close()
                    self.tracking_uri = self.remote_tracking_uri
                    print(f"[MLflow] Connected to remote tracking server: {self.remote_tracking_uri}")
                except Exception as conn_err:
                    print(f"[MLflow] Notice: Remote server at {self.remote_tracking_uri} unreachable ({conn_err}).")
                    print(f"[MLflow] Persisting tracking to local SQLite: {db_file}")

        if self.enabled:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                # Create experiment with explicit artifact location if not already existing
                try:
                    client = MlflowClient(tracking_uri=self.tracking_uri)
                    exp = client.get_experiment_by_name(self.experiment_name)
                    if exp is None:
                        client.create_experiment(
                            name=self.experiment_name,
                            artifact_location=self.artifact_location
                        )
                        print(f"[MLflow] Created experiment '{self.experiment_name}'")
                except Exception:
                    pass

                mlflow.set_experiment(self.experiment_name)
                print(f"[MLflow] Persistent tracking active at: {self.tracking_uri}")
                print(f"[MLflow] Experiment: '{self.experiment_name}' | Artifacts: {self.artifact_location}")
            except Exception as e:
                print(f"[MLflow] Warning: Failed to initialize tracking ({e}). Tracking disabled for this session.")
                self.enabled = False

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """Starts an MLflow run with standard tags."""
        if not self.enabled:
            return None

        default_tags = {
            "project": "COD",
            "task": "camouflage_detection",
            "framework": "tensorflow",
            "training_platform": os.getenv("TRAINING_PLATFORM", "huggingface"),
            "model_stage": "segmentation",
            "s3_storage_bucket": "s3://COD_dataset/dvc-store",
            "s3_artifact_location": self.artifact_location,
        }
        default_tags.update(get_hardware_info())

        git_sha = get_git_commit_sha()
        if git_sha:
            default_tags["git_commit"] = git_sha

        if tags:
            default_tags.update(tags)

        try:
            self.active_run = mlflow.start_run(
                run_name=run_name,
                tags=default_tags,
                description=description
            )
            print(f"[MLflow] Started Run ID: {self.active_run.info.run_id} (Name: '{run_name}')")
            return self.active_run
        except Exception as e:
            print(f"[MLflow] Warning: Could not start run ({e}). Disabling tracker for this session.")
            self.enabled = False
            return None

    def log_params(self, params: Dict[str, Any]):
        """Logs configuration parameters and hyperparameters to the active run."""
        if not self.enabled or not self.active_run:
            return

        # Flatten or stringify complex objects to stay within MLflow limits
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                clean_params[k] = v
            elif isinstance(v, (list, tuple)):
                clean_params[k] = str(v)
            elif v is None:
                clean_params[k] = "None"
            else:
                clean_params[k] = str(v)

        try:
            mlflow.log_params(clean_params)
        except Exception as e:
            print(f"[MLflow] Warning: Failed to log params ({e})")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs epoch-by-epoch training and validation metrics."""
        if not self.enabled or not self.active_run:
            return

        clean_metrics = {}
        for k, v in metrics.items():
            try:
                clean_metrics[k] = float(v)
            except (ValueError, TypeError):
                continue

        try:
            mlflow.log_metrics(clean_metrics, step=step)
        except Exception as e:
            print(f"[MLflow] Warning: Failed to log metrics at step {step} ({e})")

    def log_tags(self, tags: Dict[str, Any]):
        """Sets or updates tags on the active run."""
        if not self.enabled or not self.active_run:
            return

        clean_tags = {str(k): str(v) for k, v in tags.items()}
        try:
            mlflow.set_tags(clean_tags)
        except Exception as e:
            print(f"[MLflow] Warning: Failed to log tags ({e})")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Logs a local file or directory as an MLflow artifact."""
        if not self.enabled or not self.active_run:
            return

        p = Path(local_path)
        if not p.exists():
            return

        try:
            if p.is_dir():
                mlflow.log_artifacts(str(p), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(p), artifact_path=artifact_path)
        except Exception as e:
            print(f"[MLflow] Warning: Failed to log artifact '{local_path}' ({e})")

    def log_configs(self, config_paths: List[str], artifact_path: str = "configs"):
        """Logs configuration files (YAML, JSON) to reproduce the experiment."""
        if not self.enabled or not self.active_run:
            return

        for cp in config_paths:
            self.log_artifact(cp, artifact_path=artifact_path)

    def log_checkpoint_metadata(
        self,
        best_weights_path: Optional[str] = None,
        best_epoch: Optional[int] = None,
        best_val_loss: Optional[float] = None,
        best_val_iou: Optional[float] = None,
        best_val_dice: Optional[float] = None,
        s3_bucket_uri: Optional[str] = None,
        dvc_remote: Optional[str] = None,
    ):
        """
        Logs references to existing DVC/S3 checkpoint locations without duplicating
        heavy binary files directly into MLflow.
        """
        if not self.enabled or not self.active_run:
            return

        metadata = {}
        if best_weights_path:
            metadata["checkpoint_local_path"] = str(best_weights_path)
            metadata["checkpoint_filename"] = Path(best_weights_path).name
        if best_epoch is not None:
            metadata["best_epoch"] = str(best_epoch)
        if best_val_loss is not None:
            metadata["best_val_loss"] = f"{best_val_loss:.4f}"
        if best_val_iou is not None:
            metadata["best_val_iou"] = f"{best_val_iou:.4f}"
        if best_val_dice is not None:
            metadata["best_val_dice"] = f"{best_val_dice:.4f}"
        if s3_bucket_uri:
            metadata["s3_checkpoint_uri"] = str(s3_bucket_uri)
        if dvc_remote:
            metadata["dvc_remote_store"] = str(dvc_remote)

        self.log_tags(metadata)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = "DGNet-Camouflage-Detector"
    ):
        """
        Logs a TensorFlow/Keras model artifact and registers it in the MLflow Model Registry.
        Standard production practice for ML Engineers.
        """
        if not self.enabled or not self.active_run:
            return

        try:
            import mlflow.tensorflow
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
            print(f"[MLflow] Logged & Registered model '{registered_model_name}' under artifact path '{artifact_path}'.")
        except Exception as e:
            print(f"[MLflow] Notice: Could not register model with MLflow flavor ({e}). Logging as raw artifact...")
            try:
                if hasattr(model, "save"):
                    temp_path = PROJECT_ROOT / "metrics" / "temp_dgnet.keras"
                    model.save(str(temp_path))
                    self.log_artifact(str(temp_path), artifact_path=artifact_path)
            except Exception as save_err:
                print(f"[MLflow] Warning: Model artifact logging fallback failed: {save_err}")

    def log_model_summary(self, model: Any):
        """Logs the text summary of the model architecture (layers, parameters, shapes)."""
        if not self.enabled or not self.active_run:
            return

        try:
            summary_path = PROJECT_ROOT / "metrics" / "model_summary.txt"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
            self.log_artifact(str(summary_path), artifact_path="architecture")
            print("[MLflow] Logged model architecture summary artifact.")
        except Exception as e:
            print(f"[MLflow] Warning: Failed to log model summary ({e})")

    def log_plots(self, plots_dir: Optional[str] = None):
        """Logs evaluation metric plots, confusion matrices, and IoU curves to MLflow UI."""
        if not self.enabled or not self.active_run:
            return

        p_dir = Path(plots_dir) if plots_dir else (PROJECT_ROOT / "metrics")
        if not p_dir.exists():
            return

        for plot_file in p_dir.glob("*.png"):
            self.log_artifact(str(plot_file), artifact_path="plots")

    def log_quantized_models(self, quantized_dir: Optional[str] = None):
        """Logs TFLite quantized edge models (Int8/FP16) as production deployment artifacts."""
        if not self.enabled or not self.active_run:
            return

        q_dir = Path(quantized_dir) if quantized_dir else (PROJECT_ROOT / "models" / "quantized")
        if not q_dir.exists():
            for cand in [Path("/data/models/quantized"), Path("/app/models/quantized")]:
                if cand.exists():
                    q_dir = cand
                    break

        if q_dir.exists():
            for tflite_file in q_dir.glob("*.tflite"):
                self.log_artifact(str(tflite_file), artifact_path="quantized_models")
                print(f"[MLflow] Logged edge model artifact: {tflite_file.name}")

    def end_run(self, status: str = "FINISHED"):
        """Ends the active MLflow run."""
        if not self.enabled or not self.active_run:
            return

        try:
            mlflow.end_run(status=status)
            print(f"[MLflow] Completed Run ID: {self.active_run.info.run_id} (Status: {status})")
        except Exception as e:
            print(f"[MLflow] Warning: Failed to end run ({e})")
        finally:
            self.active_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.end_run(status=status)
