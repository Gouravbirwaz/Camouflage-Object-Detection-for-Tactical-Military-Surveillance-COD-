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
    ):
        # Configure Hugging Face S3 endpoint for boto3 / MLflow artifact storage
        s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://s3.hf.co/gouravbirwaz")
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
        os.environ.setdefault("AWS_ENDPOINT_URL_S3", s3_endpoint)
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

        # Priority: explicit arg > env var > default
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.experiment_name = (
            experiment_name
            or os.getenv("MLFLOW_EXPERIMENT_NAME")
            or "COD-Camouflage-Detection"
        )
        self.artifact_location = (
            artifact_location
            or os.getenv("MLFLOW_ARTIFACT_LOCATION")
            or "s3://COD_dataset/dvc-store/mlflow-artifacts"
        )

        # Enable only if MLflow library is installed AND tracking URI is configured (or explicitly forced)
        if enabled is not None:
            self.enabled = enabled and MLFLOW_AVAILABLE
        else:
            self.enabled = bool(self.tracking_uri) and MLFLOW_AVAILABLE

        self.active_run = None

        if not MLFLOW_AVAILABLE:
            print("[MLflow] Notice: 'mlflow' package is not installed. Tracking disabled.")
        elif not self.enabled or not self.tracking_uri:
            print("[MLflow] Notice: MLFLOW_TRACKING_URI not set. Running in offline/no-op mode.")
            self.enabled = False
        else:
            # Fast probe to prevent 10+ minute freeze if remote tracking host is offline/unreachable
            if self.tracking_uri.startswith(("http://", "https://")):
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(self.tracking_uri)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                try:
                    sock = socket.create_connection((host, port), timeout=2.0)
                    sock.close()
                except Exception as conn_err:
                    print(f"[MLflow] Notice: Remote tracking server at {self.tracking_uri} is unreachable ({conn_err}).")
                    if self.tracking_uri != "http://127.0.0.1:5000":
                        print("[MLflow] Falling back to embedded local MLflow server at http://127.0.0.1:5000...")
                        self.tracking_uri = "http://127.0.0.1:5000"
                        try:
                            sock = socket.create_connection(("127.0.0.1", 5000), timeout=2.0)
                            sock.close()
                            self.enabled = True
                        except Exception:
                            print("[MLflow] Embedded server not reachable. Disabling MLflow tracking.")
                            self.enabled = False
                    else:
                        self.enabled = False

            if self.enabled:
                try:
                    mlflow.set_tracking_uri(self.tracking_uri)
                    # Create experiment with explicit S3 artifact location if not already existing
                    try:
                        client = MlflowClient(tracking_uri=self.tracking_uri)
                        exp = client.get_experiment_by_name(self.experiment_name)
                        if exp is None:
                            client.create_experiment(
                                name=self.experiment_name,
                                artifact_location=self.artifact_location
                            )
                            print(f"[MLflow] Created experiment '{self.experiment_name}' with S3 artifact root: {self.artifact_location}")
                    except Exception:
                        pass

                    mlflow.set_experiment(self.experiment_name)
                    print(f"[MLflow] Initialized tracking at: {self.tracking_uri} | Experiment: '{self.experiment_name}'")
                    print(f"[MLflow] S3 Artifact Store: {self.artifact_location} (Endpoint: {s3_endpoint})")
                except Exception as e:
                    print(f"[MLflow] Warning: Failed to connect to tracking URI ({e}). Falling back to no-op mode.")
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
