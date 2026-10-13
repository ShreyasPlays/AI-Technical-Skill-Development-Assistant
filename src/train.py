"""
Model Training Module with MLflow Experiment Tracking.
Trains Decision Tree and Random Forest classifiers on the processed skills dataset.
Logs hyperparameters, metrics, dataset hashes, and model artifacts to MLflow.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig, load_config


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 content hash of a dataset for strict reproducibility provenance."""
    if not file_path.exists():
        return "not_found"
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def train_models(config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """
    Train Decision Tree and Random Forest models with MLflow experiment tracking.
    """
    if config is None:
        config = load_config()

    train_path = config.paths.processed_train_path
    if not train_path.exists():
        raise FileNotFoundError(f"Processed training data not found at {train_path}. Run feature engineering first.")

    logger.info(f"Loading processed training data from {train_path}...")
    train_df = pd.read_parquet(train_path)

    target_col = config.data.target_column
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    dataset_hash = compute_file_sha256(config.paths.raw_data_path)
    logger.info(f"Raw dataset SHA-256 hash: {dataset_hash[:16]}...")

    # Configure MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(f"MLflow experiment configured: '{config.mlflow.experiment_name}' at '{config.mlflow.tracking_uri}'")

    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    models_summary: Dict[str, Any] = {}

    # ----------------------------------------------------
    # 1. Decision Tree Classifier
    # ----------------------------------------------------
    dt_cfg = config.models.decision_tree
    dt_params = {
        "max_depth": dt_cfg.max_depth,
        "min_samples_split": dt_cfg.min_samples_split,
        "min_samples_leaf": dt_cfg.min_samples_leaf,
        "criterion": dt_cfg.criterion,
        "random_state": config.models.random_state,
    }

    logger.info(f"Training Decision Tree with params: {dt_params}")
    with mlflow.start_run(run_name="Decision_Tree_Baseline") as dt_run:
        dt_model = DecisionTreeClassifier(**dt_params)
        dt_model.fit(X_train, y_train)

        train_acc = float(dt_model.score(X_train, y_train))
        logger.info(f"Decision Tree Train Accuracy: {train_acc:.4f}")

        # Log to MLflow
        mlflow.log_params(dt_params)
        mlflow.log_param("model_name", "DecisionTreeClassifier")
        mlflow.log_param("dataset_sha256", dataset_hash)
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("train_accuracy", train_acc)

        # Save local model artifact
        dt_model_path = config.paths.models_dir / "decision_tree.joblib"
        joblib.dump(dt_model, dt_model_path)
        mlflow.sklearn.log_model(dt_model, artifact_path="model")

        models_summary["decision_tree"] = {
            "run_id": dt_run.info.run_id,
            "artifact_path": str(dt_model_path),
            "train_accuracy": train_acc,
        }

    # ----------------------------------------------------
    # 2. Random Forest Classifier
    # ----------------------------------------------------
    rf_cfg = config.models.random_forest
    rf_params = {
        "n_estimators": rf_cfg.n_estimators,
        "max_depth": rf_cfg.max_depth,
        "min_samples_split": rf_cfg.min_samples_split,
        "min_samples_leaf": rf_cfg.min_samples_leaf,
        "criterion": rf_cfg.criterion,
        "random_state": config.models.random_state,
        "n_jobs": rf_cfg.n_jobs,
    }

    logger.info(f"Training Random Forest with params: {rf_params}")
    with mlflow.start_run(run_name="Random_Forest_Baseline") as rf_run:
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)

        train_acc = float(rf_model.score(X_train, y_train))
        logger.info(f"Random Forest Train Accuracy: {train_acc:.4f}")

        # Log to MLflow
        mlflow.log_params(rf_params)
        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("dataset_sha256", dataset_hash)
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_metric("train_accuracy", train_acc)

        # Save local model artifact
        rf_model_path = config.paths.models_dir / "random_forest.joblib"
        joblib.dump(rf_model, rf_model_path)
        mlflow.sklearn.log_model(rf_model, artifact_path="model")

        models_summary["random_forest"] = {
            "run_id": rf_run.info.run_id,
            "artifact_path": str(rf_model_path),
            "train_accuracy": train_acc,
        }

    # Persist summary
    summary_path = config.paths.models_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(models_summary, f, indent=2)

    logger.success(f"Models saved under {config.paths.models_dir}. Summary written to {summary_path}")
    return models_summary


if __name__ == "__main__":
    cfg = load_config()
    train_models(cfg)
