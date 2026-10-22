"""
Model Training Module with MLflow Experiment Tracking (Phase 1 Rework).

Binary classification task: predict whether a student will pass the online
technical assessment to qualify for the interview round.

Trains three classifiers:
  1. Decision Tree (interpretable baseline)
  2. Random Forest  (ensemble champion)
  3. Logistic Regression (linear baseline)

All hyperparameters are intentionally constrained (via configs/config.yaml)
to target 70-80% test accuracy — reflecting realistic assessment uncertainty.
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import AppConfig, load_config


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 content hash for reproducibility provenance tracking."""
    if not file_path.exists():
        return "not_found"
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def train_models(config: Optional[AppConfig] = None) -> Dict[str, Any]:
    """
    Train Decision Tree, Random Forest, and Logistic Regression models with MLflow tracking.

    Returns a summary dict containing run_id, artifact path, and train accuracy per model.
    """
    if config is None:
        config = load_config()

    train_path = config.paths.processed_train_path
    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed training data not found at {train_path}. Run feature engineering first."
        )

    logger.info(f"Loading processed training data from {train_path}...")
    train_df = pd.read_parquet(train_path)

    target_col = config.data.target_column  # "assessment_pass"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values.astype(int)

    n_pass = int(y_train.sum())
    n_fail = len(y_train) - n_pass
    logger.info(
        f"Training set | N={len(y_train)} | Pass: {n_pass} ({n_pass/len(y_train)*100:.1f}%) "
        f"| Fail: {n_fail} ({n_fail/len(y_train)*100:.1f}%) | Features: {X_train.shape[1]}"
    )

    dataset_hash = compute_file_sha256(config.paths.raw_data_path)
    logger.info(f"Raw dataset SHA-256: {dataset_hash[:16]}...")

    # Configure MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    logger.info(
        f"MLflow experiment: '{config.mlflow.experiment_name}' @ '{config.mlflow.tracking_uri}'"
    )

    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    models_summary: Dict[str, Any] = {}

    # ── Shared MLflow metadata ──────────────────────────────────────────
    shared_params = {
        "dataset_sha256": dataset_hash,
        "num_features": X_train.shape[1],
        "train_samples": len(X_train),
        "task": "binary_classification",
        "target": "assessment_pass",
    }

    # ── 1. Decision Tree ────────────────────────────────────────────────
    dt_cfg = config.models.decision_tree
    dt_params = {
        "max_depth": dt_cfg.max_depth,
        "min_samples_split": dt_cfg.min_samples_split,
        "min_samples_leaf": dt_cfg.min_samples_leaf,
        "criterion": dt_cfg.criterion,
        "random_state": config.models.random_state,
    }
    logger.info(f"Training Decision Tree | params: {dt_params}")
    with mlflow.start_run(run_name="Decision_Tree_Binary") as dt_run:
        dt_model = DecisionTreeClassifier(**dt_params)
        dt_model.fit(X_train, y_train)

        train_acc = float(dt_model.score(X_train, y_train))
        logger.info(f"Decision Tree Train Accuracy: {train_acc:.4f}")

        mlflow.log_params({**dt_params, "model_name": "DecisionTreeClassifier", **shared_params})
        mlflow.log_metric("train_accuracy", train_acc)

        dt_path = config.paths.models_dir / "decision_tree.joblib"
        joblib.dump(dt_model, dt_path)
        mlflow.sklearn.log_model(dt_model, artifact_path="model")

        models_summary["decision_tree"] = {
            "run_id": dt_run.info.run_id,
            "artifact_path": str(dt_path),
            "train_accuracy": train_acc,
        }

    # ── 2. Random Forest ────────────────────────────────────────────────
    rf_cfg = config.models.random_forest
    rf_params = {
        "n_estimators": rf_cfg.n_estimators,
        "max_depth": rf_cfg.max_depth,
        "max_features": rf_cfg.max_features,
        "min_samples_split": rf_cfg.min_samples_split,
        "min_samples_leaf": rf_cfg.min_samples_leaf,
        "criterion": rf_cfg.criterion,
        "random_state": config.models.random_state,
        "n_jobs": rf_cfg.n_jobs,
    }
    logger.info(f"Training Random Forest | params: {rf_params}")
    with mlflow.start_run(run_name="Random_Forest_Binary") as rf_run:
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)

        train_acc = float(rf_model.score(X_train, y_train))
        logger.info(f"Random Forest Train Accuracy: {train_acc:.4f}")

        mlflow.log_params({**rf_params, "model_name": "RandomForestClassifier", **shared_params})
        mlflow.log_metric("train_accuracy", train_acc)

        rf_path = config.paths.models_dir / "random_forest.joblib"
        joblib.dump(rf_model, rf_path)
        mlflow.sklearn.log_model(rf_model, artifact_path="model")

        models_summary["random_forest"] = {
            "run_id": rf_run.info.run_id,
            "artifact_path": str(rf_path),
            "train_accuracy": train_acc,
        }

    # ── 3. Logistic Regression ──────────────────────────────────────────
    lr_cfg = config.models.logistic_regression
    lr_params = {
        "C": lr_cfg.C,
        "max_iter": lr_cfg.max_iter,
        "solver": lr_cfg.solver,
        "random_state": config.models.random_state,
    }
    logger.info(f"Training Logistic Regression | params: {lr_params}")
    with mlflow.start_run(run_name="Logistic_Regression_Binary") as lr_run:
        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(X_train, y_train)

        train_acc = float(lr_model.score(X_train, y_train))
        logger.info(f"Logistic Regression Train Accuracy: {train_acc:.4f}")

        mlflow.log_params({**lr_params, "model_name": "LogisticRegression", **shared_params})
        mlflow.log_metric("train_accuracy", train_acc)

        lr_path = config.paths.models_dir / "logistic_regression.joblib"
        joblib.dump(lr_model, lr_path)
        mlflow.sklearn.log_model(lr_model, artifact_path="model")

        models_summary["logistic_regression"] = {
            "run_id": lr_run.info.run_id,
            "artifact_path": str(lr_path),
            "train_accuracy": train_acc,
        }

    # Persist summary
    summary_path = config.paths.models_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(models_summary, f, indent=2)

    logger.success(f"All 3 models saved under '{config.paths.models_dir}'.")
    logger.success(f"Training summary written to {summary_path}")
    return models_summary


if __name__ == "__main__":
    cfg = load_config()
    summary = train_models(cfg)
    print("\n=== Training Summary ===")
    for model_name, info in summary.items():
        print(f"{model_name}: train_acc={info['train_accuracy']:.4f} | run_id={info['run_id'][:8]}")
