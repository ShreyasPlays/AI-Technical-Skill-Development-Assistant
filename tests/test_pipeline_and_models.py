"""Integration tests for binary ML models and evaluation reporting (Phase 1 Rework)."""

import json
import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluate import evaluate_single_model


def test_trained_models_exist_and_predict_binary():
    """Verify all 3 saved models load and produce valid binary predictions (0/1)."""
    cfg = load_config()
    model_paths = {
        "decision_tree":       cfg.paths.models_dir / "decision_tree.joblib",
        "random_forest":       cfg.paths.models_dir / "random_forest.joblib",
        "logistic_regression": cfg.paths.models_dir / "logistic_regression.joblib",
    }

    for model_name, path in model_paths.items():
        assert path.exists(), f"{model_name} artifact not found at {path}. Run 'dvc repro' first."

    test_df = pd.read_parquet(cfg.paths.processed_test_path)
    X_test = test_df.drop(columns=[cfg.data.target_column])

    for model_name, path in model_paths.items():
        model = joblib.load(path)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), f"{model_name}: prediction length mismatch."
        assert set(preds).issubset({0, 1}), (
            f"{model_name}: predictions must be binary (0=Fail, 1=Pass). Got: {set(preds)}"
        )

    # Verify predict_proba is available on all models (required for ROC-AUC)
    for model_name, path in model_paths.items():
        model = joblib.load(path)
        assert hasattr(model, "predict_proba"), (
            f"{model_name} must support predict_proba for ROC-AUC computation."
        )
        probs = model.predict_proba(X_test)
        assert probs.shape == (len(X_test), 2), (
            f"{model_name}: predict_proba must return shape (n_samples, 2). Got: {probs.shape}"
        )
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_model_evaluation_metrics():
    """Verify binary evaluation metrics meet expected quality bar for the champion model."""
    cfg = load_config()
    rf_path = cfg.paths.models_dir / "random_forest.joblib"
    rf_model = joblib.load(rf_path)

    test_df = pd.read_parquet(cfg.paths.processed_test_path)
    X_test = test_df.drop(columns=[cfg.data.target_column])
    y_test = test_df[cfg.data.target_column].values.astype(int)

    metrics = evaluate_single_model(rf_model, X_test, y_test, model_name="random_forest")

    # Accuracy must be in the realistic 70-80% target range for binary assessment prediction
    assert metrics["accuracy"] >= 0.65, (
        f"Random Forest accuracy should be ≥65%. Got: {metrics['accuracy']:.4f}"
    )
    assert metrics["accuracy"] <= 0.95, (
        f"Accuracy >95% suggests overfitting or label leakage. Got: {metrics['accuracy']:.4f}"
    )

    # ROC-AUC should be meaningfully above random (0.5)
    assert metrics["roc_auc"] is not None, "ROC-AUC must be computed for the Random Forest."
    assert metrics["roc_auc"] > 0.65, (
        f"ROC-AUC should be >0.65 for a meaningful binary classifier. Got: {metrics['roc_auc']:.4f}"
    )

    # F1-Score should be reasonable
    assert metrics["f1_score"] > 0.60, (
        f"F1-Score should be >0.60. Got: {metrics['f1_score']:.4f}"
    )

    # Confusion matrix is 2×2 for binary
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_dvc_metrics_and_evaluation_report_files():
    """Verify DVC metrics.json and Markdown evaluation report are generated with correct structure."""
    cfg = load_config()

    assert cfg.paths.metrics_path.exists(), (
        f"metrics.json not found at {cfg.paths.metrics_path}. Run 'dvc repro' first."
    )
    assert cfg.paths.evaluation_report_path.exists(), (
        f"Evaluation report not found at {cfg.paths.evaluation_report_path}."
    )

    with open(cfg.paths.metrics_path) as f:
        metrics = json.load(f)

    # All 3 models must be present
    assert "decision_tree" in metrics
    assert "random_forest" in metrics
    assert "logistic_regression" in metrics
    assert "champion_model" in metrics

    # Binary metric keys for each model
    expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    for model_name in ["decision_tree", "random_forest", "logistic_regression"]:
        model_metrics = metrics[model_name]
        for key in expected_keys:
            assert key in model_metrics, (
                f"metrics.json['{model_name}'] missing key '{key}'."
            )
        assert 0.0 <= model_metrics["accuracy"] <= 1.0
        assert 0.0 <= model_metrics["f1_score"] <= 1.0
