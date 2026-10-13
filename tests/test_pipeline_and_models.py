"""Unit and integration tests for baseline ML models and evaluation reporting."""

import json
import joblib
import pandas as pd

from src.config import load_config
from src.evaluate import evaluate_single_model


def test_trained_models_exist_and_predict():
    """Verify saved models load correctly and produce valid predictions."""
    cfg = load_config()
    dt_path = cfg.paths.models_dir / "decision_tree.joblib"
    rf_path = cfg.paths.models_dir / "random_forest.joblib"

    assert dt_path.exists(), "Decision Tree model artifact must exist"
    assert rf_path.exists(), "Random Forest model artifact must exist"

    dt_model = joblib.load(dt_path)
    rf_model = joblib.load(rf_path)

    # Load test data
    test_df = pd.read_parquet(cfg.paths.processed_test_path)
    X_test = test_df.drop(columns=[cfg.data.target_column])

    dt_preds = dt_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)

    assert len(dt_preds) == len(X_test)
    assert len(rf_preds) == len(X_test)
    assert set(dt_preds).issubset({0, 1, 2, 3, 4})
    assert set(rf_preds).issubset({0, 1, 2, 3, 4})


def test_model_evaluation_metrics():
    """Verify evaluation metrics computation satisfies expected baselines."""
    cfg = load_config()
    rf_path = cfg.paths.models_dir / "random_forest.joblib"
    rf_model = joblib.load(rf_path)

    meta_path = cfg.paths.features_meta_path
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = {int(k): v for k, v in meta["classes"].items()}

    test_df = pd.read_parquet(cfg.paths.processed_test_path)
    X_test = test_df.drop(columns=[cfg.data.target_column])
    y_test = test_df[cfg.data.target_column].values

    metrics = evaluate_single_model(rf_model, X_test, y_test, class_names)

    assert metrics["accuracy"] > 0.80, "Random Forest baseline should exceed 80% accuracy"
    assert metrics["f1_macro"] > 0.80, "Random Forest macro F1 should exceed 0.80"
    assert len(metrics["confusion_matrix"]) == 5


def test_dvc_metrics_and_evaluation_report_files():
    """Verify DVC metrics.json and Markdown evaluation report are generated."""
    cfg = load_config()
    assert cfg.paths.metrics_path.exists(), "metrics.json must be generated"
    assert cfg.paths.evaluation_report_path.exists(), "phase1_evaluation.md must be generated"

    with open(cfg.paths.metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "decision_tree" in metrics
    assert "random_forest" in metrics
    assert "champion_model" in metrics
    assert metrics["random_forest"]["accuracy"] > 0.85
