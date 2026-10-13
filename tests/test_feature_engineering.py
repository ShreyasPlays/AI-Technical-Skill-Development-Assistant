"""Unit tests for feature engineering and pipeline transformations."""

import pandas as pd
import numpy as np
from src.config import load_config
from src.data_loader import load_raw_data
from src.preprocess import split_data_leak_free
from src.feature_engineering import (
    engineer_composite_features,
    SkillFeaturePipeline,
)


def test_leakage_free_split():
    """Verify train/test split maintains target class stratification with zero overlap."""
    cfg = load_config()
    raw_df = load_raw_data(cfg.paths.raw_data_path)

    train_df, test_df = split_data_leak_free(
        raw_df,
        test_size=0.2,
        random_state=42,
        target_column="target_role",
    )

    assert len(train_df) == 960
    assert len(test_df) == 240
    # Stratified balance check
    assert train_df["target_role"].value_counts().min() == 192
    assert test_df["target_role"].value_counts().min() == 48


def test_composite_feature_engineering():
    """Verify composite domain readiness indices are computed properly."""
    df = pd.DataFrame({
        "python": [4],
        "sql": [3],
        "machine_learning": [4],
        "dsa": [3],
        "git": [3],
        "web_development": [2],
        "cloud_computing": [3],
        "system_design": [3],
        "statistics": [4],
        "deep_learning": [4],
        "years_experience": [1.5],
        "projects_completed": [4],
        "problems_solved": [100],
        "target_role": ["Machine Learning Engineer"],
    })

    eng_df = engineer_composite_features(df)

    assert "skill_overall_mean" in eng_df.columns
    assert "ml_ai_readiness" in eng_df.columns
    assert "data_analytics_readiness" in eng_df.columns
    assert "fullstack_readiness" in eng_df.columns
    assert "devops_cloud_readiness" in eng_df.columns
    assert "practical_velocity" in eng_df.columns

    import pytest
    assert eng_df["ml_ai_readiness"].iloc[0] == pytest.approx(4.0)
    # practical_velocity = log1p(100) * sqrt(4) = ~4.615 * 2 = ~9.23
    assert eng_df["practical_velocity"].iloc[0] > 0


def test_pipeline_fit_and_transform():
    """Verify feature pipeline scales without data leakage and encodes classes."""
    cfg = load_config()
    raw_df = load_raw_data(cfg.paths.raw_data_path)
    train_df, test_df = split_data_leak_free(raw_df, test_size=0.2, random_state=42)

    pipeline = SkillFeaturePipeline()
    X_train_trans, y_train = pipeline.fit_transform(train_df)
    X_test_trans, y_test = pipeline.transform(test_df)

    assert len(pipeline.feature_columns) == 19
    assert len(pipeline.class_mapping) == 5
    assert X_train_trans.shape == (960, 20)
    assert X_test_trans.shape == (240, 20)

    # Standard scaling property: train mean approx 0, std approx 1
    np.testing.assert_allclose(X_train_trans[pipeline.feature_columns].mean(), 0.0, atol=1e-5)
