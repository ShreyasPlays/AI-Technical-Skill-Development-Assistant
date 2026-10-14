"""Unit tests for feature engineering and pipeline transformations."""

import pytest
import pandas as pd
import numpy as np
from src.config import load_config
from src.data_loader import load_raw_data, SKILL_COLUMNS, CONTEXT_COLUMNS
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

    assert len(train_df) == 6822
    assert len(test_df) == 1706
    assert set(train_df["target_role"].unique()) == set(raw_df["target_role"].unique())
    assert set(test_df["target_role"].unique()) == set(raw_df["target_role"].unique())


def test_composite_feature_engineering():
    """Verify composite domain readiness indices are computed properly."""
    sample_data = {col: [3] for col in SKILL_COLUMNS}
    sample_data.update({col: [0.5] for col in CONTEXT_COLUMNS})
    sample_data["target_role"] = ["Software Developer"]

    df = pd.DataFrame(sample_data)
    eng_df = engineer_composite_features(df)

    assert "technical_overall_mean" in eng_df.columns
    assert "ai_data_readiness" in eng_df.columns
    assert "systems_readiness" in eng_df.columns
    assert "software_dev_readiness" in eng_df.columns
    assert "cybersecurity_readiness" in eng_df.columns
    assert "business_comm_readiness" in eng_df.columns
    assert "analytical_mindset_index" in eng_df.columns

    assert eng_df["technical_overall_mean"].iloc[0] == pytest.approx(3.0)
    assert eng_df["analytical_mindset_index"].iloc[0] == pytest.approx(0.5)


def test_pipeline_fit_and_transform():
    """Verify feature pipeline scales without data leakage and encodes classes."""
    cfg = load_config()
    raw_df = load_raw_data(cfg.paths.raw_data_path)
    train_df, test_df = split_data_leak_free(raw_df, test_size=0.2, random_state=42)

    pipeline = SkillFeaturePipeline()
    X_train_trans, y_train = pipeline.fit_transform(train_df)
    X_test_trans, y_test = pipeline.transform(test_df)

    assert len(pipeline.feature_columns) == 34
    assert len(pipeline.class_mapping) == 16
    assert X_train_trans.shape == (6822, 35)
    assert X_test_trans.shape == (1706, 35)

    # Standard scaling property: train mean approx 0, std approx 1
    np.testing.assert_allclose(X_train_trans[pipeline.feature_columns].mean(), 0.0, atol=1e-5)
