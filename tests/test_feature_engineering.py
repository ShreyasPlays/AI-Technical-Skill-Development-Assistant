"""Unit tests for feature engineering and binary assessment label generation pipeline."""

import pytest
import pandas as pd
import numpy as np
from src.config import load_config
from src.data_loader import load_raw_data, SKILL_COLUMNS, CONTEXT_COLUMNS
from src.preprocess import split_data_leak_free, AssessmentLabelGenerator
from src.feature_engineering import (
    engineer_composite_features,
    SkillFeaturePipeline,
)


# ── AssessmentLabelGenerator tests ──────────────────────────────────────

def _make_skill_df(skill_level: int = 3, n: int = 100) -> pd.DataFrame:
    """Helper to build a synthetic skill DataFrame."""
    data = {col: [skill_level] * n for col in SKILL_COLUMNS}
    data.update({col: [0.5] * n for col in CONTEXT_COLUMNS})
    data["target_role"] = ["Software Developer"] * n
    return pd.DataFrame(data)


def test_assessment_label_generator_binary_values():
    """Verify generated labels are strictly binary (0 or 1)."""
    df = _make_skill_df(skill_level=3, n=200)
    gen = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    result = gen.generate(df)

    assert "assessment_pass" in result.columns
    assert "assessment_score" in result.columns
    assert set(result["assessment_pass"].unique()).issubset({0, 1}), (
        "assessment_pass must be binary (0 or 1)."
    )


def test_assessment_label_generator_score_bounds():
    """Verify assessment_score is bounded [0, 1]."""
    df = _make_skill_df(skill_level=4, n=50)
    gen = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    result = gen.generate(df)

    assert result["assessment_score"].between(0.0, 1.0).all(), (
        "assessment_score must be in [0, 1] — it is a normalised weighted skill score."
    )


def test_assessment_label_reproducibility():
    """Verify labels are deterministic for the same random seed."""
    df = _make_skill_df(skill_level=3, n=100)
    gen_a = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    gen_b = AssessmentLabelGenerator(threshold=0.50, random_state=42)

    labels_a = gen_a.generate(df)["assessment_pass"].values
    labels_b = gen_b.generate(df)["assessment_pass"].values

    np.testing.assert_array_equal(labels_a, labels_b)


def test_high_skill_students_mostly_pass():
    """Verify that students with maximum skills (6/6) pass with very high probability."""
    df = _make_skill_df(skill_level=6, n=500)
    gen = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    result = gen.generate(df)

    pass_rate = result["assessment_pass"].mean()
    assert pass_rate > 0.85, (
        f"Expert-level students (skill=6) should pass >85% of the time. Got: {pass_rate:.2%}"
    )


def test_zero_skill_students_mostly_fail():
    """Verify that students with zero skills (0/6) fail with very high probability."""
    df = _make_skill_df(skill_level=0, n=500)
    gen = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    result = gen.generate(df)

    fail_rate = 1 - result["assessment_pass"].mean()
    assert fail_rate > 0.85, (
        f"Zero-skill students should fail >85% of the time. Got fail rate: {fail_rate:.2%}"
    )


# ── Leakage-free split tests ─────────────────────────────────────────────

def test_leakage_free_split():
    """Verify train/test split preserves assessment_pass stratification and no overlap."""
    cfg = load_config()
    raw_df = load_raw_data(cfg.paths.raw_data_path)

    gen = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    labeled_df = gen.generate(raw_df)

    train_df, test_df = split_data_leak_free(
        labeled_df,
        test_size=0.2,
        random_state=42,
        target_column="assessment_pass",
    )

    assert len(train_df) == 6822
    assert len(test_df) == 1706
    # No index overlap
    assert len(set(train_df.index) & set(test_df.index)) == 0
    # Both splits contain both classes
    assert set(train_df["assessment_pass"].unique()) == {0, 1}
    assert set(test_df["assessment_pass"].unique()) == {0, 1}


# ── Composite feature engineering tests ─────────────────────────────────

def test_composite_feature_engineering():
    """Verify composite domain readiness indices and aggregate stats are computed correctly."""
    sample_data = {col: [3] for col in SKILL_COLUMNS}
    sample_data.update({col: [0.5] for col in CONTEXT_COLUMNS})
    sample_data["target_role"] = ["Software Developer"]

    df = pd.DataFrame(sample_data)
    eng_df = engineer_composite_features(df)

    # Aggregate statistics
    assert "technical_overall_mean" in eng_df.columns
    assert "technical_breadth" in eng_df.columns
    assert "technical_depth" in eng_df.columns

    # Domain composites
    assert "ai_data_readiness" in eng_df.columns
    assert "systems_readiness" in eng_df.columns
    assert "software_dev_readiness" in eng_df.columns
    assert "cybersecurity_readiness" in eng_df.columns
    assert "business_comm_readiness" in eng_df.columns

    # Values for uniform skill=3
    assert eng_df["technical_overall_mean"].iloc[0] == pytest.approx(3.0)
    assert eng_df["technical_breadth"].iloc[0] == 17  # all 17 skills >= 3
    assert eng_df["technical_depth"].iloc[0] == 0    # none >= 5


# ── Pipeline fit/transform tests ─────────────────────────────────────────

def test_pipeline_fit_and_transform():
    """Verify feature pipeline scales without leakage on binary target."""
    cfg = load_config()
    raw_df = load_raw_data(cfg.paths.raw_data_path)

    gen = AssessmentLabelGenerator(threshold=0.50, random_state=42)
    labeled_df = gen.generate(raw_df)

    train_df, test_df = split_data_leak_free(
        labeled_df, test_size=0.2, random_state=42, target_column="assessment_pass"
    )

    pipeline = SkillFeaturePipeline()
    X_train_trans, y_train = pipeline.fit_transform(train_df, target_column="assessment_pass")
    X_test_trans, y_test = pipeline.transform(test_df, target_column="assessment_pass")

    # Feature count: 17 raw skills + 5 composites + 3 aggregates = 25
    assert len(pipeline.feature_columns) == 25, (
        f"Expected 25 features. Got: {len(pipeline.feature_columns)}"
    )

    # Target is binary
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})

    # Correct number of samples
    assert X_train_trans.shape[0] == 6822
    assert X_test_trans.shape[0] == 1706

    # Standard scaling: train feature means ≈ 0
    feature_cols = [c for c in X_train_trans.columns if c != "assessment_pass"]
    np.testing.assert_allclose(X_train_trans[feature_cols].mean(), 0.0, atol=1e-4)

    # Personality columns must NOT be in the feature set
    from src.data_loader import CONTEXT_COLUMNS
    for ctx_col in CONTEXT_COLUMNS:
        assert ctx_col not in pipeline.feature_columns, (
            f"Personality column '{ctx_col}' must not appear as a model feature."
        )

    # assessment_score must NOT be in the feature set (label proxy — would cause leakage)
    assert "assessment_score" not in pipeline.feature_columns, (
        "assessment_score encodes the target and must be excluded from features."
    )
