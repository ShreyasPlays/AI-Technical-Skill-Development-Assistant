"""Unit tests for dataset loading, validation, ingestion, and label engineering constants."""

from pathlib import Path
import pytest
import pandas as pd

from src.config import load_config
from src.data_loader import (
    SKILL_COLUMNS,
    CONTEXT_COLUMNS,
    ROLE_CLASSES,
    CORE_ASSESSMENT_WEIGHTS,
    ROLE_REQUIREMENTS,
    load_raw_data,
)


def test_config_loading():
    """Verify configuration file loads and validates schema correctly."""
    cfg = load_config()
    assert cfg.project_name == "AI-Technical-Skill-Development-Assistant"
    assert cfg.data.random_state == 42
    assert cfg.data.target_column == "assessment_pass"
    assert 0.0 < cfg.data.assessment_threshold < 1.0
    assert len(cfg.roles) == 16
    assert "AI ML Specialist" in cfg.roles
    assert "Software Developer" in cfg.roles


def test_core_assessment_weights_sum_to_one():
    """Verify CORE_ASSESSMENT_WEIGHTS are well-defined and sum to 1.0."""
    assert len(CORE_ASSESSMENT_WEIGHTS) == 6
    total_weight = sum(CORE_ASSESSMENT_WEIGHTS.values())
    assert abs(total_weight - 1.0) < 1e-9, (
        f"Weights must sum to 1.0 for a valid weighted mean. Got: {total_weight}"
    )
    for skill in CORE_ASSESSMENT_WEIGHTS:
        assert skill in SKILL_COLUMNS, (
            f"Core assessment skill '{skill}' is not in the 17 SKILL_COLUMNS list."
        )


def test_role_requirements_coverage():
    """Verify all 16 career roles are represented in ROLE_REQUIREMENTS."""
    assert len(ROLE_REQUIREMENTS) == 16
    for role in ROLE_CLASSES:
        assert role in ROLE_REQUIREMENTS, f"Role '{role}' missing from ROLE_REQUIREMENTS."
    # All requirement values must be valid (1-6)
    for role, reqs in ROLE_REQUIREMENTS.items():
        for skill, level in reqs.items():
            assert 1 <= level <= 6, (
                f"Requirement for '{skill}' in '{role}' must be 1-6. Got: {level}"
            )
            assert skill in SKILL_COLUMNS, (
                f"Required skill '{skill}' in '{role}' is not in SKILL_COLUMNS."
            )


def test_dataset_loading():
    """Verify loading raw Kaggle dataset validates schema and column types."""
    cfg = load_config()
    raw_csv = cfg.paths.raw_data_path

    assert raw_csv.exists(), f"Raw CSV file must exist at {raw_csv}"
    df = load_raw_data(raw_csv)

    assert len(df) == 8528
    assert set(SKILL_COLUMNS).issubset(df.columns)
    assert set(CONTEXT_COLUMNS).issubset(df.columns)
    assert "target_role" in df.columns
    assert set(df["target_role"].unique()).issubset(set(ROLE_CLASSES))


def test_skill_bounds_validation(tmp_path: Path):
    """Verify out-of-bounds skill values trigger validation error."""
    valid_row = {col: 3 for col in SKILL_COLUMNS}
    valid_row.update({col: 0.5 for col in CONTEXT_COLUMNS})
    valid_row["target_role"] = "Software Developer"

    bad_row = valid_row.copy()
    bad_row["programming_skills"] = 7  # 7 is out of bounds [0, 6]

    bad_df = pd.DataFrame([bad_row])
    bad_csv = tmp_path / "bad_bounds.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="outside expected range"):
        load_raw_data(bad_csv)


def test_fractional_skill_validation(tmp_path: Path):
    """Verify fractional skill ratings raise validation error."""
    valid_row = {col: 3 for col in SKILL_COLUMNS}
    valid_row.update({col: 0.5 for col in CONTEXT_COLUMNS})
    valid_row["target_role"] = "Software Developer"

    bad_row = valid_row.copy()
    bad_row["programming_skills"] = 3.5  # fractional value

    bad_df = pd.DataFrame([bad_row])
    bad_csv = tmp_path / "fractional.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="non-integer proficiency"):
        load_raw_data(bad_csv)


def test_unsupported_target_role_validation(tmp_path: Path):
    """Verify unknown or unlisted target role raises validation error."""
    valid_row = {col: 3 for col in SKILL_COLUMNS}
    valid_row.update({col: 0.5 for col in CONTEXT_COLUMNS})
    valid_row["target_role"] = "Quantum Cryptographer"  # unsupported role

    bad_df = pd.DataFrame([valid_row])
    bad_csv = tmp_path / "bad_role.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="unsupported roles"):
        load_raw_data(bad_csv)


def test_duplicate_columns_validation(tmp_path: Path):
    """Verify duplicate column names in CSV raise validation error."""
    cols = SKILL_COLUMNS + CONTEXT_COLUMNS + ["target_role", "programming_skills"]
    csv_header = ",".join(cols) + "\n"
    csv_row = ",".join(
        ["3"] * len(SKILL_COLUMNS)
        + ["0.5"] * len(CONTEXT_COLUMNS)
        + ["Software Developer", "3"]
    ) + "\n"

    bad_csv = tmp_path / "dup_cols.csv"
    bad_csv.write_text(csv_header + csv_row, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_raw_data(bad_csv)
