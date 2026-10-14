"""Unit tests for dataset loading, validation, and ingestion."""

from pathlib import Path
import pytest
import pandas as pd

from src.config import load_config
from src.data_loader import (
    SKILL_COLUMNS,
    CONTEXT_COLUMNS,
    ROLE_CLASSES,
    load_raw_data,
)


def test_config_loading():
    """Verify configuration file loads and validates schema correctly."""
    cfg = load_config()
    assert cfg.project_name == "AI-Technical-Skill-Development-Assistant"
    assert cfg.data.random_state == 42
    assert len(cfg.roles) == 16
    assert "AI ML Specialist" in cfg.roles
    assert "Software Developer" in cfg.roles


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
    csv_row = ",".join(["3"] * len(SKILL_COLUMNS) + ["0.5"] * len(CONTEXT_COLUMNS) + ["Software Developer", "3"]) + "\n"

    bad_csv = tmp_path / "dup_cols.csv"
    bad_csv.write_text(csv_header + csv_row, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_raw_data(bad_csv)
