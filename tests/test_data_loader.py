"""Unit tests for dataset loading, validation, and generation."""

from pathlib import Path
import pytest
import pandas as pd

from src.config import load_config
from src.data_loader import (
    SKILL_COLUMNS,
    CONTEXT_COLUMNS,
    load_raw_data,
    generate_benchmark_dataset,
)


def test_config_loading():
    """Verify configuration file loads and validates schema correctly."""
    cfg = load_config()
    assert cfg.project_name == "AI-Technical-Skill-Development-Assistant"
    assert cfg.data.random_state == 42
    assert len(cfg.roles) == 5
    assert "Machine Learning Engineer" in cfg.roles


def test_dataset_loading(tmp_path: Path):
    """Verify loading raw dataset validates schema and column types."""
    cfg = load_config()
    raw_csv = cfg.paths.raw_data_path

    assert raw_csv.exists(), f"Raw CSV file must exist at {raw_csv}"
    df = load_raw_data(raw_csv)

    assert len(df) == 1200
    assert set(SKILL_COLUMNS).issubset(df.columns)
    assert set(CONTEXT_COLUMNS).issubset(df.columns)
    assert "target_role" in df.columns


def test_skill_bounds_validation(tmp_path: Path):
    """Verify invalid skill values trigger validation error."""
    bad_df = pd.DataFrame({
        "python": [6, 2],  # 6 is out of bounds [1, 5]
        "sql": [3, 4],
        "machine_learning": [1, 2],
        "dsa": [2, 3],
        "git": [2, 3],
        "web_development": [1, 1],
        "cloud_computing": [2, 2],
        "system_design": [1, 1],
        "statistics": [2, 2],
        "deep_learning": [1, 1],
        "years_experience": [1.0, 2.0],
        "projects_completed": [3, 4],
        "problems_solved": [50, 100],
        "target_role": ["Data Analyst", "Data Scientist"],
    })
    bad_csv = tmp_path / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="outside expected range"):
        load_raw_data(bad_csv)


def test_reproducible_dataset_generation(tmp_path: Path):
    """Verify dataset generation is deterministic with fixed random seed."""
    csv1 = tmp_path / "data1.csv"
    csv2 = tmp_path / "data2.csv"

    df1 = generate_benchmark_dataset(csv1, n_samples=100, random_state=42)
    df2 = generate_benchmark_dataset(csv2, n_samples=100, random_state=42)

    pd.testing.assert_frame_equal(df1, df2)
