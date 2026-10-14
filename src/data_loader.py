"""
Data loading and dataset generation module for Technical Skill Development Assistant.
Ensures reproducible data acquisition and validation.
"""

from pathlib import Path
from typing import List
from loguru import logger
import numpy as np
import pandas as pd

from src.config import load_config

# Standard 10 technical skills evaluated
SKILL_COLUMNS: List[str] = [
    "python",
    "sql",
    "machine_learning",
    "dsa",
    "git",
    "web_development",
    "cloud_computing",
    "system_design",
    "statistics",
    "deep_learning",
]

CONTEXT_COLUMNS: List[str] = [
    "years_experience",
    "projects_completed",
    "problems_solved",
]

ROLE_CLASSES: List[str] = [
    "Machine Learning Engineer",
    "Data Scientist",
    "Full Stack Developer",
    "DevOps Engineer",
    "Data Analyst",
]


def generate_benchmark_dataset(
    output_path: Path,
    n_samples: int = 1200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic, industry-aligned technical skill benchmark dataset.
    Simulates students and junior developers targeting 5 prominent career paths.

    Skill ratings are on a 1-5 integer proficiency scale:
      1: Novice / Beginner
      2: Advanced Beginner
      3: Competent / Intermediate
      4: Proficient
      5: Expert
    """
    logger.info(f"Generating benchmark dataset with {n_samples} records (seed={random_state})...")
    rng = np.random.default_rng(seed=random_state)

    # Role archetypes with mean proficiency levels per skill (1-5 scale)
    archetypes = {
        "Machine Learning Engineer": {
            "means": [4.0, 3.2, 4.2, 3.8, 3.5, 1.8, 3.4, 3.2, 3.9, 4.0],
            "exp_mean": 1.8, "proj_mean": 5.5, "prob_mean": 240,
        },
        "Data Scientist": {
            "means": [4.1, 4.0, 3.9, 3.0, 3.2, 1.6, 2.4, 2.2, 4.5, 3.1],
            "exp_mean": 1.6, "proj_mean": 5.0, "prob_mean": 180,
        },
        "Full Stack Developer": {
            "means": [3.2, 3.3, 1.6, 3.9, 4.0, 4.5, 3.2, 3.8, 2.0, 1.4],
            "exp_mean": 1.7, "proj_mean": 6.2, "prob_mean": 220,
        },
        "DevOps Engineer": {
            "means": [3.4, 2.5, 1.5, 3.1, 4.6, 2.3, 4.6, 3.9, 2.1, 1.3],
            "exp_mean": 1.9, "proj_mean": 5.0, "prob_mean": 150,
        },
        "Data Analyst": {
            "means": [3.1, 4.5, 2.2, 2.1, 2.8, 1.5, 2.2, 1.8, 4.1, 1.4],
            "exp_mean": 1.4, "proj_mean": 4.2, "prob_mean": 110,
        },
    }

    records = []
    samples_per_role = n_samples // len(ROLE_CLASSES)
    std_dev = 0.75  # realistic variability among students

    for role_name in ROLE_CLASSES:
        role_spec = archetypes[role_name]
        means = np.array(role_spec["means"])

        for _ in range(samples_per_role):
            # Sample skills with noise and clip to [1, 5]
            raw_skills = rng.normal(loc=means, scale=std_dev)
            skills = np.clip(np.round(raw_skills), 1, 5).astype(int)

            # Sample contextual features
            years_exp = round(max(0.0, float(rng.normal(loc=role_spec["exp_mean"], scale=0.8))), 1)
            projects = int(max(1, round(float(rng.normal(loc=role_spec["proj_mean"], scale=1.8)))))
            problems = int(max(10, round(float(rng.normal(loc=role_spec["prob_mean"], scale=65)))))

            row = {skill: int(skills[idx]) for idx, skill in enumerate(SKILL_COLUMNS)}
            row["years_experience"] = years_exp
            row["projects_completed"] = projects
            row["problems_solved"] = problems
            row["target_role"] = role_name
            records.append(row)

    df = pd.DataFrame(records)
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Benchmark dataset successfully saved to: {output_path} (shape: {df.shape})")
    return df


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load raw CSV dataset and validate schema, types, and value bounds.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}. Generate or provide dataset first.")

    # Check for duplicate column names in CSV header
    with open(data_path, "r", encoding="utf-8") as f:
        header_cols = [c.strip() for c in f.readline().strip().split(",")]
    if len(header_cols) != len(set(header_cols)):
        raise ValueError(f"Dataset contains duplicate column names: {header_cols}")

    df = pd.read_csv(data_path)

    expected_cols = SKILL_COLUMNS + CONTEXT_COLUMNS + ["target_role"]

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    unexpected_cols = set(df.columns) - set(expected_cols)
    if unexpected_cols:
        logger.warning(f"Dataset contains unexpected additional columns: {unexpected_cols}")

    # Validate skill ranges and integer constraints
    for col in SKILL_COLUMNS:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or not vals.between(1, 5).all():
            raise ValueError(f"Skill column {col} contains values outside expected range [1, 5]")
        if not np.all(np.equal(vals, np.floor(vals))):
            raise ValueError(f"Skill column {col} contains non-integer proficiency ratings")

    # Validate contextual features constraints
    if "years_experience" in df.columns:
        exp = pd.to_numeric(df["years_experience"], errors="coerce")
        if exp.isna().any() or (exp < 0).any():
            raise ValueError("Context column 'years_experience' contains negative or non-numeric values")

    for col in ["projects_completed", "problems_solved"]:
        if col in df.columns:
            c_vals = pd.to_numeric(df[col], errors="coerce")
            if c_vals.isna().any() or (c_vals < 1).any() or not np.all(np.equal(c_vals, np.floor(c_vals))):
                raise ValueError(f"Context column '{col}' must contain positive integer values (>= 1)")

    # Validate target roles
    if "target_role" in df.columns:
        invalid_roles = set(df["target_role"].dropna().unique()) - set(ROLE_CLASSES)
        if invalid_roles:
            raise ValueError(f"target_role contains unsupported roles: {invalid_roles}")

    # Check missing values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Dataset contains {null_count} null values. Preprocessing will handle imputation.")
    else:
        logger.info("Dataset schema and value validation passed with zero null values.")

    # Check duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Dataset contains {dup_count} duplicate records.")

    logger.info(f"Loaded {len(df)} records across {len(df['target_role'].unique())} target roles.")
    return df


if __name__ == "__main__":
    cfg = load_config()
    raw_csv = cfg.paths.raw_data_path
    if not raw_csv.exists():
        generate_benchmark_dataset(raw_csv, n_samples=1200, random_state=cfg.data.random_state)
    loaded_df = load_raw_data(raw_csv)
    print(loaded_df.head())
    print("\nRole distribution:")
    print(loaded_df["target_role"].value_counts())
