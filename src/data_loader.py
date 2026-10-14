"""
Data loading and dataset ingestion module for Technical Skill Development Assistant.
Handles Kaggle Career Mapping benchmark dataset ingestion, schema validation, and data verification.
"""

from pathlib import Path
from typing import List
from loguru import logger
import numpy as np
import pandas as pd

from src.config import load_config

# Standard 17 technical skills evaluated from the Kaggle dataset (0-6 scale)
SKILL_COLUMNS: List[str] = [
    "database_fundamentals",
    "computer_architecture",
    "distributed_computing_systems",
    "cyber_security",
    "networking",
    "software_development",
    "programming_skills",
    "project_management",
    "computer_forensics_fundamentals",
    "technical_communication",
    "ai_ml",
    "software_engineering",
    "business_analysis",
    "communication_skills",
    "data_science",
    "troubleshooting_skills",
    "graphics_designing",
]

# 10 psychometric / context indicators (0.0 - 1.0 scale)
CONTEXT_COLUMNS: List[str] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "emotional_range",
    "conversation",
    "openness_to_change",
    "hedonism",
    "self_enhancement",
    "self_transcendence",
]

# 16 industry tech career role classes
ROLE_CLASSES: List[str] = [
    "AI ML Specialist",
    "Application Support Engineer",
    "Networking Engineer",
    "Software Developer",
    "Project Manager",
    "Business Analyst",
    "Cyber Security Specialist",
    "Database Administrator",
    "Software tester",
    "Graphics Designer",
    "Hardware Engineer",
    "Technical Writer",
    "Helpdesk Engineer",
    "Customer Service Executive",
    "API Specialist",
    "Information Security Specialist",
]

KAGGLE_RAW_URL: str = (
    "https://raw.githubusercontent.com/TuhinPatra633/Career-Prediction-Using-Machine-Learning/master/Data/CareerMapping.csv"
)


def ingest_kaggle_dataset(output_path: Path, force_download: bool = False) -> pd.DataFrame:
    """
    Ingest and curate the Kaggle Career Mapping dataset.
    Downloads from repository if needed, cleans headers to snake_case, drops duplicates, and saves CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force_download:
        logger.info(f"Using existing raw dataset at {output_path}")
        return pd.read_csv(output_path)

    logger.info(f"Downloading Kaggle Career Mapping dataset from {KAGGLE_RAW_URL}...")
    try:
        raw_df = pd.read_csv(KAGGLE_RAW_URL)
    except Exception as e:
        logger.error(f"Failed to fetch from remote URL: {e}. Checking local fallback...")
        if output_path.exists():
            return pd.read_csv(output_path)
        raise

    col_map = {
        "Database Fundamentals": "database_fundamentals",
        "Computer Architecture": "computer_architecture",
        "Distributed Computing Systems": "distributed_computing_systems",
        "Cyber Security": "cyber_security",
        "Networking": "networking",
        "Software Development": "software_development",
        "Programming Skills": "programming_skills",
        "Project Management": "project_management",
        "Computer Forensics Fundamentals": "computer_forensics_fundamentals",
        "Technical Communication": "technical_communication",
        "AI ML": "ai_ml",
        "Software Engineering": "software_engineering",
        "Business Analysis": "business_analysis",
        "Communication skills": "communication_skills",
        "Data Science": "data_science",
        "Troubleshooting skills": "troubleshooting_skills",
        "Graphics Designing": "graphics_designing",
        "Openness": "openness",
        "Conscientousness": "conscientiousness",
        "Extraversion": "extraversion",
        "Agreeableness": "agreeableness",
        "Emotional_Range": "emotional_range",
        "Conversation": "conversation",
        "Openness to Change": "openness_to_change",
        "Hedonism": "hedonism",
        "Self-enhancement": "self_enhancement",
        "Self-transcendence": "self_transcendence",
        "Role": "target_role",
    }
    df = raw_df.rename(columns=col_map)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Ingested and saved {len(df)} records to {output_path}")
    return df


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load raw CSV dataset and validate schema, types, and value bounds.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}. Ingest dataset first.")

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

    # Validate skill ranges [0, 6] and integer constraints
    for col in SKILL_COLUMNS:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or not vals.between(0, 6).all():
            raise ValueError(f"Skill column {col} contains values outside expected range [0, 6]")
        if not np.all(np.equal(vals, np.floor(vals))):
            raise ValueError(f"Skill column {col} contains non-integer proficiency ratings")

    # Validate contextual/personality scores [0.0, 1.0]
    for col in CONTEXT_COLUMNS:
        p_vals = pd.to_numeric(df[col], errors="coerce")
        if p_vals.isna().any() or not p_vals.between(0.0, 1.0).all():
            raise ValueError(f"Context column {col} contains values outside expected range [0.0, 1.0]")

    # Validate target roles
    if "target_role" in df.columns:
        invalid_roles = set(df["target_role"].dropna().unique()) - set(ROLE_CLASSES)
        if invalid_roles:
            raise ValueError(f"target_role contains unsupported roles: {invalid_roles}")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Dataset contains {null_count} null values.")
    else:
        logger.info("Dataset schema and value validation passed with zero null values.")

    logger.info(f"Loaded {len(df)} records across {len(df['target_role'].unique())} target roles.")
    return df


if __name__ == "__main__":
    cfg = load_config()
    raw_csv = cfg.paths.raw_data_path
    if not raw_csv.exists():
        ingest_kaggle_dataset(raw_csv)
    loaded_df = load_raw_data(raw_csv)
    print(loaded_df.head())
    print("\nRole distribution:")
    print(loaded_df["target_role"].value_counts())
