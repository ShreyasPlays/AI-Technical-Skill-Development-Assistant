"""
Feature Engineering module for Technical Skill Development Assistant (Phase 1 Rework).

New binary classification target: assessment_pass (0 = Fail, 1 = Pass).

Feature set (25 total):
  - 17 raw technical skill scores (integer 0-6)
  - 5  domain-readiness composite indices
  - 3  aggregate skill statistics (overall_mean, technical_breadth, technical_depth)

Personality/context columns are excluded from model features:
  the online assessment is purely technical; including personality would reduce
  interpretability and could introduce fairness concerns.
  (They remain in the raw data for EDA in notebook 02.)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig, load_config
from src.data_loader import CONTEXT_COLUMNS, SKILL_COLUMNS
from src.preprocess import run_preprocessing_pipeline


# -----------------------------------------------------------------------
# Composite feature engineering
# -----------------------------------------------------------------------

def engineer_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-informed composite features and aggregate statistics from raw skill scores.

    New features added:
      - technical_overall_mean   : mean of all 17 skills (continuous signal of breadth)
      - technical_breadth        : count of skills ≥ 3  (intermediate-or-above competency)
      - technical_depth          : count of skills ≥ 5  (advanced competency indicators)
      - ai_data_readiness        : AI / Data Science domain composite
      - systems_readiness        : Systems / Infrastructure composite
      - software_dev_readiness   : Software Development composite
      - cybersecurity_readiness  : Cyber Security composite
      - business_comm_readiness  : Business & Communication composite
    """
    df = df.copy()
    available_skills = [c for c in SKILL_COLUMNS if c in df.columns]

    # ── Aggregate skill statistics ────────────────────────────────────────
    if available_skills:
        skill_vals = df[available_skills]
        df["technical_overall_mean"] = skill_vals.mean(axis=1).round(4)
        df["technical_breadth"] = (skill_vals >= 3).sum(axis=1).astype(int)
        df["technical_depth"] = (skill_vals >= 5).sum(axis=1).astype(int)

    # ── Domain readiness composites ───────────────────────────────────────
    if "ai_ml" in df.columns and "data_science" in df.columns and "programming_skills" in df.columns:
        df["ai_data_readiness"] = (
            0.45 * df["ai_ml"]
            + 0.45 * df["data_science"]
            + 0.10 * df["programming_skills"]
        ).round(4)

    if all(c in df.columns for c in ["networking", "distributed_computing_systems", "computer_architecture"]):
        df["systems_readiness"] = (
            0.35 * df["networking"]
            + 0.35 * df["distributed_computing_systems"]
            + 0.30 * df["computer_architecture"]
        ).round(4)

    if all(c in df.columns for c in ["software_development", "programming_skills", "software_engineering"]):
        df["software_dev_readiness"] = (
            0.40 * df["software_development"]
            + 0.35 * df["programming_skills"]
            + 0.25 * df["software_engineering"]
        ).round(4)

    if all(c in df.columns for c in ["cyber_security", "computer_forensics_fundamentals", "networking"]):
        df["cybersecurity_readiness"] = (
            0.50 * df["cyber_security"]
            + 0.30 * df["computer_forensics_fundamentals"]
            + 0.20 * df["networking"]
        ).round(4)

    if all(c in df.columns for c in ["business_analysis", "project_management", "communication_skills"]):
        df["business_comm_readiness"] = (
            0.40 * df["business_analysis"]
            + 0.35 * df["project_management"]
            + 0.25 * df["communication_skills"]
        ).round(4)

    return df


# -----------------------------------------------------------------------
# Leakage-free feature pipeline
# -----------------------------------------------------------------------

class SkillFeaturePipeline:
    """
    Leakage-free Feature Engineering & Scaling Pipeline.

    - Fits StandardScaler exclusively on the training partition.
    - Applies the fitted scaler to the test partition (no re-fitting).
    - Excludes personality/context columns, target_role, and the continuous
      assessment_score from model features (assessment_score encodes the label
      and must not be used as a predictor).
    """

    # Columns that must never appear as model features
    _ALWAYS_EXCLUDE = {"target_role", "assessment_score"}

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        target_column: str = "assessment_pass",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit scaler on training partition and return transformed feature matrix."""
        logger.info("Fitting feature engineering pipeline on training set...")
        train_eng = engineer_composite_features(train_df)

        excluded = self._ALWAYS_EXCLUDE | {target_column} | set(CONTEXT_COLUMNS)
        self.feature_columns = [
            col for col in train_eng.columns if col not in excluded
        ]

        scaled = self.scaler.fit_transform(train_eng[self.feature_columns])
        transformed = pd.DataFrame(scaled, columns=self.feature_columns, index=train_eng.index)
        y_train = train_eng[target_column].values.astype(int)
        transformed[target_column] = y_train

        self.is_fitted = True
        logger.info(
            f"Pipeline fitted on {len(self.feature_columns)} features "
            f"| Train samples: {len(train_df)} | Pass rate: {y_train.mean()*100:.1f}%"
        )
        return transformed, y_train

    def transform(
        self,
        test_df: pd.DataFrame,
        target_column: str = "assessment_pass",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform test data using parameters learned strictly from training set."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming test data.")

        logger.info("Transforming test set using fitted pipeline parameters...")
        test_eng = engineer_composite_features(test_df)

        scaled = self.scaler.transform(test_eng[self.feature_columns])
        transformed = pd.DataFrame(scaled, columns=self.feature_columns, index=test_eng.index)
        y_test = test_eng[target_column].values.astype(int)
        transformed[target_column] = y_test

        logger.info(f"Test set transformed | Samples: {len(test_df)} | Pass rate: {y_test.mean()*100:.1f}%")
        return transformed, y_test

    def save_metadata(self, metadata_path: Path) -> None:
        """Persist feature column names and scaler parameters for reproducible inference."""
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_columns": self.feature_columns,
            "num_features": len(self.feature_columns),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "target_column": "assessment_pass",
            "label_description": {
                "0": "Fail — student unlikely to clear the online assessment",
                "1": "Pass — student likely to clear the online assessment",
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.success(f"Feature metadata saved to {metadata_path}")


# -----------------------------------------------------------------------
# End-to-end pipeline runner
# -----------------------------------------------------------------------

def run_feature_engineering_pipeline(
    config: Optional[AppConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end preprocessing, label engineering, feature construction, and storage."""
    if config is None:
        config = load_config()

    train_df, test_df = run_preprocessing_pipeline(config)

    pipeline = SkillFeaturePipeline()
    train_processed, _ = pipeline.fit_transform(train_df, target_column=config.data.target_column)
    test_processed, _ = pipeline.transform(test_df, target_column=config.data.target_column)

    # Persist processed parquets
    config.paths.processed_train_path.parent.mkdir(parents=True, exist_ok=True)
    train_processed.to_parquet(config.paths.processed_train_path, index=False)
    test_processed.to_parquet(config.paths.processed_test_path, index=False)

    # Persist feature metadata
    pipeline.save_metadata(config.paths.features_meta_path)

    logger.success(
        f"Processed training set: {config.paths.processed_train_path} {train_processed.shape}"
    )
    logger.success(
        f"Processed test set:     {config.paths.processed_test_path} {test_processed.shape}"
    )
    logger.info(f"Feature columns ({len(pipeline.feature_columns)}): {pipeline.feature_columns}")
    return train_processed, test_processed


if __name__ == "__main__":
    cfg = load_config()
    run_feature_engineering_pipeline(cfg)
