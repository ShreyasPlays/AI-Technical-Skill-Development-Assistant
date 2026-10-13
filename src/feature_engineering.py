"""
Feature Engineering module for Technical Skill Development Assistant.
Constructs domain-specific composite indices and prepares leakage-free feature pipelines.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import AppConfig, load_config
from src.data_loader import SKILL_COLUMNS
from src.preprocess import run_preprocessing_pipeline


def engineer_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-informed technical composite features from raw proficiency scores.
    """
    df = df.copy()

    # 1. Overall mean skill proficiency across all evaluated disciplines
    df["skill_overall_mean"] = df[SKILL_COLUMNS].mean(axis=1)

    # 2. Domain readiness indices based on target career competencies
    df["ml_ai_readiness"] = (
        0.35 * df["python"]
        + 0.35 * df["machine_learning"]
        + 0.20 * df["deep_learning"]
        + 0.10 * df["statistics"]
    )

    df["data_analytics_readiness"] = (
        0.40 * df["sql"]
        + 0.30 * df["statistics"]
        + 0.20 * df["python"]
        + 0.10 * df["machine_learning"]
    )

    df["fullstack_readiness"] = (
        0.40 * df["web_development"]
        + 0.25 * df["dsa"]
        + 0.20 * df["git"]
        + 0.15 * df["system_design"]
    )

    df["devops_cloud_readiness"] = (
        0.40 * df["cloud_computing"]
        + 0.30 * df["git"]
        + 0.20 * df["system_design"]
        + 0.10 * df["python"]
    )

    # 3. Practical velocity and project engagement index
    df["practical_velocity"] = np.log1p(df["problems_solved"]) * np.sqrt(df["projects_completed"])

    return df


class SkillFeaturePipeline:
    """
    Leakage-free Feature Engineering & Scaling Pipeline.
    Fits exclusively on the training partition and transforms test data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.class_mapping: Dict[int, str] = {}
        self.is_fitted = False

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        target_column: str = "target_role",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit scalers and encoders on training partition and return transformed features."""
        logger.info("Fitting feature engineering pipeline on training set...")
        train_eng = engineer_composite_features(train_df)

        self.feature_columns = [
            col for col in train_eng.columns if col != target_column
        ]

        # Fit and transform features
        scaled_features = self.scaler.fit_transform(train_eng[self.feature_columns])
        transformed_train = pd.DataFrame(
            scaled_features, columns=self.feature_columns, index=train_eng.index
        )

        # Fit and transform target
        y_train = self.label_encoder.fit_transform(train_eng[target_column])
        self.class_mapping = {
            int(idx): str(label) for idx, label in enumerate(self.label_encoder.classes_)
        }
        transformed_train[target_column] = y_train

        self.is_fitted = True
        logger.info(f"Pipeline fitted on {len(self.feature_columns)} features. Classes: {self.class_mapping}")
        return transformed_train, y_train

    def transform(
        self,
        test_df: pd.DataFrame,
        target_column: str = "target_role",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform test data partition using parameters learned from training set."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before transforming test data.")

        logger.info("Transforming test set using fitted pipeline parameters...")
        test_eng = engineer_composite_features(test_df)

        scaled_features = self.scaler.transform(test_eng[self.feature_columns])
        transformed_test = pd.DataFrame(
            scaled_features, columns=self.feature_columns, index=test_eng.index
        )

        y_test = self.label_encoder.transform(test_eng[target_column])
        transformed_test[target_column] = y_test
        return transformed_test, y_test

    def save_metadata(self, metadata_path: Path) -> None:
        """Save feature column names, scaler parameters, and label mappings."""
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_columns": self.feature_columns,
            "classes": self.class_mapping,
            "num_features": len(self.feature_columns),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.success(f"Feature metadata saved to {metadata_path}")


def run_feature_engineering_pipeline(config: Optional[AppConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end preprocessing, feature engineering, and persistent parquet storage."""
    if config is None:
        config = load_config()

    train_df, test_df = run_preprocessing_pipeline(config)

    pipeline = SkillFeaturePipeline()
    train_processed, _ = pipeline.fit_transform(train_df, target_column=config.data.target_column)
    test_processed, _ = pipeline.transform(test_df, target_column=config.data.target_column)

    # Save processed parquets
    config.paths.processed_train_path.parent.mkdir(parents=True, exist_ok=True)
    train_processed.to_parquet(config.paths.processed_train_path, index=False)
    test_processed.to_parquet(config.paths.processed_test_path, index=False)

    # Save feature metadata
    pipeline.save_metadata(config.paths.features_meta_path)
    logger.success(f"Processed training set saved: {config.paths.processed_train_path} ({train_processed.shape})")
    logger.success(f"Processed test set saved: {config.paths.processed_test_path} ({test_processed.shape})")

    return train_processed, test_processed


if __name__ == "__main__":
    cfg = load_config()
    run_feature_engineering_pipeline(cfg)
