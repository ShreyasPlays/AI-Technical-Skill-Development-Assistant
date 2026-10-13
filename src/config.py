"""Configuration loader and schema validator using Pydantic v2 and PyYAML."""

from pathlib import Path
from typing import Any, Dict, List
import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_data_path: Path = Path("data/raw/skills_dataset.csv")
    processed_train_path: Path = Path("data/processed/train.parquet")
    processed_test_path: Path = Path("data/processed/test.parquet")
    features_meta_path: Path = Path("data/processed/feature_metadata.json")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    metrics_path: Path = Path("reports/metrics.json")
    evaluation_report_path: Path = Path("reports/phase1_evaluation.md")


class DataConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)
    stratify: bool = Field(default=True)
    target_column: str = Field(default="target_role")
    skill_features: List[str] = Field(default_factory=list)
    context_features: List[str] = Field(default_factory=list)


class DecisionTreeConfig(BaseModel):
    max_depth: int = Field(default=6)
    min_samples_split: int = Field(default=10)
    min_samples_leaf: int = Field(default=5)
    criterion: str = Field(default="gini")


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(default=100)
    max_depth: int = Field(default=8)
    min_samples_split: int = Field(default=8)
    min_samples_leaf: int = Field(default=4)
    criterion: str = Field(default="gini")
    n_jobs: int = Field(default=-1)


class ModelsConfig(BaseModel):
    random_state: int = Field(default=42)
    decision_tree: DecisionTreeConfig = Field(default_factory=DecisionTreeConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)


class MLflowConfig(BaseModel):
    experiment_name: str = Field(default="Technical_Skill_Assistant")
    tracking_uri: str = Field(default="mlruns")


class AppConfig(BaseModel):
    project_name: str = Field(default="AI-Technical-Skill-Development-Assistant")
    version: str = Field(default="1.0.0")
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    roles: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_config(config_path: str | Path = "configs/config.yaml") -> AppConfig:
    """Load and validate the YAML configuration using Pydantic."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(**data)
