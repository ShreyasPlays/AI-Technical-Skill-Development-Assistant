"""
Data Preprocessing and Skill-Gap Analysis Module.
Contains:
1. Rule-based Skill-Gap Analyzer (calculates G_i and G_avg, ranks gap priorities).
2. Leakage-free dataset splitting and validation for the MLOps pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import AppConfig, load_config
from src.data_loader import load_raw_data, ingest_kaggle_dataset


class SkillGapAnalyzer:
    """
    Rule-based Skill Gap Analyzer.
    Compares candidate proficiency S_i against target role requirement R_i.

    Formulas:
        G_i = max(0, R_i - S_i)
        G_avg = (1 / n) * sum(G_i)
    """

    def __init__(self, role_requirements: Dict[str, Dict[str, int]]):
        self.role_requirements = role_requirements

    def get_supported_roles(self) -> List[str]:
        return list(self.role_requirements.keys())

    def analyze_gap(
        self,
        student_skills: Dict[str, int],
        target_role: str,
    ) -> Dict[str, Any]:
        """
        Analyze skill gaps for a given student profile and target role.

        Args:
            student_skills: Dict mapping skill names to proficiency (0-6).
            target_role: Target career role name.

        Returns:
            Dict containing detailed gap metrics, G_avg, and prioritized recommendations.
        """
        if target_role not in self.role_requirements:
            raise ValueError(
                f"Role '{target_role}' not supported. Available roles: {self.get_supported_roles()}"
            )

        requirements = self.role_requirements[target_role]
        gap_details: List[Dict[str, Any]] = []
        total_gap = 0
        num_skills = len(requirements)

        for skill, required_prof in requirements.items():
            current_prof = int(student_skills.get(skill, 0))
            # G_i = max(0, R_i - S_i)
            gap = max(0, required_prof - current_prof)
            total_gap += gap

            status = "Proficient"
            if gap >= 3:
                status = "Critical Gap"
            elif gap >= 1:
                status = "Minor Gap"

            gap_details.append({
                "skill": skill,
                "current_proficiency": current_prof,
                "required_proficiency": required_prof,
                "gap": gap,
                "status": status,
            })

        # Rank skills descending by gap size, then by required proficiency
        gap_details.sort(key=lambda x: (x["gap"], x["required_proficiency"]), reverse=True)

        g_avg = round(total_gap / num_skills, 2)
        # Scale max proficiency is 6
        readiness_score = round(max(0.0, 100.0 * (1.0 - (total_gap / (num_skills * 6.0)))), 1)

        recommendations = [
            f"Focus on '{item['skill']}' (Current: {item['current_proficiency']}/6, Required: {item['required_proficiency']}/6, Gap: +{item['gap']})"
            for item in gap_details if item["gap"] > 0
        ]

        return {
            "target_role": target_role,
            "average_skill_gap": g_avg,
            "readiness_percentage": readiness_score,
            "skill_gaps": gap_details,
            "recommended_focus_order": [item["skill"] for item in gap_details if item["gap"] > 0],
            "actionable_recommendations": recommendations,
        }


def split_data_leak_free(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = "target_role",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test partitions BEFORE fitting any transformations,
    ensuring zero data leakage between evaluation and training sets.
    """
    logger.info(f"Splitting dataset (N={len(df)}) with test_size={test_size}, seed={random_state}...")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    logger.info(f"Split complete. Train set: {train_df.shape}, Test set: {test_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_preprocessing_pipeline(config: Optional[AppConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full data ingestion and train/test splitting pipeline stage."""
    if config is None:
        config = load_config()

    raw_path = config.paths.raw_data_path
    if not raw_path.exists():
        logger.info(f"Raw data file not found at {raw_path}. Ingesting Kaggle dataset...")
        ingest_kaggle_dataset(output_path=raw_path)

    df = load_raw_data(raw_path)
    train_df, test_df = split_data_leak_free(
        df=df,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        target_column=config.data.target_column,
    )

    # Save initial partitioned splits
    config.paths.processed_train_path.parent.mkdir(parents=True, exist_ok=True)
    temp_train_path = config.paths.processed_train_path.parent / "train_split.parquet"
    temp_test_path = config.paths.processed_test_path.parent / "test_split.parquet"

    train_df.to_parquet(temp_train_path, index=False)
    test_df.to_parquet(temp_test_path, index=False)
    logger.success(f"Saved initial splits: {temp_train_path} and {temp_test_path}")

    return train_df, test_df


if __name__ == "__main__":
    cfg = load_config()
    run_preprocessing_pipeline(cfg)

    # Demonstration of the rule-based SkillGapAnalyzer on example student profile
    analyzer = SkillGapAnalyzer(cfg.roles)
    example_student = {
        "ai_ml": 2,
        "data_science": 1,
        "programming_skills": 3,
        "database_fundamentals": 2,
        "distributed_computing_systems": 1,
        "software_engineering": 2,
    }
    result = analyzer.analyze_gap(example_student, target_role="AI ML Specialist")
    print(f"\n--- Skill Gap Analysis for Target Role: {result['target_role']} ---")
    print(f"Average Skill Gap G_avg: {result['average_skill_gap']}")
    print(f"Readiness Score: {result['readiness_percentage']}%")
    print("\nActionable Recommendations:")
    for rec in result["actionable_recommendations"]:
        print(f" - {rec}")
