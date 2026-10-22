"""
Data Preprocessing and Label Engineering Module.

Contains:
1. AssessmentLabelGenerator — probabilistic binary pass/fail label for the online
   technical assessment gate, derived from a weighted combination of core technical skills.
2. SkillGapAnalyzer — rule-based gap calculator (G_i and G_avg) used by predict.py
   to generate personalised learning roadmaps independent of ML model predictions.
3. Leakage-free dataset splitting and validation for the MLOps pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import AppConfig, load_config
from src.data_loader import (
    CORE_ASSESSMENT_WEIGHTS,
    SKILL_COLUMNS,
    load_raw_data,
    ingest_kaggle_dataset,
)


# -----------------------------------------------------------------------
# Assessment Label Generator
# -----------------------------------------------------------------------

class AssessmentLabelGenerator:
    """
    Probabilistic binary label generator for online technical assessment pass/fail prediction.

    Design rationale
    ----------------
    Real-world online assessments (e.g., HackerRank, CodeSignal campus screens) evaluate
    a focused subset of core technical competencies rather than every possible skill.
    This generator:
      1. Computes a weighted core-skill assessment score for each student.
      2. Applies a sigmoid function around a configurable threshold to assign a
         probabilistic pass probability — capturing the inherent uncertainty of
         assessment outcomes (test anxiety, question variance, time pressure).
      3. Samples the final binary label from that probability distribution with a
         fixed random seed, ensuring full reproducibility via DVC.

    Assessment score formula (normalised 0–1):
        score = Σ (student_skill_i × weight_i) / 6.0
        prob_pass = sigmoid(sharpness × (score − threshold))
        assessment_pass ~ Bernoulli(prob_pass)

    Core assessment skills and weights are defined in data_loader.CORE_ASSESSMENT_WEIGHTS.
    """

    # Sigmoid sharpness: controls how soft/hard the threshold boundary is.
    # 5.0 → moderate boundary (students near threshold have genuine uncertainty).
    _SHARPNESS: float = 5.0

    def __init__(self, threshold: float = 0.50, random_state: int = 42):
        """
        Args:
            threshold: Normalised score threshold (0–1). Students whose score
                       equals threshold have a 50% chance of passing.
            random_state: Seed for reproducible Bernoulli sampling.
        """
        self.threshold = threshold
        self.random_state = random_state

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'assessment_score' (continuous) and 'assessment_pass' (binary) columns to df.

        Args:
            df: Raw dataframe containing at minimum the 17 SKILL_COLUMNS.

        Returns:
            Copy of df with two new columns:
              - assessment_score (float): normalised weighted core-skill score [0.0–1.0]
              - assessment_pass  (int):  1 = Pass, 0 = Fail
        """
        df = df.copy()
        rng = np.random.default_rng(self.random_state)

        # Step 1: Compute weighted core-assessment score
        raw_score = pd.Series(0.0, index=df.index)
        for skill, weight in CORE_ASSESSMENT_WEIGHTS.items():
            if skill in df.columns:
                raw_score += df[skill].astype(float) * weight
            else:
                logger.warning(f"Core assessment skill '{skill}' not found in DataFrame — skipping.")

        # Normalise to [0, 1]  (max possible raw_score = 6.0 × Σweights = 6.0)
        score_norm = raw_score / 6.0

        # Step 2: Probabilistic pass via sigmoid around threshold
        prob_pass = self._sigmoid(self._SHARPNESS * (score_norm - self.threshold))

        # Step 3: Bernoulli sample
        assessment_pass = (rng.random(len(df)) < prob_pass.values).astype(int)

        df["assessment_score"] = score_norm.round(4)
        df["assessment_pass"] = assessment_pass

        n_pass = int(assessment_pass.sum())
        n_fail = len(assessment_pass) - n_pass
        logger.info(
            f"Assessment labels generated | Pass: {n_pass} ({n_pass / len(df) * 100:.1f}%) "
            f"| Fail: {n_fail} ({n_fail / len(df) * 100:.1f}%)"
        )
        return df


# -----------------------------------------------------------------------
# Rule-Based Skill Gap Analyzer (unchanged from Phase 1)
# -----------------------------------------------------------------------

class SkillGapAnalyzer:
    """
    Rule-based Skill Gap Analyzer.
    Compares candidate proficiency S_i against target role requirement R_i.

    Formulas:
        G_i   = max(0, R_i - S_i)
        G_avg = (1 / n) × Σ G_i
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
            Dict containing detailed gap metrics, G_avg, and prioritised recommendations.
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
        readiness_score = round(max(0.0, 100.0 * (1.0 - (total_gap / (num_skills * 6.0)))), 1)

        recommendations = [
            f"Focus on '{item['skill']}' (Current: {item['current_proficiency']}/6, "
            f"Required: {item['required_proficiency']}/6, Gap: +{item['gap']})"
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


# -----------------------------------------------------------------------
# Leakage-free split helper
# -----------------------------------------------------------------------

def split_data_leak_free(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = "assessment_pass",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test partitions BEFORE fitting any transformations,
    ensuring zero data leakage between evaluation and training sets.
    Stratification preserves the pass/fail class ratio in both splits.
    """
    logger.info(
        f"Splitting dataset (N={len(df)}) with test_size={test_size}, "
        f"stratify='{target_column}', seed={random_state}..."
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column],
    )

    logger.info(f"Split complete — Train: {train_df.shape}, Test: {test_df.shape}")
    logger.info(
        f"Train class balance | Pass: {train_df[target_column].mean()*100:.1f}% "
        f"| Fail: {(1-train_df[target_column].mean())*100:.1f}%"
    )
    return train_df, test_df


# -----------------------------------------------------------------------
# End-to-end preprocessing pipeline stage
# -----------------------------------------------------------------------

def run_preprocessing_pipeline(config: Optional[AppConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data ingestion, label engineering, and train/test splitting stage.

    Steps:
      1. Ingest raw CSV (download if absent).
      2. Validate schema and value bounds.
      3. Generate binary 'assessment_pass' label via AssessmentLabelGenerator.
      4. Stratified 80/20 train/test split on 'assessment_pass'.
      5. Persist initial split parquets.
    """
    if config is None:
        config = load_config()

    raw_path = config.paths.raw_data_path
    if not raw_path.exists():
        logger.info(f"Raw data file not found at {raw_path}. Ingesting Kaggle dataset...")
        ingest_kaggle_dataset(output_path=raw_path)

    df = load_raw_data(raw_path)

    # Generate assessment pass/fail labels
    label_gen = AssessmentLabelGenerator(
        threshold=config.data.assessment_threshold,
        random_state=config.data.random_state,
    )
    df = label_gen.generate(df)

    train_df, test_df = split_data_leak_free(
        df=df,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        target_column="assessment_pass",
    )

    # Persist initial splits
    config.paths.processed_train_path.parent.mkdir(parents=True, exist_ok=True)
    temp_train_path = config.paths.processed_train_path.parent / "train_split.parquet"
    temp_test_path = config.paths.processed_test_path.parent / "test_split.parquet"

    train_df.to_parquet(temp_train_path, index=False)
    test_df.to_parquet(temp_test_path, index=False)
    logger.success(f"Saved initial splits: {temp_train_path} and {temp_test_path}")

    return train_df, test_df


if __name__ == "__main__":
    cfg = load_config()
    train, test = run_preprocessing_pipeline(cfg)

    # Demo: SkillGapAnalyzer on an example student profile
    analyzer = SkillGapAnalyzer(cfg.roles)
    example_student = {
        "ai_ml": 2, "data_science": 1, "programming_skills": 3,
        "database_fundamentals": 2, "distributed_computing_systems": 1,
        "software_engineering": 2,
    }
    result = analyzer.analyze_gap(example_student, target_role="AI ML Specialist")
    print(f"\n--- Skill Gap Analysis for: {result['target_role']} ---")
    print(f"G_avg: {result['average_skill_gap']} | Readiness: {result['readiness_percentage']}%")
    for rec in result["actionable_recommendations"]:
        print(f"  {rec}")
