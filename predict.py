"""
Interactive Prediction and Skill-Gap Analysis Demo.
Evaluates student technical competencies against Kaggle Career benchmarks:
1. Predicts the best-fitting career role using the trained Random Forest Champion model.
2. Performs mathematical skill-gap analysis: G_i = max(0, R_i - S_i).
3. Produces a prioritized, actionable skill development roadmap.
"""

import sys
import json
from typing import Dict
import joblib
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import load_config
from src.feature_engineering import engineer_composite_features
from src.preprocess import SkillGapAnalyzer


def predict_profile(skills_dict: Dict[str, float]) -> None:
    """Predict career role and analyze skill gaps for a given candidate profile."""
    cfg = load_config()

    meta_path = cfg.paths.features_meta_path
    rf_path = cfg.paths.models_dir / "random_forest.joblib"

    if not meta_path.exists() or not rf_path.exists():
        print("Error: Models or feature metadata not found. Please train models first.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rf_model = joblib.load(rf_path)

    # 1. Fill missing context features with average defaults if not specified
    for ctx in cfg.data.context_features:
        if ctx not in skills_dict:
            skills_dict[ctx] = 0.5

    for skill in cfg.data.skill_features:
        if skill not in skills_dict:
            skills_dict[skill] = 2

    # 2. Engineer composite features
    df_raw = pd.DataFrame([skills_dict])
    df_eng = engineer_composite_features(df_raw)

    feature_cols = meta["feature_columns"]
    X = df_eng[feature_cols].copy()

    # 3. Scale features using saved training parameters
    mean = np.array(meta["scaler_mean"])
    scale = np.array(meta["scaler_scale"])
    X_scaled = pd.DataFrame((X.values - mean) / scale, columns=feature_cols)

    # 4. Model Prediction
    pred_idx = rf_model.predict(X_scaled)[0]
    probabilities = rf_model.predict_proba(X_scaled)[0]
    predicted_role = meta["classes"][str(pred_idx)]

    print("\n" + "=" * 68)
    print(f"[*] PREDICTED CAREER ROLE: {predicted_role.upper()}")
    print("=" * 68)
    print("\n[+] Top Model Confidence Probabilities:")
    top_indices = np.argsort(probabilities)[::-1][:5]
    for idx in top_indices:
        prob = probabilities[idx]
        role_name = meta["classes"][str(idx)]
        bar = "#" * int(prob * 30)
        print(f"  {role_name:<32} : {prob * 100:>5.1f}% | {bar}")

    # 5. Skill Gap Analysis for the predicted role
    analyzer = SkillGapAnalyzer(cfg.roles)
    gap_result = analyzer.analyze_gap(skills_dict, target_role=predicted_role)

    print("\n" + "-" * 68)
    print(f"[i] Skill-Gap Breakdown for {predicted_role}:")
    print(f"  * Overall Readiness Score : {gap_result['readiness_percentage']}%")
    print(f"  * Average Skill Gap G_avg : {gap_result['average_skill_gap']}")
    print("\n[*] Priority Recommendations to Bridge Gaps:")
    recs = gap_result["actionable_recommendations"]
    if recs:
        for rec in recs:
            print(f"  - {rec}")
    else:
        print("  - [EXCELLENT] Candidate already meets or exceeds all role benchmarks!")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 68)
    print("  AI TECHNICAL SKILL DEVELOPMENT ASSISTANT - INFERENCE DEMO")
    print("=" * 68)
    print("\nChoose an option:")
    print("  1. Test Preset Profile: AI & Data Science Candidate")
    print("  2. Test Preset Profile: Cyber & Network Security Candidate")
    print("  3. Test Preset Profile: Software & App Developer Candidate")
    print("  4. Enter Your Own Custom Skill Ratings (0 = None, 6 = Expert)")

    choice = input("\nEnter choice [1-4] (default=1): ").strip() or "1"

    if choice == "1":
        candidate = {
            "ai_ml": 5, "data_science": 5, "programming_skills": 4, "database_fundamentals": 3,
            "distributed_computing_systems": 3, "software_engineering": 3, "networking": 1,
            "cyber_security": 1, "computer_architecture": 2, "project_management": 2,
            "technical_communication": 3, "communication_skills": 4, "troubleshooting_skills": 3,
            "graphics_designing": 1, "business_analysis": 2, "computer_forensics_fundamentals": 1,
            "software_development": 3, "openness": 0.8, "conscientiousness": 0.7, "extraversion": 0.5,
            "agreeableness": 0.6, "emotional_range": 0.3, "conversation": 0.4, "openness_to_change": 0.7,
            "hedonism": 0.2, "self_enhancement": 0.5, "self_transcendence": 0.4,
        }
    elif choice == "2":
        candidate = {
            "ai_ml": 1, "data_science": 1, "programming_skills": 3, "database_fundamentals": 2,
            "distributed_computing_systems": 3, "software_engineering": 2, "networking": 5,
            "cyber_security": 6, "computer_architecture": 3, "project_management": 2,
            "technical_communication": 3, "communication_skills": 3, "troubleshooting_skills": 5,
            "graphics_designing": 0, "business_analysis": 2, "computer_forensics_fundamentals": 5,
            "software_development": 2, "openness": 0.6, "conscientiousness": 0.8, "extraversion": 0.4,
            "agreeableness": 0.5, "emotional_range": 0.2, "conversation": 0.3, "openness_to_change": 0.5,
            "hedonism": 0.1, "self_enhancement": 0.6, "self_transcendence": 0.3,
        }
    elif choice == "3":
        candidate = {
            "ai_ml": 2, "data_science": 2, "programming_skills": 5, "database_fundamentals": 4,
            "distributed_computing_systems": 3, "software_engineering": 5, "networking": 2,
            "cyber_security": 2, "computer_architecture": 3, "project_management": 3,
            "technical_communication": 3, "communication_skills": 4, "troubleshooting_skills": 4,
            "graphics_designing": 2, "business_analysis": 3, "computer_forensics_fundamentals": 1,
            "software_development": 6, "openness": 0.7, "conscientiousness": 0.7, "extraversion": 0.5,
            "agreeableness": 0.6, "emotional_range": 0.3, "conversation": 0.5, "openness_to_change": 0.6,
            "hedonism": 0.3, "self_enhancement": 0.5, "self_transcendence": 0.4,
        }
    else:
        cfg = load_config()
        print("\nEnter skill ratings (0 = Novice, 6 = Expert):")
        candidate = {}
        for skill in cfg.data.skill_features:
            val = input(f"  {skill:<32} (0-6): ").strip()
            candidate[skill] = int(val) if val.isdigit() and 0 <= int(val) <= 6 else 3

    predict_profile(candidate)
