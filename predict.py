"""
Real-Time Binary Assessment Pass/Fail Prediction CLI (Phase 1 Rework).

Usage:
    python predict.py

The student enters their 17 technical skill proficiency scores (0-6 each).
The system:
  1. Builds the feature vector and applies the saved StandardScaler.
  2. Predicts assessment_pass (Pass / Fail) using the champion Random Forest model.
  3. Shows confidence probabilities from all 3 trained classifiers.
  4. Optionally performs a SkillGapAnalyzer roadmap for a target role of their choice.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional
import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_loader import SKILL_COLUMNS, ROLE_REQUIREMENTS
from src.feature_engineering import engineer_composite_features
from src.preprocess import SkillGapAnalyzer
from src.roadmap import RoadmapGenerator


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

SKILL_DISPLAY_NAMES: Dict[str, str] = {
    "database_fundamentals":          "Database Fundamentals",
    "computer_architecture":          "Computer Architecture",
    "distributed_computing_systems":  "Distributed Computing Systems",
    "cyber_security":                 "Cyber Security",
    "networking":                     "Networking",
    "software_development":           "Software Development",
    "programming_skills":             "Programming Skills",
    "project_management":             "Project Management",
    "computer_forensics_fundamentals":"Computer Forensics Fundamentals",
    "technical_communication":        "Technical Communication",
    "ai_ml":                          "AI / Machine Learning",
    "software_engineering":           "Software Engineering",
    "business_analysis":              "Business Analysis",
    "communication_skills":           "Communication Skills",
    "data_science":                   "Data Science",
    "troubleshooting_skills":         "Troubleshooting Skills",
    "graphics_designing":             "Graphics Designing",
}


def _print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║   AI Technical Skill Development Assistant — Assessment Predictor   ║
║             Phase 1 Rework | Binary Classification                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def _get_skill_input(skill_name: str, display_name: str) -> int:
    while True:
        try:
            val = int(input(f"  {display_name:<40} [0-6]: ").strip())
            if 0 <= val <= 6:
                return val
            print("    ⚠  Please enter an integer between 0 and 6.")
        except ValueError:
            print("    ⚠  Invalid input. Please enter a whole number (0-6).")


def build_feature_vector(
    skill_scores: Dict[str, int],
    feature_columns: list,
    scaler_mean: list,
    scaler_scale: list,
) -> pd.DataFrame:
    """
    Build a scaled feature vector from raw skill scores that matches the
    trained pipeline's feature space.
    """
    # Create a single-row DataFrame with skill columns
    row = pd.DataFrame([skill_scores])

    # Engineer composite features (same logic as feature_engineering.py)
    row_eng = engineer_composite_features(row)

    # Align to the exact feature columns the pipeline was trained on
    # Fill any missing columns with 0
    for col in feature_columns:
        if col not in row_eng.columns:
            row_eng[col] = 0.0

    X = row_eng[feature_columns].values.astype(float)

    # Apply saved StandardScaler parameters
    mean = np.array(scaler_mean)
    scale = np.array(scaler_scale)
    X_scaled = (X - mean) / scale

    return pd.DataFrame(X_scaled, columns=feature_columns)


def predict_assessment(
    skill_scores: Dict[str, int],
    models_dir: Path,
    meta_path: Path,
) -> Dict:
    """
    Run inference with all 3 trained classifiers and return predictions.

    Returns:
        dict with keys: prediction (int), confidence (float), model_votes (dict)
    """
    with open(meta_path) as f:
        meta = json.load(f)

    feature_columns = meta["feature_columns"]
    scaler_mean = meta["scaler_mean"]
    scaler_scale = meta["scaler_scale"]

    X = build_feature_vector(skill_scores, feature_columns, scaler_mean, scaler_scale)

    model_files = {
        "Random Forest":       models_dir / "random_forest.joblib",
        "Decision Tree":       models_dir / "decision_tree.joblib",
        "Logistic Regression": models_dir / "logistic_regression.joblib",
    }

    results = {}
    votes = []
    for name, path in model_files.items():
        if not path.exists():
            continue
        model = joblib.load(path)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None
        results[name] = {"prediction": pred, "pass_probability": prob}
        votes.append(pred)

    # Majority vote as final prediction
    final_prediction = int(round(sum(votes) / len(votes)))
    # Champion (Random Forest) confidence as primary confidence
    champion_prob = results.get("Random Forest", {}).get("pass_probability", 0.5)

    return {
        "prediction": final_prediction,
        "pass_probability": champion_prob,
        "model_votes": results,
    }


# -----------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------

def main(config_path: str = "configs/config.yaml"):
    _print_banner()

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure you run this from the project root directory.")
        sys.exit(1)

    models_dir = config.paths.models_dir
    meta_path = config.paths.features_meta_path

    if not meta_path.exists():
        print("ERROR: Feature metadata not found. Please run 'dvc repro' first to train models.")
        sys.exit(1)

    # ── Collect skill scores ────────────────────────────────────────────
    print("Enter your technical skill proficiency scores on a scale of 0 to 6:")
    print("  0 = No experience  |  3 = Intermediate  |  6 = Expert\n")

    skill_scores = {}
    for skill_key in SKILL_COLUMNS:
        display = SKILL_DISPLAY_NAMES.get(skill_key, skill_key.replace("_", " ").title())
        skill_scores[skill_key] = _get_skill_input(skill_key, display)

    # ── Predict ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  🔍  Analysing your profile...")
    print("─" * 70)

    result = predict_assessment(skill_scores, models_dir, meta_path)

    pred = result["prediction"]
    prob = result["pass_probability"]
    label = "✅  PASS" if pred == 1 else "❌  FAIL"

    print(f"\n  Assessment Prediction  :  {label}")
    print(f"  Pass Probability       :  {prob * 100:.1f}%  (Random Forest champion)")
    print(f"\n  Individual Model Votes:")
    for model_name, vote_info in result["model_votes"].items():
        vote_label = "Pass" if vote_info["prediction"] == 1 else "Fail"
        prob_str = f"{vote_info['pass_probability']*100:.1f}%" if vote_info["pass_probability"] else "N/A"
        print(f"    {model_name:<25} →  {vote_label}  ({prob_str} pass confidence)")

    roadmap_gen = RoadmapGenerator()

    # ── If predicted Fail or borderline, display assessment acceleration roadmap
    if pred == 0 or prob < 0.60:
        print("\n  💡  RECOMMENDED ASSESSMENT ACCELERATION ROADMAP:")
        print("  To cross the 50% passing threshold for the interview round,")
        print("  prioritize closing these core assessment competency gaps:\n")
        assess_roadmap = roadmap_gen.generate_assessment_roadmap(skill_scores, target_score_level=4)
        for i, area in enumerate(assess_roadmap["priority_areas"][:3], 1):
            res_info = area["resource_info"]
            print(f"  {i}. {area['title']} (Current: {area['current']}/6 → Target: {area['target']}/6 | Gap: {area['gap']})")
            if res_info.get("resources"):
                top_res = res_info["resources"][0]
                print(f"     • Curated Resource: {top_res['name']} ({top_res.get('url', '')})")
            if res_info.get("practice"):
                print(f"     • Practice Track:   {res_info['practice'][0]}")
            if res_info.get("project"):
                print(f"     • Mini-Project:     {res_info['project']}")
            print()

    print("─" * 70)

    # ── Optional skill gap analysis ─────────────────────────────────────
    print("\n  Would you like a personalized multi-week learning roadmap for a target role?")
    roles = sorted(ROLE_REQUIREMENTS.keys())
    for i, role in enumerate(roles, 1):
        print(f"    {i:2d}. {role}")
    print("     0. Skip")

    while True:
        try:
            choice = int(input("\n  Enter role number: ").strip())
            if choice == 0:
                break
            if 1 <= choice <= len(roles):
                target_role = roles[choice - 1]
                analyzer = SkillGapAnalyzer(ROLE_REQUIREMENTS)
                gap_result = analyzer.analyze_gap(skill_scores, target_role)

                print(f"\n{'─'*70}")
                print(f"  📊  Skill Gap Roadmap — Target Role: {target_role}")
                print(f"{'─'*70}")
                print(f"  Role Readiness     : {gap_result['readiness_percentage']}%")
                print(f"  Average Skill Gap  : {gap_result['average_skill_gap']} points")

                role_roadmap = roadmap_gen.generate_role_roadmap(skill_scores, target_role)

                if role_roadmap["priority_areas"]:
                    print("\n  🚀  Structured Multi-Week Learning Sprints:")
                    for sprint in role_roadmap["sprints"]:
                        print(f"\n  [Week {sprint['week']}] {sprint['focus_area']}")
                        print(f"    • Core Topics : {', '.join(sprint['key_topics'])}")
                        print(f"    • Practice    : {', '.join(sprint['practice_targets'])}")
                        if sprint.get("recommended_resources"):
                            r = sprint["recommended_resources"][0]
                            print(f"    • Resource    : {r['name']} ({r.get('url', '')})")
                        print(f"    • Deliverable : {sprint['milestone_deliverable']}")
                else:
                    print("\n  🎉  You already meet all requirements for this role!")
                break
            else:
                print(f"    ⚠  Enter a number between 0 and {len(roles)}.")
        except ValueError:
            print("    ⚠  Invalid input. Please enter a number.")

    print(f"\n{'─'*70}")
    print("  Thank you for using the AI Technical Skill Development Assistant!")
    print(f"{'─'*70}\n")


if __name__ == "__main__":
    main()
