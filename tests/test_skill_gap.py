"""Unit tests for rule-based Skill Gap Analyzer."""

import pytest
from src.config import load_config
from src.preprocess import SkillGapAnalyzer


@pytest.fixture
def analyzer():
    cfg = load_config()
    return SkillGapAnalyzer(cfg.roles)


def test_skill_gap_formula_calculation(analyzer: SkillGapAnalyzer):
    """
    Test G_i = max(0, R_i - S_i) and G_avg formula.

    Example candidate targeting AI ML Specialist:
      requirements: ai_ml: 5, data_science: 5, programming_skills: 4,
                    database_fundamentals: 3, distributed_computing_systems: 3, software_engineering: 3
    """
    student_profile = {
        "ai_ml": 2,
        "data_science": 1,
        "programming_skills": 4,
        "database_fundamentals": 3,
        "distributed_computing_systems": 3,
        "software_engineering": 3,
    }

    result = analyzer.analyze_gap(student_profile, target_role="AI ML Specialist")

    gaps_by_skill = {item["skill"]: item["gap"] for item in result["skill_gaps"]}
    assert gaps_by_skill["data_science"] == 4
    assert gaps_by_skill["ai_ml"] == 3
    assert gaps_by_skill["programming_skills"] == 0
    assert gaps_by_skill["database_fundamentals"] == 0

    # Total gap = 4 + 3 = 7. 6 skills -> G_avg = 7 / 6 = 1.17
    assert result["average_skill_gap"] == 1.17
    assert len(result["actionable_recommendations"]) == 2


def test_ranking_priority_order(analyzer: SkillGapAnalyzer):
    """Verify that skills with larger gaps are ranked with highest priority."""
    student_profile = {
        "ai_ml": 2,
        "data_science": 1,
        "programming_skills": 2,
        "database_fundamentals": 1,
        "distributed_computing_systems": 3,
        "software_engineering": 3,
    }
    result = analyzer.analyze_gap(student_profile, target_role="AI ML Specialist")
    focus_order = result["recommended_focus_order"]

    # Top focus skill should have largest gap (data_science gap = 4)
    assert focus_order[0] == "data_science"
    assert focus_order[1] == "ai_ml"


def test_unsupported_role_raises_error(analyzer: SkillGapAnalyzer):
    """Verify requesting an unconfigured role raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        analyzer.analyze_gap({"ai_ml": 3}, target_role="Quantum Architect")


def test_exceeding_proficiency_has_zero_gap(analyzer: SkillGapAnalyzer):
    """Verify student exceeding requirement receives 0 gap (G_i = max(0, R_i - S_i))."""
    expert_profile = {
        "ai_ml": 6,
        "data_science": 6,
        "programming_skills": 6,
        "database_fundamentals": 6,
        "distributed_computing_systems": 6,
        "software_engineering": 6,
    }
    result = analyzer.analyze_gap(expert_profile, target_role="AI ML Specialist")
    assert result["average_skill_gap"] == 0.0
    assert len(result["recommended_focus_order"]) == 0
    assert result["readiness_percentage"] == 100.0
