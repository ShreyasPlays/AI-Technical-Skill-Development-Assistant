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

    Example student profile:
      python: 2, sql: 3, machine_learning: 1, dsa: 4, git: 2, etc.
    """
    student_profile = {
        "python": 2,
        "sql": 3,
        "machine_learning": 1,
        "dsa": 4,
        "git": 2,
        "web_development": 1,
        "cloud_computing": 1,
        "system_design": 2,
        "statistics": 2,
        "deep_learning": 1,
    }

    result = analyzer.analyze_gap(student_profile, target_role="Machine Learning Engineer")

    # In ML Engineer config:
    # python required = 4 -> gap = max(0, 4 - 2) = 2
    # sql required = 3 -> gap = max(0, 3 - 3) = 0
    # machine_learning required = 4 -> gap = max(0, 4 - 1) = 3
    # dsa required = 4 -> gap = max(0, 4 - 4) = 0
    # git required = 3 -> gap = max(0, 3 - 2) = 1
    # web_dev required = 1 -> gap = max(0, 1 - 1) = 0
    # cloud required = 3 -> gap = max(0, 3 - 1) = 2
    # system_design required = 3 -> gap = max(0, 3 - 2) = 1
    # statistics required = 4 -> gap = max(0, 4 - 2) = 2
    # deep_learning required = 4 -> gap = max(0, 4 - 1) = 3
    # Total gap = 2 + 0 + 3 + 0 + 1 + 0 + 2 + 1 + 2 + 3 = 14
    # G_avg = 14 / 10 = 1.40

    gaps_by_skill = {item["skill"]: item["gap"] for item in result["skill_gaps"]}
    assert gaps_by_skill["machine_learning"] == 3
    assert gaps_by_skill["deep_learning"] == 3
    assert gaps_by_skill["python"] == 2
    assert gaps_by_skill["sql"] == 0
    assert gaps_by_skill["dsa"] == 0

    assert result["average_skill_gap"] == 1.40
    assert len(result["actionable_recommendations"]) > 0


def test_ranking_priority_order(analyzer: SkillGapAnalyzer):
    """Verify that skills with larger gaps are ranked with highest priority."""
    student_profile = {
        "python": 1,
        "sql": 4,
        "machine_learning": 2,
        "dsa": 3,
        "git": 3,
        "web_development": 1,
        "cloud_computing": 1,
        "system_design": 1,
        "statistics": 3,
        "deep_learning": 1,
    }
    result = analyzer.analyze_gap(student_profile, target_role="Machine Learning Engineer")
    focus_order = result["recommended_focus_order"]

    # Top focus skills should have largest gaps
    assert focus_order[0] in ["python", "deep_learning", "machine_learning"]


def test_unsupported_role_raises_error(analyzer: SkillGapAnalyzer):
    """Verify requesting an unconfigured role raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        analyzer.analyze_gap({"python": 3}, target_role="Quantum Architect")


def test_exceeding_proficiency_has_zero_gap(analyzer: SkillGapAnalyzer):
    """Verify student exceeding requirement receives 0 gap (G_i = max(0, R_i - S_i))."""
    expert_profile = {
        "python": 5,
        "sql": 5,
        "machine_learning": 5,
        "dsa": 5,
        "git": 5,
        "web_development": 5,
        "cloud_computing": 5,
        "system_design": 5,
        "statistics": 5,
        "deep_learning": 5,
    }
    result = analyzer.analyze_gap(expert_profile, target_role="Machine Learning Engineer")
    assert result["average_skill_gap"] == 0.0
    assert len(result["recommended_focus_order"]) == 0
    assert result["readiness_percentage"] == 100.0
