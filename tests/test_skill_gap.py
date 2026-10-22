"""
Unit tests for rule-based Skill Gap Analyzer.

The SkillGapAnalyzer is unchanged in the rework — it is used by predict.py
to generate personalised learning roadmaps AFTER the binary pass/fail
assessment prediction is made.
"""

import pytest
from src.config import load_config
from src.data_loader import ROLE_REQUIREMENTS
from src.preprocess import SkillGapAnalyzer


@pytest.fixture
def analyzer():
    """Fixture using the ROLE_REQUIREMENTS from data_loader (same as predict.py uses)."""
    return SkillGapAnalyzer(ROLE_REQUIREMENTS)


@pytest.fixture
def config_analyzer():
    """Fixture using roles from config.yaml (for config-consistency tests)."""
    cfg = load_config()
    return SkillGapAnalyzer(cfg.roles)


def test_skill_gap_formula_calculation(analyzer: SkillGapAnalyzer):
    """
    Test G_i = max(0, R_i - S_i) and G_avg formula.

    Example candidate targeting AI ML Specialist:
      requirements: ai_ml: 5, data_science: 5, programming_skills: 4,
                    database_fundamentals: 3, distributed_computing_systems: 3,
                    software_engineering: 3
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

    # Total gap = 4 + 3 = 7.  6 skills → G_avg = 7 / 6 = 1.17
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

    # Top focus skill should have the largest gap (data_science gap = 4)
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


def test_all_16_roles_are_supported(analyzer: SkillGapAnalyzer):
    """Verify all 16 career roles are supported for gap analysis."""
    from src.data_loader import ROLE_CLASSES
    supported = set(analyzer.get_supported_roles())
    for role in ROLE_CLASSES:
        assert role in supported, f"Role '{role}' is missing from ROLE_REQUIREMENTS."


def test_config_and_data_loader_roles_consistent(config_analyzer: SkillGapAnalyzer):
    """Verify config.yaml roles dict matches the 16 roles in ROLE_REQUIREMENTS."""
    from src.data_loader import ROLE_REQUIREMENTS as data_reqs
    config_roles = set(config_analyzer.get_supported_roles())
    data_roles = set(data_reqs.keys())
    assert config_roles == data_roles, (
        f"Roles in config.yaml and data_loader.ROLE_REQUIREMENTS must match.\n"
        f"  In config only: {config_roles - data_roles}\n"
        f"  In data_loader only: {data_roles - config_roles}"
    )
