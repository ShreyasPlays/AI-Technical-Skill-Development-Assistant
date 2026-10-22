"""
Unit tests for RoadmapGenerator and learning recommendation engine.
"""

import pytest
from src.roadmap import RoadmapGenerator
from src.data_loader import SKILL_COLUMNS, ROLE_REQUIREMENTS


def test_roadmap_generator_initialization():
    """Verify RoadmapGenerator loads all 17 technical attributes from YAML."""
    generator = RoadmapGenerator()
    assert len(generator.resources) >= 17
    for skill in SKILL_COLUMNS:
        res = generator.get_skill_resource(skill)
        assert "title" in res
        assert "core_topics" in res
        assert "resources" in res
        assert len(res["resources"]) >= 1


def test_assessment_roadmap_generation_for_weak_skills():
    """Verify assessment roadmap prioritizes core assessment technical gaps."""
    generator = RoadmapGenerator()
    low_skills = {s: 1 for s in SKILL_COLUMNS}

    roadmap = generator.generate_assessment_roadmap(low_skills, target_score_level=4)
    assert roadmap["type"] == "assessment_gate_roadmap"
    assert roadmap["total_gap_points"] > 0
    assert len(roadmap["priority_areas"]) == 6  # 6 core assessment skills
    assert len(roadmap["sprints"]) == 4

    # The highest weighted impact skill should be ranked first (programming_skills weight 0.22)
    top_skill = roadmap["priority_areas"][0]["skill"]
    assert top_skill == "programming_skills"


def test_role_roadmap_generation():
    """Verify role-specific roadmap correctly identifies gaps for Software Developer."""
    generator = RoadmapGenerator()
    skills = {s: 2 for s in SKILL_COLUMNS}

    roadmap = generator.generate_role_roadmap(skills, "Software Developer")
    assert roadmap["type"] == "role_roadmap"
    assert roadmap["target_role"] == "Software Developer"
    assert len(roadmap["priority_areas"]) > 0
    assert len(roadmap["sprints"]) >= 4


def test_no_gap_student_roadmap():
    """Verify student with expert skills gets completion milestone."""
    generator = RoadmapGenerator()
    expert_skills = {s: 6 for s in SKILL_COLUMNS}

    roadmap = generator.generate_assessment_roadmap(expert_skills, target_score_level=4)
    assert roadmap["total_gap_points"] == 0
    assert len(roadmap["priority_areas"]) == 0
    assert roadmap["sprints"][0]["title"] == "All Requirements Met"
