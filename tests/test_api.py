"""
Comprehensive unit and integration tests for FastAPI service routes and validation (Phase 2).
Tests route handlers, request validation, error states, and response schemas.
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.app import (
    batch_predict,
    check_input_risk,
    generate_roadmap,
    get_model_info,
    healthcheck,
    init_state,
    predict_assessment_pass,
    root_overview,
)
from src.api.schemas import (
    BatchPredictRequest,
    Century21SkillProfile,
    PredictRequest,
    RiskCheckRequest,
    RoadmapRequest,
)
from src.data_loader import CENTURY_21_SKILLS


@pytest.fixture(autouse=True, scope="module")
def setup_api_state():
    """Ensure microservice state and model artifacts are preloaded."""
    init_state()


def test_root_overview():
    """Verify root endpoint returns service overview and documentation links."""
    data = root_overview()
    assert data["service"] == "AI-Technical-Skill-Development-Assistant"
    assert data["version"] == "2.1.0"
    assert data["status"] == "online"
    assert "documentation" in data
    assert "endpoints" in data


def test_healthcheck():
    """Verify healthcheck probe returns healthy status and active champion model."""
    res = healthcheck()
    assert res.status == "healthy"
    assert res.version == "2.1.0"
    assert res.champion_model == "random_forest"
    assert res.uptime_seconds >= 0.0


def test_model_info():
    """Verify model lineage metadata returns active champion details."""
    res = get_model_info()
    assert res.model_name == "random_forest"
    assert res.stage == "Production"
    assert res.algorithm == "RandomForestClassifier"
    assert res.quality_gates_passed is True
    assert len(res.century_21_skills) == 10
    assert set(res.century_21_skills) == set(CENTURY_21_SKILLS)


def test_predict_pass_scenario():
    """Verify high-performing profile receives a Pass prediction with high probability."""
    strong_skills = Century21SkillProfile(**{skill: 5 for skill in CENTURY_21_SKILLS})
    req = PredictRequest(
        student_id="TEST_PASS_001",
        skills=strong_skills,
        target_role="AI ML Specialist",
    )
    res = predict_assessment_pass(req)
    assert res.student_id == "TEST_PASS_001"
    assert res.prediction == "Pass"
    assert res.pass_binary == 1
    assert res.pass_probability > 0.50
    assert res.champion_model == "random_forest"
    assert "random_forest" in res.ensemble_votes
    assert "decision_tree" in res.ensemble_votes
    assert "logistic_regression" in res.ensemble_votes
    assert res.roadmap_summary is not None
    assert res.roadmap_summary["target_role"] == "AI ML Specialist"


def test_predict_fail_scenario():
    """Verify a novice profile with low skill ratings receives a Fail prediction."""
    weak_skills = Century21SkillProfile(**{skill: 1 for skill in CENTURY_21_SKILLS})
    req = PredictRequest(
        student_id="TEST_FAIL_001",
        skills=weak_skills,
    )
    res = predict_assessment_pass(req)
    assert res.student_id == "TEST_FAIL_001"
    assert res.prediction == "Fail"
    assert res.pass_binary == 0
    assert res.pass_probability < 0.50


def test_predict_validation_error_out_of_bounds():
    """Verify ratings > 6 trigger a Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        Century21SkillProfile(
            ai_ml=3,
            programming_skills=9,  # Out of bounds!
        )


def test_predict_validation_error_negative():
    """Verify negative ratings trigger a ValidationError."""
    with pytest.raises(ValidationError):
        Century21SkillProfile(
            ai_ml=-1,  # Out of bounds!
        )


def test_batch_predict():
    """Verify batch inference handles multiple candidate profiles concurrently."""
    strong = Century21SkillProfile(**{skill: 5 for skill in CENTURY_21_SKILLS})
    weak = Century21SkillProfile(**{skill: 1 for skill in CENTURY_21_SKILLS})

    req = BatchPredictRequest(
        students=[
            PredictRequest(student_id="BATCH_01", skills=strong),
            PredictRequest(student_id="BATCH_02", skills=weak),
        ]
    )
    res = batch_predict(req)
    assert res.total_records == 2
    assert len(res.results) == 2
    assert res.total_passed >= 1
    assert res.total_failed >= 1


def test_roadmap_endpoint():
    """Verify roadmap endpoint computes gaps and recommends curated resources."""
    skills = Century21SkillProfile(**{skill: 2 for skill in CENTURY_21_SKILLS})
    req = RoadmapRequest(
        student_id="ROADMAP_001",
        skills=skills,
        target_role="Software Developer",
    )
    res = generate_roadmap(req)
    assert res.student_id == "ROADMAP_001"
    assert res.target_role == "Software Developer"
    assert res.average_skill_gap > 0.0
    assert len(res.gap_breakdown) > 0
    assert len(res.recommended_learning_plan) > 0


def test_roadmap_invalid_role():
    """Verify requesting an unsupported role raises an HTTPException (400)."""
    skills = Century21SkillProfile(**{skill: 3 for skill in CENTURY_21_SKILLS})
    req = RoadmapRequest(
        skills=skills,
        target_role="Quantum Chef Specialist",
    )
    with pytest.raises(HTTPException) as exc:
        generate_roadmap(req)
    assert exc.value.status_code == 400
    assert "Unsupported role" in exc.value.detail


def test_risk_check_endpoint():
    """Verify risk auditing detects out-of-bounds ratings and computes risk score."""
    clean_skills = {skill: 4 for skill in CENTURY_21_SKILLS}
    req = RiskCheckRequest(skills=clean_skills)
    res = check_input_risk(req)
    assert res.is_valid is True
    assert res.risk_level in ["LOW", "MEDIUM"]

    # Test dirty input with an out-of-bounds rating
    dirty_skills = {"ai_ml": 10, "programming_skills": 3}
    dirty_req = RiskCheckRequest(skills=dirty_skills)
    dirty_res = check_input_risk(dirty_req)
    assert dirty_res.is_valid is False
    assert dirty_res.risk_level == "HIGH"
    assert "ai_ml" in dirty_res.out_of_bounds
