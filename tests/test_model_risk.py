"""
Unit tests for Model Behavioral Testing, Quality Gates, and Risk Audit (Phase 2).
"""

import pytest
from src.data_loader import CENTURY_21_SKILLS
from src.model_risk import ModelQualityAuditor


@pytest.fixture(scope="module")
def auditor():
    return ModelQualityAuditor()


def test_boundary_extremes_behavior(auditor: ModelQualityAuditor):
    """Verify zero skills produces low pass prob and max skills produces high pass prob."""
    res = auditor.test_boundary_extremes()
    assert res["passed"] is True
    assert res["zero_skill_pass_prob"] < 0.20
    assert res["max_skill_pass_prob"] > 0.80


def test_directional_monotonicity_programming(auditor: ModelQualityAuditor):
    """Verify increasing programming skills correlates positively with pass probability."""
    res = auditor.test_monotonicity("programming_skills")
    assert res["passed"] is True
    assert res["correlation"] > 0.70
    assert res["prob_level_6"] > res["prob_level_0"]


def test_invariance_on_non_core_skill(auditor: ModelQualityAuditor):
    """Verify perturbations on non-core skill do not flip high confidence predictions."""
    res = auditor.test_invariance("graphics_designing")
    assert res["passed"] is True
    assert all(p == 1 for p in res["predictions"])


def test_inference_latency_benchmark(auditor: ModelQualityAuditor):
    """Verify single prediction latency is within acceptable production limits."""
    res = auditor.test_inference_latency(sample_size=50)
    assert res["passed"] is True
    assert res["avg_latency_ms"] <= res["max_allowed_ms"]


def test_input_auditing_clean_profile(auditor: ModelQualityAuditor):
    """Verify a clean 21st-century profile passes audit with LOW risk."""
    clean = {s: 4 for s in CENTURY_21_SKILLS}
    res = auditor.audit_input_profile(clean)
    assert res.is_valid is True
    assert res.risk_level in ["LOW", "MEDIUM"]
    assert len(res.out_of_bounds) == 0


def test_input_auditing_out_of_bounds(auditor: ModelQualityAuditor):
    """Verify out-of-bounds ratings trigger invalid status and HIGH risk."""
    dirty = {s: 3 for s in CENTURY_21_SKILLS}
    dirty["data_science"] = 12
    res = auditor.audit_input_profile(dirty)
    assert res.is_valid is False
    assert res.risk_level == "HIGH"
    assert "data_science" in res.out_of_bounds

