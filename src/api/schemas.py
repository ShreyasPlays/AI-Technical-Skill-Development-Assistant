"""
Pydantic v2 schemas for FastAPI inference and diagnostic service.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Century21SkillProfile(BaseModel):
    """
    Candidate ratings across the 10 core 21st-century technical skill domains.
    Ratings must be integers on a 0 to 6 scale.
    """
    ai_ml: int = Field(default=3, ge=0, le=6, description="Artificial Intelligence & Machine Learning proficiency (0-6)")
    data_science: int = Field(default=3, ge=0, le=6, description="Data Science, Big Data & Analytics proficiency (0-6)")
    software_development: int = Field(default=3, ge=0, le=6, description="Full-stack & Modern Software Development proficiency (0-6)")
    programming_skills: int = Field(default=3, ge=0, le=6, description="Algorithmic problem solving & core programming proficiency (0-6)")
    cyber_security: int = Field(default=3, ge=0, le=6, description="Cybersecurity, DevSecOps & Privacy proficiency (0-6)")
    distributed_computing_systems: int = Field(default=3, ge=0, le=6, description="Distributed Systems & Cloud Computing proficiency (0-6)")
    database_fundamentals: int = Field(default=3, ge=0, le=6, description="Modern Databases, SQL & NoSQL proficiency (0-6)")
    software_engineering: int = Field(default=3, ge=0, le=6, description="Systems Architecture, Testing & CI/CD proficiency (0-6)")
    troubleshooting_skills: int = Field(default=3, ge=0, le=6, description="Observability, Debugging & Systems Troubleshooting proficiency (0-6)")
    technical_communication: int = Field(default=3, ge=0, le=6, description="Agile Collaboration & Technical Communication proficiency (0-6)")

    # Optional legacy skills with sensible defaults (0) for backward compatibility
    computer_architecture: Optional[int] = Field(default=0, ge=0, le=6)
    networking: Optional[int] = Field(default=0, ge=0, le=6)
    project_management: Optional[int] = Field(default=0, ge=0, le=6)
    computer_forensics_fundamentals: Optional[int] = Field(default=0, ge=0, le=6)
    business_analysis: Optional[int] = Field(default=0, ge=0, le=6)
    communication_skills: Optional[int] = Field(default=0, ge=0, le=6)
    graphics_designing: Optional[int] = Field(default=0, ge=0, le=6)

    def to_dict(self) -> Dict[str, int]:
        """Return clean dict of all skills."""
        return self.model_dump()


class PredictRequest(BaseModel):
    """Request payload for real-time assessment pass/fail prediction."""
    student_id: Optional[str] = Field(default="RVCE26CS001", description="Optional student identifier (e.g., RVCE26CS001)")
    skills: Century21SkillProfile = Field(..., description="Student technical skill ratings (0-6 scale)")
    target_role: Optional[str] = Field(default=None, description="Optional target industry role for roadmap analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "student_id": "STUDENT_RVCE_01",
                "skills": {
                    "ai_ml": 4,
                    "data_science": 4,
                    "software_development": 4,
                    "programming_skills": 5,
                    "cyber_security": 3,
                    "distributed_computing_systems": 3,
                    "database_fundamentals": 4,
                    "software_engineering": 4,
                    "troubleshooting_skills": 4,
                    "technical_communication": 4,
                },
                "target_role": None,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response payload containing assessment prediction and confidence probabilities."""
    student_id: Optional[str] = None
    prediction: str = Field(..., description="'Pass' or 'Fail'")
    pass_binary: int = Field(..., description="1 = Pass, 0 = Fail")
    pass_probability: float = Field(..., description="Estimated probability of passing [0.0 - 1.0]")
    confidence_pct: float = Field(..., description="Decision confidence percentage [0.0 - 100.0%]")
    champion_model: str = Field(..., description="Active production model used for primary decision")
    ensemble_votes: Dict[str, str] = Field(..., description="Pass/Fail votes across all 3 trained classifiers")
    uncertainty_flag: bool = Field(..., description="True if student is near the 0.50 decision threshold")
    risk_level: str = Field(..., description="Quality/Risk assessment: LOW, MEDIUM, or HIGH")
    roadmap_summary: Optional[Dict[str, Any]] = None


class BatchPredictRequest(BaseModel):
    """Batch prediction request containing multiple candidate profiles."""
    students: List[PredictRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Batch prediction response containing results for all candidates."""
    total_records: int
    total_passed: int
    total_failed: int
    pass_rate_pct: float
    results: List[PredictionResponse]


class RoadmapRequest(BaseModel):
    """Request payload for personalized skill gap roadmap."""
    student_id: Optional[str] = None
    skills: Century21SkillProfile
    target_role: str = Field(..., description="Target role (e.g. 'AI ML Specialist', 'Software Developer')")


class RoadmapResponse(BaseModel):
    """Response payload containing targeted learning priorities and curated resources."""
    student_id: Optional[str] = None
    target_role: str
    average_skill_gap: float
    role_readiness_status: str
    gap_breakdown: List[Dict[str, Any]]
    recommended_learning_plan: Dict[str, Any]


class RiskCheckRequest(BaseModel):
    """Request payload for input risk and data drift inspection."""
    skills: Dict[str, Any] = Field(..., description="Dictionary of skill ratings")


class RiskCheckResponse(BaseModel):
    """Response payload for input risk and distribution verification."""
    is_valid: bool
    risk_level: str
    risk_score: float
    uncertainty_flag: bool
    warnings: List[str]
    missing_skills: List[str]
    out_of_bounds: Dict[str, Any]


class HealthResponse(BaseModel):
    """Service health and uptime response."""
    status: str
    service_name: str
    version: str
    champion_model: str
    mlflow_tracking_active: bool
    server_time: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Detailed production model metadata and registry lineage."""
    model_name: str
    stage: str
    version: str
    algorithm: str
    artifact_sha256: str
    mlflow_run_id: str
    test_metrics: Dict[str, float]
    quality_gates_passed: bool
    feature_count: int
    century_21_skills: List[str]
