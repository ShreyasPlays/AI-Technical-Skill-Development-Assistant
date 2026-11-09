"""
FastAPI production inference and diagnostic microservice (Phase 2).

Provides:
- GET  /              : Service overview and interactive docs links
- GET  /health        : Liveness and readiness healthcheck probe
- GET  /model/info    : Active production model metadata & registry lineage
- POST /predict       : Real-time student assessment pass/fail prediction
- POST /predict/batch : High-throughput batch inference
- POST /roadmap       : Targeted 21st-century skill gap roadmap generation
- POST /risk/check    : Input risk & quality audit probe
"""

from contextlib import asynccontextmanager
from datetime import datetime
import json
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
from loguru import logger
import numpy as np
import pandas as pd

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictionResponse,
    RiskCheckRequest,
    RiskCheckResponse,
    RoadmapRequest,
    RoadmapResponse,
)
from src.config import AppConfig, load_config
from src.data_loader import CENTURY_21_SKILLS, ROLE_CLASSES, ROLE_REQUIREMENTS, SKILL_COLUMNS
from src.feature_engineering import engineer_composite_features
from src.model_registry import ModelRegistryManager
from src.model_risk import ModelQualityAuditor
from src.preprocess import SkillGapAnalyzer
from src.roadmap import RoadmapGenerator


# Global state container
class ServiceState:
    start_time: float = time.time()
    config: Optional[AppConfig] = None
    champion_model: Any = None
    all_models: Dict[str, Any] = {}
    feature_columns: List[str] = []
    scaler_mean: np.ndarray = np.array([])
    scaler_scale: np.ndarray = np.array([])
    registry_manifest: Dict[str, Any] = {}
    auditor: Optional[ModelQualityAuditor] = None
    roadmap_gen: Optional[RoadmapGenerator] = None
    skill_gap_analyzer: Optional[SkillGapAnalyzer] = None


state = ServiceState()


def init_state(force_reload: bool = False):
    """Initialize or reload model artifacts and auxiliary services."""
    if state.champion_model is not None and not force_reload:
        return

    logger.info("Initializing AI Technical Skill Assistant state...")
    state.start_time = time.time()
    state.config = load_config()

    models_dir = state.config.paths.models_dir
    champion_name = state.config.phase2.api.champion_model_name

    # Load champion and candidate models
    for name in ["random_forest", "decision_tree", "logistic_regression"]:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            state.all_models[name] = joblib.load(path)
            logger.info(f"Loaded estimator: '{name}'")
        else:
            logger.warning(f"Model artifact not found for '{name}' at {path}")

    state.champion_model = state.all_models.get(champion_name)
    if state.champion_model is None:
        logger.error(f"FATAL: Champion model '{champion_name}' could not be loaded!")

    # Load feature metadata & scaler params
    meta_path = state.config.paths.features_meta_path
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        state.feature_columns = meta.get("feature_columns", [])
        state.scaler_mean = np.array(meta.get("scaler_mean", []))
        state.scaler_scale = np.array(meta.get("scaler_scale", []))
        logger.info(f"Loaded feature contract: {len(state.feature_columns)} features")

    # Load registry manifest
    manifest_path = state.config.phase2.model_registry.manifest_path
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            state.registry_manifest = json.load(f)
    else:
        reg_mgr = ModelRegistryManager(state.config)
        state.registry_manifest = reg_mgr.build_registry_manifest()

    # Initialize auxiliary services
    state.auditor = ModelQualityAuditor(state.config)
    state.roadmap_gen = RoadmapGenerator()
    state.skill_gap_analyzer = SkillGapAnalyzer(ROLE_REQUIREMENTS)
    logger.success("Service state initialized successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    init_state()
    yield
    logger.info("Service shutting down cleanly.")


app = FastAPI(
    title="AI Technical Skill Development Assistant API",
    description="Engineering-grade MLOps inference and skill-gap diagnostic service for campus placement readiness.",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web dashboards (e.g. React, Vue, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _transform_skill_dict_to_input(skills: Dict[str, int]) -> pd.DataFrame:
    """Preprocess and scale raw skills dictionary matching the model feature contract."""
    row = pd.DataFrame([skills])
    for s in SKILL_COLUMNS:
        if s not in row.columns:
            row[s] = 0

    row_eng = engineer_composite_features(row)
    for col in state.feature_columns:
        if col not in row_eng.columns:
            row_eng[col] = 0.0

    X_raw = row_eng[state.feature_columns].values.astype(float)
    X_scaled = (X_raw - state.scaler_mean) / state.scaler_scale
    return pd.DataFrame(X_scaled, columns=state.feature_columns)


# ── Route Handlers ───────────────────────────────────────────────────────────

@app.get("/", summary="Root Overview")
def root_overview():
    """Welcome and quick links to API documentation."""
    return {
        "service": "AI-Technical-Skill-Development-Assistant",
        "version": "2.1.0",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "roadmap": "POST /roadmap",
            "risk_check": "POST /risk/check",
            "model_info": "GET /model/info",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="Liveness & Readiness Probe")
def healthcheck():
    """Returns microservice health status, active champion model, and uptime."""
    is_ready = state.champion_model is not None and len(state.feature_columns) > 0
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service_name="AI-Technical-Skill-Development-Assistant",
        version="2.1.0",
        champion_model=state.config.phase2.api.champion_model_name if state.config else "random_forest",
        mlflow_tracking_active=True,
        server_time=datetime.utcnow().isoformat() + "Z",
        uptime_seconds=round(time.time() - state.start_time, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model Registry Lineage")
def get_model_info():
    """Returns detailed production metadata for the registered champion model."""
    champ = state.registry_manifest.get("active_champion", {})
    return ModelInfoResponse(
        model_name=champ.get("model_name", "random_forest"),
        stage=champ.get("stage", "Production"),
        version=champ.get("version", "1.0.0"),
        algorithm=champ.get("model_class", "RandomForestClassifier"),
        artifact_sha256=champ.get("sha256", "UNKNOWN"),
        mlflow_run_id=champ.get("mlflow_run_id", "UNKNOWN"),
        test_metrics=champ.get("metrics", {}),
        quality_gates_passed=champ.get("quality_passed", True),
        feature_count=len(state.feature_columns),
        century_21_skills=CENTURY_21_SKILLS,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Real-Time Assessment Prediction")
def predict_assessment_pass(payload: PredictRequest):
    """
    Predict whether a student will pass or fail the online technical assessment screening gate.
    Evaluates confidence percentages and queries ensemble models.
    """
    if state.champion_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model is not loaded. Please verify model training and registry artifacts.",
        )

    skills_dict = payload.skills.to_dict()
    X_df = _transform_skill_dict_to_input(skills_dict)

    # Primary prediction via Champion Random Forest
    prob_pass = float(state.champion_model.predict_proba(X_df)[0, 1])
    pass_binary = int(prob_pass >= 0.50)
    prediction_label = "Pass" if pass_binary == 1 else "Fail"

    # Confidence calculation: distance from decision threshold 0.50
    confidence = abs(prob_pass - 0.50) * 2.0 * 100.0

    # Ensemble voting across all 3 classifiers
    votes = {}
    for name, model in state.all_models.items():
        v = int(model.predict(X_df)[0])
        votes[name] = "Pass" if v == 1 else "Fail"

    # Risk & uncertainty check
    uncertainty_flag = (0.45 <= prob_pass <= 0.55)
    risk_level = "HIGH" if uncertainty_flag else ("MEDIUM" if 0.40 <= prob_pass <= 0.60 else "LOW")

    # Optional roadmap generation if target role is provided
    roadmap_summary = None
    if payload.target_role:
        if payload.target_role not in ROLE_CLASSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target role '{payload.target_role}' not supported. Allowed roles: {ROLE_CLASSES}",
            )
        gap_res = state.skill_gap_analyzer.analyze_gap(skills_dict, payload.target_role)
        roadmap_summary = {
            "target_role": payload.target_role,
            "average_skill_gap": gap_res["average_skill_gap"],
            "top_gap_skills": [g["skill"] for g in gap_res["skill_gaps"] if g["gap"] > 0][:3],
        }

    return PredictionResponse(
        student_id=payload.student_id,
        prediction=prediction_label,
        pass_binary=pass_binary,
        pass_probability=round(prob_pass, 4),
        confidence_pct=round(confidence, 2),
        champion_model=state.config.phase2.api.champion_model_name,
        ensemble_votes=votes,
        uncertainty_flag=uncertainty_flag,
        risk_level=risk_level,
        roadmap_summary=roadmap_summary,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch Assessment Prediction")
def batch_predict(payload: BatchPredictRequest):
    """Processes batch inference for multiple student candidates."""
    results: List[PredictionResponse] = []
    for item in payload.students:
        res = predict_assessment_pass(item)
        results.append(res)

    total = len(results)
    passed = sum(1 for r in results if r.pass_binary == 1)
    failed = total - passed
    pass_rate = round((passed / total) * 100.0, 2) if total > 0 else 0.0

    return BatchPredictResponse(
        total_records=total,
        total_passed=passed,
        total_failed=failed,
        pass_rate_pct=pass_rate,
        results=results,
    )


@app.post("/roadmap", response_model=RoadmapResponse, summary="Generate Personalised Learning Roadmap")
def generate_roadmap(payload: RoadmapRequest):
    """
    Computes mathematical skill gaps (G_i = max(0, R_i - S_i)) and recommends
    actionable, multi-modal learning resources for the student's chosen career role.
    """
    if payload.target_role not in ROLE_CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported role '{payload.target_role}'. Must be one of: {ROLE_CLASSES}",
        )

    skills_dict = payload.skills.to_dict()
    gap_result = state.skill_gap_analyzer.analyze_gap(skills_dict, payload.target_role)

    # Generate curated resource recommendations
    roadmap_items = state.roadmap_gen.generate_role_roadmap(skills_dict, payload.target_role)

    # Readiness status categorization
    avg_gap = gap_result["average_skill_gap"]
    if avg_gap == 0.0:
        readiness = "Industry Ready (Meets or exceeds all baseline requirements)"
    elif avg_gap < 1.0:
        readiness = "Near Ready (Minor skill brush-up recommended)"
    elif avg_gap < 2.5:
        readiness = "Moderate Gap (Requires targeted 4–8 week development)"
    else:
        readiness = "Significant Gap (Requires comprehensive 12+ week foundational curriculum)"

    return RoadmapResponse(
        student_id=payload.student_id,
        target_role=payload.target_role,
        average_skill_gap=avg_gap,
        role_readiness_status=readiness,
        gap_breakdown=gap_result["skill_gaps"],
        recommended_learning_plan=roadmap_items,
    )


@app.post("/risk/check", response_model=RiskCheckResponse, summary="Input Quality & Drift Check")
def check_input_risk(payload: RiskCheckRequest):
    """Audits input ratings for schema consistency, numerical bounds [0-6], and decision risk."""
    if state.auditor is None:
        state.auditor = ModelQualityAuditor(state.config)

    audit = state.auditor.audit_input_profile(payload.skills)
    return RiskCheckResponse(
        is_valid=audit.is_valid,
        risk_level=audit.risk_level,
        risk_score=audit.risk_score,
        uncertainty_flag=audit.uncertainty_flag,
        warnings=audit.warnings,
        missing_skills=audit.missing_skills,
        out_of_bounds=audit.out_of_bounds,
    )
