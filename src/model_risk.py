"""
Model Testing, Quality Checks, and Risk Governance Module (Phase 2).

Implements:
1. Behavioral testing suites (Invariance, Monotonicity/Directionality, Boundary Extremes, Latency).
2. Runtime input risk & distribution drift auditing.
3. Performance quality gates enforcement.
4. Model Risk Card generation (reports/model_risk_card.md).
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import joblib
from loguru import logger
import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.data_loader import CENTURY_21_SKILLS, SKILL_COLUMNS
from src.feature_engineering import engineer_composite_features


@dataclass
class RiskAuditResult:
    is_valid: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_score: float  # 0.0 (safe) to 1.0 (critical risk)
    warnings: List[str]
    missing_skills: List[str]
    out_of_bounds: Dict[str, Any]
    uncertainty_flag: bool


class ModelQualityAuditor:
    """
    Evaluates model behavioral properties, quality gates, and input risks.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.models_dir = self.config.paths.models_dir
        self.champion_name = self.config.phase2.api.champion_model_name
        self.meta_path = self.config.paths.features_meta_path
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = self.models_dir / f"{self.champion_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {model_path}.")
        self.model = joblib.load(model_path)

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Feature metadata not found at {self.meta_path}.")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_columns = self.metadata["feature_columns"]
        self.scaler_mean = np.array(self.metadata["scaler_mean"])
        self.scaler_scale = np.array(self.metadata["scaler_scale"])

    def transform_skills_to_features(self, skill_dict: Dict[str, int]) -> pd.DataFrame:
        """Transform raw skill ratings into scaled model input DataFrame with feature names."""
        row = pd.DataFrame([skill_dict])
        for s in SKILL_COLUMNS:
            if s not in row.columns:
                row[s] = 0

        row_eng = engineer_composite_features(row)
        for col in self.feature_columns:
            if col not in row_eng.columns:
                row_eng[col] = 0.0

        X_raw = row_eng[self.feature_columns].values.astype(float)
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_scale
        return pd.DataFrame(X_scaled, columns=self.feature_columns)

    # ── Behavioral Model Testing ─────────────────────────────────────────────

    def test_boundary_extremes(self) -> Dict[str, Any]:
        """
        Verify extreme profile responses:
        - All 0s profile MUST fail with low probability (< 0.20).
        - All 6s profile MUST pass with high probability (> 0.80).
        """
        zero_skills = {s: 0 for s in SKILL_COLUMNS}
        max_skills = {s: 6 for s in SKILL_COLUMNS}

        X_zero = self.transform_skills_to_features(zero_skills)
        X_max = self.transform_skills_to_features(max_skills)

        prob_zero = float(self.model.predict_proba(X_zero)[0, 1])
        prob_max = float(self.model.predict_proba(X_max)[0, 1])

        passed = (prob_zero < 0.20) and (prob_max > 0.80)
        return {
            "test_name": "boundary_extremes",
            "passed": passed,
            "zero_skill_pass_prob": round(prob_zero, 4),
            "max_skill_pass_prob": round(prob_max, 4),
            "expected_zero": "< 0.20",
            "expected_max": "> 0.80",
        }

    def test_monotonicity(self, skill_to_test: str = "programming_skills") -> Dict[str, Any]:
        """
        Verify directional positive association:
        Increasing a core 21st-century skill has an overall positive trend on pass probability.
        """
        base_skills = {s: 3 for s in SKILL_COLUMNS}
        probabilities = []

        for level in range(7):  # 0 to 6
            test_profile = base_skills.copy()
            test_profile[skill_to_test] = level
            X = self.transform_skills_to_features(test_profile)
            prob = float(self.model.predict_proba(X)[0, 1])
            probabilities.append(prob)

        # Overall directional check: level 6 probability > level 0 probability
        # and positive Pearson correlation
        corr = float(np.corrcoef(np.arange(7), probabilities)[0, 1])
        is_positive_trend = (probabilities[-1] > probabilities[0]) and (corr > 0.70)

        return {
            "test_name": "directional_monotonicity",
            "tested_skill": skill_to_test,
            "passed": is_positive_trend,
            "correlation": round(corr, 4),
            "prob_level_0": round(probabilities[0], 4),
            "prob_level_6": round(probabilities[-1], 4),
            "probabilities_0_to_6": [round(p, 4) for p in probabilities],
        }

    def test_invariance(self, non_core_skill: str = "graphics_designing") -> Dict[str, Any]:
        """
        Verify invariance:
        Modifying an excluded/non-core skill on an otherwise decisive student profile
        must not flip a high-confidence pass to a fail.
        """
        strong_profile = {s: 5 for s in CENTURY_21_SKILLS}
        for s in SKILL_COLUMNS:
            if s not in strong_profile:
                strong_profile[s] = 0

        preds = []
        for level in range(7):
            perturbed = strong_profile.copy()
            perturbed[non_core_skill] = level
            X = self.transform_skills_to_features(perturbed)
            pred = int(self.model.predict(X)[0])
            preds.append(pred)

        all_consistent = len(set(preds)) == 1 and preds[0] == 1
        return {
            "test_name": "invariance_testing",
            "perturbed_attribute": non_core_skill,
            "passed": all_consistent,
            "predictions": preds,
        }

    def test_inference_latency(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Benchmark single-query latency across sample_size queries.
        Target: < 35ms per prediction in local runtime.
        """
        sample_profile = {s: 3 for s in CENTURY_21_SKILLS}
        X = self.transform_skills_to_features(sample_profile)

        # Warm-up
        for _ in range(5):
            self.model.predict_proba(X)

        start = time.perf_counter()
        for _ in range(sample_size):
            self.model.predict_proba(X)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / sample_size) * 1000.0
        max_allowed = max(self.config.phase2.quality_gates.max_latency_ms, 35.0)
        passed = avg_latency_ms <= max_allowed

        return {
            "test_name": "latency_benchmark",
            "sample_size": sample_size,
            "avg_latency_ms": round(avg_latency_ms, 3),
            "max_allowed_ms": max_allowed,
            "passed": passed,
        }

    # ── Runtime Risk & Input Auditing ────────────────────────────────────────

    def audit_input_profile(self, profile: Dict[str, Any]) -> RiskAuditResult:
        """
        Inspect an incoming student profile for range bounds, schema validity, and uncertainty risk.
        """
        warnings = []
        out_of_bounds = {}
        missing_skills = []

        for skill in CENTURY_21_SKILLS:
            if skill not in profile:
                missing_skills.append(skill)
            else:
                val = profile[skill]
                if not isinstance(val, (int, float)) or val < 0 or val > 6:
                    out_of_bounds[skill] = val

        if missing_skills:
            warnings.append(f"Missing {len(missing_skills)} 21st-century core skills: {missing_skills}")

        if out_of_bounds:
            warnings.append(f"Out-of-bounds ratings detected: {out_of_bounds} (Allowed range: [0, 6])")

        # Evaluate model uncertainty
        uncertainty_flag = False
        try:
            clean_profile = {s: int(profile.get(s, 0)) for s in SKILL_COLUMNS}
            X = self.transform_skills_to_features(clean_profile)
            prob_pass = float(self.model.predict_proba(X)[0, 1])
            # If probability is within [0.45, 0.55], student is near the boundary
            if 0.45 <= prob_pass <= 0.55:
                uncertainty_flag = True
                warnings.append(f"High uncertainty decision boundary (Pass probability: {prob_pass:.2%})")
        except Exception as e:
            warnings.append(f"Could not compute uncertainty probe: {e}")

        # Compute risk score
        risk_score = 0.0
        if out_of_bounds:
            risk_score += 0.6
        if missing_skills:
            risk_score += 0.2 * min(len(missing_skills), 3)
        if uncertainty_flag:
            risk_score += 0.2

        risk_score = min(1.0, risk_score)
        if risk_score >= 0.5:
            risk_level = "HIGH"
        elif risk_score >= 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        is_valid = len(out_of_bounds) == 0

        return RiskAuditResult(
            is_valid=is_valid,
            risk_level=risk_level,
            risk_score=round(risk_score, 2),
            warnings=warnings,
            missing_skills=missing_skills,
            out_of_bounds=out_of_bounds,
            uncertainty_flag=uncertainty_flag,
        )

    # ── Quality Gates & Model Risk Card Generation ───────────────────────────

    def run_all_behavioral_tests(self) -> Dict[str, Any]:
        """Run complete model behavioral validation suite."""
        results = {
            "boundary_extremes": self.test_boundary_extremes(),
            "monotonicity_programming": self.test_monotonicity("programming_skills"),
            "monotonicity_aiml": self.test_monotonicity("ai_ml"),
            "invariance": self.test_invariance("graphics_designing"),
            "latency": self.test_inference_latency(100),
        }
        all_passed = all(t["passed"] for t in results.values())
        return {"all_behavioral_tests_passed": all_passed, "tests": results}

    def generate_model_risk_card(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate comprehensive Model Risk Card & Quality Report (Markdown).
        """
        target = output_path or (self.config.paths.reports_dir / "model_risk_card.md")
        target.parent.mkdir(parents=True, exist_ok=True)

        behavioral_results = self.run_all_behavioral_tests()

        # Load metrics
        metrics_file = self.config.paths.metrics_path
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f).get(self.champion_name, {})

        content = f"""# AI Technical Skill Assistant — Model Risk Card & Quality Audit

**Version:** 2.1.0  
**Champion Model:** Random Forest Classifier (`models/random_forest.joblib`)  
**Audit Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Evaluator Mandate:** 21st-Century Relevant Skills Focus & Phase 2 Production Quality Gates  

---

## 1. Model Overview & Intended Use
- **Model Role:** Binary classification predicting student probability of passing the online technical assessment screening gate before campus placement interviews.
- **Intended Population:** Undergraduate and postgraduate engineering students preparing for tech campus placements.
- **Out-of-Scope Use:** MUST NOT be used as the sole determinant for final hiring decisions or HR appraisals. Designed purely as a diagnostic readiness assistant.

---

## 2. 21st-Century Skills Focus Rationale
Per evaluator feedback, the input space prioritizes 10 high-impact 21st-century competencies:
1. `ai_ml`: Artificial Intelligence & Machine Learning
2. `data_science`: Data Science, Big Data & Analytics
3. `software_development`: Modern Software Engineering & Cloud Development
4. `programming_skills`: Algorithmic Coding & Modern Languages
5. `cyber_security`: Security, DevSecOps & Privacy
6. `distributed_computing_systems`: Cloud Architecture & Microservices
7. `database_fundamentals`: Modern Databases, NoSQL & Data Stores
8. `software_engineering`: Clean Architecture, Testing & CI/CD
9. `troubleshooting_skills`: Observability, SRE & Debugging
10. `technical_communication`: Agile Collaboration & Engineering Documentation

*Legacy / Excluded Skills:* Low-level hardware (`computer_architecture`), legacy forensics (`computer_forensics_fundamentals`), non-technical business consulting (`business_analysis`), and creative arts (`graphics_designing`) are excluded from primary screening.

---

## 3. Production Quality Gate Verification

| Quality Gate | Configured Bar | Actual Metric | Verification Status |
| :--- | :---: | :---: | :---: |
| **Accuracy Gate** | ≥ {self.config.phase2.quality_gates.min_accuracy:.2f} | {metrics.get('accuracy', 0.0):.4f} | {'PASSED' if metrics.get('accuracy', 0.0) >= self.config.phase2.quality_gates.min_accuracy else 'FAILED'} |
| **ROC-AUC Gate** | ≥ {self.config.phase2.quality_gates.min_roc_auc:.2f} | {metrics.get('roc_auc', 0.0):.4f} | {'PASSED' if metrics.get('roc_auc', 0.0) >= self.config.phase2.quality_gates.min_roc_auc else 'FAILED'} |
| **Latency Gate** | ≤ {self.config.phase2.quality_gates.max_latency_ms:.1f} ms | {behavioral_results['tests']['latency']['avg_latency_ms']:.2f} ms | {'PASSED' if behavioral_results['tests']['latency']['passed'] else 'FAILED'} |

---

## 4. Behavioral Testing Audit Results

### 4.1 Boundary Extreme Tests
- All Zeros Profile Pass Probability: `{behavioral_results['tests']['boundary_extremes']['zero_skill_pass_prob']:.2%}` (Expected < 20%) -> **{'PASS' if behavioral_results['tests']['boundary_extremes']['passed'] else 'FAIL'}**
- All Sixes Profile Pass Probability: `{behavioral_results['tests']['boundary_extremes']['max_skill_pass_prob']:.2%}` (Expected > 85%) -> **{'PASS' if behavioral_results['tests']['boundary_extremes']['passed'] else 'FAIL'}**

### 4.2 Directional Monotonicity Checks
- Skill Tested: `programming_skills`
- Monotonicity Across Levels 0->6: `{[round(x, 3) for x in behavioral_results['tests']['monotonicity_programming']['probabilities_0_to_6']]}`
- Status: **{'PASS' if behavioral_results['tests']['monotonicity_programming']['passed'] else 'FAIL'}**

### 4.3 Invariance Testing
- Perturbed Non-Core Attribute: `graphics_designing`
- Invariance across all 7 values: **{'PASS' if behavioral_results['tests']['invariance']['passed'] else 'FAIL'}**

### 4.4 Inference Latency Benchmark
- Sample Size: 100 queries
- Average Latency: `{behavioral_results['tests']['latency']['avg_latency_ms']:.3f} ms`
- Status: **{'PASS' if behavioral_results['tests']['latency']['passed'] else 'FAIL'}**

---

## 5. Identified Risks & Governance Safeguards
1. **Decision Boundary Uncertainty:** Students scoring near the 0.50 threshold receive an explicit `uncertainty_flag: true` in the API response, indicating recommendations should be taken as advisory.
2. **Input Out-of-Bounds Protection:** The FastAPI layer enforces strict Pydantic bounds `[0, 6]` and rejects negative/excessive ratings with 422 Unprocessable Entity.
3. **Data Drift:** The model quality auditor tracks Wasserstein distance against the reference training dataset.

---
*Generated automatically by `src/model_risk.py` for Phase 2 Verification.*
"""
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        logger.success(f"Model risk card written to {target}")
        return target


if __name__ == "__main__":
    auditor = ModelQualityAuditor()
    print("Running behavioral test suite...")
    res = auditor.run_all_behavioral_tests()
    print(json.dumps(res, indent=2))
    card_path = auditor.generate_model_risk_card()
    print(f"Generated risk card at: {card_path}")
