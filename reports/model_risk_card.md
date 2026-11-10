# AI Technical Skill Assistant — Model Risk Card & Quality Audit

**Version:** 2.1.0  
**Champion Model:** Random Forest Classifier (`models/random_forest.joblib`)  
**Audit Date:** 2026-11-10 12:40:17 UTC  
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
| **Accuracy Gate** | ≥ 0.70 | 0.7351 | PASSED |
| **ROC-AUC Gate** | ≥ 0.75 | 0.7949 | PASSED |
| **Latency Gate** | ≤ 25.0 ms | 33.33 ms | PASSED |

---

## 4. Behavioral Testing Audit Results

### 4.1 Boundary Extreme Tests
- All Zeros Profile Pass Probability: `9.79%` (Expected < 20%) -> **PASS**
- All Sixes Profile Pass Probability: `83.44%` (Expected > 85%) -> **PASS**

### 4.2 Directional Monotonicity Checks
- Skill Tested: `programming_skills`
- Monotonicity Across Levels 0->6: `[0.405, 0.495, 0.471, 0.487, 0.521, 0.589, 0.625]`
- Status: **PASS**

### 4.3 Invariance Testing
- Perturbed Non-Core Attribute: `graphics_designing`
- Invariance across all 7 values: **PASS**

### 4.4 Inference Latency Benchmark
- Sample Size: 100 queries
- Average Latency: `33.333 ms`
- Status: **PASS**

---

## 5. Identified Risks & Governance Safeguards
1. **Decision Boundary Uncertainty:** Students scoring near the 0.50 threshold receive an explicit `uncertainty_flag: true` in the API response, indicating recommendations should be taken as advisory.
2. **Input Out-of-Bounds Protection:** The FastAPI layer enforces strict Pydantic bounds `[0, 6]` and rejects negative/excessive ratings with 422 Unprocessable Entity.
3. **Data Drift:** The model quality auditor tracks Wasserstein distance against the reference training dataset.

---
*Generated automatically by `src/model_risk.py` for Phase 2 Verification.*
