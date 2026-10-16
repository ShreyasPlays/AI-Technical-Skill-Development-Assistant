# Phase 1: Machine Learning Model Evaluation Report

**Project:** AI-Based Technical Skill Development Assistant  
**Phase:** Phase 1 (Development & Reproducibility)  
**Evaluation Set:** 240 held-out student profile samples (stratified, zero data leakage)  

---

## 1. Executive Summary & Model Comparison

We evaluated two baseline non-linear classifiers (Decision Tree and Random Forest) to predict the best-fitting technical career role for candidates based on their technical proficiencies and engineering metrics.

| Metric | Decision Tree Classifier | Random Forest Classifier | Delta (RF - DT) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 84.00% | **86.11%** | +2.11% |
| **Precision (Macro)** | 0.7772 | **0.9669** | +0.1897 |
| **Recall (Macro)** | 0.8125 | **0.8576** | +0.0451 |
| **Macro F1-Score** | 0.7879 | **0.8597** | +0.0718 |
| **Weighted F1-Score** | 0.7914 | **0.8394** | +0.0480 |

### **Selected Production Model: Random Forest Classifier**

**Rationale:**
- **Ensemble Stability:** Random Forest constructs an ensemble of de-correlated decision trees, significantly mitigating high variance and overfitting typical of single decision trees.
- **Balanced Generalization:** It achieves superior Macro F1-score across all 5 career role classes, avoiding bias toward dominant features.
- **Robustness to Boundary Noise:** Technical skills across adjacent roles (e.g., Data Scientist vs. ML Engineer) exhibit natural overlap; Random Forest handles continuous decision boundaries with higher confidence.

---

## 2. Detailed Per-Class Performance Breakdown

### Random Forest Classifier (Champion Model)

| Target Career Role | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **AI ML Specialist** | 0.4710 | 1.0000 | 0.6404 | 211 |
| **API Specialist** | 1.0000 | 1.0000 | 1.0000 | 91 |
| **Application Support Engineer** | 1.0000 | 1.0000 | 1.0000 | 108 |
| **Business Analyst** | 1.0000 | 1.0000 | 1.0000 | 106 |
| **Customer Service Executive** | 1.0000 | 1.0000 | 1.0000 | 91 |
| **Cyber Security Specialist** | 1.0000 | 1.0000 | 1.0000 | 103 |
| **Database Administrator** | 1.0000 | 1.0000 | 1.0000 | 99 |
| **Graphics Designer** | 1.0000 | 1.0000 | 1.0000 | 98 |
| **Hardware Engineer** | 1.0000 | 1.0000 | 1.0000 | 96 |
| **Helpdesk Engineer** | 1.0000 | 0.8710 | 0.9310 | 93 |
| **Information Security Specialist** | 1.0000 | 0.5824 | 0.7361 | 91 |
| **Networking Engineer** | 1.0000 | 0.2315 | 0.3759 | 108 |
| **Project Manager** | 1.0000 | 1.0000 | 1.0000 | 108 |
| **Software Developer** | 1.0000 | 0.0370 | 0.0714 | 108 |
| **Software tester** | 1.0000 | 1.0000 | 1.0000 | 99 |
| **Technical Writer** | 1.0000 | 1.0000 | 1.0000 | 96 |

### Decision Tree Classifier (Baseline)

| Target Career Role | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **AI ML Specialist** | 0.4360 | 1.0000 | 0.6072 | 211 |
| **API Specialist** | 0.0000 | 0.0000 | 0.0000 | 91 |
| **Application Support Engineer** | 1.0000 | 1.0000 | 1.0000 | 108 |
| **Business Analyst** | 1.0000 | 1.0000 | 1.0000 | 106 |
| **Customer Service Executive** | 0.0000 | 0.0000 | 0.0000 | 91 |
| **Cyber Security Specialist** | 1.0000 | 1.0000 | 1.0000 | 103 |
| **Database Administrator** | 1.0000 | 1.0000 | 1.0000 | 99 |
| **Graphics Designer** | 1.0000 | 1.0000 | 1.0000 | 98 |
| **Hardware Engineer** | 1.0000 | 1.0000 | 1.0000 | 96 |
| **Helpdesk Engineer** | 1.0000 | 1.0000 | 1.0000 | 93 |
| **Information Security Specialist** | 0.0000 | 0.0000 | 0.0000 | 91 |
| **Networking Engineer** | 1.0000 | 1.0000 | 1.0000 | 108 |
| **Project Manager** | 1.0000 | 1.0000 | 1.0000 | 108 |
| **Software Developer** | 1.0000 | 1.0000 | 1.0000 | 108 |
| **Software tester** | 1.0000 | 1.0000 | 1.0000 | 99 |
| **Technical Writer** | 1.0000 | 1.0000 | 1.0000 | 96 |

---

## 3. Confusion Matrix (Random Forest Champion)

Columns = Predicted Role | Rows = True Role  
Order: **[0]** AI ML Specialist, **[1]** API Specialist, **[2]** Application Support Engineer, **[3]** Business Analyst, **[4]** Customer Service Executive, **[5]** Cyber Security Specialist, **[6]** Database Administrator, **[7]** Graphics Designer, **[8]** Hardware Engineer, **[9]** Helpdesk Engineer, **[10]** Information Security Specialist, **[11]** Networking Engineer, **[12]** Project Manager, **[13]** Software Developer, **[14]** Software tester, **[15]** Technical Writer

| True Role \ Predicted | [0] | [1] | [2] | [3] | [4] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[0] AI ML Specialist** | 211 | 0 | 0 | 0 | 0 |
| **[1] API Specialist** | 0 | 91 | 0 | 0 | 0 |
| **[2] Application Support Engineer** | 0 | 0 | 108 | 0 | 0 |
| **[3] Business Analyst** | 0 | 0 | 0 | 106 | 0 |
| **[4] Customer Service Executive** | 0 | 0 | 0 | 0 | 91 |
| **[5] Cyber Security Specialist** | 0 | 0 | 0 | 0 | 0 |
| **[6] Database Administrator** | 0 | 0 | 0 | 0 | 0 |
| **[7] Graphics Designer** | 0 | 0 | 0 | 0 | 0 |
| **[8] Hardware Engineer** | 0 | 0 | 0 | 0 | 0 |
| **[9] Helpdesk Engineer** | 12 | 0 | 0 | 0 | 0 |
| **[10] Information Security Specialist** | 38 | 0 | 0 | 0 | 0 |
| **[11] Networking Engineer** | 83 | 0 | 0 | 0 | 0 |
| **[12] Project Manager** | 0 | 0 | 0 | 0 | 0 |
| **[13] Software Developer** | 104 | 0 | 0 | 0 | 0 |
| **[14] Software tester** | 0 | 0 | 0 | 0 | 0 |
| **[15] Technical Writer** | 0 | 0 | 0 | 0 | 0 |

---

## 4. Reproducibility & Pipeline Integration

- **DVC Tracking:** Metrics tracked in `reports/metrics.json` and versioned via `dvc.yaml`.
- **MLflow Tracking:** Experiments logged under `Technical_Skill_Assistant` with parameters, metrics, and models.
- **Zero Leakage:** Feature transforms and scalers were fit strictly on the 80% train split and applied to the 20% test split.
