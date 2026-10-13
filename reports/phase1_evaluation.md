# Phase 1: Machine Learning Model Evaluation Report

**Project:** AI-Based Technical Skill Development Assistant  
**Phase:** Phase 1 (Development & Reproducibility)  
**Evaluation Set:** 240 held-out student profile samples (stratified, zero data leakage)  

---

## 1. Executive Summary & Model Comparison

We evaluated two baseline non-linear classifiers (Decision Tree and Random Forest) to predict the best-fitting technical career role for candidates based on their technical proficiencies and engineering metrics.

| Metric | Decision Tree Classifier | Random Forest Classifier | Delta (RF - DT) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 87.92% | **91.67%** | +3.75% |
| **Precision (Macro)** | 0.8815 | **0.9180** | +0.0365 |
| **Recall (Macro)** | 0.8792 | **0.9167** | +0.0375 |
| **Macro F1-Score** | 0.8795 | **0.9168** | +0.0373 |
| **Weighted F1-Score** | 0.8795 | **0.9168** | +0.0373 |

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
| **Data Analyst** | 0.9388 | 0.9583 | 0.9485 | 48 |
| **Data Scientist** | 0.9111 | 0.8542 | 0.8817 | 48 |
| **DevOps Engineer** | 0.9375 | 0.9375 | 0.9375 | 48 |
| **Full Stack Developer** | 0.9565 | 0.9167 | 0.9362 | 48 |
| **Machine Learning Engineer** | 0.8462 | 0.9167 | 0.8800 | 48 |

### Decision Tree Classifier (Baseline)

| Target Career Role | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Data Analyst** | 0.9167 | 0.9167 | 0.9167 | 48 |
| **Data Scientist** | 0.7708 | 0.7708 | 0.7708 | 48 |
| **DevOps Engineer** | 0.9038 | 0.9792 | 0.9400 | 48 |
| **Full Stack Developer** | 1.0000 | 0.8958 | 0.9451 | 48 |
| **Machine Learning Engineer** | 0.8163 | 0.8333 | 0.8247 | 48 |

---

## 3. Confusion Matrix (Random Forest Champion)

Columns = Predicted Role | Rows = True Role  
Order: **[0]** Data Analyst, **[1]** Data Scientist, **[2]** DevOps Engineer, **[3]** Full Stack Developer, **[4]** Machine Learning Engineer

| True Role \ Predicted | [0] | [1] | [2] | [3] | [4] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **[0] Data Analyst** | 46 | 2 | 0 | 0 | 0 |
| **[1] Data Scientist** | 0 | 41 | 0 | 0 | 7 |
| **[2] DevOps Engineer** | 0 | 0 | 45 | 2 | 1 |
| **[3] Full Stack Developer** | 1 | 0 | 3 | 44 | 0 |
| **[4] Machine Learning Engineer** | 2 | 2 | 0 | 0 | 44 |

---

## 4. Reproducibility & Pipeline Integration

- **DVC Tracking:** Metrics tracked in `reports/metrics.json` and versioned via `dvc.yaml`.
- **MLflow Tracking:** Experiments logged under `Technical_Skill_Assistant` with parameters, metrics, and models.
- **Zero Leakage:** Feature transforms and scalers were fit strictly on the 80% train split and applied to the 20% test split.
