# AI-Based Technical Skill Development Assistant

[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/)
[![DVC 3.59](https://img.shields.io/badge/DVC-3.59.1-purple.svg)](https://dvc.org/)
[![MLflow 2.17](https://img.shields.io/badge/MLflow-2.17.2-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An engineering-grade MLOps system that evaluates student technical competencies, computes mathematical skill gaps against industry role requirements, provides targeted learning paths, and predicts career role fit using versioned, reproducible machine learning pipelines.

---

## 1. Project Overview & Motivation

In technical education and campus placements, students frequently face uncertainty regarding whether their skills meet industry standards for roles such as **Machine Learning Engineer**, **Data Scientist**, **Full Stack Developer**, **DevOps Engineer**, and **Data Analyst**.

This project provides:
1. **Rule-Based Skill-Gap Analysis:** Mathematical quantification of individual skill deficits ($G_i$) and cohort average gaps ($G_{avg}$), generating ranked learning recommendations independent of model predictions.
2. **Predictive Role Fit Classification:** Supervised baseline classifiers (Decision Tree & Random Forest) trained to map multi-dimensional skill profiles to optimal career tracks.
3. **Reproducible MLOps Infrastructure:** Strict dependency locking, data versioning with **DVC**, experiment tracking with **MLflow**, and multi-stage orchestration via **DVC Pipelines**.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[data/raw/skills_dataset.csv] -->|DVC Tracking| B[data/raw/skills_dataset.csv.dvc]
    end

    subgraph Pipeline Orchestration [DVC Pipeline: dvc repro]
        A -->|Stage 1: Preprocess| C[src/preprocess.py & src/feature_engineering.py]
        C --> D[data/processed/train.parquet]
        C --> E[data/processed/test.parquet]
        C --> F[data/processed/feature_metadata.json]
        
        D -->|Stage 2: Train| G[src/train.py]
        G --> H[models/decision_tree.joblib]
        G --> I[models/random_forest.joblib]
        G --> J[(MLflow Tracking: Technical_Skill_Assistant)]
        
        E -->|Stage 3: Evaluate| K[src/evaluate.py]
        F --> K
        H --> K
        I --> K
        K --> L[reports/metrics.json]
        K --> M[reports/phase1_evaluation.md]
        K --> J
    end

    subgraph Rule-Based Inference
        N[Student Input Profile] --> O[SkillGapAnalyzer]
        O --> P[Skill Gap Metric G_i = max(0, R_i - S_i)]
        P --> Q[Prioritized Improvement Recommendations]
    end
```

---

## 3. Mathematical Skill-Gap Formulation

Given a student proficiency vector $S = [S_1, S_2, \dots, S_n]$ and a target role requirement vector $R = [R_1, R_2, \dots, R_n]$ on a 1–5 scale:

1. **Individual Skill Gap ($G_i$):**
   $$G_i = \max(0, R_i - S_i)$$
   *Ensures that proficiencies exceeding requirements do not penalize the candidate.*

2. **Average Skill Gap ($G_{avg}$):**
   $$G_{avg} = \frac{1}{n} \sum_{i=1}^{n} G_i$$

3. **Readiness Percentage:**
   $$\text{Readiness} = \max\left(0, 100 \times \left(1 - \frac{\sum G_i}{4n}\right)\right)$$

### Role Benchmark Profiles

| Target Career Role | Python | SQL | ML | DSA | Git | Web Dev | Cloud | System Design | Stats | Deep Learning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Machine Learning Engineer** | 4 | 3 | 4 | 4 | 3 | 1 | 3 | 3 | 4 | 4 |
| **Data Scientist** | 4 | 4 | 4 | 3 | 3 | 1 | 2 | 2 | 5 | 3 |
| **Full Stack Developer** | 3 | 3 | 1 | 4 | 4 | 5 | 3 | 4 | 2 | 1 |
| **DevOps Engineer** | 3 | 2 | 1 | 3 | 5 | 2 | 5 | 4 | 2 | 1 |
| **Data Analyst** | 3 | 5 | 2 | 2 | 2 | 1 | 2 | 1 | 4 | 1 |

---

## 4. Verified Technology Stack & Dependency Pins

The environment is strictly pinned and tested on Python 3.11.9:

| Package | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.11.9 | Core runtime |
| **NumPy** | 1.26.4 | Numerical arrays & vector math |
| **Pandas** | 2.2.3 | Tabular data manipulation & schema validation |
| **scikit-learn** | 1.5.2 | Preprocessing, classification models, evaluation metrics |
| **DVC** | 3.59.1 | Data versioning & pipeline orchestration |
| **MLflow** | 2.17.2 | Experiment tracking, model registry, metric provenance |
| **pathspec** | 0.12.1 | Pattern matching compatibility layer for DVC |
| **Pydantic** | 2.9.2 | Strongly typed configuration validation |
| **PyYAML** | 6.0.2 | Human-readable configuration parsing |
| **Loguru** | 0.7.2 | Structured and standardized logging |
| **Rich** | 13.9.4 | Terminal formatting and status output |
| **Ruff** | 0.7.1 | Fast Python linter & code quality enforcer |
| **pytest** | 9.1.1 | Unit and integration test suite |
| **ydata-profiling**| 4.12.2 | Automated exploratory profiling |

---

## 5. Dataset Provenance & Characteristics

- **Name:** Technical Skill Development Benchmark Dataset (`data/raw/skills_dataset.csv`)
- **Version Control:** Managed by DVC via pointer `data/raw/skills_dataset.csv.dvc` (never committed as raw binary to Git)
- **Cohort Size:** 1,200 records (stratified: exactly 240 samples per target role)
- **Features (14 total):**
  - **10 Technical Skills (1–5 scale):** Python, SQL, ML, DSA, Git, Web Development, Cloud Computing, System Design, Statistics, Deep Learning.
  - **3 Contextual Metrics:** Years of Experience, Projects Completed, Problems Solved.
  - **Target Variable:** `target_role` (Multi-class: 5 career paths).
- **Missing Values:** 0 nulls, 0 duplicate records.
- **Dataset Card:** Formally documented in [reports/dataset_card.md](reports/dataset_card.md)
- **Validation Notebook:** Thorough 26-section diagnostic report in [notebooks/01_dataset_acquisition_validation.ipynb](notebooks/01_dataset_acquisition_validation.ipynb)

---

## 6. Project Structure

```
AI-Technical-Skill-Development-Assistant/
├── configs/
│   └── config.yaml               # Centralized pipeline configuration
├── data/
│   ├── raw/                      # Managed by DVC
│   │   ├── skills_dataset.csv    # (Tracked in DVC cache)
│   │   └── skills_dataset.csv.dvc# DVC pointer tracked in Git
│   └── processed/
│       ├── train.parquet         # Leak-free train partition (960 rows)
│       ├── test.parquet          # Leak-free test partition (240 rows)
│       └── feature_metadata.json # Scaler parameters & class mappings
├── models/
│   ├── decision_tree.joblib      # Baseline 1 model artifact
│   ├── random_forest.joblib      # Champion baseline 2 model artifact
│   └── training_summary.json     # MLflow Run IDs and training stats
├── notebooks/
│   ├── 01_dataset_acquisition_validation.ipynb # Member 1: Complete 26-section validation deliverable
│   ├── 01_data_exploration.ipynb # Schema, distributions, correlations
│   └── 02_baseline_experiments.ipynb # Model comparison & evaluation
├── reports/
│   ├── dataset_card.md           # Member 1: Formal benchmark dataset card
│   ├── metrics.json              # DVC tracked metrics
│   └── phase1_evaluation.md      # Detailed evaluation report
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic configuration loader
│   ├── data_loader.py            # Data ingestion & schema validation
│   ├── preprocess.py             # Skill-gap formulas & leak-free split
│   ├── feature_engineering.py    # Domain composite scores & scalers
│   ├── train.py                  # MLflow experiment training
│   └── evaluate.py               # Comprehensive metric reporting
├── tests/
│   ├── test_data_loader.py       # Data loading & schema tests
│   ├── test_skill_gap.py         # Mathematical gap formula tests
│   ├── test_feature_engineering.py # Pipeline & transformation tests
│   └── test_pipeline_and_models.py # Model predictions & evaluation tests
├── .dvc/                         # DVC configuration and metadata
├── .dvcignore
├── .gitignore                    # Robust gitignore (ignores venv, raw data, logs)
├── .python-version               # Pinned Python version (3.11.9)
├── dvc.yaml                      # Multi-stage reproducible pipeline
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Pinned dependencies
├── journey.md                    # Engineering milestone log
└── README.md
```

---

## 7. Installation & Quickstart

### Step 1: Clone Repository
```bash
git clone https://github.com/ShreyasPlays/AI-Technical-Skill-Development-Assistant.git
cd AI-Technical-Skill-Development-Assistant
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Pinned Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Pipeline Execution (DVC)

The entire workflow from preprocessing to evaluation is captured in `dvc.yaml`:

```powershell
# Reproduce full pipeline end-to-end
dvc repro
```

### Inspect Pipeline Metrics
```powershell
dvc metrics show
```

Output:
```
Path                  champion_model            decision_tree.accuracy    decision_tree.f1_macro    random_forest.accuracy    random_forest.f1_macro
reports\metrics.json  Random Forest Classifier  0.8792                    0.8795                    0.9167                    0.9168
```

---

## 9. MLflow Experiment Tracking

Every model run logs parameters, metrics, and models to MLflow experiment `Technical_Skill_Assistant`.

### Launch the MLflow UI
```powershell
mlflow ui --port 5000
```
Then open [http://localhost:5000](http://localhost:5000) in your browser to interactively compare runs, view confusion matrices, and review artifact provenance.

---

## 10. Phase 1 Evaluation Results

Models were evaluated on a held-out, stratified test set of **240 candidate profiles** with zero data leakage:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | 87.92% | 0.8815 | 0.8792 | 0.8795 | Baseline |
| **Random Forest Classifier** | **91.67%** | **0.9180** | **0.9167** | **0.9168** | **Champion Model** |

**Champion Rationale:** Random Forest achieves superior generalization (+3.75% accuracy, +0.0373 macro F1) by mitigating high variance on subtle skill boundaries between adjacent roles (e.g., Data Scientist vs. ML Engineer).

---

## 11. Running Automated Tests

Run the full test suite with verbose reporting:

```powershell
pytest
```

Run code formatting and lint verification:

```powershell
ruff check src tests
```

---

## 12. License & Contributors

- **Author & Project Lead:** ShreyasPlays (<shreyas662006@gmail.com>)
- **Member 1 (Dataset Acquisition & Validation):** Siddharth Dhanush R (<siddharthdhanush7@gmail.com>)
- **Project:** Engineering College Project Review - Phase 1: Development & Reproducibility
- **License:** MIT License (Note: Dataset reuse permissions remain unverified pending committed root LICENSE file)
