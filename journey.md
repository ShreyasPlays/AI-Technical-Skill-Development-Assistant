# Engineering Journey Log: Phase 1 Development & Reproducibility

**Project:** AI-Based Technical Skill Development Assistant  
**Repository:** [AI-Technical-Skill-Development-Assistant](https://github.com/ShreyasPlays/AI-Technical-Skill-Development-Assistant)  
**Author:** ShreyasPlays (<shreyas662006@gmail.com>)  
**Milestone:** Phase 1 (Development & Reproducibility)  

---

## 1. Implementation Milestones

| Milestone | Status | Description |
| :--- | :---: | :--- |
| **Milestone 1: Environment & Identity Setup** | Completed | Inspected tools (`git`, `gh`, Python 3.11.9). Configured Git identity (`ShreyasPlays`, `shreyas662006@gmail.com`). Created local repo directory. |
| **Milestone 2: Remote GitHub Creation** | Completed | Authenticated with `gh CLI` and initialized remote repository `AI-Technical-Skill-Development-Assistant` with public visibility and origin remote. |
| **Milestone 3: Dependency Resolution & Locking** | Completed | Validated proposed pins, tested dry-runs, resolved transitive package incompatibilities, and created an isolated virtual environment (`.venv`). Pinned `requirements.txt`. |
| **Milestone 4: Kaggle Benchmark Dataset Ingestion** | Completed | Ingested Kaggle Career Mapping benchmark dataset (`data/raw/skills_dataset.csv`, $N=8,528$ records, 17 technical disciplines, 10 personality metrics, 16 career roles). Enforced schema constraints and duplicate purging. |
| **Milestone 5: Preprocessing & Skill-Gap Analysis** | Completed | Implemented rule-based `SkillGapAnalyzer` ($G_i = \max(0, R_i - S_i)$ and $G_{avg}$) independent of ML models. Implemented leak-free train/test splitting ($80/20$ stratified split). |
| **Milestone 6: Feature Engineering Pipeline** | Completed | Built composite readiness indices (`ai_data_readiness`, `systems_readiness`, `software_dev_readiness`, `cybersecurity_readiness`, `business_comm_readiness`, `analytical_mindset_index`). Learned scalers strictly on training split ($N=6,822$). |
| **Milestone 7: Baseline Model Training & MLflow** | Completed | Trained Decision Tree and Random Forest classifiers. Logged parameters, dataset SHA-256 hashes, models, and metrics to MLflow experiment `Technical_Skill_Assistant`. |
| **Milestone 8: Evaluation & Reporting** | Completed | Evaluated models on 1,706 held-out samples. Generated `reports/metrics.json` and comprehensive review report `reports/phase1_evaluation.md`. Random Forest selected as Champion (100.0% test accuracy, 1.0000 Macro F1 vs Decision Tree 84.0% accuracy). |
| **Milestone 9: DVC Tracking & Pipeline** | Completed | Initialized DVC, tracked raw dataset, created multi-stage `dvc.yaml` (`preprocess`, `train`, `evaluate`), and verified end-to-end execution via `dvc repro`. |
| **Milestone 10: Automated Testing & Code Quality** | Completed | Maintained 16 automated unit and integration tests across data loader, skill gap, feature engineering, and models. 100% tests passing in pytest (3.31s). |
| **Milestone 11: Real-Time Interactive Inference** | Completed | Created `predict.py` CLI demo supporting live profile testing, career role diagnostic prediction with probability distributions, and personalized skill-gap roadmapping. |
| **Milestone 12: Team Role Artifact Synchronization** | Completed | Delivered dedicated notebooks for Member 1 (`01_data_exploration.ipynb`), Member 2 (`02_eda_feature_engineering.ipynb`), and Member 3 (`03_baseline_experiments.ipynb`). |

---

## 2. Problems Encountered & Engineering Solutions

### Problem 1: Unconfigured Global Git User & Email
- **Symptom:** Running `git config --global user.name` and `user.email` returned empty strings, blocking automated commit authoring.
- **Root Cause:** Fresh environment with no global Git identity set.
- **Solution:** Set `git config --global user.name "ShreyasPlays"` and `user.email "shreyas662006@gmail.com"`.

### Problem 2: `pathspec` Incompatibility with DVC 3.59.1
- **Symptom:** Executing `dvc init` failed with:
  `ERROR: unexpected error - cannot import name '_DIR_MARK' from 'pathspec.patterns.gitwildmatch'`.
- **Root Cause:** A newer version of `pathspec` (v1.1.1) deprecated internal attribute `_DIR_MARK`, which `scmrepo` / `dvc 3.59.1` imports.
- **Solution:** Downgraded and pinned `pathspec==0.12.1` in the virtual environment and added it explicitly to `requirements.txt`. Tested `dvc --version` which verified clean operation on 3.59.1.

### Problem 3: DVC File Staging Blocked by Root `.gitignore`
- **Symptom:** `dvc add data/raw/skills_dataset.csv` failed with:
  `ERROR: bad DVC file name 'data\raw\skills_dataset.csv.dvc' is git-ignored`.
- **Root Cause:** An overly broad `/data/raw/*` pattern in root `.gitignore` blocked Git from tracking the `.dvc` pointer file.
- **Solution:** Updated `.gitignore` with negative ignore exemptions (`!/data/raw/*.dvc`, `!/data/raw/.gitignore`, `!/data/processed/feature_metadata.json`), ensuring DVC manages the binary dataset while Git versions the tracking metadata.

### Problem 4: Subshell Python Resolution in DVC Pipeline on Windows
- **Symptom:** Running `dvc repro` on stage `train` threw `ModuleNotFoundError: No module named 'mlflow'`.
- **Root Cause:** In PowerShell on Windows, subshell commands spawned by DVC default to the global system Python executable when the virtual environment is not explicitly in `PATH`.
- **Solution:** Updated `dvc.yaml` stage commands to invoke `.venv\Scripts\python.exe` directly, guaranteeing that all pipeline stages resolve to the isolated project environment with MLflow installed.

### Problem 5: Floating Point Exact Equality in Feature Engineering Test
- **Symptom:** `test_composite_feature_engineering` failed with `assert 3.9999999999999996 == 4.0`.
- **Root Cause:** Floating point arithmetic artifact during linear combination calculations.
- **Solution:** Refactored test assertion to use `pytest.approx()`, adhering to standard numerical testing practices.

---

## 3. Dependency Specification & Version Audit

| Library | Installed Pin | Status | Notes |
| :--- | :---: | :---: | :--- |
| Python | 3.11.9 | Exact | Active in `.venv` |
| NumPy | 1.26.4 | Exact | Linear algebra and vector operations |
| Pandas | 2.2.3 | Exact | Compatible with PyArrow parquet engines |
| scikit-learn | 1.5.2 | Exact | Decision Tree and Random Forest estimators |
| DVC | 3.59.1 | Exact | Reproducible pipeline and data versioning |
| MLflow | 2.17.2 | Exact | Experiment tracking and model registry |
| pathspec | 0.12.1 | Pinned | Required for DVC scmrepo compatibility |
| Pydantic | 2.9.2 | Exact | Configuration schema validation |
| PyYAML | 6.0.2 | Exact | Configuration file loader |
| Loguru | 0.7.2 | Exact | Structured console logging |
| pytest | 9.1.1 | Exact | Automated unit and integration test runner |

---

## 4. Test Results Summary

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\shrey\.gemini\antigravity-ide\scratch\AI-Technical-Skill-Development-Assistant
configfile: pytest.ini
testpaths: tests
collected 16 items

tests/test_data_loader.py::test_config_loading PASSED                    [  6%]
tests/test_data_loader.py::test_dataset_loading PASSED                   [ 12%]
tests/test_data_loader.py::test_skill_bounds_validation PASSED           [ 18%]
tests/test_data_loader.py::test_fractional_skill_validation PASSED       [ 25%]
tests/test_data_loader.py::test_unsupported_target_role_validation PASSED [ 31%]
tests/test_data_loader.py::test_duplicate_columns_validation PASSED      [ 37%]
tests/test_feature_engineering.py::test_leakage_free_split PASSED        [ 43%]
tests/test_feature_engineering.py::test_composite_feature_engineering PASSED [ 50%]
tests/test_feature_engineering.py::test_pipeline_fit_and_transform PASSED [ 56%]
tests/test_pipeline_and_models.py::test_trained_models_exist_and_predict PASSED [ 62%]
tests/test_pipeline_and_models.py::test_model_evaluation_metrics PASSED  [ 68%]
tests/test_pipeline_and_models.py::test_dvc_metrics_and_evaluation_report_files PASSED [ 75%]
tests/test_skill_gap.py::test_skill_gap_formula_calculation PASSED       [ 81%]
tests/test_skill_gap.py::test_ranking_priority_order PASSED              [ 87%]
tests/test_skill_gap.py::test_unsupported_role_raises_error PASSED       [ 93%]
tests/test_skill_gap.py::test_exceeding_proficiency_has_zero_gap PASSED  [100%]

============================= 16 passed in 3.31s ==============================
```

---

## 5. Phase 1 Team Task Tracker & Member Contributions

| Member | Task ID & Title | Status | Work Completed | Files Changed | Evidence Link |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Member 1** | `M1-T01`: Kaggle Data Ingestion & Schema Auditing | Done | Ingested Kaggle Career Mapping dataset ($N=8,528$), built schema validation, zero-duplicate checks, and initial exploratory notebook. | `data/raw/skills_dataset.csv`, `src/data_loader.py`, `notebooks/01_data_exploration.ipynb` | [01_data_exploration.ipynb](file:///C:/Users/shrey/.gemini/antigravity-ide/scratch/AI-Technical-Skill-Development-Assistant/notebooks/01_data_exploration.ipynb) |
| **Member 2** | `M2-T01`: EDA & Statistical Profiling | Done | Analyzed 17 skill distributions, correlation matrix, Tukey IQR outliers, and role benchmark heatmaps across career paths. | `notebooks/02_eda_feature_engineering.ipynb` | [02_eda_feature_engineering.ipynb](file:///C:/Users/shrey/.gemini/antigravity-ide/scratch/AI-Technical-Skill-Development-Assistant/notebooks/02_eda_feature_engineering.ipynb) |
| **Member 2** | `M2-T02`: Feature Engineering & Leakage-Free Pipeline | Done | Engineered 6 domain readiness indices. Fit standard scalers and label encoders strictly on $N=6,822$ training partition. Transformed held-out test partition ($N=1,706$). | `src/feature_engineering.py`, `data/processed/` | [src/feature_engineering.py](file:///C:/Users/shrey/.gemini/antigravity-ide/scratch/AI-Technical-Skill-Development-Assistant/src/feature_engineering.py) |
| **Member 3** | `M3-T01`: Model Training & MLflow Tracking | Done | Trained Decision Tree and Random Forest classifiers on 34 engineered features across 16 classes. Tracked runs, parameters, and metrics in MLflow. | `src/train.py`, `models/`, `mlruns/` | [models/](file:///C:/Users/shrey/.gemini/antigravity-ide/scratch/AI-Technical-Skill-Development-Assistant/models) |
| **Member 3** | `M3-T02`: Multi-Metric Test Evaluation & Matrices | Done | Evaluated on held-out test split ($N=1,706$). Plotted side-by-side confusion matrix heatmaps. Generated `reports/metrics.json` and `reports/phase1_evaluation.md`. | `notebooks/03_baseline_experiments.ipynb`, `reports/` | [reports/phase1_evaluation.md](file:///C:/Users/shrey/.gemini/antigravity-ide/scratch/AI-Technical-Skill-Development-Assistant/reports/phase1_evaluation.md) |
| **Member 3** | `M3-T03`: Champion Selection & Inference Demo | Done | Selected Random Forest Champion ($100.0\%$ test accuracy, $1.000$ Macro F1 vs $84.0\%$ Decision Tree). Created `predict.py` interactive career diagnostic and roadmap generator. | `predict.py`, `reports/metrics.json` | [predict.py](file:///C:/Users/shrey/.gemini/antigravity-ide/scratch/AI-Technical-Skill-Development-Assistant/predict.py) |
