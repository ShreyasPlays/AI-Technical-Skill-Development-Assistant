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
| **Milestone 4: Benchmark Dataset Ingestion** | Completed | Created `data/raw/skills_dataset.csv` (1,200 records, 10 technical skills, 3 context features, 5 balanced career roles). Tested schema validation. |
| **Milestone 5: Preprocessing & Skill-Gap Analysis** | Completed | Implemented rule-based `SkillGapAnalyzer` ($G_i = \max(0, R_i - S_i)$ and $G_{avg}$) independent of ML models. Implemented leak-free train/test splitting. |
| **Milestone 6: Feature Engineering Pipeline** | Completed | Built composite readiness indices (`ml_ai_readiness`, `data_analytics_readiness`, `fullstack_readiness`, `devops_cloud_readiness`, `practical_velocity`). Learned scalers strictly on training split. |
| **Milestone 7: Baseline Model Training & MLflow** | Completed | Implemented Decision Tree and Random Forest classifiers. Logged parameters, dataset SHA-256 hashes, models, and metrics to MLflow experiment `Technical_Skill_Assistant`. |
| **Milestone 8: Evaluation & Reporting** | Completed | Evaluated models on 240 held-out samples. Generated `reports/metrics.json` and comprehensive review report `reports/phase1_evaluation.md`. Selected Random Forest as Champion. |
| **Milestone 9: DVC Tracking & Pipeline** | Completed | Initialized DVC, tracked raw dataset with DVC pointer, created multi-stage `dvc.yaml` (`preprocess`, `train`, `evaluate`), and verified end-to-end execution via `dvc repro`. |
| **Milestone 10: Automated Testing & Code Quality** | Completed | Created 14 unit and integration tests across data loader, skill gap, feature engineering, and models. Passed 100% tests with pytest. Enforced Ruff linting. |
| **Milestone 11: Dataset Acquisition & Validation** | Completed | Contributor: Siddharth Dhanush R (Member 1). Cryptographic provenance verification (SHA-256/MD5), hardened data loader with bounds/duplicate checks, 5 new unit tests (19 total passed), comprehensive 26-section notebook (`notebooks/01_dataset_acquisition_validation.ipynb`), and verified dataset card (`reports/dataset_card.md`). |

---

## 2. Problems Encountered & Engineering Solutions

### Problem 1: Unconfigured Global Git User & Email
- **Symptom:** Running `git config --global user.name` and `user.email` returned empty strings, blocking automated commit authoring.
- **Root Cause:** Fresh environment with no global Git identity set.
- **Solution:** Queried previous projects in the workspace to discover the user's authentic handle (`ShreyasPlays` / `shreyas662006@gmail.com`), asked the user for explicit confirmation via interactive dialog, and set `git config --global user.name "ShreyasPlays"` and `user.email "shreyas662006@gmail.com"`.

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
- **Solution:** In pipeline execution scripts and documentation, prepended `$env:PATH = "$pwd\.venv\Scripts;$env:PATH"` so all pipeline invocations resolve strictly to the project's isolated virtual environment.

### Problem 5: Floating Point Exact Equality in Feature Engineering Test
- **Symptom:** `test_composite_feature_engineering` failed with `assert 3.9999999999999996 == 4.0`.
- **Root Cause:** Floating point arithmetic artifact during linear combination calculations.
- **Solution:** Refactored test assertion to use `pytest.approx(4.0)`, adhering to standard numerical testing practices.

---

## 3. Dependency Specification & Version Audit

| Library | Proposed Pin | Actual Installed Pin | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| Python | 3.11.9 | 3.11.9 | Exact | Active in `.venv` |
| NumPy | 1.26.4 | 1.26.4 | Exact | Compatible with all downstream packages |
| Pandas | 2.2.3 | 2.2.3 | Exact | Compatible with PyArrow parquet engines |
| scikit-learn | 1.5.2 | 1.5.2 | Exact | Model estimators and evaluation metrics |
| DVC | 3.59.1 | 3.59.1 | Exact | Pipeline and data versioning |
| MLflow | 2.17.2 | 2.17.2 | Exact | Experiment tracking |
| pathspec | Unspecified | 0.12.1 | Added Pin | Required to fix `_DIR_MARK` import error in DVC |
| Pydantic | 2.9.2 | 2.9.2 | Exact | Configuration schema validation |
| PyYAML | 6.0.2 | 6.0.2 | Exact | Config loader |
| Loguru | 0.7.2 | 0.7.2 | Exact | Standardized logging |
| Rich | 13.9.4 | 13.9.4 | Exact | Formatting |
| Ruff | 0.7.1 | 0.7.1 | Exact | Linting and code quality |
| pytest | Unspecified | 9.1.1 | Added Pin | Test runner |
| ydata-profiling | 4.12.2 | 4.12.2 | Exact | Profiling compatibility verified |

---

## 4. Test Results Summary

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\shrey\.gemini\antigravity-ide\scratch\AI-Technical-Skill-Development-Assistant
configfile: pytest.ini
testpaths: tests
collected 14 items

tests/test_data_loader.py::test_config_loading PASSED                    [  7%]
tests/test_data_loader.py::test_dataset_loading PASSED                   [ 14%]
tests/test_data_loader.py::test_skill_bounds_validation PASSED           [ 21%]
tests/test_data_loader.py::test_reproducible_dataset_generation PASSED   [ 28%]
tests/test_feature_engineering.py::test_leakage_free_split PASSED        [ 35%]
tests/test_feature_engineering.py::test_composite_feature_engineering PASSED [ 42%]
tests/test_feature_engineering.py::test_pipeline_fit_and_transform PASSED [ 50%]
tests/test_pipeline_and_models.py::test_trained_models_exist_and_predict PASSED [ 57%]
tests/test_pipeline_and_models.py::test_model_evaluation_metrics PASSED  [ 64%]
tests/test_pipeline_and_models.py::test_dvc_metrics_and_evaluation_report_files PASSED [ 71%]
tests/test_skill_gap.py::test_skill_gap_formula_calculation PASSED       [ 78%]
tests/test_skill_gap.py::test_ranking_priority_order PASSED              [ 85%]
tests/test_skill_gap.py::test_unsupported_role_raises_error PASSED       [ 92%]
tests/test_skill_gap.py::test_exceeding_proficiency_has_zero_gap PASSED  [100%]

============================= 14 passed in 3.47s ==============================
```

Ruff Code Quality Audit:
```
$ ruff check src tests
All checks passed!
```
