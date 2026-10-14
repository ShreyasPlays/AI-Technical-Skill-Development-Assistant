# Dataset Card: Technical Skill Development Benchmark Dataset

## Dataset Title and Identifier
- **Dataset Title:** AI-Based Technical Skill Development Assistant Benchmark Dataset
- **Standard Identifier:** `skills_dataset.csv`
- **Canonical Repository Path:** `data/raw/skills_dataset.csv`
- **Task Deliverable:** Member 1 — DATASET ACQUISITION AND VALIDATION
- **Author / Analyst:** Siddharth Dhanush R (Member 1)
- **Contact:** `siddharthdhanush7@gmail.com`

---

## Dataset Version and DVC Reference
- **DVC Tracking Pointer:** `data/raw/skills_dataset.csv.dvc`
- **Committed Pointer MD5 Hash:** `1280aa1c7cb0b66dcda792cd0a9ddd11`
- **Pointer File Size:** `59019` bytes
- **Independent Cryptographic Hash (SHA-256):** `20dfd52c6e60949b1c934e5d1a78725940175eed43d98340ffee3e84156f92e3`
- **DVC Pipeline Stage Dependency:** `stages.preprocess.deps: data/raw/skills_dataset.csv` (defined in `dvc.yaml`)
- **Pipeline Lock Reference:** Verified in `dvc.lock` under stage `preprocess`

---

## Executive Summary
The **Technical Skill Development Benchmark Dataset** is a structured tabular benchmark engineered to support Phase 1 development, data validation, pipeline reproducibility, and baseline model benchmarking for the **AI-Based Technical Skill Development Assistant**. It contains **1,200 candidate profile records** characterized across **10 technical competency skills** on a 1–5 integer rating scale and **3 contextual experience metrics**, mapped to **5 balanced career tracks** (240 samples per role).

Every measurement documented in this card has been programmatically computed and verified by executing `notebooks/01_dataset_acquisition_validation.ipynb` against the canonical artifact. The dataset satisfies all schema, range, completeness, and non-duplication rules required by the Phase 1 engineering contract.

> [!CAUTION]
> **Prominent Non-Negotiable Methodological Warning:**
> This dataset is synthetic/simulated and generated from predefined technical-role archetypes. Its class labels are produced by the generation process. Models trained and evaluated on these data may learn those artificial rules. Metrics from this dataset do not, by themselves, demonstrate that the system can assess real students, recommend suitable careers, predict employability, or measure genuine technical competency.

---

## Dataset Purpose
The primary purposes of this dataset are:
1. **Pipeline Reproducibility:** Serving as the versioned raw data artifact for end-to-end MLOps orchestration via **DVC Pipelines** (`dvc repro`).
2. **Schema & Contract Testing:** Validating data ingestion interfaces, defensive error handling, and type safety in `src/data_loader.py`.
3. **Feature Engineering Validation:** Providing input signals for computing domain composite readiness indices (`ml_ai_readiness`, `data_analytics_readiness`, `fullstack_readiness`, `devops_cloud_readiness`, `practical_velocity`) in `src/feature_engineering.py`.
4. **Baseline Classification:** Benchmarking multi-class classifiers (Decision Tree and Random Forest) to demonstrate model tracking via **MLflow**.
5. **Skill-Gap Demonstration:** Supplying realistic input vectors for evaluating the mathematical rule-based `SkillGapAnalyzer` ($G_i = \max(0, R_i - S_i)$ and $G_{avg}$) against role requirements.

---

## Intended Use
- Educational demonstrations of tabular ML pipelines and MLOps tooling (Git, DVC, MLflow, pytest, Ruff).
- Automated integration testing of preprocessing, training, and evaluation scripts.
- Verifying mathematical skill-gap formulations against role-requirement vectors.
- Comparing tree-based classifier baselines under controlled, reproducible conditions.

---

## Origin and Provenance
- **Origin Category:** Internally generated synthetic tabular benchmark.
- **Generator Implementation:** `src/data_loader.py` via `generate_benchmark_dataset()`.
- **Primary Generator Parameters:** `n_samples = 1200`, `random_state = 42`.
- **External Data Sources:** None. No records originate from Kaggle, LinkedIn, GitHub scraping, college placement offices, student surveys, or commercial hiring databases.

---

## Generation Procedure
The dataset generation workflow in `src/data_loader.py` follows a disciplined five-step simulation:
1. **Archetype Definitions:** Five role profiles define baseline Gaussian mean proficiencies across all 10 skills:
   - *Machine Learning Engineer:* high ML (4.2), Python (4.0), Deep Learning (4.0), DSA (3.8), Stats (3.9).
   - *Data Scientist:* high Statistics (4.5), Python (4.1), SQL (4.0), ML (3.9).
   - *Full Stack Developer:* high Web Dev (4.5), Git (4.0), DSA (3.9), System Design (3.8).
   - *DevOps Engineer:* high Git (4.6), Cloud (4.6), System Design (3.9), Python (3.4).
   - *Data Analyst:* high SQL (4.5), Statistics (4.1), Python (3.1), Git (2.8).
2. **Skill Perturbation & Clipping:** For each candidate in a role cohort, raw skill proficiencies are sampled from $N(\mu_{role}, \sigma=0.75)$, rounded to the nearest integer, and clipped to the valid proficiency boundary $[1, 5]$.
3. **Contextual Sampling:**
   - `years_experience`: Sampled from $N(\mu_{exp}, \sigma=0.8)$, clipped at $0.0$, rounded to 1 decimal place.
   - `projects_completed`: Sampled from $N(\mu_{proj}, \sigma=1.8)$, clipped at $1$, rounded to integer.
   - `problems_solved`: Sampled from $N(\mu_{prob}, \sigma=65)$, clipped at $10$, rounded to integer.
4. **Target Label Assignment:** The categorical label `target_role` is assigned directly from the generation loop.
5. **Cohort Shuffling:** The combined 1,200 records are deterministically shuffled using `random_state=42` and serialized to CSV.

---

## Record Count and Feature Count
- **Total Records (Rows):** `1,200`
- **Total Columns:** `14`
- **Input Predictors:** `13` (10 technical skills + 3 contextual metrics)
- **Target Variable:** `1` (`target_role`)
- **Total Data Cells:** `16,800`

---

## Schema Overview
The table below specifies the verified schema layout:

| # | Column Name | Role in Pipeline | Inferred Storage Dtype | Null Count |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `python` | Input Feature | `int64` | 0 |
| 2 | `sql` | Input Feature | `int64` | 0 |
| 3 | `machine_learning` | Input Feature | `int64` | 0 |
| 4 | `dsa` | Input Feature | `int64` | 0 |
| 5 | `git` | Input Feature | `int64` | 0 |
| 6 | `web_development` | Input Feature | `int64` | 0 |
| 7 | `cloud_computing` | Input Feature | `int64` | 0 |
| 8 | `system_design` | Input Feature | `int64` | 0 |
| 9 | `statistics` | Input Feature | `int64` | 0 |
| 10 | `deep_learning` | Input Feature | `int64` | 0 |
| 11 | `years_experience` | Input Feature | `float64` | 0 |
| 12 | `projects_completed` | Input Feature | `int64` | 0 |
| 13 | `problems_solved` | Input Feature | `int64` | 0 |
| 14 | `target_role` | Target Classification Label | `object` | 0 |

---

## Technical Skill Feature Descriptions
Each technical skill represents self-reported or rubric-evaluated proficiency on a 1–5 integer Dreyfus scale:
- **1 — Novice / Beginner:** Little or no practical exposure; understands basic terminology.
- **2 — Advanced Beginner:** Can execute simple tasks with guidance or documentation.
- **3 — Competent / Intermediate:** Independent practical ability on standard development tasks.
- **4 — Proficient:** Deep conceptual grasp; solves complex edge cases and architectural tasks.
- **5 — Expert:** Authoritative master; designs systems, optimizes performance, mentors others.

### Individual Skill Statistics (Measured from CSV)
| Skill Feature | Observed Min | Observed Max | Mean | Std Dev | Median | % Rating $\ge 4$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `python` | 1 | 5 | 3.52 | 0.89 | 4 | 55.4% |
| `sql` | 1 | 5 | 3.44 | 1.00 | 4 | 50.1% |
| `machine_learning` | 1 | 5 | 2.70 | 1.33 | 3 | 31.8% |
| `dsa` | 1 | 5 | 3.15 | 0.97 | 3 | 37.8% |
| `git` | 1 | 5 | 3.57 | 0.98 | 4 | 56.6% |
| `web_development` | 1 | 5 | 2.35 | 1.26 | 2 | 21.8% |
| `cloud_computing` | 1 | 5 | 3.14 | 1.10 | 3 | 38.3% |
| `system_design` | 1 | 5 | 3.02 | 1.14 | 3 | 34.7% |
| `statistics` | 1 | 5 | 3.29 | 1.27 | 3 | 47.7% |
| `deep_learning` | 1 | 5 | 2.33 | 1.26 | 2 | 20.9% |

---

## Contextual Feature Descriptions
Contextual metrics capture simulated candidate experience and velocity beyond raw skill ratings:

| Feature Name | Description | Inferred Dtype | Min | Q25 | Median | Mean | Q75 | Max | Std | Outliers (IQR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `years_experience` | Relevant professional / internship tenure (years) | `float64` | 0.0 | 1.1 | 1.6 | 1.68 | 2.2 | 4.4 | 0.79 | 15 (1.2%) |
| `projects_completed` | Completed software / data science projects | `int64` | 1 | 4 | 5 | 5.19 | 6 | 11 | 1.84 | 11 (0.9%) |
| `problems_solved` | Algorithmic coding problems solved on platforms | `int64` | 10 | 119 | 176 | 180.38 | 239 | 443 | 82.25 | 4 (0.3%) |

---

## Target Definition and Supported Classes
- **Target Column:** `target_role`
- **Type:** Multi-class categorical string
- **Number of Supported Classes:** Exactly `5`
- **Supported Class Identifiers:**
  1. `Data Analyst`
  2. `Data Scientist`
  3. `DevOps Engineer`
  4. `Full Stack Developer`
  5. `Machine Learning Engineer`

---

## Actual Target-Class Distribution
| Career Role | Measured Record Count | Observed Proportion (%) | Expected Balanced Count | Delta | Balance Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Data Analyst** | 240 | 20.00% | 240 | 0 | Perfectly Balanced |
| **Data Scientist** | 240 | 20.00% | 240 | 0 | Perfectly Balanced |
| **DevOps Engineer** | 240 | 20.00% | 240 | 0 | Perfectly Balanced |
| **Full Stack Developer** | 240 | 20.00% | 240 | 0 | Perfectly Balanced |
| **Machine Learning Engineer** | 240 | 20.00% | 240 | 0 | Perfectly Balanced |
| **Total** | **1,200** | **100.00%** | **1,200** | **0** | **Strict Stratification** |

---

## Data Types and Value Ranges
- **Technical Skills:** Stored as `int64`. Validated range $[1, 5]$. No values $< 1$, no values $> 5$, zero fractional values.
- **`years_experience`:** Stored as `float64`. Validated non-negative $[0.0, 4.4]$.
- **`projects_completed`:** Stored as `int64`. Validated positive integer $[1, 11]$.
- **`problems_solved`:** Stored as `int64`. Validated positive integer $[10, 443]$.
- **`target_role`:** Stored as `object`. Validated against `ROLE_CLASSES`. Zero whitespace or casing defects.

---

## Quality-Validation Methodology
Data quality was audited through a multi-tier automated validation protocol executed in `notebooks/01_dataset_acquisition_validation.ipynb` and enforced in `src/data_loader.py`:
1. **Cryptographic Integrity:** Hashing file bytes via SHA-256 and MD5, comparing against DVC tracking metadata.
2. **Schema & Typographic Verification:** Header inspection ensuring exact column presence, order, and absence of duplicate names.
3. **Range & Type Constraints:** Programmatic type assertion, integer-equality testing, and boundary clipping checks.
4. **Completeness & Duplication Audit:** Null cell counting, string whitespace inspection, and row/profile duplicate detection.
5. **Defensive Rejection Testing:** Passing synthetic malformed fixtures to ensure the validator rejects invalid inputs.

---

## Measured Data-Quality Results
All 18 programmatic checks executed in the validation notebook passed:

| Check ID | Category | Description | Measured Outcome | Status |
| :---: | :--- | :--- | :--- | :---: |
| **VAL-01** | Schema | All 14 required columns present | Exactly 14 columns observed | **PASS** |
| **VAL-02** | Schema | No unexpected additional columns | 0 unexpected columns | **PASS** |
| **VAL-03** | Schema | No duplicate column headers in CSV | 0 duplicate column names | **PASS** |
| **VAL-04** | Schema | Standard column order preserved | Exactly matches project schema contract | **PASS** |
| **VAL-05** | Dtype | Skill columns stored as integer types | `int64` across all 10 skills | **PASS** |
| **VAL-06** | Dtype | `years_experience` stored as float | `float64` observed | **PASS** |
| **VAL-07** | Dtype | Project & problem counts stored as integers | `int64` observed | **PASS** |
| **VAL-08** | Range | Technical skills lower bound $\ge 1$ | Minimum observed = 1 | **PASS** |
| **VAL-09** | Range | Technical skills upper bound $\le 5$ | Maximum observed = 5 | **PASS** |
| **VAL-10** | Range | Technical skills integer value constraint | 0 fractional values observed | **PASS** |
| **VAL-11** | Context | `years_experience` non-negative | Minimum observed = 0.0 years | **PASS** |
| **VAL-12** | Context | `projects_completed` positive integer | Minimum observed = 1 project | **PASS** |
| **VAL-13** | Context | `problems_solved` positive integer | Minimum observed = 10 problems | **PASS** |
| **VAL-14** | Target | Target role valid categories | Exactly 5 documented roles observed | **PASS** |
| **VAL-15** | Completeness | Zero missing cells in dataset | 0 nulls across 16,800 cells (0.0%) | **PASS** |
| **VAL-16** | Duplicates | Zero full duplicate rows | 0 complete duplicate rows | **PASS** |
| **VAL-17** | Duplicates | Zero duplicate feature profiles | 0 duplicate input vectors | **PASS** |
| **VAL-18** | Integrity | DVC MD5 hash matching pointer | Exactly matches `1280aa1c...` | **PASS** |

---

## Duplicate Records and Feature-Profile Risks
- **Full Row Duplicates:** `0` (0.0%).
- **Duplicate Feature Profiles:** `0` (0.0%).
- **Contradictory Label Assignments:** `0`.
- **Methodological Context:** In real-world survey datasets, identical feature profiles frequently appear with divergent career outcomes due to unmeasured latent factors (communication skills, institution tier, interview presence). In this simulated benchmark, zero collisions exist, meaning class boundaries are clean and free from contradictory supervisory signal.

---

## Missing-Value and Invalid-Value Findings
- **Missing Cells:** `0`
- **Missing Rows:** `0`
- **Empty String Values:** `0`
- **Out-of-Bounds Ratings:** `0`
- **Fractional Ratings:** `0`
- **Negative Context Values:** `0`

---

## Potential Target Leakage
- **Direct Target Leakage:** **None.** No column in the raw CSV directly leaks or duplicates `target_role`.
- **Proxy Leakage & Rule Encoding:** **High structural correlation.** Because records were generated around role-specific mean vectors, the combination of skill ratings closely mirrors the generator's internal lookup dictionary. Classifiers effectively invert the simulation rather than discovering empirical relationships.
- **Downstream Preprocessing Safeguard:** As verified in `src/preprocess.py` and `tests/test_feature_engineering.py`, train/test splitting occurs *before* composite feature scaling, ensuring zero downstream data leakage.

---

## Synthetic-Data Limitations and Expected Generalization Risks
1. **Shortcut Learning:** Tree classifiers achieve elevated accuracy ($\sim 92\%$ Random Forest test accuracy) by learning the generator's normal distribution boundaries.
2. **Distribution Shift:** Real candidates exhibit irregular, multi-modal, or interdisciplinary skill sets (e.g., full-stack engineers transitioning to ML) not captured by standard normal archetypes.
3. **Absence of Human Psychometric Calibration:** Self-evaluations contain cognitive biases (Dunning-Kruger effect, impostor syndrome), whereas the benchmark assumes uniform Gaussian error.
4. **Artificial Class Parity:** The job market has unequal demand ratios, unlike the 20% flat distribution in this benchmark.

---

## Privacy and Ethical Considerations
- **Personal Identifiable Information (PII):** 0 PII elements. No student names, emails, IP addresses, demographics, or academic records are present.
- **Ethical Use Constraints:** This synthetic dataset is developed strictly for algorithmic and pipeline verification. It must **never** be used for student placement gatekeeping, candidate rejection, course access throttling, or automated hiring decisions.

---

## License and Reuse Permissions
- **Repository Documentation:** The project `README.md` displays an MIT License badge.
- **Repository Inspection:** A formal root `LICENSE` file is currently missing from the repository tree.
- **Current Legal Status:**
> [!IMPORTANT]
> **Dataset license and reuse permissions: UNVERIFIED — confirm with the project owner before external redistribution or reuse.**
>
> A software badge in the README does not automatically confer open data redistribution rights. The repository owner has been requested to commit a formal `LICENSE` file covering both software and data artifacts.

---

## Appropriate Uses
- Verifying pipeline orchestration via `dvc repro`.
- Testing data validation logic in `src/data_loader.py`.
- Training and tracking baseline models in MLflow.
- Computing mathematical skill gaps ($G_i$) against role requirements.
- Educational demonstrations of MLOps best practices.

---

## Inappropriate Uses
- Evaluating real human students or job candidates.
- Making employment or admissions decisions.
- Publishing real-world career-fit claims based on model accuracy.
- Training production models intended for campus recruitment.

---

## Known Issues and Open Questions
1. **Inaccessible DVC Remote Path:** `.dvc/config` points to `C:\Users\shrey\...`, preventing cross-workstation `dvc pull`. A centralized cloud remote is needed in Phase 2.
2. **Missing Root `LICENSE` File:** Repository lacks a committed license document.
3. **Empirical Validation Gap:** The system lacks a real-world student validation cohort for ground-truth benchmarking.

---

## Reproduction Instructions
To reproduce data validation and inspect the dataset:
```powershell
# 1. Activate project environment
.\.venv\Scripts\Activate.ps1

# 2. Run unit tests
pytest -v tests/test_data_loader.py

# 3. Execute validation notebook
jupyter nbconvert --to notebook --execute notebooks/01_dataset_acquisition_validation.ipynb
```

---

## Validation Report Reference
For the complete technical investigation, interactive visualizations, and code executions, refer to:
- [notebooks/01_dataset_acquisition_validation.ipynb](file:///c:/Users/dhanu/OneDrive/Desktop/MLOps/AI-Technical-Skill-Development-Assistant/notebooks/01_dataset_acquisition_validation.ipynb)
- [reports/phase1_evaluation.md](file:///c:/Users/dhanu/OneDrive/Desktop/MLOps/AI-Technical-Skill-Development-Assistant/reports/phase1_evaluation.md)

---

## Responsible Owner and Maintenance Notes
- **Responsible Owner:** Siddharth Dhanush R (Member 1 — Dataset Acquisition & Validation)
- **Review Date:** 14 October 2026
- **Submission Target:** 15 October 2026 Milestone Review
- **Maintenance Policy:** The raw dataset artifact and its committed DVC pointer (`skills_dataset.csv.dvc`) must be preserved without manual alteration. Any future modifications to generator archetypes or schema must be versioned through DVC and approved by the team.
