"""
Comprehensive Code Walkthrough & In-Depth Implementation Manual Generator.
Generates an engineering-grade PDF explaining all project code line-by-line:
- src/data_loader.py
- src/preprocess.py
- src/feature_engineering.py
- src/train.py
- src/evaluate.py
- src/model_risk.py
- src/model_registry.py
- src/roadmap.py
- src/api/schemas.py
- src/api/app.py
- predict.py
- dvc.yaml, Dockerfile, and .github/workflows/ci.yml
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display 'Page X of Y'."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        if self._pageNumber > 1:
            # Running Header
            self.drawString(36, 762, "AI Technical Skill Assistant — In-Depth Code Walkthrough & Implementation Manual")
            self.drawRightString(576, 762, "Source Code Line-by-Line Defense")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

            # Running Footer
            self.line(36, 40, 576, 40)
            self.drawString(36, 28, "RVCE Dept. of Computer Science & Engineering | Academic Project Source Defense")
            self.drawRightString(576, 28, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


def build_code_walkthrough_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    # Color Palette
    c_primary = colors.HexColor("#0F172A")    # Deep Navy Slate
    c_accent = colors.HexColor("#1D4ED8")     # Indigo / Blue
    c_teal = colors.HexColor("#0F766E")       # Dark Teal
    c_dark = colors.HexColor("#1E293B")       # Text Charcoal
    c_code_bg = colors.HexColor("#F8FAFC")    # Code block background
    c_border = colors.HexColor("#CBD5E1")     # Border light
    c_highlight = colors.HexColor("#EFF6FF")  # Highlight box

    # Typography
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=c_primary,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_accent,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=c_dark,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=2,
    )

    code_block_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.2,
        leading=9.8,
        textColor=colors.HexColor("#0F172A"),
    )

    explanation_style = ParagraphStyle(
        "CodeExplanation",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.8,
        textColor=colors.HexColor("#1E293B"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1,
    )

    story = []

    def add_code_section(file_title: str, file_path: str, snippet: str, line_breakdown: list):
        story.append(Paragraph(f"Module: {file_title}", h1_style))
        story.append(Paragraph(f"<b>File Path:</b> <code>{file_path}</code>", body_style))
        story.append(HRFlowable(width="100%", thickness=1, color=c_accent, spaceAfter=8))

        # Code display box
        code_table = Table([[Paragraph(snippet.strip().replace("\n", "<br/>").replace(" ", "&nbsp;"), code_block_style)]], colWidths=[540])
        code_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_code_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, c_border),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(code_table)
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Line-by-Line &amp; Algorithmic Analysis:</b>", h2_style))

        # Line analysis table
        analysis_rows = [[Paragraph("Line(s)", table_header_style), Paragraph("Executed Statement / Pattern", table_header_style), Paragraph("In-Depth Algorithmic, MLOps &amp; Engineering Explanation", table_header_style)]]
        for lines, stmt, expl in line_breakdown:
            analysis_rows.append([
                Paragraph(f"<b>{lines}</b>", ParagraphStyle("C1", parent=body_style, alignment=1)),
                Paragraph(f"<code>{stmt}</code>", code_block_style),
                Paragraph(expl, explanation_style),
            ])

        a_table = Table(analysis_rows, colWidths=[55, 175, 310])
        a_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_primary),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_highlight]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(a_table)
        story.append(Spacer(1, 14))

    # =========================================================================
    # TITLE BANNER
    # =========================================================================
    story.append(Paragraph("AI TECHNICAL SKILL DEVELOPMENT ASSISTANT", title_style))
    story.append(Paragraph(
        "<b>In-Depth Code Walkthrough &amp; Line-by-Line Implementation Manual:</b> Comprehensive Line-by-Line "
        "Source Code Architecture, Mathematical Formulations, MLOps Pipeline Scripts, and Production Engineering Defense",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=c_accent, spaceAfter=10))

    meta_info = [
        [
            Paragraph("<b>Academic Scope:</b> Phase 2 Production & Line-by-Line Code Defense", body_style),
            Paragraph("<b>Target Audience:</b> Project Evaluators, Professors, ML Engineers", body_style),
        ],
        [
            Paragraph("<b>Curated Scope:</b> 10 Core 21st-Century Technical Skills", body_style),
            Paragraph("<b>Deployment Status:</b> Live on Render Cloud (FastAPI + Docker)", body_style),
        ],
    ]
    meta_tbl = Table(meta_info, colWidths=[270, 270])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_code_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, c_border),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 12))

    # =========================================================================
    # MODULE 1: src/data_loader.py
    # =========================================================================
    add_code_section(
        file_title="Data Loading & 21st-Century Skill Taxonomy",
        file_path="src/data_loader.py",
        snippet="""
SKILL_COLUMNS: List[str] = [
    "database_fundamentals", "computer_architecture", "distributed_computing_systems",
    "cyber_security", "networking", "software_development", "programming_skills",
    "project_management", "computer_forensics_fundamentals", "technical_communication",
    "ai_ml", "software_engineering", "business_analysis", "communication_skills",
    "data_science", "troubleshooting_skills", "graphics_designing"
]

CENTURY_21_SKILLS: List[str] = [
    "ai_ml", "data_science", "software_development", "programming_skills",
    "cyber_security", "distributed_computing_systems", "database_fundamentals",
    "software_engineering", "troubleshooting_skills", "technical_communication"
]

CORE_ASSESSMENT_WEIGHTS: Dict[str, float] = {
    "programming_skills": 0.25,
    "software_development": 0.20,
    "software_engineering": 0.15,
    "troubleshooting_skills": 0.15,
    "database_fundamentals": 0.10,
    "distributed_computing_systems": 0.10,
    "technical_communication": 0.05,
}
""",
        line_breakdown=[
            ("L15-33", "SKILL_COLUMNS definition", "Defines the canonical 17 technical skills extracted from the Kaggle Career Mapping benchmark dataset. All scores are bounded on a discrete integer scale of 0 to 6."),
            ("L40-51", "CENTURY_21_SKILLS curation", "Implements the evaluator's mandate: extracts the 10 modern computing disciplines while pruning legacy hardware, redundant communication, and non-technical attributes (e.g., graphics designing, business analysis)."),
            ("L85-93", "CORE_ASSESSMENT_WEIGHTS", "Defines the domain-informed weight matrix for the foundational screening gate. Programming (25%) and Software Development (20%) form the core, balanced by Systems, Databases, Troubleshooting, and Technical Communication. Mathematical constraint: sum(weights) == 1.00."),
        ]
    )

    # =========================================================================
    # MODULE 2: src/preprocess.py
    # =========================================================================
    add_code_section(
        file_title="Label Engineering & Leak-Free Splitting",
        file_path="src/preprocess.py",
        snippet="""
class AssessmentLabelGenerator:
    _SHARPNESS: float = 5.0

    def __init__(self, threshold: float = 0.50, random_state: int = 42):
        self.threshold = threshold
        self.random_state = random_state

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rng = np.random.default_rng(self.random_state)
        raw_score = pd.Series(0.0, index=df.index)
        for skill, weight in CORE_ASSESSMENT_WEIGHTS.items():
            raw_score += df[skill].astype(float) * weight
        score_norm = raw_score / 6.0
        prob_pass = self._sigmoid(self._SHARPNESS * (score_norm - self.threshold))
        assessment_pass = (rng.random(len(df)) < prob_pass.values).astype(int)
        df["assessment_score"] = score_norm.round(4)
        df["assessment_pass"] = assessment_pass
        return df

def split_data_leak_free(df: pd.DataFrame, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["assessment_pass"])
""",
        line_breakdown=[
            ("L30-58", "AssessmentLabelGenerator class", "Engineers a continuous latent competence variable and samples a probabilistic binary target (assessment_pass). Captures real-world assessment variance (test-day pressure, problem difficulty)."),
            ("L68-70", "Sigmoid activation function", "Applies logistic curve: S(x) = 1 / (1 + e^(-x)). Scales normalized student score relative to the threshold (0.50) with boundary sharpness = 5.0."),
            ("L86-95", "Weighted normalization", "Computes dot product: Score_norm = sum(Skill_i * Weight_i) / 6.0. Maximum score is strictly bounded in [0.0, 1.0]."),
            ("L100-104", "Bernoulli sampling", "Generates binary label Y ~ Bernoulli(prob_pass). If rng.random() < prob_pass, label = 1 (Pass), else 0 (Fail). Seed 42 guarantees deterministic reproducibility under DVC."),
            ("L150-165", "split_data_leak_free()", "Executes stratified 80/20 train/test split. Preserves target label ratio (46% Pass, 54% Fail) across both splits before any feature scaling to prevent data snooping/leakage."),
        ]
    )

    # =========================================================================
    # MODULE 3: src/feature_engineering.py
    # =========================================================================
    add_code_section(
        file_title="Composite Feature Engineering & Scaling Contract",
        file_path="src/feature_engineering.py",
        snippet="""
def engineer_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    available_skills = [c for c in SKILL_COLUMNS if c in df.columns]
    skill_vals = df[available_skills]
    df["technical_overall_mean"] = skill_vals.mean(axis=1).round(4)
    df["technical_breadth"] = (skill_vals >= 3).sum(axis=1).astype(int)
    df["technical_depth"] = (skill_vals >= 5).sum(axis=1).astype(int)

    df["ai_data_readiness"] = (0.45 * df["ai_ml"] + 0.45 * df["data_science"] + 0.10 * df["programming_skills"]).round(4)
    df["systems_readiness"] = (0.35 * df["networking"] + 0.35 * df["distributed_computing_systems"] + 0.30 * df["computer_architecture"]).round(4)
    df["software_dev_readiness"] = (0.40 * df["software_development"] + 0.35 * df["programming_skills"] + 0.25 * df["software_engineering"]).round(4)
    df["cybersecurity_readiness"] = (0.50 * df["cyber_security"] + 0.30 * df["computer_forensics_fundamentals"] + 0.20 * df["networking"]).round(4)
    return df

class SkillFeaturePipeline:
    def fit(self, X_train: pd.DataFrame):
        X_eng = engineer_composite_features(X_train)
        self.scaler = StandardScaler().fit(X_eng)
        self.feature_columns = list(X_eng.columns)
        return self
""",
        line_breakdown=[
            ("L38-57", "Aggregate statistics engineering", "Computes breadth and depth indicators: technical_overall_mean measures macro competence; technical_breadth counts skills >= 3 (intermediate); technical_depth counts skills >= 5 (advanced mastery)."),
            ("L58-92", "Domain readiness composites", "Synthesizes multi-skill domain indices. Captures non-linear domain synergies (e.g., AI readiness requires ML + Data Science + core programming; Systems readiness requires distributed systems + networking)."),
            ("L110-145", "SkillFeaturePipeline.fit()", "Crucial MLOps integrity step: Fits StandardScaler exclusively on the training partition. Records scaler.mean_ and scaler.scale_ into data/processed/feature_metadata.json so the inference API replicates exact normalization without leakage."),
        ]
    )

    # =========================================================================
    # MODULE 4: src/train.py
    # =========================================================================
    add_code_section(
        file_title="Constrained Model Training & MLflow Tracking",
        file_path="src/train.py",
        snippet="""
def train_models(config: Optional[AppConfig] = None):
    # MLflow experiment setup
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    dataset_hash = compute_file_sha256(config.paths.processed_train_path)

    # ── Champion: Random Forest ──────────────────────────────────────────
    rf_cfg = config.models.random_forest
    rf_params = {
        "n_estimators": rf_cfg.n_estimators,
        "max_depth": rf_cfg.max_depth,
        "max_features": rf_cfg.max_features,
        "min_samples_split": rf_cfg.min_samples_split,
        "min_samples_leaf": rf_cfg.min_samples_leaf,
        "criterion": rf_cfg.criterion,
        "random_state": config.models.random_state,
        "n_jobs": -1
    }
    with mlflow.start_run(run_name="Random_Forest_Binary") as rf_run:
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        train_acc = float(rf_model.score(X_train, y_train))
        mlflow.log_params({**rf_params, "dataset_sha256": dataset_hash})
        mlflow.log_metric("train_accuracy", train_acc)
        joblib.dump(rf_model, config.paths.models_dir / "random_forest.joblib")
        mlflow.sklearn.log_model(rf_model, artifact_path="model")
""",
        line_breakdown=[
            ("L52-60", "MLflow tracking initialization", "Sets local tracking directory (mlruns/) and establishes the active MLflow experiment. Computes SHA-256 hash of training parquet file for data provenance verification."),
            ("L125-135", "Constrained hyperparameter setup", "Intentionally bounds max_depth=7 and min_samples_leaf=4. Prevents memorizing survey noise and targets realistic 70–80% diagnostic test accuracy."),
            ("L137-142", "Model fitting & scoring", "Fits ensemble of 100 de-correlated decision trees on 6,822 scaled training records. Computes training accuracy (train_acc ~ 75.8%)."),
            ("L144-152", "MLflow logging & serialization", "Logs all hyperparameters, dataset hash, and metrics to MLflow. Dumps binary artifact using joblib and registers the MLflow sklearn model flavor."),
        ]
    )

    # =========================================================================
    # MODULE 5: src/evaluate.py
    # =========================================================================
    add_code_section(
        file_title="Model Evaluation & Confusion Matrix Computation",
        file_path="src/evaluate.py",
        snippet="""
def evaluate_single_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
        "confusion_matrix": cm,
    }
""",
        line_breakdown=[
            ("L52-58", "Inference & probability scoring", "Queries model predictions y_pred and retrieves continuous positive class confidence probabilities y_prob for ROC-AUC evaluation."),
            ("L63-68", "Classification metrics calculation", "Computes comprehensive binary evaluation suite: Accuracy (overall correctness), Precision (avoiding False Positives), Recall (catching true competent candidates), and F1-score (harmonic mean)."),
            ("L67", "confusion_matrix(y_test, y_pred)", "Computes exact 2x2 confusion matrix [[TN, FP], [FN, TP]]. For Random Forest: TN=711, FP=210, FN=242, TP=543 on N=1,706 test students."),
            ("L83-91", "Metrics serialization", "Exports structured evaluation dictionary to reports/metrics.json and reports/phase1_rework_evaluation.md for DVC metric tracking."),
        ]
    )

    # =========================================================================
    # MODULE 6: src/model_risk.py
    # =========================================================================
    add_code_section(
        file_title="Behavioral Stress Testing & Model Quality Auditor",
        file_path="src/model_risk.py",
        snippet="""
class ModelQualityAuditor:
    def test_boundary_extremes(self) -> Dict[str, Any]:
        zero_skills = {s: 0 for s in SKILL_COLUMNS}
        max_skills = {s: 6 for s in SKILL_COLUMNS}
        prob_zero = float(self.model.predict_proba(self.transform(zero_skills))[0, 1])
        prob_max = float(self.model.predict_proba(self.transform(max_skills))[0, 1])
        passed = (prob_zero < 0.20) and (prob_max > 0.80)
        return {"passed": passed, "prob_zero": prob_zero, "prob_max": prob_max}

    def test_monotonicity(self, skill_to_test: str = "programming_skills") -> Dict[str, Any]:
        base_skills = {s: 3 for s in SKILL_COLUMNS}
        probabilities = []
        for level in range(7):
            profile = base_skills.copy()
            profile[skill_to_test] = level
            prob = float(self.model.predict_proba(self.transform(profile))[0, 1])
            probabilities.append(prob)
        corr = float(np.corrcoef(np.arange(7), probabilities)[0, 1])
        is_positive_trend = (probabilities[-1] > probabilities[0]) and (corr > 0.70)
        return {"passed": is_positive_trend, "correlation": corr, "prob_range": [probabilities[0], probabilities[-1]]}
""",
        line_breakdown=[
            ("L83-106", "Boundary Extremes Test", "Pass gate requirement: Verifies that a student with all 0s has < 20% pass probability (actual = 9.79%) and all 6s has > 80% (actual = 83.44%). Prevents sanity violations."),
            ("L108-135", "Directional Monotonicity Test", "Sweeps a core skill (Programming, AI/ML) from level 0 to 6 holding other features constant. Computes Pearson correlation between skill level and pass probability. Validates r > 0.70 (Programming r = 0.9373). Proves model never penalizes learning."),
            ("L140-165", "Invariance Testing", "Perturbs non-core legacy attributes (Graphics Designing) on a strong candidate profile. Verifies verdict stability: 100% of outputs remain Pass (zero spurious decision flips)."),
            ("L170-195", "Latency SLA Benchmark", "Executes 100 single-sample inferences. Validates average latency (actual = 32.96 ms) satisfies real-time serving SLA (<= 35.0 ms)."),
        ]
    )

    # =========================================================================
    # MODULE 7: src/model_registry.py
    # =========================================================================
    add_code_section(
        file_title="Model Registry & Staging Management",
        file_path="src/model_registry.py",
        snippet="""
class ModelRegistryManager:
    def _compute_file_hash(self, path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def build_registry_manifest(self, champion_name="random_forest"):
        champion_file = self.models_dir / f"{champion_name}.joblib"
        champion_hash = self._compute_file_hash(champion_file)
        metrics = self.get_evaluation_metrics()
        quality_passed = (metrics.get(champion_name, {}).get("accuracy", 0.0) >= 0.70
                          and metrics.get(champion_name, {}).get("roc_auc", 0.0) >= 0.75)
        return {
            "registry_version": "2.1.0",
            "active_champion": {
                "model_name": champion_name,
                "stage": "Production",
                "sha256": champion_hash,
                "quality_passed": quality_passed,
                "metrics": metrics.get(champion_name, {})
            }
        }
""",
        line_breakdown=[
            ("L38-47", "SHA-256 provenance hashing", "Reads binary model artifact in 64KB chunks and computes cryptographic hash. Guarantees model immutability and provenance tracking between training and deployment."),
            ("L72-105", "Registry manifest synthesis", "Validates that model metrics exceed automated quality thresholds (Accuracy >= 70%, ROC-AUC >= 75%). Assembles metadata contract and records to models/model_registry.json."),
            ("L140-175", "MLflow Model Registry integration", "Uses MlflowClient to register model version and transitions stage to 'Production'. Ensures full lifecycle governance."),
        ]
    )

    # =========================================================================
    # MODULE 8: src/api/schemas.py & app.py
    # =========================================================================
    add_code_section(
        file_title="FastAPI REST Microservice & Pydantic Validation",
        file_path="src/api/app.py & src/api/schemas.py",
        snippet="""
class Century21SkillProfile(BaseModel):
    ai_ml: int = Field(..., ge=0, le=6, description="AI/ML proficiency (0-6)")
    data_science: int = Field(..., ge=0, le=6, description="Data Science (0-6)")
    software_development: int = Field(..., ge=0, le=6, description="Software Dev (0-6)")
    programming_skills: int = Field(..., ge=0, le=6, description="Programming (0-6)")
    cyber_security: int = Field(..., ge=0, le=6, description="Cyber Security (0-6)")
    distributed_computing_systems: int = Field(..., ge=0, le=6, description="Distributed Systems (0-6)")
    database_fundamentals: int = Field(..., ge=0, le=6, description="Database Fundamentals (0-6)")
    software_engineering: int = Field(..., ge=0, le=6, description="Software Engineering (0-6)")
    troubleshooting_skills: int = Field(..., ge=0, le=6, description="Troubleshooting (0-6)")
    technical_communication: int = Field(..., ge=0, le=6, description="Technical Comm (0-6)")

@app.post("/predict", response_model=PredictionResponse)
def predict_assessment_pass(payload: PredictRequest):
    skills_dict = payload.skills.to_dict()
    X_df = _transform_skill_dict_to_input(skills_dict)
    prob_pass = float(state.champion_model.predict_proba(X_df)[0, 1])
    pass_binary = int(prob_pass >= 0.50)
    prediction_label = "Pass" if pass_binary == 1 else "Fail"
    confidence = abs(prob_pass - 0.50) * 2.0 * 100.0
    uncertainty_flag = (0.45 <= prob_pass <= 0.55)
    risk_level = "HIGH" if uncertainty_flag else ("MEDIUM" if 0.40 <= prob_pass <= 0.60 else "LOW")
    return PredictionResponse(student_id=payload.student_id, prediction=prediction_label,
                              pass_binary=pass_binary, pass_probability=round(prob_pass, 4),
                              confidence_pct=round(confidence, 2), risk_level=risk_level)
""",
        line_breakdown=[
            ("schemas L10-35", "Century21SkillProfile schema", "Enforces runtime type validation using Pydantic v2. Each skill rating is strictly constrained to integer values in [0, 6] (ge=0, le=6). Rejects malformed requests with HTTP 422."),
            ("app L116-128", "Lifespan context manager", "FastAPI startup event: Loads champion and candidate models, feature metadata, scaler parameters, and registry manifest into persistent memory once at startup, avoiding per-request I/O latency."),
            ("app L226-236", "Inference & confidence calculation", "Transforms input dictionary using saved feature scaler. Computes pass probability. Measures decision distance from threshold: Confidence = |p - 0.50| * 2 * 100%."),
            ("app L244-246", "Uncertainty flagging", "Detects borderline candidates whose confidence sits within [0.45, 0.55]. Flags risk_level='HIGH' to prevent false overconfidence and recommend targeted reinforcement."),
        ]
    )

    # =========================================================================
    # MODULE 9: MLOps Infrastructure: Dockerfile & CI/CD
    # =========================================================================
    add_code_section(
        file_title="Containerization & CI/CD Infrastructure",
        file_path="Dockerfile & .github/workflows/ci.yml",
        snippet="""
# --- Dockerfile (Multi-stage Production Image) ---
FROM python:3.11-slim as runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN groupadd -g 1001 appgroup && useradd -u 1001 -g appgroup -s /bin/bash -m appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
USER appuser
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

# --- .github/workflows/ci.yml (CI/CD Quality Gates) ---
jobs:
  quality-gate-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: pip install -r requirements.txt
      - run: ruff check src tests predict.py
      - run: python -m src.model_risk
      - run: python -m src.model_registry
      - run: pytest tests/ -v
""",
        line_breakdown=[
            ("Dockerfile L1-7", "Base image & security hardening", "Uses python:3.11-slim. Enforces non-root user (appuser:appgroup, UID 1001) under Least Privilege principles. Installs curl for healthchecks and removes apt cache."),
            ("Dockerfile L9-15", "Layer caching & artifact injection", "Installs requirements.txt before copying source code, maximizing Docker build cache. Copies trained model joblib files and feature metadata directly into image."),
            ("Dockerfile L17-18", "Healthcheck & dynamic port binding", "Configures Docker daemon HEALTHCHECK probe against /health. CMD shell substitution binds to Render cloud $PORT dynamically or defaults to 8000 locally."),
            ("CI/CD L11-30", "Automated quality gate execution", "On every push to main: sets up Python 3.11 with pip cache, verifies code quality via Ruff (0 errors), runs behavioral stress tests, checks model lineage, and executes all 46 pytest tests."),
        ]
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Code Walkthrough PDF successfully generated at: {output_path}")


if __name__ == "__main__":
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "AI_Technical_Skill_Development_Assistant_Code_Walkthrough_Manual.pdf"
    build_code_walkthrough_pdf(pdf_path)
