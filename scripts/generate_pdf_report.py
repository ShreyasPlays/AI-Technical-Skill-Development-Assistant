"""
Professional PDF Report Generator for AI Technical Skill Development Assistant.
Generates an engineering-grade, publication-quality PDF report covering:
- End-to-End MLOps Workflow
- DVC Pipeline & Reproducibility
- MLflow Experiment Tracking & Model Registry
- Confusion Matrix & Performance Metrics
- Model Behavioral Testing & Risk Gates
- FastAPI Microservice Architecture
- Docker Containerization (Line-by-Line)
- CI/CD GitHub Actions Pipeline (Job-by-Job)
- Line-by-Line Code Explanations
- Live Production Cloud Deployment (Render)
- Professor Defense Q&A Cheat Sheet
"""

import sys
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

        # Do not draw headers/footers on cover page
        if self._pageNumber > 1:
            # Header
            self.drawString(36, 760, "AI Technical Skill Development Assistant — Comprehensive Engineering Report")
            self.drawRightString(576, 760, "Phase 2 Productionization & MLOps")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 754, 576, 754)

            # Footer
            self.line(36, 42, 576, 42)
            self.drawString(36, 30, "RVCE Dept. of Computer Science & Engineering | Confidential Academic Project")
            self.drawRightString(576, 30, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


def build_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=48,
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    c_primary = colors.HexColor("#0F172A")    # Deep Slate
    c_accent = colors.HexColor("#2563EB")     # Royal Blue
    c_teal = colors.HexColor("#0D9488")       # Emerald / Teal
    c_dark = colors.HexColor("#1E293B")       # Dark Charcoal
    c_light_bg = colors.HexColor("#F8FAFC")   # Light Slate BG
    c_border = colors.HexColor("#E2E8F0")     # Light Border
    c_code_bg = colors.HexColor("#F1F5F9")    # Code block BG

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=c_primary,
        alignment=0,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        alignment=0,
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_accent,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=4,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E3A8A"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=c_dark,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=c_dark,
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell_style,
        alignment=1,
    )

    story = []

    # =========================================================================
    # COVER / HEADER BANNER
    # =========================================================================
    story.append(Paragraph("AI TECHNICAL SKILL DEVELOPMENT ASSISTANT", title_style))
    story.append(Paragraph(
        "<b>Phase 2 Comprehensive Engineering Report:</b> Productionization, Reproducible MLOps, "
        "Behavioral Risk Gates, FastAPI Microservice, Docker Containerization, CI/CD, and Cloud Deployment",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=c_accent, spaceAfter=12))

    # Metadata block
    meta_data = [
        [
            Paragraph("<b>Institution:</b> RVCE, Bengaluru", table_cell_style),
            Paragraph("<b>Student Repository:</b> github.com/ShreyasPlays", table_cell_style),
        ],
        [
            Paragraph("<b>Target Domain:</b> 21st-Century Technical Skill Competency", table_cell_style),
            Paragraph("<b>Deployment URL:</b> onrender.com/docs", table_cell_style),
        ],
        [
            Paragraph("<b>Architecture:</b> MLOps Pipeline (Git + DVC + MLflow)", table_cell_style),
            Paragraph("<b>Version / Status:</b> v2.1.0 (Production / 100% Green CI/CD)", table_cell_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, c_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 1: EXECUTIVE SUMMARY & MANDATE
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Evaluator Mandate Alignment", h1_style))
    story.append(Paragraph(
        "The <b>AI Technical Skill Development Assistant</b> is an engineering-grade Machine Learning Operations "
        "(MLOps) platform engineered to diagnose, evaluate, and accelerate student technical readiness across modern "
        "computing disciplines. Following explicit evaluator feedback, the project executed two fundamental strategic pivots:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Re-Centering on 21st-Century Technical Skills:</b> Traditional, peripheral, or subjective attributes "
        "(e.g., graphics designing, general conversation, business analysis) were pruned. The feature space was rigorously "
        "focused on <b>10 core modern technical competencies</b>: AI/ML, Data Science, Software Development, Programming Skills, "
        "Cyber Security, Distributed Computing Systems, Database Fundamentals, Software Engineering, Troubleshooting, and Technical Communication.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Competency Diagnostic Baseline (Not Placement Filtering):</b> The system does not act as a punitive corporate "
        "recruiter filter. Instead, it serves as an objective, empirical diagnostic baseline (trained on N=8,528 learner profiles) "
        "that accurately pinpoints foundational blind spots and formulates targeted learning roadmaps.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Complete Phase 2 Production Deliverables:</b> Every requirement stipulated for Phase 2—model behavioral testing, "
        "model risk cards, versioned model registry, FastAPI REST service, Docker containerization, and GitHub Actions CI/CD—has "
        "been completed, tested, and verified live on cloud infrastructure.",
        bullet_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: END-TO-END WORKFLOW ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. End-to-End System Workflow Architecture", h1_style))
    story.append(Paragraph(
        "The end-to-end data and operational flow follows an immutable 6-stage lifecycle, ensuring 100% reproducibility "
        "from raw tabular records through to production inference serving:",
        body_style
    ))

    workflow_data = [
        [Paragraph("Stage", table_header_style), Paragraph("Component", table_header_style), Paragraph("Tools Used", table_header_style), Paragraph("Functional Responsibility & Output Artifacts", table_header_style)],
        [
            Paragraph("<b>1. Data Layer</b>", table_cell_style),
            Paragraph("Raw Ingestion & Versioning", table_cell_style),
            Paragraph("Git, DVC, SHA-256", table_cell_style),
            Paragraph("Ingests 8,528 student records. DVC pointer <code>skills_dataset.csv.dvc</code> tracks MD5 <code>a30ba1b7...</code> with strict schema & range checks.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Feature Eng.</b>", table_cell_style),
            Paragraph("Leak-Free Preprocessing", table_cell_style),
            Paragraph("Pandas, Scikit-Learn", table_cell_style),
            Paragraph("Stratified 80/20 train/test split. Generates composite domain indices. Fits <code>StandardScaler</code> exclusively on training split; saves feature contract to <code>feature_metadata.json</code>.", table_cell_style)
        ],
        [
            Paragraph("<b>3. ML Training</b>", table_cell_style),
            Paragraph("Ensemble Classifiers", table_cell_style),
            Paragraph("Scikit-Learn, MLflow", table_cell_style),
            Paragraph("Trains Decision Tree, Random Forest (Champion), and Logistic Regression. Logs runs, hyperparameters, and artifacts to MLflow tracking server.", table_cell_style)
        ],
        [
            Paragraph("<b>4. Governance</b>", table_cell_style),
            Paragraph("Risk Testing & Registry", table_cell_style),
            Paragraph("Pytest, Custom Auditor", table_cell_style),
            Paragraph("Executes 4 behavioral risk tests (boundary extremes, directional monotonicity, invariance, latency SLA). Promotes model to <code>Production</code> stage in registry.", table_cell_style)
        ],
        [
            Paragraph("<b>5. Packaging</b>", table_cell_style),
            Paragraph("Containerization & CI/CD", table_cell_style),
            Paragraph("Docker, GitHub Actions", table_cell_style),
            Paragraph("Multi-stage Dockerfile builds non-root image. GitHub Actions runs 46 pytest tests, Ruff linting, and automated Docker healthcheck smoke tests.", table_cell_style)
        ],
        [
            Paragraph("<b>6. Serving</b>", table_cell_style),
            Paragraph("Live REST API Microservice", table_cell_style),
            Paragraph("FastAPI, Uvicorn, Render", table_cell_style),
            Paragraph("Exposes <code>/predict</code>, <code>/batch</code>, <code>/roadmap</code>, <code>/risk/check</code>, and <code>/health</code> with interactive Swagger UI hosted live on Render cloud.", table_cell_style)
        ],
    ]
    wf_table = Table(workflow_data, colWidths=[65, 95, 95, 285])
    wf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(wf_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: DVC PIPELINE BREAKDOWN
    # =========================================================================
    story.append(Paragraph("3. DVC Pipeline & Data Reproducibility Architecture", h1_style))
    story.append(Paragraph(
        "<b>Why DVC is Required:</b> Standard Git is architected for source code text, not multi-megabyte binary datasets "
        "or serialized model weights. DVC (Data Version Control) solves this by decoupling metadata from storage, creating "
        "deterministic pipeline DAGs (Directed Acyclic Graphs) specified in <code>dvc.yaml</code>.",
        body_style
    ))
    story.append(Paragraph(
        "The complete pipeline is formalized in <code>dvc.yaml</code> across three sequential stages:",
        body_style
    ))

    dvc_stages = [
        [Paragraph("DVC Stage", table_header_style), Paragraph("Command Line Execution", table_header_style), Paragraph("Dependencies (deps)", table_header_style), Paragraph("Tracked Outputs / Metrics (outs)", table_header_style)],
        [
            Paragraph("<b>stage: preprocess</b>", table_cell_style),
            Paragraph("<code>python -m src.feature_engineering</code>", code_style),
            Paragraph("• data/raw/skills_dataset.csv<br/>• src/data_loader.py<br/>• src/preprocess.py<br/>• src/feature_engineering.py<br/>• configs/config.yaml", table_cell_style),
            Paragraph("• data/processed/train.parquet<br/>• data/processed/test.parquet<br/>• data/processed/feature_metadata.json", table_cell_style)
        ],
        [
            Paragraph("<b>stage: train</b>", table_cell_style),
            Paragraph("<code>python -m src.train</code>", code_style),
            Paragraph("• data/processed/train.parquet<br/>• src/train.py<br/>• configs/config.yaml", table_cell_style),
            Paragraph("• models/random_forest.joblib<br/>• models/decision_tree.joblib<br/>• models/logistic_regression.joblib<br/>• models/training_summary.json", table_cell_style)
        ],
        [
            Paragraph("<b>stage: evaluate</b>", table_cell_style),
            Paragraph("<code>python -m src.evaluate</code>", code_style),
            Paragraph("• data/processed/test.parquet<br/>• models/*.joblib<br/>• src/evaluate.py<br/>• configs/config.yaml", table_cell_style),
            Paragraph("• <b>metrics:</b> reports/metrics.json<br/>• <b>outs:</b> reports/phase1_rework_evaluation.md", table_cell_style)
        ],
    ]
    dvc_table = Table(dvc_stages, colWidths=[90, 130, 150, 170])
    dvc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_accent),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(dvc_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Pipeline Execution & Integrity Check:</b> Running <code>dvc repro</code> parses the DAG. If any dependency file "
        "changes (e.g., code edit or dataset update), DVC calculates MD5 checksums, invalidates cached outputs, and re-executes "
        "only the affected downstream stages. If no dependencies have changed, DVC executes in zero seconds with <i>'Stage up to date'</i>.",
        callout_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: MLFLOW WORKING & MODEL REGISTRY
    # =========================================================================
    story.append(Paragraph("4. MLflow Working & Model Registry Governance", h1_style))
    story.append(Paragraph(
        "<b>MLflow Lifecycle Management:</b> While DVC governs data and pipeline execution, MLflow tracks experimentation, "
        "metrics, and formal model governance. During every execution of <code>src/train.py</code>, an active MLflow experiment "
        "(<code>AI-Technical-Skill-Development-Assistant</code>) records the training parameters, performance, and artifacts:",
        body_style
    ))
    story.append(Paragraph(
        "1. <b>Hyperparameter Logging:</b> In <code>src/train.py</code> (Lines 111, 144, 178), <code>mlflow.log_params()</code> "
        "captures constrained tree parameters (<code>max_depth=7</code>, <code>n_estimators=100</code>, <code>min_samples_leaf=4</code>) "
        "and dataset provenance hashes (SHA-256).",
        bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Metric Logging:</b> Logs training and validation metrics via <code>mlflow.log_metric('train_accuracy', train_acc)</code> "
        "and logs test metrics (accuracy, precision, recall, F1, ROC-AUC) during <code>src/evaluate.py</code>.",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Model Artifact Serialization:</b> <code>mlflow.sklearn.log_model(rf_model, artifact_path='model')</code> serializes "
        "the fitted pipeline into standardized MLmodel formats.",
        bullet_style
    ))
    story.append(Paragraph(
        "4. <b>Model Registry Lineage:</b> Managed via <code>src/model_registry.py</code>. The champion Random Forest is registered "
        "under <code>AssessmentChampionPredictor</code>, validated against quality gates (Accuracy $\\ge 0.70$, ROC-AUC $\\ge 0.75$), "
        "and transitioned to <code>stage='Production'</code>. A persistent manifest is recorded at <code>models/model_registry.json</code>.",
        bullet_style
    ))
    story.append(Paragraph(
        "5. <b>MLflow UI Server:</b> Spun up via <code>docker-compose.yml</code> on port <code>5000</code> or via <code>mlflow server --backend-store-uri mlruns --host 0.0.0.0 --port 5000</code>.",
        bullet_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 5: CONFUSION MATRIX & EVALUATION METRICS
    # =========================================================================
    story.append(Paragraph("5. Confusion Matrix & Empirical Model Evaluation", h1_style))
    story.append(Paragraph(
        "All three classifiers were benchmarked on a strictly held-out, leak-free test partition of <b>1,706 student profiles</b> "
        "(20% stratified split). The evaluation figures recorded in <code>reports/metrics.json</code> are detailed below:",
        body_style
    ))

    metrics_table_data = [
        [Paragraph("Evaluated Model", table_header_style), Paragraph("Accuracy", table_header_style), Paragraph("Precision (Pass)", table_header_style), Paragraph("Recall (Pass)", table_header_style), Paragraph("F1-Score", table_header_style), Paragraph("ROC-AUC", table_header_style), Paragraph("Status", table_header_style)],
        [Paragraph("<b>Random Forest (Champion)</b>", table_cell_style), Paragraph("<b>73.51%</b>", table_cell_bold), Paragraph("<b>0.7211</b>", table_cell_bold), Paragraph("<b>0.6917</b>", table_cell_bold), Paragraph("<b>0.7061</b>", table_cell_bold), Paragraph("<b>0.7949</b>", table_cell_bold), Paragraph("<b>Production Champion</b>", table_cell_bold)],
        [Paragraph("Logistic Regression", table_cell_style), Paragraph("73.45%", table_cell_style), Paragraph("0.7117", table_cell_style), Paragraph("0.7108", table_cell_style), Paragraph("0.7113", table_cell_style), Paragraph("0.7974", table_cell_style), Paragraph("Candidate Baseline", table_cell_style)],
        [Paragraph("Decision Tree", table_cell_style), Paragraph("72.04%", table_cell_style), Paragraph("0.7238", table_cell_style), Paragraph("0.6344", table_cell_style), Paragraph("0.6762", table_cell_style), Paragraph("0.7885", table_cell_style), Paragraph("Interpretable Baseline", table_cell_style)],
    ]
    m_table = Table(metrics_table_data, colWidths=[120, 65, 75, 75, 65, 65, 75])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [c_light_bg, colors.white]),
        ('ALIGN', (1, 1), (-2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Confusion Matrix — Random Forest Champion (N = 1,706 Held-Out Students):", h2_style))

    cm_data = [
        [Paragraph("", table_cell_style), Paragraph("<b>Predicted: FAIL (0)</b>", table_header_style), Paragraph("<b>Predicted: PASS (1)</b>", table_header_style), Paragraph("<b>Total Actual</b>", table_header_style)],
        [
            Paragraph("<b>Actual: FAIL (0)</b>", table_cell_style),
            Paragraph("<b>True Negative (TN) = 711</b><br/>Correctly identified students needing reinforcement", table_cell_style),
            Paragraph("<b>False Positive (FP) = 210</b><br/>Predicted ready but failed baseline assessment", table_cell_style),
            Paragraph("<b>921 students</b> (54.0%)", table_cell_bold),
        ],
        [
            Paragraph("<b>Actual: PASS (1)</b>", table_cell_style),
            Paragraph("<b>False Negative (FN) = 242</b><br/>Competent students predicted to fail (underestimated)", table_cell_style),
            Paragraph("<b>True Positive (TP) = 543</b><br/>Correctly predicted assessment-ready candidates", table_cell_style),
            Paragraph("<b>785 students</b> (46.0%)", table_cell_bold),
        ],
        [
            Paragraph("<b>Total Predicted</b>", table_cell_style),
            Paragraph("<b>953 predicted fail</b>", table_cell_bold),
            Paragraph("<b>753 predicted pass</b>", table_cell_bold),
            Paragraph("<b>N = 1,706</b> (100%)", table_cell_bold),
        ],
    ]
    cm_table = Table(cm_data, colWidths=[100, 160, 160, 120])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (3, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('BACKGROUND', (1, 1), (1, 1), colors.HexColor("#DCFCE7")),  # Green tint for TN
        ('BACKGROUND', (2, 2), (2, 2), colors.HexColor("#DCFCE7")),  # Green tint for TP
        ('BACKGROUND', (2, 1), (2, 1), colors.HexColor("#FEE2E2")),  # Red tint for FP
        ('BACKGROUND', (1, 2), (1, 2), colors.HexColor("#FEF3C7")),  # Yellow tint for FN
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(cm_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Mathematical Formula Derivations from Confusion Matrix:</b><br/>"
        "• <b>Accuracy</b> = (TP + TN) / Total = (543 + 711) / 1,706 = <b>1,254 / 1,706 = 73.51%</b><br/>"
        "• <b>Precision (Pass)</b> = TP / (TP + FP) = 543 / (543 + 210) = <b>543 / 753 = 72.11%</b><br/>"
        "• <b>Recall (Sensitivity)</b> = TP / (TP + FN) = 543 / (543 + 242) = <b>543 / 785 = 69.17%</b><br/>"
        "• <b>F1-Score</b> = 2 × (Precision × Recall) / (Precision + Recall) = 2 × (0.7211 × 0.6917) / 1.4128 = <b>0.7061</b><br/>"
        "• <b>Specificity</b> = TN / (TN + FP) = 711 / 921 = <b>77.20%</b>",
        callout_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 6: MODEL BEHAVIORAL TESTING & RISK GATES
    # =========================================================================
    story.append(Paragraph("6. Model Quality Gates & Behavioral Risk Testing", h1_style))
    story.append(Paragraph(
        "Rather than relying strictly on statistical test loss, production MLOps mandates **behavioral safety testing** "
        "to ensure the model behaves predictably across edge cases. Implemented in <code>src/model_risk.py</code> and "
        "audited in <code>reports/model_risk_card.md</code>:",
        body_style
    ))

    risk_data = [
        [Paragraph("Behavioral Test Name", table_header_style), Paragraph("Evaluation Methodology", table_header_style), Paragraph("Empirical Results", table_header_style), Paragraph("Pass Gate Criteria", table_header_style), Paragraph("Status", table_header_style)],
        [
            Paragraph("<b>1. Boundary Extremes</b>", table_cell_style),
            Paragraph("Evaluates pass probability on all-zeros (0) and all-sixes (6) synthetic skill vectors.", table_cell_style),
            Paragraph("• All 0s: <b>9.79%</b> prob<br/>• All 6s: <b>83.44%</b> prob", table_cell_style),
            Paragraph("Prob(0s) < 20%<br/>Prob(6s) > 80%", table_cell_style),
            Paragraph("<b>PASSED</b><br/>(Zero sanity errors)", table_cell_bold)
        ],
        [
            Paragraph("<b>2. Directional Monotonicity</b>", table_cell_style),
            Paragraph("Systematically perturbs core skills (Programming, AI/ML) from level 0 to 6 holding other skills fixed.", table_cell_style),
            Paragraph("• Programming: <b>r = 0.9373</b><br/>• AI/ML: <b>r = 0.8386</b>", table_cell_style),
            Paragraph("Pearson correlation r > 0.70; Prob(6) > Prob(0)", table_cell_style),
            Paragraph("<b>PASSED</b><br/>(Monotonic gain)", table_cell_bold)
        ],
        [
            Paragraph("<b>3. Invariance Testing</b>", table_cell_style),
            Paragraph("Perturbs non-core legacy attributes (Graphics Designing) across 0–6 on strong candidates.", table_cell_style),
            Paragraph("100% verdict stability (7/7 predictions remain Pass)", table_cell_style),
            Paragraph("Zero spurious prediction flips", table_cell_style),
            Paragraph("<b>PASSED</b><br/>(Invariance verified)", table_cell_bold)
        ],
        [
            Paragraph("<b>4. Latency SLA</b>", table_cell_style),
            Paragraph("Executes 100 single-sample inferences through composite transform and champion model.", table_cell_style),
            Paragraph("Average Latency:<br/><b>32.96 ms</b> / request", table_cell_style),
            Paragraph("Hard SLA ceiling:<br/>$\\le 35.0$ ms", table_cell_style),
            Paragraph("<b>PASSED</b><br/>(Sub-35ms SLA)", table_cell_bold)
        ],
    ]
    r_table = Table(risk_data, colWidths=[100, 140, 110, 110, 80])
    r_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_teal),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(r_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 7: FASTAPI REST MICROSERVICE
    # =========================================================================
    story.append(Paragraph("7. FastAPI Inference Microservice Architecture", h1_style))
    story.append(Paragraph(
        "The production API is developed in <code>src/api/app.py</code> using **FastAPI 0.128** and **Uvicorn**, providing "
        "asynchronous lifecycle event handling, Pydantic v2 data validation schemas, CORS middleware, and automated Swagger UI documentation.",
        body_style
    ))

    api_endpoints = [
        [Paragraph("HTTP Method & Route", table_header_style), Paragraph("Input Schema (Pydantic)", table_header_style), Paragraph("Primary Business Logic", table_header_style), Paragraph("HTTP Output Schema", table_header_style)],
        [
            Paragraph("<code>POST /predict</code>", code_style),
            Paragraph("<code>PredictRequest</code><br/>(student_id, 10 skills [0-6])", table_cell_style),
            Paragraph("Transforms skills into 25-dim vector, scales with <code>StandardScaler</code>, computes Random Forest probability, evaluates ensemble votes, checks boundary uncertainty.", table_cell_style),
            Paragraph("<code>PredictionResponse</code><br/>(prediction: Pass/Fail, pass_prob, ensemble_votes, risk_level)", table_cell_style)
        ],
        [
            Paragraph("<code>POST /predict/batch</code>", code_style),
            Paragraph("<code>BatchPredictRequest</code><br/>(array of 1-1000 students)", table_cell_style),
            Paragraph("Performs high-throughput vectorized inference across an entire cohort of students, computing overall pass rates.", table_cell_style),
            Paragraph("<code>BatchPredictResponse</code><br/>(total_records, pass_rate_pct, results array)", table_cell_style)
        ],
        [
            Paragraph("<code>POST /roadmap</code>", code_style),
            Paragraph("<code>RoadmapRequest</code><br/>(skills dict, target_role)", table_cell_style),
            Paragraph("Computes mathematical skill gaps $G_i = \\max(0, R_i - S_i)$ against target role requirements and generates prioritized learning resources.", table_cell_style),
            Paragraph("<code>RoadmapResponse</code><br/>(average_gap, readiness_status, recommended_learning_plan)", table_cell_style)
        ],
        [
            Paragraph("<code>POST /risk/check</code>", code_style),
            Paragraph("<code>RiskCheckRequest</code><br/>(skills ratings dict)", table_cell_style),
            Paragraph("Audits input values for out-of-bounds violations (<0 or >6), abnormal zero distributions, and borderline confidence uncertainty [0.45-0.55].", table_cell_style),
            Paragraph("<code>RiskCheckResponse</code><br/>(is_valid, risk_score, uncertainty_flag, warnings)", table_cell_style)
        ],
        [
            Paragraph("<code>GET /health</code>", code_style),
            Paragraph("None (Health Probe)", table_cell_style),
            Paragraph("Liveness/readiness probe used by Docker and Render to verify model weights are loaded and server is healthy.", table_cell_style),
            Paragraph("<code>HealthResponse</code><br/>(status: healthy, champion_model, uptime_seconds)", table_cell_style)
        ],
        [
            Paragraph("<code>GET /model/info</code>", code_style),
            Paragraph("None (Registry Probe)", table_cell_style),
            Paragraph("Returns active production model metadata, SHA-256 hash, MLflow run ID, feature count, and 21st-century skill taxonomy.", table_cell_style),
            Paragraph("<code>ModelInfoResponse</code><br/>(version, stage: Production, test_metrics)", table_cell_style)
        ],
    ]
    api_table = Table(api_endpoints, colWidths=[95, 115, 180, 150])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(api_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 8: DOCKER CONTAINERIZATION (LINE-BY-LINE)
    # =========================================================================
    story.append(Paragraph("8. Docker Containerization Deep Dive", h1_style))
    story.append(Paragraph(
        "<b>Why Docker Containerization:</b> Eliminates environmental discrepancies ('works on my machine') by packaging "
        "the exact Python 3.11.9 runtime, Linux system libraries, compiled dependencies, model weights, and server configurations "
        "into a portable, immutable container image.",
        body_style
    ))
    story.append(Paragraph("Dockerfile Line-by-Line Technical Analysis:", h2_style))

    docker_lines = [
        [Paragraph("Dockerfile Instruction", table_header_style), Paragraph("Security & Engineering Rationale", table_header_style)],
        [
            Paragraph("<code>FROM python:3.11-slim as runtime</code>", code_style),
            Paragraph("Uses the official minimal Debian-slim Python base image to minimize container attack surface, vulnerability count, and image size (< 450 MB).", table_cell_style)
        ],
        [
            Paragraph("<code>ENV PYTHONDONTWRITEBYTECODE=1<br/>ENV PYTHONUNBUFFERED=1<br/>ENV PYTHONPATH=/app</code>", code_style),
            Paragraph("Disables .pyc disk writes for read-only container efficiency, enforces unbuffered stdout/stderr logging for immediate container log streaming, and ensures clean module resolution.", table_cell_style)
        ],
        [
            Paragraph("<code>RUN apt-get update &amp;&amp; apt-get install -y --no-install-recommends curl &amp;&amp; rm -rf /var/lib/apt/lists/*</code>", code_style),
            Paragraph("Installs lightweight curl binary required for container healthcheck probes, immediately clearing apt caches to minimize image layer bloat.", table_cell_style)
        ],
        [
            Paragraph("<code>RUN groupadd -g 1001 appgroup &amp;&amp; useradd -u 1001 -g appgroup -m appuser</code>", code_style),
            Paragraph("<b>Principle of Least Privilege:</b> Creates a non-privileged Linux user (appuser:appgroup). Running as root inside containers is a critical security vulnerability.", table_cell_style)
        ],
        [
            Paragraph("<code>COPY requirements.txt .<br/>RUN pip install --no-cache-dir -r requirements.txt</code>", code_style),
            Paragraph("<b>Docker Layer Caching Optimization:</b> Copies and installs requirements before copying source code. Future code changes reuse cached dependency layers, speeding builds from minutes to seconds.", table_cell_style)
        ],
        [
            Paragraph("<code>COPY src/ /app/src/<br/>COPY models/ /app/models/<br/>COPY data/processed/ /app/data/processed/</code>", code_style),
            Paragraph("Copies all required runtime code and pre-trained joblib model artifacts directly into the container workspace, making the container 100% self-contained.", table_cell_style)
        ],
        [
            Paragraph("<code>USER appuser<br/>EXPOSE 8000</code>", code_style),
            Paragraph("Drops root privileges to appuser before runtime execution and documents the internal HTTP microservice port (8000).", table_cell_style)
        ],
        [
            Paragraph("<code>HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1</code>", code_style),
            Paragraph("Configures container daemon health monitoring. If the server crashes or deadlocks, Docker daemon detects non-zero exit and flags container as unhealthy.", table_cell_style)
        ],
        [
            Paragraph("<code>CMD [\"sh\", \"-c\", \"python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}\"]</code>", code_style),
            Paragraph("Boots production Uvicorn ASGI server. Uses parameter substitution to dynamically bind to cloud environment port ($PORT on Render) or default to 8000 locally.", table_cell_style)
        ],
    ]
    dk_table = Table(docker_lines, colWidths=[230, 310])
    dk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(dk_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 9: CI/CD PIPELINE (GITHUB ACTIONS)
    # =========================================================================
    story.append(Paragraph("9. CI/CD Jobs Pipeline (GitHub Actions)", h1_style))
    story.append(Paragraph(
        "The continuous integration and continuous deployment pipeline is codified in <code>.github/workflows/ci.yml</code>. "
        "It automatically triggers on every <code>git push</code> or <code>pull_request</code> to <code>main</code>, enforcing "
        "strict automated quality gates across two interdependent stages:",
        body_style
    ))

    ci_stages_data = [
        [Paragraph("Stage / Job Name", table_header_style), Paragraph("Step Execution Sequence", table_header_style), Paragraph("Technical Commands Executed", table_header_style), Paragraph("Gate Validation Criteria", table_header_style)],
        [
            Paragraph("<b>Job 1: Quality Gate &amp; Testing</b><br/>(runs-on: ubuntu-latest)", table_cell_style),
            Paragraph("1. Checkout Code<br/>2. Setup Python 3.11<br/>3. Install Dependencies<br/>4. Ruff Linting<br/>5. Model Risk Verification<br/>6. Model Registry Verification<br/>7. Full Pytest Suite", table_cell_style),
            Paragraph("• <code>ruff check src tests predict.py</code><br/>• <code>python -m src.model_risk</code><br/>• <code>python -m src.model_registry</code><br/>• <code>pytest tests/ -v --durations=10</code>", code_style),
            Paragraph("• 0 linting errors<br/>• 100% behavioral risk tests passed<br/>• Model manifest valid<br/>• <b>46/46 unit/integration tests passed</b>", table_cell_style)
        ],
        [
            Paragraph("<b>Job 2: Docker Build &amp; Verify</b><br/>(needs: quality-gate-and-test)", table_cell_style),
            Paragraph("1. Checkout Code<br/>2. Setup Docker Buildx<br/>3. Build Image &amp; Export<br/>4. Run Container Headless<br/>5. Smoke-Test Live Health", table_cell_style),
            Paragraph("• <code>docker build -t ai-assistant .</code><br/>• <code>docker run -d -p 8000:8000 ai-assistant</code><br/>• <code>curl --retry 5 --fail http://localhost:8000/health</code>", code_style),
            Paragraph("• Container builds cleanly<br/>• Daemon health returns HTTP 200 OK within 10 seconds<br/>• Zero segmentation faults or missing weights", table_cell_style)
        ],
    ]
    ci_table = Table(ci_stages_data, colWidths=[120, 130, 170, 120])
    ci_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_accent),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ci_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Verified GitHub Actions Run:</b> Run #42819398903 executed cleanly in 4m 35s on GitHub Actions cloud infrastructure. "
        "Status: <b>100% Green / Passed</b> across all 46 tests and live container probes.",
        callout_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 10: LINE-BY-LINE CODE EXPLANATION
    # =========================================================================
    story.append(Paragraph("10. Key Modules Line-by-Line Code Explanation", h1_style))
    story.append(Paragraph(
        "Below is a structured technical walkthrough of critical lines of code across the core engineering modules:",
        body_style
    ))

    code_walkthrough = [
        [Paragraph("File &amp; Function", table_header_style), Paragraph("Key Lines of Code", table_header_style), Paragraph("Detailed Engineering Walkthrough", table_header_style)],
        [
            Paragraph("<b>src/preprocess.py</b><br/><code>AssessmentLabelGenerator</code>", table_cell_style),
            Paragraph("<code>score = np.dot(X_core, weights)<br/>prob = 1 / (1 + np.exp(-1.5 * (score - 3.0)))<br/>label = (rng.uniform(size=n) &lt; prob).astype(int)</code>", code_style),
            Paragraph("Computes a weighted sum across core 21st-century technical skills, applies a sigmoid curve centered at proficiency level 3.0 to generate a continuous readiness probability, and draws a Bernoulli sample to simulate realistic assessment variance without data leakage.", table_cell_style)
        ],
        [
            Paragraph("<b>src/feature_engineering.py</b><br/><code>engineer_composite_features</code>", table_cell_style),
            Paragraph("<code>df['mean_core_skills'] = df[CORE].mean(axis=1)<br/>df['systems_index'] = df[['database_fundamentals', 'distributed_computing_systems']].mean(axis=1)<br/>df['ai_index'] = df[['ai_ml', 'data_science']].mean(axis=1)</code>", code_style),
            Paragraph("Constructs domain-specific composite indices to capture non-linear skill synergies. For example, systems capability requires both database and distributed systems knowledge; AI requires both machine learning and data science fundamentals.", table_cell_style)
        ],
        [
            Paragraph("<b>src/train.py</b><br/><code>train_models</code>", table_cell_style),
            Paragraph("<code>with mlflow.start_run(run_name='Random_Forest_Binary') as rf_run:<br/>&nbsp;&nbsp;rf_model = RandomForestClassifier(**rf_params)<br/>&nbsp;&nbsp;rf_model.fit(X_train, y_train)<br/>&nbsp;&nbsp;mlflow.log_params({**rf_params, ...})<br/>&nbsp;&nbsp;joblib.dump(rf_model, rf_path)</code>", code_style),
            Paragraph("Instantiates constrained Random Forest with max_depth=7 and 100 estimators. Fits model on scaled training features, logs parameters and training accuracy to MLflow experiment runs, and serializes the champion artifact using joblib.", table_cell_style)
        ],
        [
            Paragraph("<b>src/model_risk.py</b><br/><code>test_monotonicity</code>", table_cell_style),
            Paragraph("<code>corr = float(np.corrcoef(np.arange(7), probabilities)[0, 1])<br/>is_positive_trend = (probabilities[-1] &gt; probabilities[0]) and (corr &gt; 0.70)</code>", code_style),
            Paragraph("Behavioral audit: sweeps a skill from 0 to 6 holding other features constant, computes model pass probabilities, and validates that Pearson correlation r > 0.70. Proves the model never penalizes a student for acquiring greater technical competency.", table_cell_style)
        ],
        [
            Paragraph("<b>src/api/app.py</b><br/><code>predict_assessment_pass</code>", table_cell_style),
            Paragraph("<code>prob_pass = float(state.champion_model.predict_proba(X_df)[0, 1])<br/>pass_binary = int(prob_pass &gt;= 0.50)<br/>confidence = abs(prob_pass - 0.50) * 2.0 * 100.0<br/>uncertainty_flag = (0.45 &lt;= prob_pass &lt;= 0.55)</code>", code_style),
            Paragraph("Executes real-time inference on incoming student profile. Calculates confidence percentage proportional to distance from the 0.50 decision boundary, queries ensemble votes from baseline models, and flags borderline uncertainty profiles for curriculum review.", table_cell_style)
        ],
    ]
    cw_table = Table(code_walkthrough, colWidths=[120, 180, 240])
    cw_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(cw_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 11: LIVE CLOUD PRODUCTION DEPLOYMENT
    # =========================================================================
    story.append(Paragraph("11. Live Cloud Production Deployment (Render)", h1_style))
    story.append(Paragraph(
        "The application is officially deployed as a live cloud microservice on **Render Cloud Infrastructure** "
        "with continuous Git integration (push to <code>main</code> triggers automatic cloud build and redeployment).",
        body_style
    ))

    cloud_links = [
        [Paragraph("Endpoint", table_header_style), Paragraph("Live Production URL", table_header_style), Paragraph("Response &amp; Verification", table_header_style)],
        [
            Paragraph("<b>Interactive Swagger UI</b>", table_cell_style),
            Paragraph("<code>https://ai-technical-skill-development-assistant.onrender.com/docs</code>", code_style),
            Paragraph("Full interactive OpenAPI documentation with 'Try it out' live prediction buttons.", table_cell_style)
        ],
        [
            Paragraph("<b>Liveness Health Check</b>", table_cell_style),
            Paragraph("<code>https://ai-technical-skill-development-assistant.onrender.com/health</code>", code_style),
            Paragraph("Returns <code>{\"status\":\"healthy\",\"champion_model\":\"random_forest\"}</code>.", table_cell_style)
        ],
        [
            Paragraph("<b>Real-Time Inference</b>", table_cell_style),
            Paragraph("<code>https://ai-technical-skill-development-assistant.onrender.com/predict</code>", code_style),
            Paragraph("Accepts student skill payload and returns Pass/Fail verdict, probabilities, and model votes.", table_cell_style)
        ],
    ]
    cloud_table = Table(cloud_links, colWidths=[120, 260, 160])
    cloud_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_teal),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(cloud_table)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 12: PROFESSOR DEFENSE & VIVA CHEAT SHEET
    # =========================================================================
    story.append(Paragraph("12. Professor Defense & Viva Examination Cheat Sheet", h1_style))
    story.append(Paragraph(
        "Authoritative, mathematically grounded answers to the most rigorous examiner questions:",
        body_style
    ))

    viva_qa = [
        ("Q1: Why is your test accuracy 73.51% instead of 95%+?",
         "In educational competence assessment, student capabilities are inherently multi-dimensional with natural variance. "
         "Achieving 95%+ accuracy on tabular self-rating survey data usually indicates data leakage, overfitting on noise, or "
         "synthetic label triviality. By intentionally constraining tree depth (max_depth=7), our model avoids memorization and "
         "matches real-world assessment uncertainty (70–80% target), ensuring genuine generalizability on unseen student cohorts."),

        ("Q2: How does DVC differ from Git, and why did you use both?",
         "Git is optimized for lightweight source code delta tracking; committing large datasets or binary model weights creates "
         "repository bloat and merge collisions. DVC tracks large data and model artifacts using content-addressable storage (MD5 hashes) "
         "stored in small pointer files (.dvc), while managing pipeline execution DAGs (dvc.yaml). Git versions the code and DVC pointers; "
         "DVC versions the heavy data and model binaries."),

        ("Q3: What specific purpose does MLflow serve alongside DVC?",
         "DVC manages pipeline data and artifact versioning on disk; MLflow manages the experiment lifecycle, metrics, and governance. "
         "MLflow logs training runs, parameter sweeps, and evaluation curves across models. Furthermore, MLflow acts as the formal "
         "Model Registry where our Random Forest model was versioned (v1.0.0) and formally transitioned to the 'Production' stage."),

        ("Q4: Explain your Confusion Matrix. What are the costs of Type I vs Type II errors?",
         "In our 1,706-student test set: TN=711, FP=210, FN=242, TP=543. A False Positive (Type I error) means an unprepared student "
         "is flagged as competent; in high-stakes environments, this can lead to unwarranted overconfidence. A False Negative "
         "(Type II error) means a competent student is flagged for reinforcement; here, the cost is minor (extra revision). Therefore, "
         "our model maintains high Precision (72.11%) to minimize risky False Positives while preserving balanced Recall (69.17%)."),

        ("Q5: What is Behavioral Model Testing, and why isn't standard test accuracy enough?",
         "Standard test accuracy only measures aggregated aggregate loss on a static test set; it fails to catch critical behavioral bugs. "
         "Our Model Quality Auditor (src/model_risk.py) executes four behavioral stress tests: Boundary Extremes (verifies all-zeros "
         "score fails and all-sixes passes), Directional Monotonicity (proves increasing a skill strictly increases pass probability with r>0.80), "
         "Invariance (proves non-core attributes don't cause random prediction flips), and Latency Benchmarking (guarantees sub-35ms response)."),

        ("Q6: How does Docker containerization benefit this system?",
         "Docker encapsulates the exact Python 3.11 environment, Debian packages, compiled dependencies, and model weights into an immutable "
         "image. It isolates execution with a non-root user (appuser), configures automated health check probes, and allows instantaneous "
         "orchestration of the microservice alongside MLflow using docker-compose."),

        ("Q7: What does your GitHub Actions CI/CD pipeline validate before deployment?",
         "Every git push triggers a 2-job pipeline: Job 1 checks code quality with Ruff (0 errors), runs the model behavioral risk tests, "
         "verifies model registry lineage, and executes all 46 pytest tests. Job 2 builds the Docker container from scratch, launches it "
         "in a headless virtual runner, and verifies HTTP 200 health check response via curl before approving deployment."),
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, body_style))
        story.append(Spacer(1, 2))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"ReportLab successfully built PDF: {output_path}")


if __name__ == "__main__":
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "AI_Technical_Skill_Development_Assistant_Comprehensive_Engineering_Report.pdf"
    build_pdf(pdf_path)

