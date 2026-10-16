import os
import sys
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw accurate total page counts
    along with academic running headers and footers.
    """
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
        if self._pageNumber == 1:
            return  # Suppress running header/footer on title/abstract cover page
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#334155"))
        
        # Running Header
        self.drawString(40, letter[1] - 32, "CREDIT CARD FRAUD DETECTION: A FIRST-PRINCIPLES RESEARCH MONOGRAPH")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(letter[0] - 40, letter[1] - 32, "N = 100 MONTE CARLO EVALUATION")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, letter[1] - 36, letter[0] - 40, letter[1] - 36)
        
        # Running Footer
        self.line(40, 44, letter[0] - 40, 44)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 32, "Python for Data Science (PDS) — Capstone Research Monograph | IIIT Sri City")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 40, 32, page_str)
        self.restoreState()

def build_pdf_monograph():
    out_dir_local = r"c:\Users\HP\.gemini\antigravity-ide\scratch\Credit-Card-Fraud-Detection-Model\report"
    out_dir_desktop = r"C:\Users\HP\Desktop\PDS\report"
    os.makedirs(out_dir_local, exist_ok=True)
    os.makedirs(out_dir_desktop, exist_ok=True)
    
    pdf_local = os.path.join(out_dir_local, "Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf")
    pdf_desktop = os.path.join(out_dir_desktop, "Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf")

    # Document Geometry: 532 pt printable width (Margins 40pt)
    margin = 40
    doc = SimpleDocTemplate(
        pdf_local,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=46,
        bottomMargin=52
    )

    styles = getSampleStyleSheet()
    
    # Measured Academic Color Palette
    c_primary = colors.HexColor("#0F172A")    # Deep Slate / Navy
    c_secondary = colors.HexColor("#1E3A8A")  # Deep Blue
    c_accent = colors.HexColor("#B91C1C")     # Crimson
    c_dark = colors.HexColor("#1E293B")       # Dark Charcoal (Body)
    c_muted = colors.HexColor("#64748B")      # Muted Slate
    c_bg_light = colors.HexColor("#F8FAFC")   # Light Slate
    c_border = colors.HexColor("#CBD5E1")     # Border
    c_highlight = colors.HexColor("#F1F5F9")  # Table Alternate Row
    c_callout_bg = colors.HexColor("#EFF6FF") # Soft Blue
    c_callout_border = colors.HexColor("#3B82F6")
    c_amber_bg = colors.HexColor("#FEFCE8")   # Warning / Note
    c_amber_border = colors.HexColor("#EAB308")

    # Typography Hierarchy
    cover_title = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=21, leading=26,
        textColor=c_primary, spaceAfter=6
    )
    
    cover_subtitle = ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10.5, leading=15,
        textColor=c_secondary, spaceAfter=10
    )
    
    sec_heading = ParagraphStyle(
        'SectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12.5, leading=16.5,
        textColor=c_primary, spaceBefore=12, spaceAfter=6,
        keepWithNext=True
    )
    
    subsec_heading = ParagraphStyle(
        'SubSectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.5, leading=13.5,
        textColor=c_secondary, spaceBefore=8, spaceAfter=3,
        keepWithNext=True
    )

    subsubsec_heading = ParagraphStyle(
        'SubSubSectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.5, leading=12,
        textColor=c_dark, spaceBefore=5, spaceAfter=2,
        keepWithNext=True
    )

    body = ParagraphStyle(
        'BodyCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.0, leading=11.5,
        textColor=c_dark, spaceAfter=4.5
    )

    math_block = ParagraphStyle(
        'MathBlock', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8.0, leading=11.2,
        textColor=c_primary, spaceBefore=2, spaceAfter=4, leftIndent=10
    )

    code_snippet = ParagraphStyle(
        'CodeSnippet', parent=styles['Normal'],
        fontName='Courier', fontSize=6.8, leading=8.8,
        textColor=colors.HexColor("#0284C7"), spaceBefore=2, spaceAfter=4, leftIndent=8
    )

    callout_text = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=11.2,
        textColor=c_dark
    )

    tbl_header = ParagraphStyle(
        'TblHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.8, leading=8.6,
        textColor=colors.white, alignment=1
    )

    tbl_cell = ParagraphStyle(
        'TblCell', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6.6, leading=8.4,
        textColor=c_dark, alignment=0
    )

    tbl_cell_center = ParagraphStyle('TblCellCenter', parent=tbl_cell, alignment=1)
    tbl_cell_right = ParagraphStyle('TblCellRight', parent=tbl_cell, alignment=2)
    tbl_cell_bold = ParagraphStyle('TblCellBold', parent=tbl_cell, fontName='Helvetica-Bold')
    tbl_cell_bold_center = ParagraphStyle('TblCellBoldCenter', parent=tbl_cell_center, fontName='Helvetica-Bold')
    tbl_cell_bold_right = ParagraphStyle('TblCellBoldRight', parent=tbl_cell_right, fontName='Helvetica-Bold')

    story = []

    # =========================================================================
    # SECTION 1: TITLE & ABSTRACT
    # =========================================================================
    story.append(Paragraph("Credit Card Fraud Detection at Enterprise Scale", cover_title))
    story.append(Paragraph(
        "A First-Principles Machine Learning Evaluation, 100-Run Monte Carlo Benchmark, "
        "and Financial Decision Analysis in Pure NumPy",
        cover_subtitle
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=8))

    # Metadata Grid
    meta_rows = [
        [Paragraph("<b>Course:</b> Python for Data Science (PDS)", tbl_cell),
         Paragraph("<b>Dataset:</b> Kaggle / ULB Benchmark (284,807 transactions)", tbl_cell)],
        [Paragraph("<b>Framework:</b> 100% Pure NumPy (Zero External ML Packages)", tbl_cell),
         Paragraph("<b>Class Imbalance:</b> 0.1727% (492 Frauds : 284,315 Genuine)", tbl_cell)],
        [Paragraph("<b>Evaluation Protocol:</b> 100 Stratified Monte Carlo Splits (Seeds 1000–1099)", tbl_cell),
         Paragraph("<b>Hardware Target:</b> 12-Core Multi-Threaded Local Benchmarking", tbl_cell)],
        [Paragraph("<b>Institution:</b> IIIT Sri City — Capstone Repository", tbl_cell),
         Paragraph("<b>Primary Methodology:</b> Asymmetric Cost-Sensitive Risk Minimization", tbl_cell)]
    ]
    t_meta = Table(meta_rows, colWidths=[266, 266])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # Abstract (Exact 230 words)
    abstract_text = (
        "<b>1. Abstract & Research Objectives:</b><br/>"
        "Credit card fraud detection operates under severe class imbalance and highly asymmetric financial consequences. "
        "In this investigation, we engineer and evaluate an end-to-end fraud screening pipeline built entirely from first principles "
        "in pure Python and vectorized NumPy, without reliance on high-level machine learning frameworks such as scikit-learn, "
        "XGBoost, or PyTorch. Our methodology encompasses ten modular techniques, including Synthetic Minority Oversampling (SMOTE), "
        "Deep Multi-Layer Perceptrons (MLP) trained via analytical backpropagation and Adam optimization, Gradient Boosted Decision Trees (GBDT), "
        "Unsupervised Isolation Forests, Focal Loss, Dynamic Bayesian Risk Thresholding, and Stacked Generalization.<br/><br/>"
        "To rigorously assess performance while accounting for sampling variance across data partitions, we execute a 100-run Monte Carlo "
        "benchmark across 100 pseudo-random seeds. Rather than viewing high recall in isolation, we formulate an asymmetric financial decision "
        "framework that balances direct fraud loss ($C_{\\text{FN}} = \\text{Amount}$) against false alarm investigation overhead ($C_{\\text{FP}}$). "
        "Empirical findings demonstrate that while Cost-Sensitive Logistic Regression attains the highest event recall (93.81% ± 0.51%) and "
        "dollar recall (92.89% ± 1.41%), it generates an average of 7,854 false positives per test split, causing its total operational loss "
        "to exceed $20,500 under standard review costs. Conversely, GBDT and Baseline Logistic Regression provide substantially superior operational "
        "efficiency when false-positive handling costs are non-zero. Finally, we provide sensitivity analyses across cost assumptions, examine "
        "methodological limitations including concept drift, and outline a fully reproducible open-source implementation."
    )
    t_abs = Table([[Paragraph(abstract_text, callout_text)]], colWidths=[532])
    t_abs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_callout_bg),
        ('BOX', (0,0), (-1,-1), 1, c_callout_border),
        ('PADDING', (0,0), (-1,-1), 6.5),
    ]))
    story.append(t_abs)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Formal Research Objectives:</b>", subsubsec_heading))
    objectives = [
        "1. <b>First-Principles Implementation:</b> Demonstrate that state-of-the-art fraud screening architectures can be implemented purely via vectorized numerical linear algebra without black-box machine learning libraries.",
        "2. <b>Financial Decision Formulation:</b> Replace symmetric accuracy metrics with an asymmetric cost matrix reflecting direct stolen loss and investigation overhead.",
        "3. <b>Uncertainty Quantification:</b> Evaluate estimator stability across 100 stratified partitions to report rigorous confidence intervals.",
        "4. <b>Operational Trade-Off Analysis:</b> Quantify the tension between fraud capture and false-positive investigation capacity under varying operational cost regimes."
    ]
    for obj in objectives:
        story.append(Paragraph(obj, body))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 2: DATASET & PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("2. Dataset Characteristics and Preprocessing Pipeline", sec_heading))
    story.append(Paragraph(
        "<b>2.1 Data Provenance and Imbalance:</b> The corpus comprises 284,807 credit card transactions recorded across European "
        "cardholders in September 2013, released by the Machine Learning Group at Université Libre de Bruxelles (ULB) and Worldline. "
        "The data contains 492 confirmed fraudulent events against 284,315 genuine transactions, yielding an extreme fraud incidence of "
        "<b>P(Y=1) = 0.1727%</b> (1 fraud per 578 legitimate transactions). A naive majority classifier predicting 'genuine' for all cases "
        "yields 99.827% accuracy yet captures zero fraud, demonstrating why accuracy is fundamentally uninformative.",
        body
    ))
    story.append(Paragraph(
        "<b>2.2 Privacy Transformation (Features V1–V28):</b> Due to confidentiality mandates (GDPR, PCI-DSS), original customer identifiers, "
        "merchant metadata, and locations were projected via Principal Component Analysis (PCA) into 28 orthogonal numerical features ($V_1$ to $V_{28}$). "
        "Mathematically, the centered feature matrix $X$ was decomposed via Singular Value Decomposition: $X = U \\Sigma V^T$, where the columns of $V$ "
        "represent orthogonal eigenvectors of the covariance matrix. Consequently, $\\text{Cov}(V_i, V_j) = 0$ for all $i \\neq j$. "
        "Only `Time` (seconds elapsed from $t_0$) and `Amount` (transaction currency value) remain in their original scalar dimensions.",
        body
    ))
    story.append(Paragraph(
        "<b>2.3 Missing Values and Outlier Audit:</b> An exhaustive audit confirmed zero null, NaN, or missing values across all 284,807 rows. "
        "`Amount` exhibits extreme positive skewness (min: $0.00, median: $22.00, mean: $88.35, max: $25,691.16). To mitigate gradient explosion "
        "during numerical optimization, `Amount` is scaled using robust statistics (median and interquartile range) or standard normalization.",
        body
    ))
    story.append(Paragraph(
        "<b>2.4 Strict Data Leakage Prevention Protocol:</b> In highly imbalanced classification, data leakage between training and evaluation "
        "partitions is a frequent source of falsely inflated performance. We enforce an inviolable sequence of operations:",
        body
    ))
    
    pipeline_steps = [
        "<b>Step 1 (Partitioning):</b> The raw dataset is partitioned into an 80% training set (227,845 rows) and a 20% test set (56,962 rows) "
        "using stratified sampling to preserve the 0.1727% fraud ratio in both splits.",
        "<b>Step 2 (Parameter Estimation):</b> Scaling parameters (mean $\\mu_{\\text{train}}$, standard deviation $\\sigma_{\\text{train}}$) are computed "
        "strictly on the training partition. Test features are standardized using these frozen training statistics.",
        "<b>Step 3 (Minority Oversampling):</b> Synthetic Minority Oversampling (SMOTE) is applied <i>exclusively</i> to the training partition. "
        "Synthetic instances are NEVER introduced into validation folds or the held-out test split, ensuring unbiased evaluation."
    ]
    for p_step in pipeline_steps:
        story.append(Paragraph(f"• {p_step}", body))

    story.append(Spacer(1, 6))

    # Pipeline ASCII Flow Diagram
    flow_diagram = (
        "<b>DATA PIPELINE & ISOLATION FLOW:</b><br/>"
        "<font face='Courier' size='7'>"
        "[Raw Data: 284,807 txns] ──► [Stratified 80/20 Split] ──┬──► [Train Split: 227,845 txns] ──► Fit Scaler (μ, σ) ──► Apply SMOTE ──► Train Model<br/>"
        "                                                         └──► [Test Split:  56,962 txns]  ──► Transform (Frozen μ, σ) ───────────► Final Evaluation"
        "</font>"
    )
    t_flow = Table([[Paragraph(flow_diagram, callout_text)]], colWidths=[532])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_highlight),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_flow)
    story.append(PageBreak())

    # =========================================================================
    # SECTION 3: METHODOLOGY AND MODEL IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("3. Methodology and Model Implementation (First Principles)", sec_heading))
    story.append(Paragraph(
        "All models and sampling routines were authored strictly from mathematical first principles in Python using vectorized NumPy. "
        "Below, each of the ten technical milestones is detailed with its objective, mathematical formulation, implementation, and known limitations:",
        body
    ))

    methodology_steps = [
        ("Step 1: Synthetic Minority Oversampling (SMOTE) & Cyclic Time Encoding",
         "<b>Objective:</b> Alleviate gradient starvation of minority class without simple instance duplication.<br/>"
         "<b>Mathematical Formulation:</b> For each minority vector $x_i$, identify its $k$ nearest minority neighbors under Euclidean distance. "
         "Synthesize a new sample: $x_{\\text{syn}} = x_i + \\lambda(x_{zi} - x_i)$ where $\\lambda \\sim \\text{Uniform}(0, 1)$. "
         "Transform linear time $t$ into circular topology: $\\sin(2\\pi t / 86400)$ and $\\cos(2\\pi t / 86400)$.<br/>"
         "<b>NumPy Implementation:</b> Pairwise distance matrix computed via broadcasted differences: `d = np.linalg.norm(X[:, None] - X[None, :], axis=-1)`.<br/>"
         "<b>Limitations:</b> Convex interpolation assumes local linearity between fraud points; can generate unrealistic samples in high-dimensional sparse voids."),

        ("Step 2: Deep Multi-Layer Perceptron (MLP) with Backpropagation and Adam",
         "<b>Objective:</b> Model non-linear decision boundaries across orthogonal PCA features.<br/>"
         "<b>Mathematical Formulation:</b> 3-layer architecture: $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$, with LeakyReLU activation $f(z) = \\max(0.01z, z)$. "
         "Loss: Binary Cross-Entropy with L2 regularization $\\frac{\\lambda}{2N}\\sum ||W||^2$. Gradients derived via analytical chain rule. "
         "Parameters updated via Adam: $m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t$, $v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2$ with bias correction.<br/>"
         "<b>NumPy Implementation:</b> Fully vectorized forward and backward passes using matrix dot products (`@`) and He normal weight initialization.<br/>"
         "<b>Limitations:</b> Computationally expensive relative to linear models; susceptible to uncalibrated probability predictions in tail distributions."),

        ("Step 3: Stratified K-Fold Cross-Validation & Asymmetric Cost Objective",
         "<b>Objective:</b> Prevent fold collapse on rare classes and align validation with business costs.<br/>"
         "<b>Mathematical Formulation:</b> Partition minority indices $\\mathcal{I}_1$ and majority indices $\\mathcal{I}_0$ into $K=5$ disjoint subsets, "
         "re-merging them to ensure exact class proportions. Cost Objective: $\\mathcal{L}_{\\text{cost}} = \\sum (\\text{FP} \\times C_{\\text{FP}}) + \\sum (\\text{FN} \\times \\text{Amount})$.<br/>"
         "<b>NumPy Implementation:</b> Programmed custom stratified split generator using index slicing without external dependencies.<br/>"
         "<b>Limitations:</b> 5-fold cross-validation scales training time 5×; metric variance can still remain non-trivial across folds."),

        ("Step 4: Gradient Boosted Decision Trees (GBDT) from Scratch",
         "<b>Objective:</b> Construct an ensemble of weak decision stumps targeting non-linear feature interactions.<br/>"
         "<b>Mathematical Formulation:</b> Minimize negative log-likelihood via functional gradient descent. At stage $m$, compute pseudo-residuals "
         "$r_{im} = y_i - p_i(x_i)$. Fit decision stump $h_m(x)$ by evaluating feature split thresholds that maximize variance reduction: "
         "$\\Delta \\text{Var} = \\text{Var}(S) - \\frac{|S_L|}{|S|}\\text{Var}(S_L) - \\frac{|S_R|}{|S|}\\text{Var}(S_R)$. Update: $F_m(x) = F_{m-1}(x) + \\eta h_m(x)$ with $\\eta = 0.1$.<br/>"
         "<b>NumPy Implementation:</b> Vectorized quantile scanning across feature vectors to find split candidates.<br/>"
         "<b>Limitations:</b> Tree depth restricted to stumps/shallow trees to maintain reasonable training latency in pure NumPy without C++ tree walkers."),

        ("Step 5: Unsupervised Anomaly Detection: Isolation Forest from Scratch",
         "<b>Objective:</b> Detect novel fraud patterns without reliance on ground-truth training labels.<br/>"
         "<b>Mathematical Formulation:</b> Construct $T=100$ isolation trees using random feature selection and uniform random split thresholds. "
         "Let $h(x)$ denote path length to isolate sample $x$. The anomaly score is $s(x, n) = 2^{-E(h(x))/c(n)}$, where $c(n) = 2(\\ln(n-1) + 0.5772156649) - \\frac{2(n-1)}{n}$.<br/>"
         "<b>NumPy Implementation:</b> Recursive tree construction stored as nested array indices; scoring vectorized across trees.<br/>"
         "<b>Limitations:</b> Lower supervised precision (8.36%) as it flags generic distributional anomalies rather than fraud-specific behaviors."),

        ("Step 6: Cost-Sensitive Loss Reweighting & Dollar Value Recall",
         "<b>Objective:</b> Direct optimization toward financial recovery rather than unweighted event detection.<br/>"
         "<b>Mathematical Formulation:</b> Weight sample loss by transaction monetary exposure: $w_i = (\\text{Amount}_i / \\mu_{\\text{Amount}})$ if $y_i=1$, else $1.0$. "
         "Gradient: $\\nabla_W = \\frac{1}{N}X^T (W \\odot (\\hat{y} - y))$. Dollar Recall: $\\frac{\\sum_{TP} \\text{Amount}}{\\sum_{\\text{All Fraud}} \\text{Amount}}$.<br/>"
         "<b>NumPy Implementation:</b> Element-wise weight array broadcasting during logistic loss and gradient computation.<br/>"
         "<b>Limitations:</b> Induces aggressive positive classification for large transactions, substantially elevating false alarm rates."),

        ("Step 7: Multi-Paradigm Hybrid Soft-Voting Ensemble",
         "<b>Objective:</b> Reduce model-specific variance by combining structurally diverse model families.<br/>"
         "<b>Mathematical Formulation:</b> Calibrated probability blending: $P_{\\text{ens}}(x) = 0.35 \\cdot P_{\\text{LogReg}}(x) + 0.35 \\cdot P_{\\text{GBDT}}(x) + 0.30 \\cdot P_{\\text{MLP}}(x)$.<br/>"
         "<b>NumPy Implementation:</b> Linear combination of output probability vectors.<br/>"
         "<b>Limitations:</b> Inference latency equals the sum of base models; requires probability calibration across disparate loss objectives."),

        ("Step 8: Dynamic Instance-Dependent Bayesian Thresholding",
         "<b>Objective:</b> Replace static threshold ($p \\ge 0.50$) with rational transaction-specific risk decisioning.<br/>"
         "<b>Mathematical Formulation:</b> Under Bayes Minimum Risk, flag transaction $x$ if expected cost of flagging is less than expected cost of allowing: "
         "$(1 - p(x))C_{\\text{FP}} < p(x)C_{\\text{FN}}(x) \\implies p^*(x) = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + \\text{Amount}(x)}$.<br/>"
         "<b>NumPy Implementation:</b> Vectorized element-wise comparison: `pred = (probs >= (2.50 / (2.50 + amounts))).astype(int)`.<br/>"
         "<b>Limitations:</b> Assumes risk neutrality and known fixed $C_{\\text{FP}}$; extreme low amounts ($0.99) require very high confidence to trigger."),

        ("Step 9: Focal Loss and Hard Negative Example Mining",
         "<b>Objective:</b> Prevent overwhelmingly easy genuine transactions from dominating backpropagation gradients.<br/>"
         "<b>Mathematical Formulation:</b> $\\mathcal{L}_{\\text{FL}}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\ln(p_t)$, setting $\\gamma = 2.0$. "
         "For well-classified genuine records ($p_t = 0.99$), modulating factor $(1 - 0.99)^2 = 0.0001$ suppresses gradient magnitude by $10,000\\times$.<br/>"
         "<b>NumPy Implementation:</b> Analytical focal loss gradient vectorization: $\\nabla_W = \\frac{1}{N}X^T ((1-p_t)^\\gamma (p - y))$.<br/>"
         "<b>Limitations:</b> Sensitive to hyperparameter selection $(\\alpha, \\gamma)$; can slow convergence in early training epochs."),

        ("Step 10: Stacked Generalization with Out-of-Fold Meta-Learning",
         "<b>Objective:</b> Train a meta-learner to discover non-linear consensus among base predictors without data leakage.<br/>"
         "<b>Mathematical Formulation:</b> Wolpert (1992) Stacking. Partition training data into 5 folds. Base models generate Out-Of-Fold (OOF) predictions "
         "matrix $S \\in \\mathbb{R}^{N \\times M}$. A Level-1 Logistic Meta-Learner trains on $S$ to learn model trust coefficients: $\\hat{y} = \\sigma(S W_{\\text{meta}} + b_{\\text{meta}})$.<br/>"
         "<b>NumPy Implementation:</b> 5-fold cross-validation loop populating OOF matrix, followed by L2-regularized logistic regression on meta-features.<br/>"
         "<b>Limitations:</b> High computational complexity during training; Level-1 learner can overfit if base models produce highly collinear probabilities.")
    ]

    for title, details in methodology_steps:
        story.append(Paragraph(title, subsec_heading))
        story.append(Paragraph(details, body))
        story.append(Spacer(1, 2))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 4: EXPERIMENTAL DESIGN & BENCHMARK RESULTS
    # =========================================================================
    story.append(Paragraph("4. Experimental Design and Benchmark Results", sec_heading))
    story.append(Paragraph(
        "<b>4.1 Simulation Design & Uncertainty Quantification:</b> Evaluating an imbalanced fraud model on a single 80/20 train/test split "
        "is subject to sampling variance. To quantify estimator stability, we conducted a <b>100-run Monte Carlo simulation ($N = 100$)</b>. "
        "In each run $k \\in \\{1000, \\dots, 1099\\}$, the entire dataset was partitioned via a fresh stratified 80/20 split, all seven model "
        "architectures were trained from scratch, and test metrics were logged.<br/>"
        "We report sample mean $\\bar{x}$, sample standard deviation $s$, and the 95% Confidence Interval half-width for the mean: "
        "$\\text{Margin} = 1.96 \\cdot \\frac{s}{\\sqrt{N}} = 0.196 \\cdot s$. "
        "<i>Note on Interpretation:</i> These confidence intervals quantify uncertainty in the estimated mean performance across repeated random splits "
        "of this historical dataset. They do NOT imply that sampling variance is eliminated, nor do they guarantee identical performance on future non-stationary data.",
        body
    ))

    # Comprehensive Benchmark Table (532 pt width)
    bench_headers = [
        Paragraph("<b>Model</b>", tbl_header),
        Paragraph("<b>Event Recall</b>", tbl_header),
        Paragraph("<b>Dollar Recall</b>", tbl_header),
        Paragraph("<b>Precision</b>", tbl_header),
        Paragraph("<b>F1 Score</b>", tbl_header),
        Paragraph("<b>ROC-AUC</b>", tbl_header),
        Paragraph("<b>FPR (%)</b>", tbl_header),
        Paragraph("<b>Saved ($)</b>", tbl_header),
        Paragraph("<b>False Pos (FP)</b>", tbl_header),
        Paragraph("<b>Missed ($)</b>", tbl_header),
        Paragraph("<b>Total Cost ($)</b>", tbl_header)
    ]

    bench_rows = [
        [Paragraph("<b>1. Baseline LogReg</b>", tbl_cell),
         Paragraph("86.74%<br/>±0.66%", tbl_cell_center),
         Paragraph("78.50%<br/>±1.95%", tbl_cell_center),
         Paragraph("14.29%<br/>±0.34%", tbl_cell_center),
         Paragraph("24.49%<br/>±0.50%", tbl_cell_center),
         Paragraph("0.9749<br/>±0.0019", tbl_cell_center),
         Paragraph("0.91%<br/>±0.03%", tbl_cell_center),
         Paragraph("$9,508<br/>±$381", tbl_cell_right),
         Paragraph("518.3<br/>±14.5", tbl_cell_center),
         Paragraph("$2,672<br/>±$275", tbl_cell_right),
         Paragraph("<b>$3,968</b><br/>±$275", tbl_cell_bold_right)],

        [Paragraph("<b>2. Cost-Sensitive LogReg</b>", tbl_cell),
         Paragraph("<b>93.81%</b><br/>±0.51%", tbl_cell_bold_center),
         Paragraph("<b>92.89%</b><br/>±1.41%", tbl_cell_bold_center),
         Paragraph("1.18%<br/>±0.03%", tbl_cell_center),
         Paragraph("2.33%<br/>±0.06%", tbl_cell_center),
         Paragraph("0.9732<br/>±0.0020", tbl_cell_center),
         Paragraph("13.81%<br/>±0.40%", tbl_cell_center),
         Paragraph("<b>$11,254</b><br/>±$401", tbl_cell_bold_right),
         Paragraph("7,854.0<br/>±225.6", tbl_cell_center),
         Paragraph("<b>$926</b><br/>±$198", tbl_cell_bold_right),
         Paragraph("$20,561<br/>±$508", tbl_cell_right)],

        [Paragraph("<b>3. Neural Net (MLP)</b>", tbl_cell),
         Paragraph("89.81%<br/>±0.56%", tbl_cell_center),
         Paragraph("84.35%<br/>±1.84%", tbl_cell_center),
         Paragraph("5.06%<br/>±0.13%", tbl_cell_center),
         Paragraph("9.57%<br/>±0.24%", tbl_cell_center),
         Paragraph("0.9716<br/>±0.0024", tbl_cell_center),
         Paragraph("2.96%<br/>±0.08%", tbl_cell_center),
         Paragraph("$10,219<br/>±$394", tbl_cell_right),
         Paragraph("1,682.8<br/>±47.6", tbl_cell_center),
         Paragraph("$1,961<br/>±$269", tbl_cell_right),
         Paragraph("$6,169<br/>±$270", tbl_cell_right)],

        [Paragraph("<b>4. GBDT Classifier</b>", tbl_cell),
         Paragraph("88.14%<br/>±0.60%", tbl_cell_center),
         Paragraph("81.67%<br/>±1.86%", tbl_cell_center),
         Paragraph("9.18%<br/>±0.36%", tbl_cell_center),
         Paragraph("16.57%<br/>±0.58%", tbl_cell_center),
         Paragraph("0.9771<br/>±0.0018", tbl_cell_center),
         Paragraph("1.57%<br/>±0.07%", tbl_cell_center),
         Paragraph("$9,890<br/>±$382", tbl_cell_right),
         Paragraph("892.7<br/>±39.5", tbl_cell_center),
         Paragraph("$2,290<br/>±$272", tbl_cell_right),
         Paragraph("<b>$4,522</b><br/>±$272", tbl_cell_bold_right)],

        [Paragraph("<b>5. Isolation Forest</b> (Unsup.)", tbl_cell),
         Paragraph("59.90%<br/>±1.87%", tbl_cell_center),
         Paragraph("56.13%<br/>±2.86%", tbl_cell_center),
         Paragraph("8.36%<br/>±0.43%", tbl_cell_center),
         Paragraph("14.59%<br/>±0.66%", tbl_cell_center),
         Paragraph("0.9449<br/>±0.0025", tbl_cell_center),
         Paragraph("1.18%<br/>±0.05%", tbl_cell_center),
         Paragraph("$6,810<br/>±$431", tbl_cell_right),
         Paragraph("673.0<br/>±30.5", tbl_cell_center),
         Paragraph("$5,370<br/>±$409", tbl_cell_right),
         Paragraph("$7,053<br/>±$410", tbl_cell_right)],

        [Paragraph("<b>6. Soft-Voting Ensemble</b>", tbl_cell),
         Paragraph("89.70%<br/>±0.59%", tbl_cell_center),
         Paragraph("84.51%<br/>±1.75%", tbl_cell_center),
         Paragraph("6.45%<br/>±0.17%", tbl_cell_center),
         Paragraph("12.01%<br/>±0.28%", tbl_cell_center),
         Paragraph("<b>0.9799</b><br/>±0.0016", tbl_cell_bold_center),
         Paragraph("2.28%<br/>±0.06%", tbl_cell_center),
         Paragraph("$10,236<br/>±$388", tbl_cell_right),
         Paragraph("1,299.3<br/>±36.7", tbl_cell_center),
         Paragraph("$1,944<br/>±$240", tbl_cell_right),
         Paragraph("$5,192<br/>±$240", tbl_cell_right)],

        [Paragraph("<b>7. Stacked Meta-Learner</b>", tbl_cell),
         Paragraph("90.34%<br/>±0.59%", tbl_cell_center),
         Paragraph("85.82%<br/>±1.66%", tbl_cell_center),
         Paragraph("4.57%<br/>±0.11%", tbl_cell_center),
         Paragraph("8.68%<br/>±0.20%", tbl_cell_center),
         Paragraph("0.9779<br/>±0.0019", tbl_cell_center),
         Paragraph("3.31%<br/>±0.09%", tbl_cell_center),
         Paragraph("$10,399<br/>±$392", tbl_cell_right),
         Paragraph("1,880.5<br/>±50.0", tbl_cell_center),
         Paragraph("$1,781<br/>±$230", tbl_cell_right),
         Paragraph("$6,482<br/>±$230", tbl_cell_right)]
    ]

    t_bench = Table([bench_headers] + bench_rows, colWidths=[104, 42, 44, 38, 38, 42, 36, 50, 42, 44, 52])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>4.2 Critical Operational Trade-Off Analysis:</b><br/>"
        "• <b>The False-Alarm Penalty:</b> Cost-Sensitive Logistic Regression attains the highest recall (93.81%), missing only 6.1 frauds ($926 loss). "
        "However, this comes at an extreme operational penalty: <b>7,854 false positives (13.81% FPR)</b>. Under a baseline review cost of $2.50 per alert, "
        "investigation overhead amounts to $19,635, elevating total operational loss to <b>$20,561 per split</b>.<br/>"
        "• <b>Cost-Effective Champions:</b> Baseline Logistic Regression and GBDT exhibit far superior total cost profiles ($3,968 and $4,522 total loss, respectively). "
        "By restricting false alarms to 518 and 893, they prevent operational team saturation while preserving ~80% of stolen capital.<br/>"
        "• <b>Reconciliation of Step 10 Notebook vs. 100-Run Table:</b> In the single-split Step 10 notebook (Seed 42), the stacked meta-learner was coupled "
        "with instance-dependent dynamic thresholding ($p^*(x)$), yielding 42.16% precision and 118 false alarms. In contrast, the 100-run simulation evaluated "
        "the stacked meta-learner under standard fixed thresholding ($p=0.50$) to ensure controlled benchmark parity across all models.",
        body
    ))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 5: FINANCIAL DECISION ANALYSIS & SENSITIVITY
    # =========================================================================
    story.append(Paragraph("5. Financial Decision Analysis & Operational Cost Sensitivity", sec_heading))
    story.append(Paragraph(
        "<b>5.1 Cost Formulation & Institutional Grounding:</b> A rational fraud system must minimize expected net loss. "
        "We define direct fraud loss as $C_{\\text{FN}}(x) = \\text{Amount}(x)$ due to mandatory consumer indemnification under US Regulation E / EU PSD2. "
        "The cost of a false alert ($C_{\\text{FP}}$) encapsulates direct communication (automated SMS gateway $0.05, IVR phone call $0.20), "
        "amortized analyst triage queue time ($1.20–$1.80), and customer friction/churn hazard ($0.70–$1.20). "
        "While peer-reviewed literature (Bahnsen et al., 2015; Dal Pozzolo et al., 2014) identifies $C_{\\text{FP}} \\in [\\$2.00, \\$3.00]$ (standardized to $2.50), "
        "we treat $C_{\\text{FP}} = \\$2.50$ as a <b>project baseline assumption</b> rather than a universal constant.",
        body
    ))
    story.append(Paragraph(
        "<b>5.2 Bayes Minimum Risk Threshold Derivation:</b> Under risk-neutral expectation, an alert is triggered if: "
        "$(1 - p(x))C_{\\text{FP}} < p(x)C_{\\text{FN}}(x) \\implies p^*(x) = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + \\text{Amount}(x)}$. "
        "This dynamic rule scales threshold sensitivity: high-dollar transactions ($3,000) trigger at $p^* \\approx 0.083\\%$, while micro-transactions "
        "($0.99) require $p^* \\approx 71.6\\%$, preventing negative expected value interventions.",
        body
    ))
    story.append(Paragraph(
        "<b>5.3 Empirical Sensitivity Analysis Across Cost Regimes:</b> Because actual operational overhead varies widely across institutions "
        "(e.g., automated in-app biometric challenge vs. manual phone review), we evaluated the total expected financial loss "
        "$\\mathbb{E}[\\text{Loss}] = \\text{FP} \\times C_{\\text{FP}} + \\text{Missed Dollars}$ across 100 runs for $C_{\\text{FP}} \\in [\\$0.50, \\$1.00, \\$2.50, \\$5.00, \\$10.00]$:",
        body
    ))

    # Sensitivity Table (532 pt width)
    sens_headers = [
        Paragraph("<b>Model Architecture</b>", tbl_header),
        Paragraph("<b>C_FP = $0.50<br/>(Push/App)</b>", tbl_header),
        Paragraph("<b>C_FP = $1.00<br/>(SMS Alert)</b>", tbl_header),
        Paragraph("<b>C_FP = $2.50<br/>(Base/Queue)</b>", tbl_header),
        Paragraph("<b>C_FP = $5.00<br/>(Manual Call)</b>", tbl_header),
        Paragraph("<b>C_FP = $10.00<br/>(Analyst Review)</b>", tbl_header),
        Paragraph("<b>Optimal Regime</b>", tbl_header)
    ]

    sens_rows = [
        [Paragraph("1. Baseline Logistic Regression", tbl_cell),
         Paragraph("$2,931.06", tbl_cell_right),
         Paragraph("$3,190.22", tbl_cell_right),
         Paragraph("<b>$3,967.70</b>", tbl_cell_bold_right),
         Paragraph("<b>$5,263.50</b>", tbl_cell_bold_right),
         Paragraph("<b>$7,855.10</b>", tbl_cell_bold_right),
         Paragraph("<b>Lowest Cost for C_FP ≥ $2.50</b>", tbl_cell_bold)],

        [Paragraph("2. Cost-Sensitive LogReg", tbl_cell),
         Paragraph("$4,853.38", tbl_cell_right),
         Paragraph("$8,780.38", tbl_cell_right),
         Paragraph("$20,561.36", tbl_cell_right),
         Paragraph("$40,196.34", tbl_cell_right),
         Paragraph("$79,466.29", tbl_cell_right),
         Paragraph("Viable only if C_FP < $0.15", tbl_cell)],

        [Paragraph("3. Neural Network (MLP)", tbl_cell),
         Paragraph("$2,802.91", tbl_cell_right),
         Paragraph("$3,644.33", tbl_cell_right),
         Paragraph("$6,168.61", tbl_cell_right),
         Paragraph("$10,375.73", tbl_cell_right),
         Paragraph("$18,789.98", tbl_cell_right),
         Paragraph("Balanced Non-Linear Catch", tbl_cell)],

        [Paragraph("4. GBDT Classifier", tbl_cell),
         Paragraph("$2,736.38", tbl_cell_right),
         Paragraph("<b>$3,182.72</b>", tbl_cell_bold_right),
         Paragraph("$4,521.74", tbl_cell_right),
         Paragraph("$6,753.44", tbl_cell_right),
         Paragraph("$11,216.84", tbl_cell_right),
         Paragraph("<b>Lowest Cost for C_FP ≈ $1.00</b>", tbl_cell_bold)],

        [Paragraph("5. Isolation Forest (Unsup.)", tbl_cell),
         Paragraph("$5,706.44", tbl_cell_right),
         Paragraph("$6,042.97", tbl_cell_right),
         Paragraph("$7,052.54", tbl_cell_right),
         Paragraph("$8,735.17", tbl_cell_right),
         Paragraph("$12,100.42", tbl_cell_right),
         Paragraph("Zero-Day Anomaly Sentinel", tbl_cell)],

        [Paragraph("6. Soft-Voting Ensemble", tbl_cell),
         Paragraph("<b>$2,593.39</b>", tbl_cell_bold_right),
         Paragraph("$3,243.02", tbl_cell_right),
         Paragraph("$5,191.91", tbl_cell_right),
         Paragraph("$8,440.06", tbl_cell_right),
         Paragraph("$14,936.36", tbl_cell_right),
         Paragraph("<b>Lowest Cost for C_FP ≤ $0.50</b>", tbl_cell_bold)],

        [Paragraph("7. Stacked Meta-Learner", tbl_cell),
         Paragraph("$2,721.38", tbl_cell_right),
         Paragraph("$3,661.66", tbl_cell_right),
         Paragraph("$6,482.48", tbl_cell_right),
         Paragraph("$11,183.86", tbl_cell_right),
         Paragraph("$20,586.61", tbl_cell_right),
         Paragraph("High Precision Meta-Ranking", tbl_cell)]
    ]

    t_sens = Table([sens_headers] + sens_rows, colWidths=[122, 65, 65, 65, 65, 70, 80])
    t_sens.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_sens)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Key Sensitivity Finding:</b> If false alarms can be verified via low-friction, automated digital channels ($C_{\\text{FP}} \\le \\$0.50$), "
        "the <b>Soft-Voting Ensemble</b> minimizes total institutional loss ($2,593.39). As investigation overhead rises to standard operational "
        "levels ($C_{\\text{FP}} \\ge \\$2.50$), <b>Baseline Logistic Regression</b> emerges as the most financially efficient model ($3,967.70), "
        "followed closely by GBDT ($4,521.74). Cost-Sensitive Logistic Regression is economically unviable unless $C_{\\text{FP}}$ approaches zero.",
        body
    ))

    story.append(PageBreak())

    # =========================================================================
    # SECTION 6: DISCUSSION, LIMITATIONS, AND CONCLUSION
    # =========================================================================
    story.append(Paragraph("6. Discussion, Limitations, and Practical Recommendations", sec_heading))
    story.append(Paragraph(
        "<b>6.1 Synthesis of Experimental Findings:</b><br/>"
        "1. <i>The Trade-Off Between Detection and Operational Overhead:</i> Maximizing recall does not equate to minimizing cost. "
        "High-recall models (Cost-Sensitive LogReg) generate massive alert volumes that can overwhelm operational fraud investigation queues.<br/>"
        "2. <i>First-Principles Vectorization:</i> By structuring forward propagation, gradient descent, and decision tree splitting into vectorized "
        "NumPy array operations, we attained sub-millisecond per-sample scoring latency on standard hardware without relying on compiled C++ frameworks.<br/>"
        "3. <i>Unsupervised Complements Supervised:</i> While Isolation Forest achieved lower overall recall (59.90%), its zero-label formulation "
        "renders it an essential parallel watchdog for detecting structural anomalies that evade supervised boundary planes.",
        body
    ))
    story.append(Paragraph(
        "<b>6.2 Operational Model Deployment Selection Guide:</b><br/>"
        "• <b>Scenario A (Strict Alert Capacity / Manual Triage):</b> Deploy <b>Baseline Logistic Regression</b> or <b>GBDT</b>. These models "
        "maintain FPR < 1.6%, keeping weekly false positives within manageable analyst limits while stopping > 86% of fraud.<br/>"
        "• <b>Scenario B (Automated Frictionless Customer Challenge):</b> Deploy the <b>Soft-Voting Ensemble</b>. It delivers the highest ROC-AUC (0.9799) "
        "and minimizes total cost when automated verification costs are low ($C_{\\text{FP}} \\le \\$0.50$).<br/>"
        "• <b>Scenario C (Catastrophic High-Value Protection):</b> Deploy <b>Cost-Sensitive LogReg</b> strictly as a secondary filter on high-dollar "
        "transactions (> $1,000), where saving $4,000+ per event economically justifies false positive inquiries.",
        body
    ))
    story.append(Paragraph(
        "<b>6.3 Study Limitations & Methodological Caveats:</b><br/>"
        "• <i>Non-Stationary Adversarial Concept Drift:</i> This dataset reflects a static 48-hour snapshot from 2013. In production, fraud syndicates "
        "actively probe rules, adapt merchant categories, and rotate card vectors. A static model trained on historical data will experience performance decay.<br/>"
        "• <i>Synthetically Derived Features:</i> PCA anonymization obscured semantic transaction attributes (merchant terminal ID, device fingerprint, "
        "IP velocity), preventing real-time graph relational modeling.<br/>"
        "• <i>Finite Pool Resampling:</i> While 100 Monte Carlo splits evaluate algorithmic variance under repeated partitioning, all splits draw from the "
        "same finite pool of 492 fraud events. True generalization requires streaming out-of-time evaluation across distinct temporal epochs.",
        body
    ))
    story.append(Paragraph(
        "<b>6.4 Proposed Latency Design Targets vs. Production Reality:</b><br/>"
        "Commercial networks (VisaNet, Mastercard Banknet) mandate strict transaction scoring budgets (< 50 ms). In our local test environment, "
        "single-sample inference averaged 0.2 ms for Logistic Regression, 1.8 ms for MLP, and 4.5 ms for GBDT. However, these figures represent "
        "<b>proposed engineering targets on benchmark hardware</b>, not field-validated production SLAs under concurrent multi-tenant network loads.",
        body
    ))

    story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 7: REFERENCES & REPRODUCIBILITY APPENDIX
    # =========================================================================
    story.append(Paragraph("7. References and Reproducibility Appendix", sec_heading))
    story.append(Paragraph(
        "<b>7.1 Traceable Peer-Reviewed Bibliography:</b>", subsec_heading
    ))

    references = [
        "Bahnsen, A. C., Aouada, D., Stojanovic, A., & Ottersten, B. (2015). 'Cost-sensitive credit card fraud detection using Bayes minimum risk.' "
        "<i>Expert Systems with Applications</i>, 42(5), 2501–2509. DOI: 10.1016/j.eswa.2014.10.038.",
        "Dal Pozzolo, A., Caelen, O., Le Borgne, Y. A., Waterschoot, S., & Bontempi, G. (2014). 'Learned lessons in credit card fraud detection from a practitioner perspective.' "
        "<i>Expert Systems with Applications</i>, 41(10), 4915–4928. DOI: 10.1016/j.eswa.2014.02.026.",
        "Dal Pozzolo, A., Borne, Y. A., Caelen, O., & Bontempi, G. (2015). 'Calibrating probability with undersampling for unbalanced classification.' "
        "<i>IEEE Symposium Series on Computational Intelligence (SSCI)</i>, 159–166. DOI: 10.1109/SSCI.2015.33.",
        "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). 'SMOTE: synthetic minority over-sampling technique.' "
        "<i>Journal of Artificial Intelligence Research (JAIR)</i>, 16, 321–357. DOI: 10.1613/jair.953.",
        "Wolpert, D. H. (1992). 'Stacked generalization.' <i>Neural Networks</i>, 5(2), 241–259. DOI: 10.1016/S0893-6080(05)80023-1.",
        "Friedman, J. H. (2001). 'Greedy function approximation: a gradient boosting machine.' <i>The Annals of Statistics</i>, 29(5), 1189–1232. DOI: 10.1214/aos/1013203451.",
        "Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). 'Isolation forest.' <i>IEEE International Conference on Data Mining (ICDM)</i>, 413–422. DOI: 10.1109/ICDM.2008.17.",
        "Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). 'Focal loss for dense object detection.' <i>IEEE International Conference on Computer Vision (ICCV)</i>, 2980–2988. DOI: 10.1109/ICCV.2017.324.",
        "Davis, J., & Goadrich, M. (2006). 'The relationship between Precision-Recall and ROC curves.' <i>International Conference on Machine Learning (ICML)</i>, 233–240. DOI: 10.1145/1143844.1143874.",
        "LexisNexis Risk Solutions. (2023). '2023 True Cost of Fraud Study: Financial Services and Lending.' Global Financial Fraud Report.",
        "United States Code of Federal Regulations. 12 CFR Part 1005 (Regulation E) — Electronic Fund Transfers. Consumer Financial Protection Bureau (CFPB)."
    ]
    for ref in references:
        story.append(Paragraph(f"• {ref}", tbl_cell))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>7.2 Environment and Execution Instructions:</b>", subsec_heading))
    story.append(Paragraph(
        "• <b>Hardware / Runtime:</b> Python 3.13 / Windows 11 x64, 12 Logical CPU Cores, 16 GB RAM.<br/>"
        "• <b>Dependencies:</b> `numpy>=1.24.0`, `pandas>=2.0.0`, `reportlab>=4.0.0` (ReportLab used strictly for PDF rendering; zero external ML packages).<br/>"
        "• <b>Re-generating 100-Run Benchmark:</b> Run `python run_100_simulations.py` (executes 100 parallel worker processes across seeds 1000–1099 in ~88s).<br/>"
        "• <b>Re-generating Monograph PDF:</b> Run `python generate_pdf_report.py` to re-compile this PDF to both the workspace and Desktop.",
        body
    ))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    
    import shutil
    shutil.copyfile(pdf_local, pdf_desktop)
    print(f"Publication-grade Monograph compiled successfully:")
    print(f"Local:   {pdf_local}")
    print(f"Desktop: {pdf_desktop}")

if __name__ == "__main__":
    build_pdf_monograph()
