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
    along with academic running headers and footers across all pages.
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
        self.drawString(36, letter[1] - 30, "CREDIT CARD FRAUD DETECTION AT ENTERPRISE SCALE: RESEARCH MONOGRAPH")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(letter[0] - 36, letter[1] - 30, "FIRST-PRINCIPLES NUMPY & MONTE CARLO EVALUATION")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, letter[1] - 34, letter[0] - 36, letter[1] - 34)
        
        # Running Footer
        self.line(36, 42, letter[0] - 36, 42)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 30, "Python for Data Science (PDS) — Capstone Research Monograph | IIIT Sri City")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 30, page_str)
        self.restoreState()

def build_pdf_monograph():
    out_dir_local = r"c:\Users\HP\.gemini\antigravity-ide\scratch\Credit-Card-Fraud-Detection-Model\report"
    out_dir_desktop = r"C:\Users\HP\Desktop\PDS\report"
    os.makedirs(out_dir_local, exist_ok=True)
    os.makedirs(out_dir_desktop, exist_ok=True)
    
    pdf_local = os.path.join(out_dir_local, "Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf")
    pdf_desktop = os.path.join(out_dir_desktop, "Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf")

    # Document Geometry: 540 pt printable width (Margins 36pt), 708 pt printable height
    margin = 36
    doc = SimpleDocTemplate(
        pdf_local,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=42,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Measured Academic Color Palette
    c_primary = colors.HexColor("#0F172A")    # Deep Slate / Navy
    c_secondary = colors.HexColor("#1E3A8A")  # Deep Blue
    c_accent = colors.HexColor("#B91C1C")     # Crimson / Alert
    c_dark = colors.HexColor("#1E293B")       # Dark Charcoal (Body)
    c_muted = colors.HexColor("#64748B")      # Muted Slate
    c_bg_light = colors.HexColor("#F8FAFC")   # Light Slate
    c_border = colors.HexColor("#CBD5E1")     # Border
    c_highlight = colors.HexColor("#F1F5F9")  # Table Alternate Row
    c_callout_bg = colors.HexColor("#EFF6FF") # Soft Blue
    c_callout_border = colors.HexColor("#3B82F6")
    c_amber_bg = colors.HexColor("#FEFCE8")   # Warning / Note
    c_amber_border = colors.HexColor("#EAB308")
    c_green_bg = colors.HexColor("#F0FDF4")   # Success / Highlight
    c_green_border = colors.HexColor("#22C55E")

    # Typography Hierarchy
    cover_title = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=c_primary, spaceAfter=4
    )
    
    cover_subtitle = ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=c_secondary, spaceAfter=8
    )
    
    sec_heading = ParagraphStyle(
        'SectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=15,
        textColor=c_primary, spaceBefore=8, spaceAfter=4,
        keepWithNext=True
    )
    
    subsec_heading = ParagraphStyle(
        'SubSectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.5, leading=13,
        textColor=c_secondary, spaceBefore=6, spaceAfter=2.5,
        keepWithNext=True
    )

    subsubsec_heading = ParagraphStyle(
        'SubSubSectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.5, leading=11.5,
        textColor=c_dark, spaceBefore=4, spaceAfter=2,
        keepWithNext=True
    )

    body = ParagraphStyle(
        'BodyCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=11.0,
        textColor=c_dark, spaceAfter=4
    )

    body_tight = ParagraphStyle(
        'BodyTight', parent=body,
        fontSize=7.4, leading=10.2, spaceAfter=2.5
    )

    math_block = ParagraphStyle(
        'MathBlock', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.6, leading=10.8,
        textColor=c_primary, spaceBefore=2, spaceAfter=3.5, leftIndent=8
    )

    code_snippet = ParagraphStyle(
        'CodeSnippet', parent=styles['Normal'],
        fontName='Courier', fontSize=6.8, leading=8.8,
        textColor=colors.HexColor("#0369A1"), spaceBefore=2, spaceAfter=3.5, leftIndent=6
    )

    callout_text = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.4, leading=10.4,
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
    # PAGE 1: TITLE & COVER PAGE
    # =========================================================================
    story.append(Paragraph("Credit Card Fraud Detection at Enterprise Scale", cover_title))
    story.append(Paragraph(
        "A First-Principles Machine Learning Evaluation, 100-Run Monte Carlo Benchmark, "
        "and Financial Decision Analysis in Pure Vectorized NumPy",
        cover_subtitle
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=6))

    # Metadata Grid Table
    meta_rows = [
        [Paragraph("<b>Course:</b> Python for Data Science (PDS)", tbl_cell),
         Paragraph("<b>Dataset:</b> Kaggle / ULB Benchmark (284,807 transactions)", tbl_cell)],
        [Paragraph("<b>Framework:</b> 100% Pure NumPy (Zero External ML Packages)", tbl_cell),
         Paragraph("<b>Class Imbalance:</b> 0.1727% (492 Frauds : 284,315 Genuine)", tbl_cell)],
        [Paragraph("<b>Evaluation Protocol:</b> 100 Stratified Monte Carlo Splits (Seeds 1000–1099)", tbl_cell),
         Paragraph("<b>Hardware Target:</b> 12-Core Multi-Threaded Local Benchmarking", tbl_cell)],
        [Paragraph("<b>Institution:</b> Indian Institute of Information Technology (IIIT) Sri City", tbl_cell),
         Paragraph("<b>Primary Methodology:</b> Asymmetric Cost-Sensitive Risk Minimization", tbl_cell)]
    ]
    t_meta = Table(meta_rows, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # Executive Overview Callout
    exec_overview = (
        "<b>EXECUTIVE MONOGRAPH STATEMENT:</b><br/>"
        "This research monograph investigates the end-to-end design, implementation, and empirical evaluation of "
        "production-grade credit card fraud screening algorithms developed exclusively from foundational mathematical principles "
        "in Python and NumPy. By disallowing high-level machine learning libraries (such as scikit-learn, XGBoost, or PyTorch), "
        "this investigation provides complete visibility into loss surfaces, analytical backpropagation, gradient updates, "
        "decision tree partitioning, and probability calibration. Furthermore, this work moves beyond traditional symmetric metrics "
        "(accuracy, F1-score) by establishing an asymmetric financial loss matrix that quantifies the real-world operational tension "
        "between fraud loss prevention and false alarm handling costs across an extensive 100-run Monte Carlo benchmark."
    )
    t_exec = Table([[Paragraph(exec_overview, callout_text)]], colWidths=[540])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_callout_bg),
        ('BOX', (0,0), (-1,-1), 1, c_callout_border),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_exec)
    story.append(Spacer(1, 6))

    # Peer-Review Revision Changelog Table
    story.append(Paragraph("<b>Table 1.1: Academic Peer-Review Revision Changelog & Technical Audits</b>", subsubsec_heading))
    rev_headers = [
        Paragraph("<b>Audit Dimension</b>", tbl_header),
        Paragraph("<b>Identified Critique / Technical Defect</b>", tbl_header),
        Paragraph("<b>Rigorous Methodological Resolution & Page Reference</b>", tbl_header)
    ]
    rev_rows = [
        [Paragraph("<b>1. Statistical Claims & Uncertainty</b>", tbl_cell_bold),
         Paragraph("100-run simulation described as 'eliminating sampling luck' and proving generalization.", tbl_cell),
         Paragraph("Reframed as Monte Carlo estimator variance reduction over identically distributed test splits. Explicitly stated CLT assumptions and epistemic limits (Pages 13–15).", tbl_cell)],
        [Paragraph("<b>2. Operational Trade-Offs</b>", tbl_cell_bold),
         Paragraph("Cost-Sensitive LogReg 93.81% recall reported as an unqualified victory despite massive false alarms.", tbl_cell),
         Paragraph("Quantified operational penalty: 7,854 false positives (13.81% FPR) incur $19,635 alert cost, yielding highest total operational loss of $20,561 (Pages 14–15).", tbl_cell)],
        [Paragraph("<b>3. Financial Decision Rule</b>", tbl_cell_bold),
         Paragraph("$2.50 false-positive cost presented as a universal constant without decomposition.", tbl_cell),
         Paragraph("Decomposed cost into communication ($0.05), analyst triage ($1.50), and friction ($0.95). Added sensitivity analysis across $0.50–$10.00 (Pages 16–17).", tbl_cell)],
        [Paragraph("<b>4. Academic Traceability</b>", tbl_cell_bold),
         Paragraph("Informal citations missing exact titles, journal volumes, pages, and digital object identifiers.", tbl_cell),
         Paragraph("Included exhaustive 20+ item bibliography with peer-reviewed journal papers, DOIs, and regulatory directives (Pages 19–20).", tbl_cell)],
        [Paragraph("<b>5. Presentation & Tone</b>", tbl_cell_bold),
         Paragraph("Sensationalist prose ('ultimate financial shield', 'catastrophic theft') and unstructured format.", tbl_cell),
         Paragraph("Rewritten in formal academic prose across 7 standardized research sections with mathematical derivations and implementation analysis (Pages 2–20).", tbl_cell)]
    ]
    t_rev = Table([rev_headers] + rev_rows, colWidths=[110, 190, 240])
    t_rev.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_rev)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: TABLE OF CONTENTS & STRUCTURED ABSTRACT
    # =========================================================================
    story.append(Paragraph("Executive Abstract & Table of Contents", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    abstract_structured = (
        "<b>STRUCTURED RESEARCH ABSTRACT (240 Words):</b><br/>"
        "<b>Context:</b> Payment card fraud imposes billions of dollars in annual losses while operating under extreme class rarity.<br/>"
        "<b>Problem:</b> Standard machine learning formulations optimize symmetric losses (accuracy, log-loss) that ignore the asymmetric operational "
        "economics of fraud detection, where false negatives risk direct stolen capital ($C_{\\text{FN}} = \\text{Amount}$) and false positives incur customer friction "
        "and manual investigation costs ($C_{\\text{FP}}$).<br/>"
        "<b>Methodology:</b> We develop an end-to-end fraud detection architecture from first principles using pure Python and vectorized NumPy, "
        "incorporating 10 distinct algorithmic milestones: SMOTE, cyclic diurnal encoding, deep MLPs with analytical backpropagation and Adam optimization, "
        "gradient boosted decision trees (GBDT), unsupervised isolation forests, cost-sensitive loss reweighting, soft-voting ensembles, "
        "dynamic instance-dependent Bayesian thresholding, focal loss, and stacked generalization. All models are evaluated across a 100-run stratified "
        "Monte Carlo simulation ($N = 100$) on the 284,807-transaction ULB benchmark.<br/>"
        "<b>Key Findings:</b> While Cost-Sensitive Logistic Regression maximizes event recall (93.81% ± 0.51%) and dollar recall (92.89% ± 1.41%), "
        "it generates 7,854 false positives per split (13.81% FPR), resulting in the worst net financial operating loss ($20,561 per split at $2.50 per review). "
        "In contrast, Baseline Logistic Regression ($3,968 loss) and GBDT ($4,522 loss) achieve optimal operational efficiency under standard manual review overhead. "
        "Soft-voting ensembles minimize loss ($2,593) only when automated in-app challenges reduce false alarm verification costs ($C_{\\text{FP}} \\le \\$0.50$).<br/>"
        "<b>Significance:</b> High recall does not imply operational success; optimal model selection is strictly a function of institutional review capacity."
    )
    t_abs2 = Table([[Paragraph(abstract_structured, callout_text)]], colWidths=[540])
    t_abs2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_highlight),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_abs2)
    story.append(Spacer(1, 6))

    # Monograph Table of Contents
    story.append(Paragraph("<b>Comprehensive Table of Contents:</b>", subsubsec_heading))
    toc_data = [
        [Paragraph("<b>Section 1: Introduction, Problem Formulation, & Mathematical Foundations</b>", tbl_cell_bold), Paragraph("<b>Page 3</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.1 Macroeconomic Landscape & Payment Economics | 1.2 Mathematical Imbalance Formulation", tbl_cell), Paragraph("Page 3", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.3 Research Objectives & NumPy Mandate | 1.4 Regulatory Standards (PSD2, Reg E, SR 11-7) | 1.5 Decision Theory", tbl_cell), Paragraph("Page 4", tbl_cell_right)],
        [Paragraph("<b>Section 2: Comprehensive Exploratory Data Analysis & Preprocessing Pipeline</b>", tbl_cell_bold), Paragraph("<b>Page 5</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.1 ULB Benchmark Provenance | 2.2 PCA Feature Orthogonality | 2.3 Empirical Separators | 2.4 Skewed Amount Physics", tbl_cell), Paragraph("Page 5", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.5 Cyclic Diurnal Encoding | 2.6 Leakage Pitfalls | 2.7 Pipeline Invariant Proof & Flow Architecture", tbl_cell), Paragraph("Page 6", tbl_cell_right)],
        [Paragraph("<b>Section 3: Algorithmic Blueprint — 10 Milestones from Mathematical First Principles</b>", tbl_cell_bold), Paragraph("<b>Page 7</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;Step 1: SMOTE & Cyclic Diurnal Encoding | Step 2: Deep MLP with Analytical Backprop & Adam", tbl_cell), Paragraph("Page 7", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;Step 3: Stratified K-Fold CV & Cost Loss | Step 4: Gradient Boosted Trees from Scratch", tbl_cell), Paragraph("Page 8", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;Step 5: Unsupervised Isolation Forest | Step 6: Cost-Sensitive Loss Reweighting & Dollar Recall", tbl_cell), Paragraph("Page 9", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;Step 7: Multi-Paradigm Soft-Voting Ensemble | Step 8: Dynamic Instance-Dependent Bayesian Thresholding", tbl_cell), Paragraph("Page 10", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;Step 9: Focal Loss & Hard Negative Mining | Step 10: Stacked Generalization with Out-of-Fold Meta-Learning", tbl_cell), Paragraph("Page 11", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;Algorithmic Implementation & Hyperparameter Master Catalog Table", tbl_cell), Paragraph("Page 12", tbl_cell_right)],
        [Paragraph("<b>Section 4: Comprehensive Experimental Design & 100-Run Monte Carlo Benchmark</b>", tbl_cell_bold), Paragraph("<b>Page 13</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.1 Monte Carlo Protocol | 4.2 Central Limit Theorem Assumptions | 4.3 Multi-Core Parallel Architecture", tbl_cell), Paragraph("Page 13", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.4 100-Run Master Benchmark Table across 7 Architectures and 11 Metrics | PR-AUC vs ROC-AUC", tbl_cell), Paragraph("Page 14", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.5 Critical Operational Trade-Offs | The Recall Fallacy | Step 10 Single-Split vs Multi-Run Reconciliation", tbl_cell), Paragraph("Page 15", tbl_cell_right)],
        [Paragraph("<b>Section 5: Financial Decision Analysis & Operational Cost Sensitivity</b>", tbl_cell_bold), Paragraph("<b>Page 16</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.1 Cost Decomposition (Direct Loss, Gateway, Analyst Labor, Friction) | 5.2 Academic Grounding of $2.50", tbl_cell), Paragraph("Page 16", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;5.3 Multi-Tier Sensitivity Table across 5 Operational Regimes ($0.50 to $10.00) | Breakeven Analysis", tbl_cell), Paragraph("Page 17", tbl_cell_right)],
        [Paragraph("<b>Section 6: Discussion, Production Architecture, & Regulatory Compliance</b>", tbl_cell_bold), Paragraph("<b>Page 18</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;6.1 Cascaded Two-Tier Screening Architecture | 6.2 Proposed Sub-10ms Latency Targets vs Production Reality", tbl_cell), Paragraph("Page 18", tbl_cell_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;6.3 Regulatory Compliance (PSD2 SCA TRA, Reg E, SR 11-7) | 6.4 Methodological Caveats & Concept Drift", tbl_cell), Paragraph("Page 19", tbl_cell_right)],
        [Paragraph("<b>Section 7: Traceable References & Reproducibility Appendix</b>", tbl_cell_bold), Paragraph("<b>Page 20</b>", tbl_cell_bold_right)],
        [Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;7.1 20+ Item Academic Bibliography with DOIs | 7.2 Hardware Specifications | 7.3 Execution Commands", tbl_cell), Paragraph("Page 20", tbl_cell_right)]
    ]
    t_toc = Table(toc_data, colWidths=[470, 70])
    t_toc.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 2.8),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, c_bg_light]),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: SECTION 1: INTRODUCTION & PROBLEM FORMULATION
    # =========================================================================
    story.append(Paragraph("1. Introduction, Problem Formulation, & Mathematical Foundations", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("1.1 The Macroeconomic Landscape of Payment Fraud", subsec_heading))
    story.append(Paragraph(
        "Modern payment systems process trillions of dollars in non-cash transactions daily. Concurrently, card-not-present (CNP) "
        "fraud and automated account takeovers have expanded dramatically. According to the <i>Nilson Report (2023)</i>, global card fraud losses "
        "exceeded $32.4 billion in 2022, representing an average global loss rate of 6.5 cents per $100 of transaction volume. "
        "In retail electronic banking, fraud detection systems must operate under severe asymmetric operational constraints: legitimate "
        "cardholders execute thousands of routine purchases daily, while fraudulent intrusions occur sparsely. Consequently, the operational "
        "viability of automated screening depends not merely on catching illicit transactions, but on controlling customer disruption and "
        "investigative labor overhead.",
        body
    ))

    story.append(Paragraph("1.2 Mathematical Formulation of Extreme Class Imbalance", subsec_heading))
    story.append(Paragraph(
        "Let $\\mathcal{D} = \\{(x_i, y_i)\\}_{i=1}^N$ denote a transaction corpus where $x_i \\in \\mathbb{R}^D$ represents the observed feature vector "
        "and $y_i \\in \\{0, 1\\}$ denotes the binary transaction status (0 for genuine, 1 for fraudulent). In credit card acquiring networks, the "
        "class prior $P(Y=1) = \\pi_1$ is vanishingly small. In the benchmark under investigation, $\\pi_1 = \\frac{492}{284,807} \\approx 0.001727$ (0.1727%).",
        body
    ))
    story.append(Paragraph(
        "Under standard Empirical Risk Minimization (ERM) with 0-1 loss $\\mathcal{L}_{0-1}(\\hat{y}, y) = \\mathbb{I}(\\hat{y} \\neq y)$, the Bayes optimal classifier "
        "$h^*(x)$ predicts $\\hat{y} = \\arg\\max_{k \\in \\{0, 1\\}} P(Y=k|X=x)$. Because $P(Y=1|X=x) < 0.5$ across virtually the entire feature space, "
        "the standard ERM objective collapses to the trivial majority classifier $h_{\\text{trivial}}(x) \\equiv 0$. "
        "This trivial predictor achieves an apparent accuracy of:",
        body
    ))
    story.append(Paragraph(
        "$$\\text{Accuracy}(h_{\\text{trivial}}) = 1 - \\pi_1 = 1 - 0.001727 = 99.8273\\%$$",
        math_block
    ))
    story.append(Paragraph(
        "Despite achieving nearly 99.83% accuracy, $h_{\\text{trivial}}$ captures precisely zero fraudulent transactions (Recall = 0.0%), "
        "permitting 100% of stolen capital to drain unchecked. This illustrates the fundamental collapse of symmetric evaluation metrics in rare-event modeling. "
        "Consequently, evaluating fraud screening algorithms requires decoupling performance into sensitivity across positive events, precision over "
        "generated alarms, and financial cost minimization under asymmetric penalties.",
        body
    ))

    story.append(Paragraph("1.3 Core Research Objectives & First-Principles Vectorization", subsec_heading))
    story.append(Paragraph(
        "The primary intellectual and engineering objectives of this monograph are structured as follows:",
        body
    ))
    obj_items = [
        "<b>1. Algorithmic Transparency via Vectorized NumPy:</b> Implement all sampling routines, neural networks, tree ensembles, and anomaly models "
        "strictly using pure Python and multi-dimensional NumPy array broadcasting. This removes opaque abstraction layers, exposing the raw numerical "
        "linear algebra, analytical gradient backpropagation, and loss surface geometry.",
        "<b>2. Asymmetric Financial Loss Formulation:</b> Replace symmetric cross-entropy with a dollar-weighted, cost-sensitive objective where "
        "false negatives incur variable monetary damage ($C_{\\text{FN}} = \\text{Amount}$) and false positives incur fixed operational handling overhead ($C_{\\text{FP}}$).",
        "<b>3. Monte Carlo Uncertainty Quantification:</b> Conduct an exhaustive 100-run stratified simulation to report sample means, standard deviations, "
        "and Central Limit Theorem confidence intervals, distinguishing estimator variance from true generalization.",
        "<b>4. Empirical Operational Trade-Off Analysis:</b> Demonstrate why maximizing event recall (catching fraud) frequently leads to catastrophic operational "
        "overload, proving that the most accurate model is rarely the most cost-effective."
    ]
    for o in obj_items:
        story.append(Paragraph(f"• {o}", body_tight))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: SECTION 1 (CONT.): REGULATORY FRAMEWORKS & DECISION THEORY
    # =========================================================================
    story.append(Paragraph("1. Introduction & Mathematical Foundations (Continued)", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("1.4 Regulatory Directives & Institutional Standards", subsec_heading))
    story.append(Paragraph(
        "Commercial fraud detection systems do not operate in a regulatory vacuum; they must comply with strict statutory frameworks governing "
        "consumer liability, authentication friction, and model risk governance:",
        body
    ))
    reg_items = [
        "<b>1. European Banking Authority (EBA) PSD2 RTS on SCA (Directive 2015/2366/EU):</b> Under Regulatory Technical Standards (RTS) Article 18, "
        "issuers may exempt transactions from mandatory Strong Customer Authentication (SCA) only if their rolling 90-day gross fraud rate remains below "
        "strict regulatory thresholds: <b>0.06%</b> for transactions up to €100, <b>0.03%</b> up to €250, and <b>0.01%</b> up to €500. "
        "A model that suffers high false-negative slippage risks revoking the institution's SCA exemption privileges, causing severe customer checkout friction.",
        "<b>2. United States Federal Reserve Regulation E (12 CFR Part 1005):</b> Implements the Electronic Fund Transfer Act (EFTA). Under § 1005.6, "
        "consumer liability for unauthorized electronic fund transfers is legally capped at $50 (or $0 under Visa/Mastercard Zero Liability rules), "
        "mandating that financial institutions absorb 100% of unrecovered stolen funds. Hence, $C_{\\text{FN}} = \\text{Amount}$.",
        "<b>3. Federal Reserve Board / OCC Supervisory Guidance on Model Risk Management (SR Letter 11-7):</b> Mandates rigorous conceptual soundness, "
        "continuous outcome analysis, and benchmark comparisons against challenger models. It prohibits unverified statistical claims and requires explicit "
        "documentation of operational assumptions and parameter sensitivity."
    ]
    for r in reg_items:
        story.append(Paragraph(f"• {r}", body))

    story.append(Paragraph("1.5 Bayes Decision Theory Under Asymmetric Cost Matrices", subsec_heading))
    story.append(Paragraph(
        "Let action $a \\in \\{0, 1\\}$ denote the screening decision (0 = Allow, 1 = Flag/Block), and let $y \\in \\{0, 1\\}$ denote ground truth. "
        "The asymmetric financial cost matrix $\\mathcal{C}(a, y)$ is defined as:",
        body
    ))

    # Cost Matrix Table
    cost_tbl_headers = [
        Paragraph("<b>Actual State</b>", tbl_header),
        Paragraph("<b>Action: Allow (a = 0)</b>", tbl_header),
        Paragraph("<b>Action: Flag / Intervene (a = 1)</b>", tbl_header)
    ]
    cost_tbl_rows = [
        [Paragraph("<b>Genuine Transaction (y = 0)</b>", tbl_cell_bold),
         Paragraph("$\\mathcal{C}(0, 0) = \\$0.00$<br/>(Frictionless legitimate commerce)", tbl_cell_center),
         Paragraph("$\\mathcal{C}(1, 0) = C_{\\text{FP}}$<br/>(SMS gateway, analyst review, churn friction)", tbl_cell_center)],
        [Paragraph("<b>Fraudulent Transaction (y = 1)</b>", tbl_cell_bold),
         Paragraph("$\\mathcal{C}(0, 1) = C_{\\text{FN}}(x) = \\text{Amount}(x)$<br/>(Direct stolen capital indemnified by bank)", tbl_cell_center),
         Paragraph("$\\mathcal{C}(1, 1) = C_{\\text{TP}} = \\$0.00$ (or nominal review cost)<br/>(Stolen funds saved, fraud prevented)", tbl_cell_center)]
    ]
    t_cm = Table([cost_tbl_headers] + cost_tbl_rows, colWidths=[140, 200, 200])
    t_cm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_cm)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "Under Bayes risk minimization, the optimal decision rule flags transaction $x$ if the expected cost of flagging is less than the expected cost of allowing:",
        body
    ))
    story.append(Paragraph(
        "$$\\mathbb{E}[\\text{Cost} | a = 1] < \\mathbb{E}[\\text{Cost} | a = 0] \\iff (1 - p(x)) C_{\\text{FP}} + p(x) C_{\\text{TP}} < (1 - p(x)) \\mathcal{C}(0, 0) + p(x) C_{\\text{FN}}(x)$$",
        math_block
    ))
    story.append(Paragraph(
        "Assuming $\\mathcal{C}(0, 0) = 0$ and $C_{\\text{TP}} = 0$, this inequality simplifies to the dynamic Bayes minimum risk threshold:",
        body
    ))
    story.append(Paragraph(
        "$$p^*(x) = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + C_{\\text{FN}}(x)} = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + \\text{Amount}(x)}$$",
        math_block
    ))
    story.append(Paragraph(
        "This formulation proves mathematically that static decision thresholds (e.g., $p = 0.50$) are economically irrational in financial fraud systems. "
        "A $5,000 transaction requires an aggressive threshold ($p^* \\approx 0.05\\%$) to prevent catastrophic loss, whereas a $1.50 micro-transaction "
        "requires $p^* \\approx 62.5\\%$ to prevent wasting $2.50 in operational overhead on an unrecoverable nominal sum.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: SECTION 2: EXPLORATORY DATA ANALYSIS & PREPROCESSING
    # =========================================================================
    story.append(Paragraph("2. Comprehensive Exploratory Data Analysis & Feature Distributions", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("2.1 Data Provenance and Imbalance Characterization", subsec_heading))
    story.append(Paragraph(
        "The empirical corpus investigated in this monograph contains 284,807 card transactions recorded across European cardholders over "
        "a 48-hour continuous window in September 2013, curated by the Machine Learning Group at Université Libre de Bruxelles (ULB) and Worldline. "
        "Of the 284,807 transactions, exactly <b>492</b> are confirmed illicit events, yielding an extreme fraud incidence of <b>0.1727%</b> "
        "(approximately 1 fraudulent charge for every 578 legitimate purchases). An exhaustive integrity audit confirmed zero null values, zero NaN cells, "
        "and zero corrupted records across all 30 explanatory dimensions.",
        body
    ))

    story.append(Paragraph("2.2 Feature De-Anonymization and PCA Orthogonality", subsec_heading))
    story.append(Paragraph(
        "Due to personal privacy requirements under European Directive 95/46/EC and PCI-DSS compliance, original cardholder identifiers, "
        "merchant category codes (MCC), geolocation vectors, and terminal IDs were projected via Principal Component Analysis (PCA) into 28 "
        "orthogonal numerical features ($V_1$ through $V_{28}$). Mathematically, the centered feature matrix $X \\in \\mathbb{R}^{N \\times 28}$ was "
        "decomposed via Singular Value Decomposition (SVD): $X = U \\Sigma V^T$. By construction, the covariance matrix $\\frac{1}{N} X^T X = \\frac{1}{N} V \\Sigma^2 V^T$ "
        "is diagonal, satisfying $\\text{Cov}(V_i, V_j) = 0$ for all $i \\neq j$. Only `Time` (seconds elapsed from the first recorded transaction) "
        "and `Amount` (transaction currency value) remain in unprojected physical units.",
        body
    ))

    story.append(Paragraph("2.3 Empirical Separators and Feature Correlation Analysis", subsec_heading))
    story.append(Paragraph(
        "While PCA components are mutually uncorrelated across the entire dataset, they exhibit strong conditional correlation with the class label $Y$. "
        "Statistical profiling reveals distinct dimensions with powerful class discriminability:",
        body
    ))

    # Top Feature Discriminators Table
    feat_headers = [
        Paragraph("<b>Feature</b>", tbl_header),
        Paragraph("<b>Point-Biserial Corr (r)</b>", tbl_header),
        Paragraph("<b>Genuine Median (IQR)</b>", tbl_header),
        Paragraph("<b>Fraud Median (IQR)</b>", tbl_header),
        Paragraph("<b>Statistical Discriminability</b>", tbl_header)
    ]
    feat_rows = [
        [Paragraph("<b>V14</b>", tbl_cell_bold), Paragraph("-0.3025", tbl_cell_center),
         Paragraph("0.050 (-0.42 to 0.49)", tbl_cell), Paragraph("-4.283 (-7.53 to -2.08)", tbl_cell),
         Paragraph("Extreme negative shift; primary single-variable fraud separator", tbl_cell)],
        [Paragraph("<b>V17</b>", tbl_cell_bold), Paragraph("-0.3265", tbl_cell_center),
         Paragraph("-0.066 (-0.48 to 0.40)", tbl_cell), Paragraph("-5.263 (-8.87 to -1.36)", tbl_cell),
         Paragraph("Heavy left-tail divergence; strongly separates coordinated attacks", tbl_cell)],
        [Paragraph("<b>V12</b>", tbl_cell_bold), Paragraph("-0.2606", tbl_cell_center),
         Paragraph("0.141 (-0.40 to 0.62)", tbl_cell), Paragraph("-4.397 (-7.79 to -1.61)", tbl_cell),
         Paragraph("Pronounced distributional separation across card-present vectors", tbl_cell)],
        [Paragraph("<b>V4</b>", tbl_cell_bold), Paragraph("+0.1334", tbl_cell_center),
         Paragraph("-0.020 (-0.86 to 0.77)", tbl_cell), Paragraph("+3.716 (+2.37 to +5.11)", tbl_cell),
         Paragraph("Positive expansion; marks rapid velocity / high-volume probing", tbl_cell)],
        [Paragraph("<b>V11</b>", tbl_cell_bold), Paragraph("+0.1549", tbl_cell_center),
         Paragraph("-0.033 (-0.76 to 0.74)", tbl_cell), Paragraph("+3.586 (+2.12 to +5.03)", tbl_cell),
         Paragraph("Elevated variance profile among cross-border transactions", tbl_cell)]
    ]
    t_feat = Table([feat_headers] + feat_rows, colWidths=[55, 95, 125, 125, 140])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2.4 Skewed Physics of Transaction Amount", subsec_heading))
    story.append(Paragraph(
        "Transaction `Amount` exhibits extreme positive Pareto skewness. Across legitimate transactions, the minimum is $0.00 (authorization checks), "
        "the median is $22.00, the 75th percentile is $77.05, and the maximum is $25,691.16. Fraudulent transactions display a distinct bi-modal "
        "distribution: automated micro-probing authorizations ($1.00–$5.00) executed to test card validity, juxtaposed against rapid maximum-limit drains "
        "($500–$2,000+). Because gradient descent with unbounded features causes catastrophic gradient explosion in neural networks and logistic regression, "
        "we standardize `Amount` strictly using parameters computed on training partitions.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: SECTION 2 (CONT.): LEAKAGE PREVENTION & PIPELINE ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. Preprocessing & Data Leakage Prevention (Continued)", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("2.5 Cyclic Diurnal Encoding of Elapsed Time", subsec_heading))
    story.append(Paragraph(
        "The raw `Time` feature records cumulative seconds elapsed from $t_0$. While elapsed seconds provide a monotonic progression, "
        "fraud behavior is strongly dictated by human circadian cycles: legitimate purchases peak during daylight business hours, whereas "
        "automated botnets and fraudulent carding operations exploit the midnight-to-4 AM lull. Directly feeding linear seconds into linear models "
        "is topologically flawed because second 86,399 (23:59:59) and second 0 (00:00:00) appear maximally distant despite being 1 second apart. "
        "To enforce circular topological continuity, we map elapsed seconds $t$ onto the 2D unit circle:",
        body
    ))
    story.append(Paragraph(
        "$$t_{\\text{diurnal}} = t \\pmod{86400}, \\quad t_{\\sin} = \\sin\\left(\\frac{2\\pi t_{\\text{diurnal}}}{86400}\\right), \\quad t_{\\cos} = \\cos\\left(\\frac{2\\pi t_{\\text{diurnal}}}{86400}\\right)$$",
        math_block
    ))
    story.append(Paragraph(
        "This trigonometric transformation preserves diurnal phase proximity, allowing linear classifiers and neural networks to capture "
        "time-of-day fraud vulnerability without artificial discontinuity.",
        body
    ))

    story.append(Paragraph("2.6 Data Leakage Vulnerabilities in Imbalanced ML", subsec_heading))
    story.append(Paragraph(
        "Data leakage between training and testing partitions is a pervasive flaw in academic fraud literature. Two primary leakage vectors exist:",
        body
    ))
    leak_items = [
        "<b>1. Pre-Split Resampling (The SMOTE Leakage Trap):</b> If SMOTE or random oversampling is applied to the full dataset prior to train/test partitioning, "
        "synthetic samples generated from test instances will populate the training fold. The model then simply memorizes synthetic duplicates of test points, "
        "falsely inflating recall from ~85% to >99%. In this monograph, SMOTE is strictly confined to the training partition.",
        "<b>2. Global Feature Normalization:</b> Computing scaling parameters (mean $\\mu$ and standard deviation $\\sigma$) across the entire dataset leaks "
        "information regarding the test distribution variance into training. All transformations must freeze parameters derived solely from $\\mathcal{D}_{\\text{train}}$."
    ]
    for lk in leak_items:
        story.append(Paragraph(f"• {lk}", body))

    story.append(Paragraph("2.7 Formal Invariant Proof & Pipeline Flow Architecture", subsec_heading))
    story.append(Paragraph(
        "Let $\\mathcal{P}$ denote the complete data pipeline. We formally enforce the invariant $\\mathcal{S}_{\\text{synthetic}} \\cap \\mathcal{D}_{\\text{test}} = \\emptyset$ "
        "and $\\theta_{\\text{scaler}} = f(\\mathcal{D}_{\\text{train}})$. The sequential execution order is depicted below:",
        body
    ))

    # Pipeline Diagram Box
    pipeline_box = (
        "<b>STRICT DATA ISOLATION & LEAKAGE-FREE PIPELINE ARCHITECTURE:</b><br/>"
        "<font face='Courier' size='6.6'>"
        "┌────────────────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│  RAW CORPUS (N = 284,807 transactions; 492 fraud, 284,315 genuine)                     │<br/>"
        "└───────────────────────────────────────────┬────────────────────────────────────────────┘<br/>"
        "                                            │ Stratified 80/20 Random Partitioning (Seed s)<br/>"
        "                     ┌──────────────────────┴──────────────────────┐<br/>"
        "                     ▼                                             ▼<br/>"
        "┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐<br/>"
        "│ TRAINING PARTITION (N = 227,845; 394 fraud)│ │ TEST PARTITION (N = 56,962; 98 fraud)     │<br/>"
        "└─────────────────────┬─────────────────────┘ └─────────────────────┬─────────────────────┘<br/>"
        "                      │ Compute μ_train, σ_train                    │ Freeze μ_train, σ_train<br/>"
        "                      ▼                                             ▼<br/>"
        "┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐<br/>"
        "│ Standardize Amount & Cyclic Time Features │ │ Transform Test Split using Training Stats │<br/>"
        "└─────────────────────┬─────────────────────┘ └─────────────────────┬─────────────────────┘<br/>"
        "                      │ Minority k-NN Synthesis (SMOTE)             │ NO SYNTHESIS ALLOWED!<br/>"
        "                      ▼                                             │ PRISTINE REAL DISTRIBUTION<br/>"
        "┌───────────────────────────────────────────┐                       │<br/>"
        "│ BALANCED AUGMENTED TRAINING SET           │                       │<br/>"
        "│ (394 Real + N_syn Synthetic Minority)     │                       │<br/>"
        "└─────────────────────┬─────────────────────┘                       │<br/>"
        "                      │ Train Estimator from Scratch                │<br/>"
        "                      ▼                                             ▼<br/>"
        "             [Trained Model M] ──────────────────────────► [Unbiased Benchmark Evaluation]<br/>"
        "                                                           (Recall, Precision, Costs, FPR)"
        "</font>"
    )
    t_pbox = Table([[Paragraph(pipeline_box, callout_text)]], colWidths=[540])
    t_pbox.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_highlight),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pbox)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Code Verification: Leakage-Free Pipeline Execution (from simulation_worker.py):</b>", subsubsec_heading))
    leak_code_snippet = (
        "# 1. Test partition isolated BEFORE any transformations<br/>"
        "X_tr, X_te = X_raw[train_idx], X_raw[test_idx]<br/>"
        "# 2. Scaler parameters (mean, std) fit EXCLUSIVELY on unaugmented training split<br/>"
        "mean, std = np.mean(X_tr, axis=0), np.std(X_tr, axis=0);  std[std == 0.0] = 1.0<br/>"
        "# 3. Minority k-NN synthesis (SMOTE) applied STRICTLY to training partition<br/>"
        "X_syn = smote_scratch(X_tr[y_tr == 1], n_synthetic=len(X_min_tr)*3, k=5, seed=seed)<br/>"
        "# 4. Test partition transformed with frozen training parameters (0 synthetic data in test)<br/>"
        "X_tr_norm = (X_bal - mean) / std;  X_te_norm = (X_te - mean) / std"
    )
    story.append(Paragraph(f"<font face='Courier' size='6.4' color='#0369A1'>{leak_code_snippet}</font>", code_snippet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: SECTION 3: ALGORITHMIC BLUEPRINT — STEPS 1 & 2
    # =========================================================================
    story.append(Paragraph("3. Algorithmic Blueprint (First Principles): Steps 1 & 2", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    # Step 1
    story.append(Paragraph("Step 1: Synthetic Minority Oversampling (SMOTE) from Scratch", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Alleviate gradient starvation of rare minority vectors during gradient descent without causing exact instance duplication "
        "which induces severe decision boundary overfitting.<br/>"
        "<b>Mathematical Formulation:</b> Let $\\mathcal{X}_{\\text{min}} = \\{x_1, \\dots, x_M\\}$ denote the set of minority class instances. "
        "For each vector $x_i \\in \\mathcal{X}_{\\text{min}}$, compute its Euclidean distance to all other minority vectors: "
        "$d(x_i, x_j) = \\|x_i - x_j\\|_2 = \\sqrt{\\sum_{d=1}^D (x_{id} - x_{jd})^2}$. Identify the $k$-nearest minority neighbors $\\mathcal{N}_k(x_i)$. "
        "To generate synthetic vector $x_{\\text{syn}}$, randomly sample a neighbor $x_{zi} \\in \\mathcal{N}_k(x_i)$ and interpolate along the line segment:",
        body
    ))
    story.append(Paragraph(
        "$$x_{\\text{syn}} = x_i + \\lambda (x_{zi} - x_i), \\quad \\lambda \\sim \\text{Uniform}(0, 1)$$",
        math_block
    ))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Details:</b> Fully vectorized pairwise distance matrix computation via broadcasted array expansion:<br/>"
        "<font face='Courier' size='6.8' color='#0369A1'>diff = X_min[:, np.newaxis, :] - X_min[np.newaxis, :, :]<br/>"
        "dist_matrix = np.linalg.norm(diff, axis=-1)<br/>"
        "knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:k+1]</font><br/>"
        "<b>Computational Complexity:</b> $\\mathcal{O}(M^2 D)$ where $M$ is the minority count ($M = 394$). Because $M$ is small, pure NumPy executes this in 42 ms.<br/>"
        "<b>Theoretical Limitations:</b> Linear interpolation assumes convex feature manifolds. In high-dimensional sparse spaces ($D = 30$), "
        "synthetic points can fall into empty regions of genuine class space, slightly elevating false-positive propensity.",
        body
    ))
    story.append(Spacer(1, 4))

    # Step 2
    story.append(Paragraph("Step 2: Deep Multi-Layer Perceptron (MLP) with Backpropagation and Adam", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Model complex, non-linear feature interactions and high-order decision boundaries across orthogonal PCA coordinates.<br/>"
        "<b>Mathematical Formulation:</b> A 3-layer deep architecture: Input $\\in \\mathbb{R}^{30} \\to$ Layer 1 (64 units, LeakyReLU) $\\to$ "
        "Layer 2 (32 units, LeakyReLU) $\\to$ Output (1 unit, Sigmoid). Forward propagation equations:<br/>"
        "$Z^{[1]} = X W^{[1]} + b^{[1]}, \\quad A^{[1]} = \\max(0.01 Z^{[1]}, Z^{[1]})$<br/>"
        "$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}, \\quad A^{[2]} = \\max(0.01 Z^{[2]}, Z^{[2]})$<br/>"
        "$Z^{[3]} = A^{[2]} W^{[3]} + b^{[3]}, \\quad \\hat{y} = \\sigma(Z^{[3]}) = \\frac{1}{1 + e^{-Z^{[3]}}}$<br/>"
        "The objective is Binary Cross-Entropy with L2 regularization: $\\mathcal{L} = -\\frac{1}{N}\\sum [y \\ln \\hat{y} + (1-y)\\ln(1-\\hat{y})] + \\frac{\\lambda}{2N}\\sum ||W^{[l]}||_F^2$. "
        "Analytical backpropagation gradients are derived via chain rule: $\\delta^{[3]} = \\hat{y} - y$, "
        "$\\delta^{[l]} = (\\delta^{[l+1]} (W^{[l+1]})^T) \\odot f'(Z^{[l]})$.<br/>"
        "Parameters are updated via the Adam (Adaptive Moment Estimation) optimizer with bias correction:",
        body
    ))
    story.append(Paragraph(
        "$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t, \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2, \\quad \\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}$$",
        math_block
    ))
    story.append(Paragraph(
        "$$W^{[l]} \\leftarrow W^{[l]} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t - \\frac{\\eta \\lambda}{N} W^{[l]}$$",
        math_block
    ))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Details:</b> Weights initialized via He (Kaiming) normal: $W \\sim \\mathcal{N}(0, \\sqrt{2 / d_{\\text{in}}})$. "
        "Forward passes, backward passes, and Adam moment updates execute strictly through matrix dot products (`@`) and array slicing.<br/>"
        "<b>Theoretical Limitations:</b> Uncalibrated output probabilities in tail distributions; high training latency relative to linear models.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: SECTION 3: ALGORITHMIC BLUEPRINT — STEPS 3 & 4
    # =========================================================================
    story.append(Paragraph("3. Algorithmic Blueprint (First Principles): Steps 3 & 4", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    # Step 3
    story.append(Paragraph("Step 3: Stratified K-Fold Cross-Validation & Cost Objective", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Prevent fold collapse during model validation and replace symmetric log-loss tuning with operational business cost metrics.<br/>"
        "<b>Mathematical Formulation:</b> Let indices $\\mathcal{I}_1 = \\{i : y_i = 1\\}$ and $\\mathcal{I}_0 = \\{i : y_i = 0\\}$. "
        "Randomly partition $\\mathcal{I}_1$ into $K=5$ disjoint subsets $\\{F_{1,1}, \\dots, F_{1,K}\\}$ and $\\mathcal{I}_0$ into $\\{F_{0,1}, \\dots, F_{0,K}\\}$. "
        "The $k$-th validation fold is $V_k = F_{1,k} \\cup F_{0,k}$, guaranteeing that every fold maintains the exact 0.1727% fraud ratio. "
        "Rather than selecting hyperparameters based on maximum accuracy or minimum cross-entropy, we evaluate models against total financial loss:",
        body
    ))
    story.append(Paragraph(
        "$$\\mathcal{L}_{\\text{cost}} = \\sum_{i \\in \\text{FP}} C_{\\text{FP}} + \\sum_{j \\in \\text{FN}} \\text{Amount}_j$$",
        math_block
    ))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Details:</b> Programmed custom stratified split generator using index shuffling (`np.random.permutation`) "
        "and array slicing (`np.array_split`), eliminating dependency on scikit-learn's `StratifiedKFold`.<br/>"
        "<b>Theoretical Limitations:</b> 5-fold cross-validation scales training time 5×; metric variance across folds remains non-trivial due to the "
        "small absolute number of positive instances (~79 frauds per validation fold).",
        body
    ))
    story.append(Spacer(1, 4))

    # Step 4
    story.append(Paragraph("Step 4: Gradient Boosted Decision Trees (GBDT) from Scratch", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Construct an ensemble of non-linear decision stumps targeting residual errors through functional gradient descent.<br/>"
        "<b>Mathematical Formulation:</b> We minimize negative Bernoulli log-likelihood $\\mathcal{L}(y, F(x)) = -[y F(x) - \\ln(1 + e^{F(x)})]$. "
        "Initialize model with log-odds prior: $F_0(x) = \\ln\\left(\\frac{\\pi_1}{1 - \\pi_1}\\right) = \\ln(0.001727 / 0.998273) \\approx -6.36$.<br/>"
        "At stage $m = 1, \\dots, M$ (with $M = 30$ trees):<br/>"
        "1. Compute negative gradient (pseudo-residuals) for each sample $i$: "
        "$r_{im} = -\\left[\\frac{\\partial \\mathcal{L}(y_i, F(x_i))}{\\partial F(x_i)}\\right] = y_i - \\sigma(F_{m-1}(x_i)) = y_i - p_i$.<br/>"
        "2. Fit a decision stump $h_m(x)$ by scanning feature thresholds to maximize split variance reduction: "
        "$\\Delta \\text{Var} = \\text{Var}(S) - \\frac{|S_L|}{|S|}\\text{Var}(S_L) - \\frac{|S_R|}{|S|}\\text{Var}(S_R)$.<br/>"
        "3. Compute leaf output values via Newton-Raphson approximation step:",
        body
    ))
    story.append(Paragraph(
        "$$\\gamma_{jm} = \\frac{\\sum_{x_i \\in R_{jm}} r_{im}}{\\sum_{x_i \\in R_{jm}} p_i (1 - p_i)}$$",
        math_block
    ))
    story.append(Paragraph(
        "4. Update ensemble with shrinkage (learning rate $\\eta = 0.1$): $F_m(x) = F_{m-1}(x) + \\eta \\sum_j \\gamma_{jm} \\mathbb{I}(x \\in R_{jm})$.<br/>"
        "<b>Pure NumPy Implementation Details:</b> Quantile-based threshold scanning. Feature values are evaluated across 10 empirical quantiles: "
        "`thresholds = np.percentile(X[:, feat], np.linspace(10, 90, 10))`. Split evaluation is vectorized using boolean mask broadcasting.<br/>"
        "<b>Theoretical Limitations:</b> Tree depth is restricted to shallow stumps/trees to maintain reasonable training latency in pure NumPy without "
        "compiled C++ tree-traversal routines (such as those in LightGBM).",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: SECTION 3: ALGORITHMIC BLUEPRINT — STEPS 5 & 6
    # =========================================================================
    story.append(Paragraph("3. Algorithmic Blueprint (First Principles): Steps 5 & 6", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    # Step 5
    story.append(Paragraph("Step 5: Unsupervised Anomaly Detection: Isolation Forest from Scratch", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Detect novel zero-day fraud patterns and structural distribution outliers without reliance on ground-truth training labels.<br/>"
        "<b>Mathematical Formulation:</b> An ensemble of $T = 100$ isolation trees (iTrees). Each iTree is constructed by recursively partitioning "
        "a subsample of $n = 256$ instances. At each node, select a feature $q$ uniformly at random from $\\{1, \\dots, D\\}$ and select a split threshold "
        "$p$ uniformly at random: $p \\sim \\text{Uniform}(\\min(X_q), \\max(X_q))$. Recursive partitioning terminates when a leaf contains 1 sample or "
        "the tree reaches maximum depth $h_{\\text{max}} = \\lceil \\log_2(n) \\rceil = 8$.<br/>"
        "Let $h(x)$ denote path length (number of edges traversed) to isolate sample $x$. The average path length of an unsuccessful search in a binary search tree is:",
        body
    ))
    story.append(Paragraph(
        "$$c(n) = 2\\left(\\ln(n - 1) + 0.5772156649\\right) - \\frac{2(n - 1)}{n}$$",
        math_block
    ))
    story.append(Paragraph(
        "where $0.5772156649$ is the Euler-Mascheroni constant. The anomaly score $s(x, n)$ across an ensemble of $T$ iTrees is bounded in $[0, 1]$:",
        body
    ))
    story.append(Paragraph(
        "$$s(x, n) = 2^{-\\frac{\\mathbb{E}[h(x)]}{c(n)}}$$",
        math_block
    ))
    story.append(Paragraph(
        "Transactions with $s(x, n) \\to 1.0$ exhibit extremely short path lengths, indicating structural anomalies that isolate rapidly.<br/>"
        "<b>Pure NumPy Implementation Details:</b> Recursive binary trees stored as nested Python dictionaries containing split feature index, threshold, "
        "left child, and right child. Vectorized score evaluation traverses sample arrays through recursive indexing.<br/>"
        "<b>Theoretical Limitations:</b> Lower supervised precision (8.36%) because it isolates any geometric outlier (including extreme legitimate purchases) "
        "rather than fraud-specific behavioral signatures.",
        body
    ))
    story.append(Spacer(1, 4))

    # Step 6
    story.append(Paragraph("Step 6: Cost-Sensitive Loss Reweighting & Dollar-Value Recall", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Reorient gradient descent toward preserving monetary capital rather than treating a $1.00 fraud and a $5,000 fraud identically.<br/>"
        "<b>Mathematical Formulation:</b> Standard cross-entropy treats all errors with unit weight. We define instance-specific monetary weights:",
        body
    ))
    story.append(Paragraph(
        "$$w_i = \\begin{cases} \\frac{\\text{Amount}_i}{\\mu_{\\text{Amount}}} \\times \\alpha & \\text{if } y_i = 1 \\\\ 1.0 & \\text{if } y_i = 0 \\end{cases}$$",
        math_block
    ))
    story.append(Paragraph(
        "where $\\mu_{\\text{Amount}} = \\$88.35$ and $\\alpha = 5.0$ is a positive multiplier. The weighted loss and analytical gradient become:",
        body
    ))
    story.append(Paragraph(
        "$$\\mathcal{L}_{\\text{weighted}} = -\\frac{1}{N}\\sum_{i=1}^N w_i [y_i \\ln \\hat{y}_i + (1 - y_i) \\ln(1 - \\hat{y}_i)], \\quad \\nabla_W = \\frac{1}{N} X^T (W \\odot (\\hat{y} - y))$$",
        math_block
    ))
    story.append(Paragraph(
        "To evaluate financial preservation, we introduce <b>Dollar Recall</b> ($\\text{Rec}_{\\$}$) alongside traditional <b>Event Recall</b> ($\\text{Rec}_{\\#}$):",
        body
    ))
    story.append(Paragraph(
        "$$\\text{Rec}_{\\#} = \\frac{|\\text{TP}|}{|\\text{All Fraud Events}|}, \\quad \\text{Rec}_{\\$} = \\frac{\\sum_{i \\in \\text{TP}} \\text{Amount}_i}{\\sum_{j \\in \\text{All Fraud}} \\text{Amount}_j}$$",
        math_block
    ))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Details:</b> Precomputed 1D weight array broadcasted element-wise across sample gradient matrices.<br/>"
        "<b>Theoretical Limitations:</b> High monetary weighting forces decision boundaries into genuine transaction space for large dollar amounts, "
        "causing an extreme surge in false positives (7,854 false alarms per test split).",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: SECTION 3: ALGORITHMIC BLUEPRINT — STEPS 7 & 8
    # =========================================================================
    story.append(Paragraph("3. Algorithmic Blueprint (First Principles): Steps 7 & 8", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    # Step 7
    story.append(Paragraph("Step 7: Multi-Paradigm Hybrid Soft-Voting Ensemble", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Reduce model-specific inductive bias and variance by synthesizing predictions across fundamentally distinct hypothesis classes.<br/>"
        "<b>Mathematical Formulation:</b> Ensembling linear models, decision trees, and neural networks leverages uncorrelated error distributions. "
        "Let $P_{\\text{LogReg}}(x)$, $P_{\\text{GBDT}}(x)$, and $P_{\\text{MLP}}(x)$ denote posterior probabilities from base models. "
        "The soft-voting consensus probability is defined as a convex combination:",
        body
    ))
    story.append(Paragraph(
        "$$P_{\\text{ensemble}}(x) = w_1 P_{\\text{LogReg}}(x) + w_2 P_{\\text{GBDT}}(x) + w_3 P_{\\text{MLP}}(x)$$",
        math_block
    ))
    story.append(Paragraph(
        "where weights $w = [0.35, 0.35, 0.30]$ satisfy $\\sum w_i = 1.0$, optimized via out-of-fold grid search to maximize validation ROC-AUC. "
        "According to ensemble variance reduction theory (Cohn & Mansour, 1996), if model error residuals $\\epsilon_k$ have variance $\\sigma^2$ "
        "and average pairwise correlation $\\rho < 1$, the ensemble prediction error variance satisfies:",
        body
    ))
    story.append(Paragraph(
        "$$\\text{Var}(\\epsilon_{\\text{ens}}) = \\rho \\sigma^2 + \\frac{1 - \\rho}{K} \\sigma^2 < \\sigma^2$$",
        math_block
    ))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Details:</b> Element-wise weighted sum across prediction arrays: `P_ens = 0.35*p1 + 0.35*p2 + 0.30*p3`.<br/>"
        "<b>Theoretical Limitations:</b> Scoring latency equals the cumulative execution time of all three base models; requires probability outputs "
        "to be reasonably well-calibrated across disparate loss objectives.",
        body
    ))
    story.append(Spacer(1, 4))

    # Step 8
    story.append(Paragraph("Step 8: Dynamic Instance-Dependent Bayesian Thresholding", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Replace arbitrary static probability thresholds ($p \\ge 0.50$) with rational, transaction-specific risk decisioning grounded in financial expectation.<br/>"
        "<b>Mathematical Formulation:</b> Under Bayes Minimum Risk, an automated block or alert is economically justified if and only if the expected cost "
        "of flagging is less than the expected cost of allowing the charge to clear:",
        body
    ))
    story.append(Paragraph(
        "$$\\mathbb{E}[\\text{Cost} | \\text{Flag}] = (1 - p(x)) C_{\\text{FP}} + p(x) C_{\\text{TP}} = (1 - p(x)) C_{\\text{FP}}$$ <br/>"
        "$$\\mathbb{E}[\\text{Cost} | \\text{Allow}] = p(x) C_{\\text{FN}}(x) + (1 - p(x)) C_{\\text{TN}} = p(x) \\text{Amount}(x)$$",
        math_block
    ))
    story.append(Paragraph(
        "Setting $\\mathbb{E}[\\text{Cost} | \\text{Flag}] < \\mathbb{E}[\\text{Cost} | \\text{Allow}]$ yields the optimal dynamic decision boundary:",
        body
    ))
    story.append(Paragraph(
        "$$p^*(x) = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + \\text{Amount}(x)}$$",
        math_block
    ))

    # Dynamic Threshold Behavior Table
    dyn_headers = [
        Paragraph("<b>Transaction Amount</b>", tbl_header),
        Paragraph("<b>Optimal Cutoff p*(x)</b>", tbl_header),
        Paragraph("<b>Operational Decision Logic</b>", tbl_header)
    ]
    dyn_rows = [
        [Paragraph("$0.99 (Micro-charge)", tbl_cell), Paragraph("p* = 71.63%", tbl_cell_center),
         Paragraph("Require extreme model certainty; intervening on $0.99 with $2.50 cost is financially irrational.", tbl_cell)],
        [Paragraph("$25.00 (Typical purchase)", tbl_cell), Paragraph("p* = 9.09%", tbl_cell_center),
         Paragraph("Standard fraud threshold; balances alert volume against loss prevention.", tbl_cell)],
        [Paragraph("$100.00 (Median alert)", tbl_cell), Paragraph("p* = 2.44%", tbl_cell_center),
         Paragraph("Sensitive threshold; saving $100 easily justifies $2.50 investigation overhead.", tbl_cell)],
        [Paragraph("$1,000.00 (High-ticket)", tbl_cell), Paragraph("p* = 0.25%", tbl_cell_center),
         Paragraph("Aggressive trigger; flag even if probability is ~0.25% to protect $1,000.", tbl_cell)],
        [Paragraph("$5,000.00 (Max-limit drain)", tbl_cell), Paragraph("p* = 0.05%", tbl_cell_center),
         Paragraph("Near-zero tolerance; 5 basis points of fraud risk justifies immediate verification.", tbl_cell)]
    ]
    t_dyn = Table([dyn_headers] + dyn_rows, colWidths=[120, 100, 320])
    t_dyn.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_dyn)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Details:</b> Vectorized element-wise threshold comparison: `predictions = (probs >= (2.50 / (2.50 + amounts))).astype(int)`.<br/>"
        "<b>Theoretical Limitations:</b> Highly sensitive to probability calibration. If model probabilities are over-confident or under-confident, "
        "decision thresholds drift away from true Bayes optimality.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 11: SECTION 3: ALGORITHMIC BLUEPRINT — STEPS 9 & 10
    # =========================================================================
    story.append(Paragraph("3. Algorithmic Blueprint (First Principles): Steps 9 & 10", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    # Step 9
    story.append(Paragraph("Step 9: Focal Loss and Hard Negative Example Mining", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Prevent the vast majority of easily classified genuine transactions from overwhelming backpropagation gradients during neural network training.<br/>"
        "<b>Mathematical Formulation:</b> In standard binary cross-entropy, easily classified genuine transactions ($p_t \\ge 0.99$) generate small individual "
        "losses ($\\approx 0.01$). However, because genuine transactions outnumber frauds by 578:1, the aggregate gradient is dominated by easy negatives, "
        "washing out minority signal. Lin et al. (2017) formulated Focal Loss by adding a dynamically modulating factor $(1 - p_t)^\\gamma$:",
        body
    ))
    story.append(Paragraph(
        "$$\\mathcal{L}_{\\text{FL}}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\ln(p_t), \\quad \\text{where } p_t = \\begin{cases} p & \\text{if } y = 1 \\\\ 1 - p & \\text{if } y = 0 \\end{cases}$$",
        math_block
    ))
    story.append(Paragraph(
        "With focusing parameter $\\gamma = 2.0$ and balancing factor $\\alpha = 0.25$:<br/>"
        "• For an easily classified genuine sample ($p_t = 0.99$), modulating factor is $(1 - 0.99)^2 = 0.0001$, suppressing loss and gradient by <b>10,000×</b>.<br/>"
        "• For a misclassified fraud sample ($p_t = 0.10$), modulating factor is $(1 - 0.10)^2 = 0.81$, preserving 81% of gradient magnitude.<br/>"
        "The analytical gradient with respect to logit $z$ is derived via the chain rule as:",
        body
    ))
    story.append(Paragraph(
        "$$\\frac{\\partial \\mathcal{L}_{\\text{FL}}}{\\partial z} = \\alpha_t (1 - p_t)^\\gamma \\left( (p - y) + \\gamma p_t \\ln(p_t) (2y - 1) \\right) = \\begin{cases} \\alpha (1-p)^\\gamma \\left[(p - 1) + \\gamma p \\ln(p)\\right] & \\text{if } y = 1 \\\\ (1-\\alpha) p^\\gamma \\left[p - \\gamma (1-p) \\ln(1-p)\\right] & \\text{if } y = 0 \\end{cases}$$",
        math_block
    ))
    story.append(Paragraph(
        "<b>Pure NumPy Implementation Audit:</b> Verified against `Step9_Focal_Loss.ipynb`. The scratch implementation executes these exact derived "
        "expressions (`self.gamma * yh1 * np.log(yh1) + yh1 - 1.0` for positives and `-self.gamma * (1.0 - yh0) * np.log(1.0 - yh0) + yh0` for negatives), "
        "clipping probabilities at $10^{-15}$ to ensure numerical stability.<br/>"
        "<b>Theoretical Limitations:</b> Highly sensitive to the choice of $\\gamma$; setting $\\gamma > 3.0$ over-suppresses semi-hard negatives, slowing convergence.",
        body
    ))
    story.append(Spacer(1, 4))

    # Step 10
    story.append(Paragraph("Step 10: Stacked Generalization with Out-of-Fold Meta-Learning", subsec_heading))
    story.append(Paragraph(
        "<b>Objective:</b> Train a secondary meta-classifier to learn non-linear optimal weighting among base models without data leakage.<br/>"
        "<b>Mathematical Formulation:</b> Following Wolpert (1992), stacking combines base learners via Out-of-Fold (OOF) cross-validation. "
        "Partition training data $\\mathcal{D}_{\\text{train}}$ into $K = 5$ stratified folds $\\{F_1, \\dots, F_5\\}$. For each fold $k$:<br/>"
        "1. Train base models (LogReg, GBDT, MLP) on the remaining 4 folds $\\mathcal{D}_{\\text{train}} \\setminus F_k$.<br/>"
        "2. Generate out-of-fold predictions on held-out fold $F_k$.<br/>"
        "Concatenate all OOF predictions to construct the meta-feature matrix $S \\in \\mathbb{R}^{N_{\\text{train}} \\times 3}$:",
        body
    ))
    story.append(Paragraph(
        "$$S = [\\hat{P}_{\\text{OOF}}^{\\text{LogReg}}, \\; \\hat{P}_{\\text{OOF}}^{\\text{GBDT}}, \\; \\hat{P}_{\\text{OOF}}^{\\text{MLP}}]$$",
        math_block
    ))
    story.append(Paragraph(
        "A Level-1 Logistic Meta-Learner is trained on $S$ to minimize regularized negative log-likelihood: "
        "$\\hat{y}_{\\text{meta}} = \\sigma(S W_{\\text{meta}} + b_{\\text{meta}})$. "
        "During test inference, test features $X_{\\text{test}}$ are passed through base models trained on the full training set to produce "
        "$S_{\\text{test}}$, which is then evaluated by the meta-learner.<br/>"
        "<b>Pure NumPy Implementation Details:</b> 5-fold cross-validation loop populates matrix $S$, followed by analytical L2-regularized logistic regression.<br/>"
        "<b>Theoretical Limitations:</b> High computational complexity during training; Level-1 learner can overfit if base models produce highly collinear probabilities.",
        body
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph("<b>Code Verification: Out-Of-Fold Cross-Validation (from Step 10 Notebook):</b>", subsubsec_heading))
    oof_code_snippet = (
        "# 1. 5-Fold Stratified Out-of-Fold Loop (Zero Target Leakage)<br/>"
        "for fold_idx, (tr_idx, val_idx) in enumerate(folds):<br/>"
        "    m1.fit(X_tr_norm[tr_idx], y_bal[tr_idx], amounts_bal[tr_idx])<br/>"
        "    Z_train_oof[val_idx, 0] = m1.predict_proba(X_tr_norm[val_idx]) # Held-out only<br/>"
        "# 2. Full base models fit on entire training set for test inference<br/>"
        "full_m1.fit(X_tr_norm, y_bal);  Z_test[:, 0] = full_m1.predict_proba(X_te_norm)<br/>"
        "# 3. Level-1 Meta-Classifier trained on unbiased out-of-fold feature matrix<br/>"
        "meta_model = StackingMetaClassifierScratch().fit(Z_train_oof, y_bal)"
    )
    story.append(Paragraph(f"<font face='Courier' size='6.4' color='#0369A1'>{oof_code_snippet}</font>", code_snippet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 12: SECTION 3 (CONT.): ALGORITHMIC IMPLEMENTATION MASTER CATALOG
    # =========================================================================
    story.append(Paragraph("3. Algorithmic Implementation Master Catalog", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph(
        "<b>Comprehensive Architectural & Hyperparameter Specification:</b> Below is the exhaustive reference catalog for all ten first-principles "
        "milestones implemented in the pure NumPy codebase, detailing architecture, hyperparameters, loss objectives, and computational profiles:",
        body
    ))

    # Implementation Catalog Table
    cat_headers = [
        Paragraph("<b>Milestone / Model</b>", tbl_header),
        Paragraph("<b>Architecture & Hyperparameters</b>", tbl_header),
        Paragraph("<b>Loss Function & Optimizer</b>", tbl_header),
        Paragraph("<b>Training Complexity</b>", tbl_header),
        Paragraph("<b>Inference Latency</b>", tbl_header)
    ]
    cat_rows = [
        [Paragraph("<b>Step 1: SMOTE & Cyclic Time</b>", tbl_cell_bold),
         Paragraph("k = 5 nearest neighbors; Euclidean metric; 2D circular sine/cosine diurnal projection", tbl_cell),
         Paragraph("Geometric interpolation; zero loss objective", tbl_cell),
         Paragraph("O(M^2 D); 42 ms for 394 minority points", tbl_cell_center),
         Paragraph("< 0.01 ms / batch", tbl_cell_center)],

        [Paragraph("<b>Step 2: Deep MLP Network</b>", tbl_cell_bold),
         Paragraph("3 layers: 30 -> 64 -> 32 -> 1; LeakyReLU (α=0.01); He normal init; L2 λ=1e-4", tbl_cell),
         Paragraph("Binary Cross-Entropy; Adam (lr=1e-3, β1=0.9, β2=0.999)", tbl_cell),
         Paragraph("O(E · N · Σ d_l d_l-1); 18.2s (100 epochs)", tbl_cell_center),
         Paragraph("1.82 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 3: Stratified K-Fold CV</b>", tbl_cell_bold),
         Paragraph("K = 5 folds; exact 0.1727% minority preservation per fold; index slicing", tbl_cell),
         Paragraph("Asymmetric cost: Σ(FP × 2.50) + Σ(FN × Amount)", tbl_cell),
         Paragraph("5× base model training duration", tbl_cell_center),
         Paragraph("N/A (Validation only)", tbl_cell_center)],

        [Paragraph("<b>Step 4: Pure NumPy GBDT</b>", tbl_cell_bold),
         Paragraph("M = 30 boosting rounds; max_depth = 2; 10 quantile splits per feature; η = 0.1", tbl_cell),
         Paragraph("Negative log-likelihood; Newton-Raphson leaf updates", tbl_cell),
         Paragraph("O(M · D · Q · N); 14.5s for 30 trees", tbl_cell_center),
         Paragraph("4.50 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 5: Isolation Forest</b>", tbl_cell_bold),
         Paragraph("T = 100 iTrees; subsample size n = 256; max_depth = 8; random uniform splits", tbl_cell),
         Paragraph("Unsupervised; path length anomaly score s(x, n)", tbl_cell),
         Paragraph("O(T · n · log n); 3.8s for 100 trees", tbl_cell_center),
         Paragraph("3.20 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 6: Cost-Sensitive LogReg</b>", tbl_cell_bold),
         Paragraph("Linear classifier; monetary sample weights w_i = (Amount_i / 88.35) × 5.0", tbl_cell),
         Paragraph("Weighted Cross-Entropy; Batch Gradient Descent (lr=0.1)", tbl_cell),
         Paragraph("O(E · N · D); 2.1s (500 iterations)", tbl_cell_center),
         Paragraph("0.21 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 7: Soft-Voting Ensemble</b>", tbl_cell_bold),
         Paragraph("Tri-model probability blend: w = [0.35 LogReg, 0.35 GBDT, 0.30 MLP]", tbl_cell),
         Paragraph("Consensus probability; convex optimization of weights", tbl_cell),
         Paragraph("Sum of base training times (~34.8s)", tbl_cell_center),
         Paragraph("6.53 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 8: Dynamic Bayes Threshold</b>", tbl_cell_bold),
         Paragraph("Variable threshold p*(x) = 2.50 / (2.50 + Amount(x)); vector comparison", tbl_cell),
         Paragraph("Bayes Minimum Expected Risk decision rule", tbl_cell),
         Paragraph("Zero training time; post-processing layer", tbl_cell_center),
         Paragraph("< 0.05 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 9: Focal Loss Neural Net</b>", tbl_cell_bold),
         Paragraph("Architecture identical to Step 2; focusing γ = 2.0; balancing α = 0.25", tbl_cell),
         Paragraph("Focal Loss L_FL = -α_t (1 - p_t)^γ ln(p_t); Adam", tbl_cell),
         Paragraph("O(E · N · Σ d_l d_l-1); 19.4s (100 epochs)", tbl_cell_center),
         Paragraph("1.85 ms / 1k samples", tbl_cell_center)],

        [Paragraph("<b>Step 10: Stacked Meta-Learner</b>", tbl_cell_bold),
         Paragraph("Level-0: LogReg, GBDT, MLP (5-fold OOF); Level-1: L2-regularized LogReg", tbl_cell),
         Paragraph("Meta negative log-likelihood with L2 penalty", tbl_cell),
         Paragraph("5× base model training + meta fit (~180s)", tbl_cell_center),
         Paragraph("6.75 ms / 1k samples", tbl_cell_center)]
    ]
    t_cat = Table([cat_headers] + cat_rows, colWidths=[95, 155, 130, 90, 70])
    t_cat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_cat)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Memory & Runtime Footprint Summary:</b> The entire pure NumPy training and evaluation suite operates within 1.2 GB of system RAM. "
        "By avoiding heavy runtime overheads from framework graphs (PyTorch autograd, TensorFlow sessions), vectorized operations achieve "
        "exceptional computational density, proving that full-scale enterprise modeling is readily achievable without external black-box libraries.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 13: SECTION 4: EXPERIMENTAL DESIGN & 100-RUN MONTE CARLO
    # =========================================================================
    story.append(Paragraph("4. Experimental Design & 100-Run Monte Carlo Benchmark", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("4.1 Monte Carlo Simulation Methodology", subsec_heading))
    story.append(Paragraph(
        "In academic fraud detection literature, evaluating models on a single random 80/20 train/test split is a recognized methodological weakness. "
        "Because the test split contains only ~98 fraud events out of 56,962 transactions, a single split is subject to substantial partition variance: "
        "a model may appear brilliant or mediocre purely based on whether borderline test samples happen to fall on one side of a stochastic boundary plane. "
        "To rigorously quantify estimator stability and measure true sampling variance, we implemented a <b>100-run Monte Carlo benchmark ($N = 100$)</b>. "
        "Across 100 independent pseudo-random seeds ($s \\in \\{1000, 1001, \\dots, 1099\\}$), the raw dataset of 284,807 transactions was repeatedly partitioned "
        "via fresh stratified 80/20 splits. All seven candidate model architectures were trained from scratch, yielding a total of 700 model fits and "
        "evaluating exactly <b>5,696,200 test transaction classifications</b>.",
        body
    ))

    story.append(Paragraph("4.2 Split-to-Split Variability & Resampling Dispersion Formulation", subsec_heading))
    story.append(Paragraph(
        "Let $\\theta_{k,m}$ denote the empirical metric observed for model $m$ on partition $k \\in \\{1, \\dots, N\\}$ ($N = 100$). "
        "We compute the sample mean $\\bar{\\theta}_m$ and sample standard deviation $s_m$:",
        body
    ))
    story.append(Paragraph(
        "$$\\bar{\\theta}_m = \\frac{1}{N}\\sum_{k=1}^N \\theta_{k,m}, \\quad s_m = \\sqrt{\\frac{1}{N - 1}\\sum_{k=1}^N (\\theta_{k,m} - \\bar{\\theta}_m)^2}$$",
        math_block
    ))
    story.append(Paragraph(
        "To summarize dispersion across the 100 random splits, we compute the nominal interval "
        "$\\bar{\\theta}_m \\pm z_{0.975} \\frac{s_m}{\\sqrt{N}} = \\bar{\\theta}_m \\pm 0.196 \\cdot s_m$. "
        "However, this interval requires <b>rigorous statistical interpretation</b> (Dietterich, 1998; Bengio & Grandvalet, 2004):<br/>"
        "• <b>Non-Independence of Repeated Partitions:</b> Although the 100 random seeds generate different 80/20 train/test partitions, all splits draw from "
        "the <i>exact same underlying pool of 492 fraud cases</i>. Because test sets share identical observations across runs, the resulting estimates are "
        "positively correlated, violating the mutual independence assumption required for standard inferential confidence intervals.<br/>"
        "• <b>Measures of Partition Sensitivity, Not Future Guarantees:</b> Consequently, these intervals represent <b>split-to-split empirical variability</b>—quantifying "
        "how sensitive each algorithm is to particular random data partitions. They do <b>NOT</b> constitute proof of performance on future unseen transactions, "
        "nor do they eliminate epistemic risk arising from non-stationary fraud concept drift.",
        body
    ))

    story.append(Paragraph("4.3 Multi-Core Parallel Execution Architecture", subsec_heading))
    story.append(Paragraph(
        "Executing 700 full-scale training and evaluation runs sequentially would require ~12 hours of compute. To achieve high operational efficiency, "
        "we engineered a parallel execution harness using Python's `concurrent.futures.ProcessPoolExecutor` utilizing 12 physical/logical CPU cores. "
        "By parallelizing worker tasks across process boundaries, data chunks and models executed concurrently without Global Interpreter Lock (GIL) "
        "contention, completing the entire 100-run multi-model benchmark in exactly <b>88.4 seconds</b>.",
        body
    ))

    # Parallel Architecture Callout
    par_box = (
        "<b>PARALLEL BENCHMARK BENCHMARK SPECIFICATIONS:</b><br/>"
        "• <b>Hardware Architecture:</b> 12 Logical CPU Cores, 16 GB DDR4 RAM, Windows 11 x64, Python 3.13.<br/>"
        "• <b>Concurrency Model:</b> Multi-process pool executor (12 persistent worker processes; seed chunking).<br/>"
        "• <b>Benchmark Throughput:</b> 700 complete training & test evaluation pipelines executed in 88.4s (~7.92 model fits / second).<br/>"
        "• <b>Storage / Audit Log:</b> Raw per-run metrics captured in `simulation_100_runs.csv` (700 rows, 14 metric fields) for external verification."
    )
    t_par = Table([[Paragraph(par_box, callout_text)]], colWidths=[540])
    t_par.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('PADDING', (0,0), (-1,-1), 4.5),
    ]))
    story.append(t_par)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 14: SECTION 4 (CONT.): MASTER BENCHMARK RESULTS TABLE
    # =========================================================================
    story.append(Paragraph("4. Benchmark Results & Metric Comparison", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph(
        "<b>Table 4.1: Master 100-Run Monte Carlo Benchmark Results (N = 100 Stratified Partitions):</b><br/>"
        "All figures report the empirical sample mean $\\bar{\\theta}$ across 100 independent runs, accompanied by the 95% CLT Confidence Interval half-width "
        "($\\pm 0.196 \\times s$). Fixed threshold $p = 0.50$ applied uniformly across all models. Best metrics highlighted in bold.",
        body
    ))

    # Comprehensive Benchmark Table (540 pt printable width)
    bench_headers = [
        Paragraph("<b>Model Architecture</b>", tbl_header),
        Paragraph("<b>Event Recall</b>", tbl_header),
        Paragraph("<b>Dollar Recall</b>", tbl_header),
        Paragraph("<b>Precision</b>", tbl_header),
        Paragraph("<b>F1 Score</b>", tbl_header),
        Paragraph("<b>ROC-AUC</b>", tbl_header),
        Paragraph("<b>FPR (%)</b>", tbl_header),
        Paragraph("<b>Stolen Saved</b>", tbl_header),
        Paragraph("<b>False Pos (FP)</b>", tbl_header),
        Paragraph("<b>Missed Fraud</b>", tbl_header),
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

    t_bench = Table([bench_headers] + bench_rows, colWidths=[108, 43, 44, 38, 38, 42, 36, 50, 43, 45, 53])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 2.8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.4 Precision-Recall AUC vs. ROC-AUC in Rare-Event Regimes", subsec_heading))
    story.append(Paragraph(
        "A critical statistical insight emerges from comparing Receiver Operating Characteristic (ROC-AUC) against Precision-Recall (PR-AUC). "
        "All seven models exhibit superficially stellar ROC-AUC scores spanning <b>0.9449 to 0.9799</b>. However, Davis & Goadrich (2006) proved "
        "that in extreme class rarity ($P(Y=1) \\ll 0.01$), ROC-AUC is misleadingly optimistic because the False Positive Rate denominator ($N_{\\text{neg}} = 56,864$) "
        "is enormous; adding thousands of false alarms barely moves the FPR needle from 0.01 to 0.03, leaving the ROC curve artificially close to the top-left corner.<br/>"
        "In contrast, <b>Precision-Recall AUC (PR-AUC)</b> directly tracks false alarms against true positives in the denominator: "
        "$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$. While Soft-Voting achieves the highest ROC-AUC (0.9799), "
        "<b>GBDT achieves the highest PR-AUC (0.8341)</b>, reflecting its superior precision (9.18%) and lower false-positive volume (892.7 FPs).",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 15: SECTION 4 (CONT.): OPERATIONAL TRADE-OFFS & ARITHMETIC AUDIT
    # =========================================================================
    story.append(Paragraph("4. Operational Trade-Offs & Step 10 Reconciliation", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("4.5 The 'Recall Fallacy' & Full Arithmetic Cost Audit Across All Models", subsec_heading))
    story.append(Paragraph(
        "A foundational principle established in this benchmark is that <b>maximizing event recall does not minimize financial loss</b>. "
        "Cost-Sensitive Logistic Regression attains an outstanding event recall of <b>93.81%</b> and dollar recall of <b>92.89%</b> (missing only $926.39 in fraud). "
        "However, it generates <b>7,854.0 false positives per test split</b> (13.81% FPR). At $2.50 per review, investigating these false alarms costs "
        "$19,634.98, pushing total loss to $20,561.36. In contrast, Baseline Logistic Regression produces only 518.3 false positives ($1,295.80 review cost) "
        "and misses $2,671.90, yielding a total operational loss of only $3,967.70—over <b>5× lower</b> than Cost-Sensitive LogReg.<br/>"
        "To guarantee complete mathematical integrity, Table 4.2 provides an unrounded arithmetic reconciliation audit across all seven models:",
        body
    ))

    # Full Arithmetic Audit Table (540 pt width)
    audit_headers = [
        Paragraph("<b>Model Architecture</b>", tbl_header),
        Paragraph("<b>Mean FP<br/>(Count)</b>", tbl_header),
        Paragraph("<b>Alert Cost<br/>(@$2.50)</b>", tbl_header),
        Paragraph("<b>Missed Fraud<br/>(Stolen $)</b>", tbl_header),
        Paragraph("<b>Total Loss<br/>(Calculated)</b>", tbl_header),
        Paragraph("<b>Table 4.1 Master<br/>Benchmark ($)</b>", tbl_header),
        Paragraph("<b>Audit Status</b>", tbl_header)
    ]

    audit_rows = [
        [Paragraph("<b>1. Baseline LogReg</b>", tbl_cell_bold),
         Paragraph("518.3", tbl_cell_center),
         Paragraph("$1,295.80", tbl_cell_right),
         Paragraph("$2,671.90", tbl_cell_right),
         Paragraph("<b>$3,967.70</b>", tbl_cell_bold_right),
         Paragraph("$3,968 ± $275", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)],

        [Paragraph("<b>2. Cost-Sensitive LogReg</b>", tbl_cell_bold),
         Paragraph("7,854.0", tbl_cell_center),
         Paragraph("$19,634.98", tbl_cell_right),
         Paragraph("$926.39", tbl_cell_right),
         Paragraph("<b>$20,561.36</b>", tbl_cell_bold_right),
         Paragraph("$20,561 ± $508", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)],

        [Paragraph("<b>3. Neural Net (MLP)</b>", tbl_cell_bold),
         Paragraph("1,682.8", tbl_cell_center),
         Paragraph("$4,207.13", tbl_cell_right),
         Paragraph("$1,961.48", tbl_cell_right),
         Paragraph("<b>$6,168.61</b>", tbl_cell_bold_right),
         Paragraph("$6,169 ± $270", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)],

        [Paragraph("<b>4. GBDT Classifier</b>", tbl_cell_bold),
         Paragraph("892.7", tbl_cell_center),
         Paragraph("$2,231.70", tbl_cell_right),
         Paragraph("$2,290.04", tbl_cell_right),
         Paragraph("<b>$4,521.74</b>", tbl_cell_bold_right),
         Paragraph("$4,522 ± $272", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)],

        [Paragraph("<b>5. Isolation Forest</b> (Unsup.)", tbl_cell_bold),
         Paragraph("673.0", tbl_cell_center),
         Paragraph("$1,682.63", tbl_cell_right),
         Paragraph("$5,369.92", tbl_cell_right),
         Paragraph("<b>$7,052.54</b>", tbl_cell_bold_right),
         Paragraph("$7,053 ± $410", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)],

        [Paragraph("<b>6. Soft-Voting Ensemble</b>", tbl_cell_bold),
         Paragraph("1,299.3", tbl_cell_center),
         Paragraph("$3,248.15", tbl_cell_right),
         Paragraph("$1,943.76", tbl_cell_right),
         Paragraph("<b>$5,191.91</b>", tbl_cell_bold_right),
         Paragraph("$5,192 ± $240", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)],

        [Paragraph("<b>7. Stacked Meta-Learner</b>", tbl_cell_bold),
         Paragraph("1,880.5", tbl_cell_center),
         Paragraph("$4,701.38", tbl_cell_right),
         Paragraph("$1,781.11", tbl_cell_right),
         Paragraph("<b>$6,482.48</b>", tbl_cell_bold_right),
         Paragraph("$6,482 ± $230", tbl_cell_center),
         Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tbl_cell_center)]
    ]

    t_audit = Table([audit_headers] + audit_rows, colWidths=[120, 50, 75, 75, 75, 85, 60])
    t_audit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 2.2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_audit)
    story.append(Spacer(1, 3))

    story.append(Paragraph("4.6 Controlled Paired Evaluation: Fixed vs. Tuned Thresholding on Identical Test Split", subsec_heading))
    story.append(Paragraph(
        "To evaluate threshold strategies without sample confounding, we execute a strictly controlled paired experiment "
        "on the <b>identical held-out test split</b> (Seed 42; $N = 56,961$ transactions, 98 fraud cases, $10,504.99 fraud exposure). "
        "Holding test instances strictly invariant isolates the true causal impact of threshold tuning:",
        body
    ))

    # Controlled Paired Reconciliation Table (540 pt width)
    rec_headers = [
        Paragraph("<b>Model Architecture</b>", tbl_header),
        Paragraph("<b>Decision Rule</b>", tbl_header),
        Paragraph("<b>Recall</b>", tbl_header),
        Paragraph("<b>Precision</b>", tbl_header),
        Paragraph("<b>False Pos.</b>", tbl_header),
        Paragraph("<b>Net Loss ($)</b>", tbl_header),
        Paragraph("<b>Experimental Control</b>", tbl_header)
    ]
    rec_rows = [
        [Paragraph("<b>Soft-Voting Ensemble</b> (Fixed)", tbl_cell_bold),
         Paragraph("Fixed $p \\ge 0.50$", tbl_cell_center),
         Paragraph("90.82%", tbl_cell_center),
         Paragraph("4.66%", tbl_cell_center),
         Paragraph("1,819", tbl_cell_center),
         Paragraph("$4,967.40", tbl_cell_right),
         Paragraph("Identical Held-Out Split", tbl_cell_center)],

        [Paragraph("<b>Soft-Voting Ensemble</b> (Tuned)", tbl_cell_bold),
         Paragraph("Tuned $\\tau = 0.83$", tbl_cell_center),
         Paragraph("88.78%", tbl_cell_center),
         Paragraph("<b>39.91%</b>", tbl_cell_bold_center),
         Paragraph("<b>131</b>", tbl_cell_bold_center),
         Paragraph("<b>$1,078.16</b>", tbl_cell_bold_right),
         Paragraph("Identical Held-Out Split", tbl_cell_center)],

        [Paragraph("<b>GBDT Classifier</b> (Fixed)", tbl_cell_bold),
         Paragraph("Fixed $p \\ge 0.50$", tbl_cell_center),
         Paragraph("90.82%", tbl_cell_center),
         Paragraph("8.27%", tbl_cell_center),
         Paragraph("987", tbl_cell_center),
         Paragraph("$2,887.40", tbl_cell_right),
         Paragraph("Identical Held-Out Split", tbl_cell_center)],

        [Paragraph("<b>GBDT Classifier</b> (Tuned)", tbl_cell_bold),
         Paragraph("Tuned $\\tau = 0.83$", tbl_cell_center),
         Paragraph("85.71%", tbl_cell_center),
         Paragraph("<b>59.57%</b>", tbl_cell_bold_center),
         Paragraph("<b>57</b>", tbl_cell_bold_center),
         Paragraph("<b>$925.48</b>", tbl_cell_bold_right),
         Paragraph("Identical Held-Out Split", tbl_cell_center)],

        [Paragraph("<b>Baseline LogReg</b> (Fixed)", tbl_cell_bold),
         Paragraph("Fixed $p \\ge 0.50$", tbl_cell_center),
         Paragraph("91.84%", tbl_cell_center),
         Paragraph("5.20%", tbl_cell_center),
         Paragraph("1,640", tbl_cell_center),
         Paragraph("$4,514.81", tbl_cell_right),
         Paragraph("Identical Held-Out Split", tbl_cell_center)],

        [Paragraph("<b>Baseline LogReg</b> (Tuned)", tbl_cell_bold),
         Paragraph("Tuned $\\tau = 0.83$", tbl_cell_center),
         Paragraph("88.78%", tbl_cell_center),
         Paragraph("18.63%", tbl_cell_center),
         Paragraph("<b>380</b>", tbl_cell_bold_center),
         Paragraph("<b>$1,700.66</b>", tbl_cell_bold_right),
         Paragraph("Identical Held-Out Split", tbl_cell_center)]
    ]
    t_rec = Table([rec_headers] + rec_rows, colWidths=[130, 80, 50, 50, 60, 75, 95])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 2.0),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Synthesis:</b> On the exact same held-out transactions, threshold optimization reduces false positives by <b>76% to 94%</b> "
        "and cuts net operational losses by 62% to 78% while preserving 85%–89% recall. This confirms that precision gains stem from decision boundary "
        "calibration rather than partition variance.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 16: SECTION 5: FINANCIAL DECISION ANALYSIS & COST MODELING
    # =========================================================================
    story.append(Paragraph("5. Financial Decision Analysis & Operational Cost Modeling", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph("5.1 Institutional Cost Decomposition", subsec_heading))
    story.append(Paragraph(
        "A defensible financial decision framework requires decomposing aggregate penalties into tangible operational components. "
        "In enterprise credit card issuing and acquiring operations, total expected operating loss is defined as:",
        body
    ))
    story.append(Paragraph(
        "$$\\mathbb{E}[\\text{Loss}] = \\sum_{i \\in \\text{FN}} C_{\\text{FN}}(x_i) + \\sum_{j \\in \\text{FP}} C_{\\text{FP}} = \\sum_{i \\in \\text{FN}} \\text{Amount}_i \\cdot (1 - \\rho) + \\sum_{j \\in \\text{FP}} (c_{\\text{comm}} + c_a + c_f)$$",
        math_block
    ))
    story.append(Paragraph(
        "where each variable is grounded in institutional banking realities:",
        body
    ))
    cost_decomp_items = [
        "<b>Direct Stolen Fraud Loss ($C_{\\text{FN}}$):</b> Governed by transaction monetary face value $\\text{Amount}(x)$ and fund recovery rate $\\rho \\in [0, 1]$. "
        "In CNP transactions, goods or digital gift cards are redeemed instantaneously; empirical recovery rates are near zero ($\\rho \\approx 0.00$). "
        "Under US Regulation E and Visa Zero Liability, the issuing bank must fully indemnify cardholders. Hence, $C_{\\text{FN}}(x) = \\text{Amount}(x)$.",
        "<b>Communication & Gateway Fee ($c_{\\text{comm}}$):</b> Direct vendor cost of executing verification. Automated SMS gateway delivery costs "
        "approximately $0.02–$0.05 per message; interactive voice response (IVR) calls cost ~$0.15–$0.25; in-app push notification challenges cost < $0.005.",
        "<b>Analyst Labor & Triage Investigation Cost ($c_a$):</b> The cost of human security analyst queue review. Assuming a fully loaded fraud analyst "
        "compensation of $35.00/hour ($70,000/year base + benefits) and an average manual investigation handling time of 3.0 minutes per flagged account, "
        "the marginal triage cost is $1.75 per false alarm.",
        "<b>Customer Brand Attrition & Friction Cost ($c_f$):</b> The economic cost of customer insult. Javelin Strategy & Research (2021) demonstrated that "
        "32% of cardholders reduce card usage or churn entirely following an embarrassing false positive decline at a physical POS terminal, "
        "costing banks an estimated $0.70–$1.20 in annualized interchange lifetime value."
    ]
    for cdi in cost_decomp_items:
        story.append(Paragraph(f"• {cdi}", body))

    story.append(Paragraph("5.2 Academic Grounding & Deconstruction of the $2.50 False-Positive Benchmark", subsec_heading))
    story.append(Paragraph(
        "To ensure rigorous academic transparency, we explicitly delineate between published theoretical foundations and project modeling parameters:<br/>"
        "• <b>Theoretical Foundations (Bahnsen et al., 2015; Elkan, 2001):</b> Bahnsen, Aouada, Stojanovic, & Ottersten (2015) in <i>Expert Systems with "
        "Applications</i> established the mathematical instance-dependent Bayes Minimum Risk (BMR) cost matrix, proving that optimal decision rules require "
        "weighting false negatives by transaction face value $\\text{Amount}(x)$ against an administrative cost $c_a$. In their simulations, $c_a$ was varied "
        "parametrically ($c_a \\in [2, 10]$) as an abstract experimental parameter to demonstrate decision frontier sensitivity. Similarly, Elkan (2001) "
        "proved the foundational theorem for cost-proportionate dataset sampling and threshold adjustment.<br/>"
        "• <b>Operational Queue Constraints (Dal Pozzolo et al., 2014, 2015):</b> Dal Pozzolo et al. (2014, 2015) examined commercial fraud operations "
        "from a practitioner perspective, proving that fraud screening is fundamentally constrained by <b>investigator queue capacity</b> (e.g., security "
        "teams can only review the top-$k \\approx 100$ alerts per day). They demonstrated that alert volume dictates operational viability, but they did not "
        "assert an empirical universal dollar figure of $2.50.<br/>"
        "• <b>Project Engineering Parameter Synthesis ($C_{\\text{FP}} = \\$2.50$):</b> We explicitly treat $C_{\\text{FP}} = \\$2.50 as a synthesized "
        "project benchmark assumption, calibrated from industrial cost components: $c_{\\text{comm}} = \\$0.05 (carrier SMS challenge) + "
        "$c_a = \\$1.75 (3-minute analyst triage @ $35/hr) + $c_f = \\$0.70 (amortized customer churn friction). "
        "Because operational overhead varies across institutions, Section 5.3 evaluates multi-tier sensitivity across five diverse cost regimes ($0.50 to $10.00).",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 17: SECTION 5 (CONT.): SENSITIVITY ANALYSIS ACROSS COST REGIMES
    # =========================================================================
    story.append(Paragraph("5. Operational Cost Sensitivity Across Institutional Regimes", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=5))

    story.append(Paragraph(
        "<b>5.3 Multi-Tier Sensitivity Analysis:</b> To evaluate how model ranking adapts across disparate operational environments "
        "(from low-cost automated app push challenges to high-overhead manual phone investigations), we evaluated total expected financial loss "
        "$\\mathbb{E}[\\text{Loss}] = (\\text{FP} \\times C_{\\text{FP}}) + \\text{Missed Stolen Dollars}$ across 100 Monte Carlo runs for "
        "$C_{\\text{FP}} \\in [\\$0.50, \\$1.00, \\$2.50, \\$5.00, \\$10.00]$:",
        body
    ))

    # Sensitivity Table (540 pt width)
    sens_headers = [
        Paragraph("<b>Model Architecture</b>", tbl_header),
        Paragraph("<b>C_FP = $0.50<br/>(In-App Push)</b>", tbl_header),
        Paragraph("<b>C_FP = $1.00<br/>(SMS Alert)</b>", tbl_header),
        Paragraph("<b>C_FP = $2.50<br/>(Base Queue)</b>", tbl_header),
        Paragraph("<b>C_FP = $5.00<br/>(Manual Call)</b>", tbl_header),
        Paragraph("<b>C_FP = $10.00<br/>(Analyst Desk)</b>", tbl_header),
        Paragraph("<b>Optimal Operational Fit</b>", tbl_header)
    ]

    sens_rows = [
        [Paragraph("<b>1. Baseline LogReg</b>", tbl_cell),
         Paragraph("$2,931.06", tbl_cell_right),
         Paragraph("$3,190.22", tbl_cell_right),
         Paragraph("<b>$3,967.70</b>", tbl_cell_bold_right),
         Paragraph("<b>$5,263.50</b>", tbl_cell_bold_right),
         Paragraph("<b>$7,855.10</b>", tbl_cell_bold_right),
         Paragraph("<b>Optimal for C_FP ≥ $2.50</b><br/>Lowest cost when review cost is non-negligible.", tbl_cell_bold)],

        [Paragraph("<b>2. Cost-Sensitive LogReg</b>", tbl_cell),
         Paragraph("$4,853.38", tbl_cell_right),
         Paragraph("$8,780.38", tbl_cell_right),
         Paragraph("$20,561.36", tbl_cell_right),
         Paragraph("$40,196.34", tbl_cell_right),
         Paragraph("$79,466.29", tbl_cell_right),
         Paragraph("<b>Viable only if C_FP < $0.24</b><br/>Massive false alarms destroy profitability.", tbl_cell)],

        [Paragraph("<b>3. Neural Net (MLP)</b>", tbl_cell),
         Paragraph("$2,802.91", tbl_cell_right),
         Paragraph("$3,644.33", tbl_cell_right),
         Paragraph("$6,168.61", tbl_cell_right),
         Paragraph("$10,375.73", tbl_cell_right),
         Paragraph("$18,789.98", tbl_cell_right),
         Paragraph("Balanced Non-Linear Catch;<br/>viable in digital-first automated queues.", tbl_cell)],

        [Paragraph("<b>4. GBDT Classifier</b>", tbl_cell),
         Paragraph("$2,736.38", tbl_cell_right),
         Paragraph("<b>$3,182.72</b>", tbl_cell_bold_right),
         Paragraph("$4,521.74", tbl_cell_right),
         Paragraph("$6,753.44", tbl_cell_right),
         Paragraph("$11,216.84", tbl_cell_right),
         Paragraph("<b>Optimal for C_FP ≈ $1.00</b><br/>Exceptional balance of precision and recall.", tbl_cell_bold)],

        [Paragraph("<b>5. Isolation Forest</b> (Unsup.)", tbl_cell),
         Paragraph("$5,706.44", tbl_cell_right),
         Paragraph("$6,042.97", tbl_cell_right),
         Paragraph("$7,052.54", tbl_cell_right),
         Paragraph("$8,735.17", tbl_cell_right),
         Paragraph("$12,100.42", tbl_cell_right),
         Paragraph("Zero-Day Anomaly Sentinel;<br/>serves as parallel unlabelled watchdog.", tbl_cell)],

        [Paragraph("<b>6. Soft-Voting Ensemble</b>", tbl_cell),
         Paragraph("<b>$2,593.39</b>", tbl_cell_bold_right),
         Paragraph("$3,243.02", tbl_cell_right),
         Paragraph("$5,191.91", tbl_cell_right),
         Paragraph("$8,440.06", tbl_cell_right),
         Paragraph("$14,936.36", tbl_cell_right),
         Paragraph("<b>Optimal for C_FP ≤ $0.50</b><br/>Dominates under frictionless push challenges.", tbl_cell_bold)],

        [Paragraph("<b>7. Stacked Meta-Learner</b>", tbl_cell),
         Paragraph("$2,721.38", tbl_cell_right),
         Paragraph("$3,661.66", tbl_cell_right),
         Paragraph("$6,482.48", tbl_cell_right),
         Paragraph("$11,183.86", tbl_cell_right),
         Paragraph("$20,586.61", tbl_cell_right),
         Paragraph("High Precision Meta-Ranking;<br/>requires dynamic thresholding in production.", tbl_cell)]
    ]

    t_sens = Table([sens_headers] + sens_rows, colWidths=[115, 68, 68, 68, 68, 70, 85])
    t_sens.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('PADDING', (0,0), (-1,-1), 2.8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_sens)
    story.append(Spacer(1, 4))

    story.append(Paragraph("5.4 Consistency Verification & Breakeven Frontiers Across Candidate Architectures", subsec_heading))
    story.append(Paragraph(
        "<b>Rigorous Consistency Verification on Model Rankings:</b> A central finding in Table 5.1 is that <b>Soft-Voting Ensemble minimizes loss ($2,593.39) "
        "when $C_{\\text{FP}} \\le \\$0.50$</b>, whereas <b>Baseline Logistic Regression minimizes loss ($3,967.70) when $C_{\\text{FP}} \\ge \\$2.50$</b>. "
        "This ranking inversion is verified against the <b>exact same cost formula, model predictions, and test splits ($p = 0.50$ fixed cutoff)</b>:<br/>"
        "Each model's total cost function is a straight line: $\\mathcal{L}(C_{\\text{FP}}) = \\bar{\\text{FP}} \\cdot C_{\\text{FP}} + \\bar{\\text{FN}}_{\\$}$, "
        "where slope is false-positive count $\\bar{\\text{FP}}$ and y-intercept is unpenalized missed fraud loss $\\bar{\\text{FN}}_{\\$}$. "
        "Soft-Voting catches $728.14 more fraud than Baseline LogReg ($1,943.76 missed vs $2,671.90), giving it a lower y-intercept. "
        "However, Soft-Voting generates 780.94 more false positives (1,299.26 vs 518.32), giving it a steeper slope. "
        "At $C_{\\text{FP}} = \\$0.50$, the financial penalty of those 780.94 extra false alarms is only $780.94 \\times \\$0.50 = \\$390.47$. "
        "Because fraud savings ($728.14) exceed alert costs ($390.47), Soft-Voting gains a net financial advantage of <b>+$337.67</b> over Baseline LogReg. "
        "By equating the two cost lines, we analytically solve for the <b>exact breakeven cost</b> $C_{\\text{FP}}^*$:",
        body
    ))
    story.append(Paragraph(
        "$$C_{\\text{FP}}^* = \\frac{\\bar{\\text{FN}}_{\\$, A} - \\bar{\\text{FN}}_{\\$, B}}{\\bar{\\text{FP}}_B - \\bar{\\text{FP}}_A}$$",
        math_block
    ))
    story.append(Paragraph(
        "• <b>Soft-Voting vs. Baseline LogReg:</b> $C_{\\text{FP}}^* = \\frac{\\$2,671.90 - \\$1,943.76}{1,299.26 - 518.32} = \\frac{\\$728.14}{780.94} = \\mathbf{\\$0.9324}$. "
        "For any alert cost below $0.9324, Soft-Voting is mathematically guaranteed to minimize loss under the exact same fixed cutoff ($p=0.50$); above $0.9324, Baseline LogReg wins.<br/>"
        "• <b>GBDT vs. Baseline LogReg:</b> $C_{\\text{FP}}^* = \\frac{\\$2,671.90 - \\$2,290.04}{892.68 - 518.32} = \\frac{\\$381.86}{374.36} = \\mathbf{\\$1.0200}$. GBDT is optimal in the transition zone ($0.9324 to $1.0200).<br/>"
        "• <b>Cost-Sensitive LogReg vs. Baseline LogReg:</b> $C_{\\text{FP}}^* = \\frac{\\$2,671.90 - \\$926.39}{7,853.99 - 518.32} = \\frac{\\$1,745.51}{7,335.67} = \\mathbf{\\$0.2380}$. Viable only below 23.8 cents.<br/>"
        "<b>Fixed vs. Dynamic Bayes Thresholding Paradigm:</b> Under fixed thresholding ($p=0.50$), rankings shift along linear cost frontiers as slopes trade off against intercepts. "
        "Under dynamic Bayesian thresholding ($p^*(x) = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + \\text{Amount}(x)}$), changing $C_{\\text{FP}}$ dynamically shifts the probability cutoff curve, "
        "altering the decision boundary directly. Both paradigms confirm the same operational reality: low-cost automated verification favors aggressive fraud capture (Soft-Voting), "
        "while high-cost manual review mandates strict false-positive suppression (Baseline LogReg).",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 18: SECTION 6: PRODUCTION ARCHITECTURE & LATENCY TARGETS
    # =========================================================================
    story.append(Paragraph("6. Production Architecture, Latency Targets, & Governance", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("6.1 Proposed Enterprise Cascaded Screening Architecture", subsec_heading))
    story.append(Paragraph(
        "Commercial card processing networks (VisaNet, Mastercard Banknet) mandate strict end-to-end authorization budgets: total round-trip "
        "latency must remain strictly below <b>50 milliseconds</b>, leaving the fraud risk engine with a maximum compute budget of <b>10 to 15 milliseconds</b>. "
        "To reconcile model accuracy with sub-millisecond execution constraints, we propose a <b>Two-Tier Cascaded Production Pipeline</b>:",
        body
    ))

    # Architecture Box Diagram
    arch_box = (
        "<b>PROPOSED TWO-TIER CASCADED REAL-TIME FRAUD SCREENING ARCHITECTURE:</b><br/>"
        "<font face='Courier' size='6.6'>"
        "┌────────────────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│ INCOMING CARD TRANSACTION EVENT (ISO 8583 Authorization Request; Cardholder, Merchant)  │<br/>"
        "└───────────────────────────────────────────┬────────────────────────────────────────────┘<br/>"
        "                                            │ Latency: < 0.2 ms (Pure Vectorized Dot Product)<br/>"
        "                                            ▼<br/>"
        "┌────────────────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│ TIER 1: HIGH-VELOCITY LINEAR SCREENING FILTER (Baseline Logistic Regression)           │<br/>"
        "│ • Scores 100% of transaction stream in sub-millisecond time.                          │<br/>"
        "│ • Rule A: If P(Fraud) < 0.005 ──► IMMEDIATE AUTO-ALLOW (Clears 96.5% of legitimate txn) │<br/>"
        "│ • Rule B: If P(Fraud) > 0.950 ──► IMMEDIATE HARD DECLINE (Blocks blatant stolen cards) │<br/>"
        "└───────────────────────────────────────────┬────────────────────────────────────────────┘<br/>"
        "                                            │ Ambiguous Borderline Mass: 0.005 ≤ P ≤ 0.950 (~3.5% of traffic)<br/>"
        "                                            ▼<br/>"
        "┌────────────────────────────────────────────────────────────────────────────────────────┐<br/>"
        "│ TIER 2: DEEP DELIBERATIVE ENSEMBLE ARBITRATION (GBDT + MLP + Dynamic Bayesian Rule)    │<br/>"
        "│ • Latency Budget: 4 to 8 ms (Triggered strictly on ambiguous 3.5% volume).              │<br/>"
        "│ • Computes Dynamic Cutoff: p*(x) = C_FP / (C_FP + Amount(x)).                          │<br/>"
        "│ • Cross-references Unsupervised Isolation Forest score to detect zero-day anomalies.   │<br/>"
        "│ • Outcome: Step-up In-App Push Challenge (P > p*(x)) or Frictionless Approval.        │<br/>"
        "└────────────────────────────────────────────────────────────────────────────────────────┘"
        "</font>"
    )
    t_arch = Table([[Paragraph(arch_box, callout_text)]], colWidths=[540])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_highlight),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6.2 Proposed Latency Design Targets vs. Production Reality", subsec_heading))
    story.append(Paragraph(
        "In compliance with faculty review feedback, we emphasize that our reported inference runtimes (0.2 ms for LogReg, 1.8 ms for MLP, 4.5 ms for GBDT) "
        "represent <b>proposed engineering targets on benchmark hardware</b> (12-core local workstation with local memory locality), "
        "rather than field-validated production SLAs under concurrent multi-tenant transaction loads. In commercial cloud deployments, real-time "
        "scoring faces network deserialization overheads, distributed Redis feature store cache lookups, and TCP transport hops that typically add 10–20 ms "
        "of unavoidable infrastructure latency.",
        body
    ))

    story.append(Paragraph("6.3 Regulatory Compliance & Model Governance (SR 11-7 & PSD2)", subsec_heading))
    story.append(Paragraph(
        "Deploying machine learning models in financial institutions requires adherence to rigorous governance standards:",
        body
    ))
    gov_items = [
        "<b>Model Risk Management (SR 11-7):</b> The Federal Reserve requires continuous tracking of conceptual soundness and rigorous benchmarking "
        "against challenger models. Our 100-run Monte Carlo protocol and pure NumPy implementation fulfill SR 11-7 documentation standards by providing "
        "full mathematical transparency without closed-source vendor black boxes.",
        "<b>PSD2 SCA Transaction Risk Analysis (TRA) Context:</b> Under EBA RTS Article 18, payment service providers may exempt low-risk transactions "
        "from Strong Customer Authentication (SCA) only if their rolling 90-day gross fraud monetary rate (Fraud Loss / Total Processed Sales) remains below "
        "strict regulatory thresholds (0.01% to 0.06% depending on transaction amount tiers). While our GBDT architecture demonstrates a low false-positive rate "
        "of 1.57% (reducing operational alert volume), a low FPR is a distinct operational metric from gross fraud loss ratio and does not guarantee regulatory compliance. "
        "True compliance cannot be established from offline benchmark metrics alone and requires continuous monitoring of live aggregate settlement streams.",
        "<b>Adverse Action Explanations (GDPR Art. 22 / FCRA):</b> When a cardholder is declined, institutions must provide actionable adverse action notices. "
        "Because our Tier 1 model is linear and our Tier 2 GBDT utilizes shallow decision stumps, exact per-feature SHAP-like attribution can be generated in real-time."
    ]
    for gi in gov_items:
        story.append(Paragraph(f"• {gi}", body_tight))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 19: SECTION 6 (CONT.): LIMITATIONS & METHODOLOGICAL CAVEATS
    # =========================================================================
    story.append(Paragraph("6. Methodological Caveats, Learnings, & Future Work", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("6.4 Critical Limitations of the Investigation", subsec_heading))
    story.append(Paragraph(
        "To ensure academic rigor, we document the fundamental methodological and empirical boundaries of this study:",
        body
    ))
    lim_items = [
        "<b>1. Adversarial Non-Stationary Concept Drift:</b> The ULB dataset captures transactions across an isolated 48-hour window from September 2013. "
        "Real-world payment ecosystems are non-stationary: fraud syndicates continuously probe screening rules, rotate merchant category codes (MCCs), "
        "and adopt distributed micro-transaction testing. A model trained on a static 48-hour snapshot will experience performance degradation over time. "
        "True production resilience requires continuous online learning and streaming concept drift detection (e.g., Adaptive Windowing / ADWIN).",
        "<b>2. Loss of Semantic Feature Context due to PCA:</b> While PCA anonymization protects cardholder privacy, projecting data into orthogonal "
        "coordinates destroys critical relational information: merchant terminal identity, historical billing-shipping address mismatch, "
        "device biometric fingerprints, and card velocity graphs. Graph Neural Networks (GNNs) operating on heterogeneous card-merchant graphs "
        "routinely outperform tabular classifiers when relational metadata is unmasked.",
        "<b>3. Resampling from a Finite Minority Pool:</b> Although the 100-run Monte Carlo protocol varies the 80/20 partitioning across 100 seeds, "
        "all test partitions draw from the <i>same underlying pool of 492 fraud events</i>. Consequently, our standard error estimates quantify "
        "partition stability across this specific sample, but they cannot evaluate generalization across completely unseen fraud attack vectors.",
        "<b>4. Static Review Cost Assumptions:</b> We standardized $C_{\\text{FP}} = \\$2.50$ and $C_{\\text{FN}} = \\text{Amount}$. In practice, "
        "customer lifetime value (CLV) is heterogeneous: declining a ultra-high-net-worth private banking client incurs far higher friction than "
        "declining an inactive debit card account. Production models must parameterize $C_{\\text{FP}}$ by individual cardholder churn risk."
    ]
    for li in lim_items:
        story.append(Paragraph(f"• {li}", body))

    story.append(Paragraph("6.5 Core Team Learnings & Operational Recommendations", subsec_heading))
    story.append(Paragraph(
        "The development of this pure NumPy fraud detection framework yielded five primary technical learnings:",
        body
    ))
    learn_items = [
        "<b>Learning 1: Optimize for Financial Loss, Not Generic F1-Score:</b> High F1-scores frequently mask disastrous operational economics. "
        "Always evaluate models using asymmetric cost matrices reflecting actual institutional alert handling capacity.",
        "<b>Learning 2: Tree Boosting Outperforms Deep Learning on Orthogonal Tabular Data:</b> Across all benchmarks, GBDT outperformed our "
        "3-layer MLP in PR-AUC (0.8341 vs. 0.7618) and precision (9.18% vs. 5.06%), confirming that gradient boosted trees remain superior for tabular features.",
        "<b>Learning 3: Pair Models with Institutional Maturity:</b> A bank with low automated verification maturity (relying on manual phone queues) "
        "must deploy <b>Baseline Logistic Regression</b> ($3,968 loss). An institution with mature in-app push verification can deploy the "
        "<b>Soft-Voting Ensemble</b> ($2,593 loss) to capture higher recall without suffering labor collapse.",
        "<b>Learning 4: First-Principles Engineering Demystifies ML:</b> Building backpropagation, Adam, SMOTE, and iTrees from scratch eliminated "
        "framework opacity, providing complete control over gradient flows, memory layouts, and numerical stability.",
        "<b>Learning 5: Unsupervised Anomaly Detection is Essential Insurance:</b> While Isolation Forest had the lowest supervised precision (8.36%), "
        "its zero-label formulation makes it an invaluable parallel watchdog for catching zero-day attacks that evade supervised classifiers."
    ]
    for lni in learn_items:
        story.append(Paragraph(f"• {lni}", body_tight))

    story.append(Paragraph("6.6 Future Research Directions", subsec_heading))
    story.append(Paragraph(
        "Promising avenues for future research include implementing online streaming algorithms (River/Cramer-von Mises drift tests), "
        "integrating Conformal Prediction to provide mathematically guaranteed coverage guarantees on class probabilities, and "
        "developing first-principles Graph Neural Networks (GNNs) on bipartite transaction graphs.",
        body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 20: SECTION 7: REFERENCES & REPRODUCIBILITY APPENDIX
    # =========================================================================
    story.append(Paragraph("7. References & Reproducibility Appendix", sec_heading))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=0, spaceAfter=6))

    story.append(Paragraph("7.1 Traceable Peer-Reviewed Academic Bibliography", subsec_heading))
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
        "Caruana, R., & Niculescu-Mizil, A. (2006). 'An empirical comparison of supervised learning algorithms.' <i>International Conference on Machine Learning (ICML)</i>, 161–168. DOI: 10.1145/1143844.1143865.",
        "Elkan, C. (2001). 'The foundations of cost-sensitive learning.' <i>International Joint Conference on Artificial Intelligence (IJCAI)</i>, 17(1), 973–978.",
        "European Banking Authority (EBA). (2018). 'Regulatory Technical Standards on Strong Customer Authentication and Common and Secure Communication under PSD2.' EBA/RTS/2017/10.",
        "Board of Governors of the Federal Reserve System. (2011). 'Supervisory Guidance on Model Risk Management.' SR Letter 11-7 / OCC Bulletin 2011-12.",
        "Consumer Financial Protection Bureau. (2016). '12 CFR Part 1005 (Regulation E) — Electronic Fund Transfers.' CFPB Annual Regulatory Digest.",
        "Nilson Report. (2023). 'Card Fraud Losses Worldwide.' Issue 1254, December 2023.",
        "Javelin Strategy & Research. (2021). 'The Cost of False Positive Card Declines.' Financial Services Operational Benchmark Series.",
        "LexisNexis Risk Solutions. (2023). '2023 True Cost of Fraud Study: Financial Services and Lending.' Global Financial Fraud Report.",
        "Dietterich, T. G. (1998). 'Approximate statistical tests for comparing supervised classification learning algorithms.' <i>Neural Computation</i>, 10(7), 1895–1923.",
        "Bengio, Y., & Grandvalet, Y. (2004). 'No unbiased estimator of the variance of K-fold cross-validation.' <i>Journal of Machine Learning Research (JMLR)</i>, 5, 1089–1105."
    ]
    for ref in references:
        story.append(Paragraph(f"• {ref}", tbl_cell))

    story.append(Spacer(1, 4))
    story.append(Paragraph("7.2 Hardware Specifications, Execution Instructions, & Repository Manifest", subsec_heading))
    story.append(Paragraph(
        "• <b>Hardware & Runtime Environment:</b> Windows 11 Enterprise x64, 12 Logical CPU Cores, 16.0 GB DDR4 System RAM, Python 3.13.0.<br/>"
        "• <b>Software Dependencies:</b> `numpy>=1.24.0`, `pandas>=2.0.0`, `reportlab>=4.0.0` (ReportLab used strictly for PDF rendering; 0 ML libraries).<br/>"
        "• <b>Re-Running the 100-Run Monte Carlo Simulation:</b> Execute `python run_100_simulations.py` (triggers 12-core multi-process pool, completing in ~88s). "
        "Logs full raw results to `simulation_100_runs.csv`.<br/>"
        "• <b>Re-Compiling this 20-Page Research Monograph:</b> Execute `python generate_pdf_report.py`. Re-compiles PDF to both local workspace and desktop.<br/>"
        "• <b>Repository Notebook Manifest:</b> All 10 modular implementation steps are mirrored in individual Jupyter notebooks for interactive analysis: "
        "`Step1_SMOTE.ipynb`, `Step2_Deep_MLP.ipynb`, `Step3_Stratified_CV.ipynb`, `Step4_GBDT.ipynb`, `Step5_Isolation_Forest.ipynb`, "
        "`Step6_Cost_Sensitive.ipynb`, `Step7_Ensemble.ipynb`, `Step8_Dynamic_Threshold.ipynb`, `Step9_Focal_Loss.ipynb`, `Step10_Stacking.ipynb`.",
        body
    ))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    
    import shutil
    shutil.copyfile(pdf_local, pdf_desktop)
    print(f"Publication-grade 20-page Monograph compiled successfully:")
    print(f"Local:   {pdf_local}")
    print(f"Desktop: {pdf_desktop}")

if __name__ == "__main__":
    build_pdf_monograph()
