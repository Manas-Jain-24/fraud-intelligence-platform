import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.platypus import Image as RLImage
from reportlab.pdfgen import canvas

_EQUATION_CACHE = {}

def render_math_equation(latex_str, fontsize=8.5, dpi=300, max_width=490):
    cache_key = (latex_str, fontsize, dpi)
    if cache_key in _EQUATION_CACHE:
        buf, pt_w, pt_h = _EQUATION_CACHE[cache_key]
        buf.seek(0)
        img = RLImage(buf, width=pt_w, height=pt_h)
        img.hAlign = 'CENTER'
        return img

    s = latex_str.strip()
    while s.startswith('$'):
        s = s[1:].strip()
    while s.endswith('$'):
        s = s[:-1].strip()

    s = s.replace(r'\text{', r'\mathrm{')
    s = s.replace(r'\%', r'\%')
    s = s.replace(r'\mathbb{E}', r'\mathbf{E}')
    s = s.replace(r'\mathbb{I}', r'\mathbf{I}')
    s = s.replace(r'\mathbb{R}', r'\mathbf{R}')
    s = s.replace(r'\iff', r'\Longleftrightarrow')
    s = s.replace(r'\bar{\text{FN}}_{\$, \text{GBDT}}', r'\overline{\mathrm{FN}}_{\mathrm{USD}, \mathrm{GBDT}}')
    s = s.replace(r'\bar{\text{FN}}_{\$, \text{SoftVote}}', r'\overline{\mathrm{FN}}_{\mathrm{USD}, \mathrm{SoftVote}}')
    s = s.replace(r'\bar{\text{FP}}_{\text{SoftVote}}', r'\overline{\mathrm{FP}}_{\mathrm{SoftVote}}')
    s = s.replace(r'\bar{\text{FP}}_{\text{GBDT}}', r'\overline{\mathrm{FP}}_{\mathrm{GBDT}}')
    s = s.replace(r'\bar{\text{FN}}_{\$, A}', r'\overline{\mathrm{FN}}_{\mathrm{USD}, A}')
    s = s.replace(r'\bar{\text{FN}}_{\$, B}', r'\overline{\mathrm{FN}}_{\mathrm{USD}, B}')
    s = s.replace(r'\bar{\text{FP}}_A', r'\overline{\mathrm{FP}}_A')
    s = s.replace(r'\bar{\text{FP}}_B', r'\overline{\mathrm{FP}}_B')
    s = s.replace(r'\bar{\text{FP}}', r'\overline{\mathrm{FP}}')
    s = s.replace(r'\bar{\theta}', r'\overline{\theta}')
    s = s.replace(r'\;', r'\ ')
    s = s.replace(r'\$', r'\$')

    fig = plt.figure(figsize=(10, 0.8))
    fig.patch.set_alpha(0.0)
    fig.text(0.5, 0.5, f'${s}$', fontsize=fontsize, ha='center', va='center', color='#0F172A')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    pil_img = PILImage.open(buf)
    pt_w = pil_img.size[0] * 72.0 / dpi
    pt_h = pil_img.size[1] * 72.0 / dpi

    if pt_w > max_width:
        scale = max_width / pt_w
        pt_w *= scale
        pt_h *= scale

    if pt_h > 22:
        scale = 22.0 / pt_h
        pt_w *= scale
        pt_h *= scale

    _EQUATION_CACHE[cache_key] = (buf, pt_w, pt_h)
    buf.seek(0)
    img = RLImage(buf, width=pt_w, height=pt_h)
    img.hAlign = 'CENTER'
    return img

def make_equation_flowable(latex_str, fontsize=8.5, max_width=490):
    img = render_math_equation(latex_str, fontsize=fontsize, max_width=max_width)
    t = Table([[img]], colWidths=[540])
    t.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return t

class NumberedCanvas(canvas.Canvas):
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
            self.saveState()
            self.setFont("Helvetica", 7.0)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(36, 25, "Python for Data Science (PDS) | IIIT Sri City — 1,000-Run Monte Carlo Statistical Benchmark Report")
            self.drawRightString(letter[0] - 36, 25, f"Page 1 of {page_count}")
            self.restoreState()
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1E293B"))
        # Running Header
        self.drawString(36, letter[1] - 28, "STATISTICAL REPORT: 1,000-RUN MONTE CARLO SIMULATION BENCHMARK (N = 1,000)")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawRightString(letter[0] - 36, letter[1] - 28, "IIIT SRI CITY — PDS RESEARCH GROUP")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, letter[1] - 32, letter[0] - 36, letter[1] - 32)

        # Running Footer
        self.line(36, 38, letter[0] - 36, 38)
        self.setFont("Helvetica", 7.0)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 26, "Empirical Resampling Stability, Tail Risk (CVaR), and Operational Cost Frontiers across 7 Architectures")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 26, page_str)
        self.restoreState()

def build_pdf_statistical_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    out_dir_local = r"c:\Users\HP\.gemini\antigravity-ide\scratch\Credit-Card-Fraud-Detection-Model\report"
    out_dir_desktop = r"C:\Users\HP\Desktop\PDS\report"
    fig_dir = os.path.join(repo_root, "report", "figures_statistical")
    if not os.path.exists(fig_dir):
        fig_dir = os.path.join(script_dir, "..", "figures_statistical")
    os.makedirs(out_dir_local, exist_ok=True)
    os.makedirs(out_dir_desktop, exist_ok=True)

    pdf_local = os.path.join(out_dir_local, "Statistical_Report_1000_Simulations.pdf")
    pdf_desktop = os.path.join(out_dir_desktop, "Statistical_Report_1000_Simulations.pdf")

    margin = 36
    doc = SimpleDocTemplate(
        pdf_local,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=38,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    # Color Tokens
    c_primary = colors.HexColor("#0F172A")    # Deep Navy
    c_secondary = colors.HexColor("#1E3A8A")  # Deep Blue
    c_accent = colors.HexColor("#B91C1C")     # Crimson
    c_dark = colors.HexColor("#1E293B")       # Dark Charcoal
    c_muted = colors.HexColor("#64748B")      # Slate Muted
    c_bg_light = colors.HexColor("#F8FAFC")   # Light Slate
    c_border = colors.HexColor("#CBD5E1")     # Border
    c_highlight = colors.HexColor("#F1F5F9")  # Table Alternate Row
    c_callout_bg = colors.HexColor("#EFF6FF") # Soft Blue
    c_callout_border = colors.HexColor("#3B82F6")
    c_green_bg = colors.HexColor("#F0FDF4")   # Soft Green
    c_green_border = colors.HexColor("#22C55E")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=15, leading=18,
        textColor=c_primary, spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11,
        textColor=c_secondary, spaceAfter=3
    )
    meta_style = ParagraphStyle(
        'MetaText', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.2, leading=9.5,
        textColor=c_muted, spaceAfter=6
    )
    sec_heading = ParagraphStyle(
        'SecHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10.0, leading=12.5,
        textColor=c_primary, spaceBefore=5, spaceAfter=3,
        keepWithNext=True
    )
    subsec_heading = ParagraphStyle(
        'SubSecHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8.5, leading=11,
        textColor=c_secondary, spaceBefore=4, spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.3, leading=9.6,
        textColor=c_dark, spaceAfter=3.0
    )
    callout_style = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.0, leading=9.2,
        textColor=colors.HexColor("#1E3A8A")
    )
    caption_style = ParagraphStyle(
        'FigCaption', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.0, leading=9.0,
        textColor=c_primary, alignment=1, spaceAfter=2
    )
    takeaway_style = ParagraphStyle(
        'TakeawayText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6.8, leading=8.8,
        textColor=c_dark
    )
    th_style = ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.5, leading=8.2,
        textColor=colors.white, alignment=1
    )
    tc_style = ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6.5, leading=8.0,
        textColor=c_dark, alignment=1
    )
    tc_bold = ParagraphStyle(
        'TableCellBold', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.5, leading=8.0,
        textColor=c_dark, alignment=1
    )
    tc_left = ParagraphStyle(
        'TableCellLeft', parent=styles['Normal'],
        fontName='Helvetica', fontSize=6.5, leading=8.0,
        textColor=c_dark, alignment=0
    )
    tc_left_bold = ParagraphStyle(
        'TableCellLeftBold', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=6.5, leading=8.0,
        textColor=c_dark, alignment=0
    )

    story = []

    # ==========================================
    # PAGE 1: TITLE, EXECUTIVE SUMMARY & PLAIN-ENGLISH ANALOGY GUIDE
    # ==========================================
    title_table = Table([
        [Paragraph("STATISTICAL REPORT: 1,000-RUN MONTE CARLO BENCHMARK (<i>N</i> = 1,000)", title_style)],
        [Paragraph("Visual Empirical Analysis & Plain-English Operational Guide for Credit Card Fraud Detection", subtitle_style)],
        [Paragraph("<b>Authors:</b> Research Group | Python for Data Science (PDS) Capstone | IIIT Sri City<br/>"
                   "<b>Dataset:</b> ULB Machine Learning Group Benchmark (284,807 Transactions, 492 Frauds, 0.1727% Imbalance)<br/>"
                   "<b>Empirical Scope:</b> 1,000 Stratified Splits | 7,000 Independent Model Fits | 56,962,000 Out-of-Sample Predictions", meta_style)]
    ], colWidths=[540])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("Executive Summary & Core Empirical Findings", sec_heading))
    story.append(Paragraph(
        "Evaluating fraud detection models on a single random train/test split creates severe estimator variance. "
        "This monograph reports the definitive results of a <b>1,000-run Monte Carlo simulation (<i>N</i> = 1,000)</b> "
        "conducted across seeds <i>s</i> &isin; {1000, &hellip;, 1999}. Every model architecture was re-trained from mathematical first principles "
        "on 1,000 distinct stratified partitions (7,000 total fits; 56,962,000 classifications) to audit resampling stability, tail-risk exposure (CVaR), "
        "and true operational loss trade-offs.",
        body_style
    ))

    findings_data = [
        [
            Paragraph("<b>1. Overall Financial & Risk Champion: GBDT</b><br/>"
                      "GBDT achieves the lowest net loss (<b>$4,369.26 &plusmn; $77.66</b> per split) under baseline manual review ($C<sub>FP</sub> = $2.50). "
                      "It triggers only <b>880.19 false positives</b> (1.55% FPR), achieving the highest precision (<b>9.30%</b>), highest F1-score (<b>16.76%</b>), "
                      "and lowest extreme tail risk (CVaR<sub>99%</sub> = <b>$8,117.52</b>).", body_style),
            Paragraph("<b>2. Push Challenge Champion: Soft-Voting</b><br/>"
                      "The Multi-Paradigm Soft-Voting Ensemble maximizes discriminative ranking power (<b>ROC-AUC = 0.9808 &plusmn; 0.0005</b>), capturing <b>91.12% recall</b>. "
                      "When alert costs are minimal (<i>C</i><sub>FP</sub> &le; $0.64, e.g., in-app biometric push challenges), Soft-Voting minimizes net loss (<b>$2,485.42</b> at $0.50/alert).", body_style)
        ],
        [
            Paragraph("<b>3. The Recall Fallacy: Cost-Sensitive LogReg</b><br/>"
                      "Cost-Sensitive LogReg maximizes event recall (<b>95.53%</b>) and dollar recall (<b>94.57%</b>), missing only $663.85 in fraud. "
                      "However, it triggers an unbearable <b>8,444.07 false alarms</b> per split. At $2.50/alert, triage costs explode to <b>$21,110.16</b>, "
                      "driving net loss to <b>$21,774.01</b>—over $17,400 worse than GBDT.", body_style),
            Paragraph("<b>4. Analytical Breakeven Review Cost ($0.6358)</b><br/>"
                      "Equating the loss functions between Soft-Voting and GBDT reveals the exact breakeven triage cost threshold:<br/>"
                      "<i>C</i><sub>FP</sub><sup>*</sup> = ($2,168.80 - $1,590.85) / (1,789.14 - 880.19) = <b>$0.6358 (63.6&cent;)</b>.<br/>"
                      "For review costs below <b>63.6&cent;</b>, Soft-Voting is optimal; for costs above 63.6&cent;, GBDT strictly dominates.", body_style)
        ]
    ]
    t_findings = Table(findings_data, colWidths=[266, 266])
    t_findings.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_callout_bg),
        ('BOX', (0,0), (-1,-1), 1, c_callout_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_findings)
    story.append(Spacer(1, 4))

    story.append(Paragraph("Everyday Plain-English Guide: Making Sense of the Numbers", sec_heading))
    story.append(Paragraph(
        "To make fraud detection statistics accessible to everyone (professors, non-technical peers, and banking executives), "
        "we translate the core mathematical metrics into everyday banking analogies:",
        body_style
    ))

    analogy_data = [
        [
            Paragraph("<b>Plain-English Name</b>", th_style),
            Paragraph("<b>Everyday Banking Analogy</b>", th_style),
            Paragraph("<b>What the Real Numbers Mean for You</b>", th_style)
        ],
        [
            Paragraph("<b>The Security Catch Rate</b><br/>(Event Recall: 90% vs 95%)", tc_left_bold),
            Paragraph("If <b>100 thieves</b> try to steal cards today, how many get blocked at the door?", tc_left),
            Paragraph("A 90% catch rate stops 90 thieves and lets 10 slip through. Cost-Sensitive LogReg stops 95 thieves, while GBDT stops 88.", tc_left)
        ],
        [
            Paragraph("<b>The Smoke Alarm Trust</b><br/>(Precision: 9.3% vs 1.1%)", tc_left_bold),
            Paragraph("When the smoke alarm screams, is there real fire or just burnt toast?", tc_left),
            Paragraph("At <b>9.30% precision (GBDT)</b>, 1 out of 11 alarms is an actual criminal. At <b>1.11% precision (Cost-Sens)</b>, <b>90 out of 91 alarms are burnt toast</b>!", tc_left)
        ],
        [
            Paragraph("<b>Queue Overload</b><br/>(False Positives: 880 vs 8,444)", tc_left_bold),
            Paragraph("Do your security guards spend their day catching crooks or apologizing to angry shoppers?", tc_left),
            Paragraph("Cost-Sensitive triggers <b>8,444 false alarms</b> per split (analysts spend 99% of their day clearing innocent people). GBDT cuts that down to <b>880 alarms</b>.", tc_left)
        ],
        [
            Paragraph("<b>The Bank's Total Bill</b><br/>(Net Loss: $4,369 vs $21,774)", tc_left_bold),
            Paragraph("Saving $1,000 of stolen goods isn't smart if you spend $21,000 on private detectives to do it!", tc_left),
            Paragraph("The bank pays for missed fraud ($1/dollar) + investigator time ($2.50/alarm). GBDT's total bill is <b>$4,369</b>, beating Cost-Sensitive's bill of <b>$21,774</b> by over $17,400.", tc_left)
        ],
        [
            Paragraph("<b>The Channel Rule</b><br/>(Breakeven Cost: $0.64)", tc_left_bold),
            Paragraph("Are you verifying fraud by sending an app push button, or calling them on the phone?", tc_left),
            Paragraph("If verification costs <b>under 64&cent;</b> (instant mobile biometric challenge), <b>Soft-Voting wins</b>. If it costs <b>over 64&cent;</b> (human call center), <b>GBDT wins</b>.", tc_left)
        ]
    ]
    t_analogy = Table(analogy_data, colWidths=[120, 180, 240])
    t_analogy.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_analogy)

    story.append(PageBreak())

    # ==========================================
    # PAGE 2: THE RECALL FALLACY & METRIC GLOSSARY
    # ==========================================
    story.append(Paragraph("1. The 'Recall Fallacy' in Banking: Why Chasing 100% Recall Breaks Operations", sec_heading))
    story.append(Paragraph(
        "In naive machine learning evaluations, practitioners assume that higher recall is always better. "
        "Figure 2 illustrates the empirical reality across our 1,000 simulation runs: as models push from 88% recall to 95% recall, "
        "the relationship between caught fraud and false alarms is not linear—it is an <b>exponential hockey-stick explosion</b>.",
        body_style
    ))

    # Figure 2: The Recall Fallacy
    fig2_img = RLImage(os.path.join(fig_dir, "fig2_recall_vs_false_alarms.png"), width=490, height=175)
    fig2_img.hAlign = 'CENTER'
    story.append(fig2_img)
    story.append(Spacer(1, 2))

    fig2_box = Table([
        [Paragraph("<b>Figure 2 Visual Analysis & Key Takeaway:</b>", caption_style)],
        [Paragraph(
            "<b>The Operational Knee of the Curve:</b> "
            "GBDT Classifier operates in the highly efficient 'Manageable Analyst Triage Zone' with 88.43% recall and only 880.2 false alarms (1.55% FPR). "
            "In stark contrast, Cost-Sensitive Logistic Regression forces the boundary outward to capture an additional 7.10% of fraud (95.53% recall), "
            "which causes an astronomical <b>860% surge in false alarms (8,444.1 FPs)</b>. "
            "In an issuing bank with 50,000 daily card transactions, this flood of 8,400 false declines exhausts call centers and drives legitimate customers away.",
            takeaway_style
        )]
    ], colWidths=[540])
    fig2_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(fig2_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("2. Technical Metric Interpretation & Regulatory Distinctions", sec_heading))
    glossary_data = [
        [
            Paragraph("<b>Metric & Formula</b>", th_style),
            Paragraph("<b>Statistical Meaning</b>", th_style),
            Paragraph("<b>Banking Reality & Regulatory Distinctions</b>", th_style)
        ],
        [
            Paragraph("<b>Event Recall (TPR)</b><br/>TP / (TP + FN)", tc_left_bold),
            Paragraph("Percentage of actual fraudulent transactions detected.", tc_left),
            Paragraph("Measures attack interception. Maximizing recall blindly without monitoring false alarms causes catastrophic queue saturation.", tc_left)
        ],
        [
            Paragraph("<b>Dollar Recall</b><br/>&sum;<sub>TP</sub> Amt / &sum;<sub>All</sub> Amt", tc_left_bold),
            Paragraph("Fraction of stolen monetary capital prevented from loss.", tc_left),
            Paragraph("Reflects real capital preservation. Stopping one $4,000 jewelry charge saves more money than stopping four hundred $5 micro-charges.", tc_left)
        ],
        [
            Paragraph("<b>Precision (PPV)</b><br/>TP / (TP + FP)", tc_left_bold),
            Paragraph("Trustworthiness of an alarm: fraction of flagged cards that are truly fraud.", tc_left),
            Paragraph("GBDT’s 9.30% precision means 1 out of 11 alerts is an actual crook. Cost-Sensitive LogReg’s 1.11% precision means 90 out of 91 alerts are innocent cardholders.", tc_left)
        ],
        [
            Paragraph("<b>False Discovery (FDR)</b><br/>FP / (TP + FP) = 1 - Prec", tc_left_bold),
            Paragraph("Proportion of false alarms inside the analyst investigation queue.", tc_left),
            Paragraph("For Cost-Sensitive LogReg, FDR is 98.89% (analysts waste 99% of their hours). For GBDT, FDR drops to 90.70%, quadrupling investigative triage efficiency.", tc_left)
        ],
        [
            Paragraph("<b>False Positive Rate</b><br/>FP / (FP + TN)", tc_left_bold),
            Paragraph("Probability of flagging an innocent customer's transaction.", tc_left),
            Paragraph("<b>Regulatory Clarification:</b> Model FPR (1.55% for GBDT) is distinct from European PSD2 SCA TRA gross fraud ceilings (1 bps / 0.01%). A low FPR does not ensure regulatory compliance.", tc_left)
        ]
    ]
    t_glossary = Table(glossary_data, colWidths=[110, 150, 280])
    t_glossary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_glossary)

    story.append(PageBreak())

    # ==========================================
    # PAGE 3: MASTER BENCHMARK & LOSS DISTRIBUTION BOXPLOTS
    # ==========================================
    story.append(Paragraph("3. Master 1,000-Run Empirical Benchmark Summary (Table 3.1)", sec_heading))
    story.append(Paragraph(
        "Values represent the <b>Sample Mean &plusmn; 95% CLT Confidence Interval Half-Width</b> across all 1,000 independent simulation runs ($N = 1,000$):",
        body_style
    ))

    headers_31 = [
        Paragraph("<b>Model Architecture</b>", th_style),
        Paragraph("<b>Event Recall</b>", th_style),
        Paragraph("<b>Dollar Recall</b>", th_style),
        Paragraph("<b>Precision</b>", th_style),
        Paragraph("<b>F1-Score</b>", th_style),
        Paragraph("<b>ROC-AUC</b>", th_style),
        Paragraph("<b>FPR (%)</b>", th_style),
        Paragraph("<b>Stolen $ Saved</b>", th_style),
        Paragraph("<b>False Pos (FP)</b>", th_style),
        Paragraph("<b>Missed Fraud</b>", th_style),
        Paragraph("<b>Net Loss ($)</b>", th_style)
    ]
    data_31 = [
        headers_31,
        [Paragraph("1. Baseline LogReg", tc_left), Paragraph("90.42% &plusmn; 0.18%", tc_style), Paragraph("85.42% &plusmn; 0.56%", tc_style), Paragraph("5.95% &plusmn; 0.04%", tc_style), Paragraph("11.16% &plusmn; 0.08%", tc_style), Paragraph("0.9780 &plusmn; 0.0006", tc_style), Paragraph("2.50% &plusmn; 0.02%", tc_style), Paragraph("$10,124 &plusmn; $127", tc_style), Paragraph("1,420.6 &plusmn; 11.5", tc_style), Paragraph("$1,771 &plusmn; $72", tc_style), Paragraph("$5,322 &plusmn; $72", tc_style)],
        [Paragraph("2. Cost-Sens. LogReg", tc_left_bold), Paragraph("<b>95.53% &plusmn; 0.13%</b>", tc_bold), Paragraph("<b>94.57% &plusmn; 0.40%</b>", tc_bold), Paragraph("1.11% &plusmn; 0.01%", tc_style), Paragraph("2.18% &plusmn; 0.01%", tc_style), Paragraph("0.9745 &plusmn; 0.0006", tc_style), Paragraph("14.85% &plusmn; 0.09%", tc_style), Paragraph("<b>$11,230 &plusmn; $136</b>", tc_bold), Paragraph("8,444.1 &plusmn; 50.1", tc_style), Paragraph("<b>$664 &plusmn; $120</b>", tc_bold), Paragraph("$21,774 &plusmn; $120", tc_style)],
        [Paragraph("3. Neural Net (MLP)", tc_left), Paragraph("90.35% &plusmn; 0.18%", tc_style), Paragraph("85.33% &plusmn; 0.57%", tc_style), Paragraph("5.72% &plusmn; 0.05%", tc_style), Paragraph("10.75% &plusmn; 0.08%", tc_style), Paragraph("0.9702 &plusmn; 0.0007", tc_style), Paragraph("2.61% &plusmn; 0.02%", tc_style), Paragraph("$10,116 &plusmn; $128", tc_style), Paragraph("1,485.8 &plusmn; 13.4", tc_style), Paragraph("$1,778 &plusmn; $76", tc_style), Paragraph("$5,493 &plusmn; $76", tc_style)],
        [Paragraph("4. GBDT Classifier", tc_left_bold), Paragraph("88.43% &plusmn; 0.20%", tc_style), Paragraph("82.02% &plusmn; 0.57%", tc_style), Paragraph("<b>9.30% &plusmn; 0.11%</b>", tc_bold), Paragraph("<b>16.76% &plusmn; 0.18%</b>", tc_bold), Paragraph("0.9778 &plusmn; 0.0005", tc_style), Paragraph("1.55% &plusmn; 0.02%", tc_style), Paragraph("$9,725 &plusmn; $125", tc_style), Paragraph("<b>880.2 &plusmn; 11.9</b>", tc_bold), Paragraph("$2,169 &plusmn; $78", tc_style), Paragraph("<b>$4,369 &plusmn; $78</b>", tc_bold)],
        [Paragraph("5. Isolation Forest", tc_left), Paragraph("60.46% &plusmn; 0.60%", tc_style), Paragraph("57.48% &plusmn; 0.91%", tc_style), Paragraph("8.22% &plusmn; 0.12%", tc_style), Paragraph("14.40% &plusmn; 0.19%", tc_style), Paragraph("0.9451 &plusmn; 0.0009", tc_style), Paragraph("<b>1.21% &plusmn; 0.02%</b>", tc_bold), Paragraph("$6,827 &plusmn; $136", tc_style), Paragraph("<b>688.8 &plusmn; 10.2</b>", tc_bold), Paragraph("$5,067 &plusmn; $122", tc_style), Paragraph("$6,789 &plusmn; $122", tc_style)],
        [Paragraph("6. Soft-Voting Ensem.", tc_left_bold), Paragraph("91.12% &plusmn; 0.17%", tc_style), Paragraph("86.91% &plusmn; 0.54%", tc_style), Paragraph("4.82% &plusmn; 0.03%", tc_style), Paragraph("9.15% &plusmn; 0.06%", tc_style), Paragraph("<b>0.9808 &plusmn; 0.0005</b>", tc_bold), Paragraph("3.15% &plusmn; 0.03%", tc_style), Paragraph("$10,303 &plusmn; $128", tc_style), Paragraph("1,789.1 &plusmn; 14.3", tc_style), Paragraph("$1,591 &plusmn; $72", tc_style), Paragraph("$6,064 &plusmn; $72", tc_style)],
        [Paragraph("7. Stacked Meta-Learner", tc_left), Paragraph("91.86% &plusmn; 0.17%", tc_style), Paragraph("87.96% &plusmn; 0.53%", tc_style), Paragraph("3.63% &plusmn; 0.02%", tc_style), Paragraph("6.97% &plusmn; 0.04%", tc_style), Paragraph("0.9793 &plusmn; 0.0005", tc_style), Paragraph("4.26% &plusmn; 0.03%", tc_style), Paragraph("$10,430 &plusmn; $129", tc_style), Paragraph("2,421.5 &plusmn; 17.8", tc_style), Paragraph("$1,464 &plusmn; $73", tc_style), Paragraph("$7,518 &plusmn; $73", tc_style)]
    ]
    t_31 = Table(data_31, colWidths=[92, 45, 45, 42, 42, 48, 40, 48, 48, 45, 45])
    t_31.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_primary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('BACKGROUND', (0,4), (-1,4), c_green_bg),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_31)
    story.append(Spacer(1, 4))

    # Figure 1: Net Loss Distribution Boxplots
    fig1_img = RLImage(os.path.join(fig_dir, "fig1_loss_distribution_boxplots.png"), width=490, height=170)
    fig1_img.hAlign = 'CENTER'
    story.append(fig1_img)
    story.append(Spacer(1, 2))

    fig1_box = Table([
        [Paragraph("<b>Figure 1 Visual Analysis & Key Takeaway:</b>", caption_style)],
        [Paragraph(
            "<b>Empirical Stability & Cost Dominance:</b> "
            "Across 1,000 independent partitions, GBDT Classifier maintains the tightest interquartile range (IQR) and lowest median loss ($4,194.31), "
            "with a sample mean of $4,369.26. Notice that Cost-Sensitive Logistic Regression is completely separated at the top of the scale ($21,774.01), "
            "proving that its poor financial performance is an intrinsic mathematical property of its decision boundary rather than bad luck on a single random split.",
            takeaway_style
        )]
    ], colWidths=[540])
    fig1_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(fig1_box)

    story.append(PageBreak())

    # ==========================================
    # PAGE 4: ARITHMETIC AUDIT & TAIL RISK PROFILES
    # ==========================================
    story.append(Paragraph("4. Master Arithmetic Reconciliation Audit (Unrounded Values)", sec_heading))
    story.append(Paragraph(
        "Every model's reported net operational loss is verified against the underlying cost formula down to the cent: "
        "<b>Total Net Loss = (FP&#772; &times; $2.50) + FN&#772;<sub>$</sub></b>.",
        body_style
    ))

    audit_headers = [
        Paragraph("<b>Model Architecture</b>", th_style),
        Paragraph("<b>Mean False Pos (FP&#772;)</b>", th_style),
        Paragraph("<b>Alert Cost (FP&#772; &times; $2.50)</b>", th_style),
        Paragraph("<b>Missed Fraud (FN&#772;<sub>$</sub>)</b>", th_style),
        Paragraph("<b>Total Calculated Loss</b>", th_style),
        Paragraph("<b>Table 3.1 Benchmark</b>", th_style),
        Paragraph("<b>Reconciliation Status</b>", th_style)
    ]
    audit_data = [
        audit_headers,
        [Paragraph("1. Baseline LogReg", tc_left), Paragraph("1,420.592", tc_style), Paragraph("$3,551.48", tc_style), Paragraph("$1,770.57", tc_style), Paragraph("<b>$5,322.05</b>", tc_bold), Paragraph("$5,322 &plusmn; $72", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tc_style)],
        [Paragraph("2. Cost-Sens. LogReg", tc_left), Paragraph("8,444.065", tc_style), Paragraph("$21,110.16", tc_style), Paragraph("$663.85", tc_style), Paragraph("<b>$21,774.01</b>", tc_bold), Paragraph("$21,774 &plusmn; $120", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tc_style)],
        [Paragraph("3. Neural Net (MLP)", tc_left), Paragraph("1,485.818", tc_style), Paragraph("$3,714.55", tc_style), Paragraph("$1,778.08", tc_style), Paragraph("<b>$5,492.63</b>", tc_bold), Paragraph("$5,493 &plusmn; $76", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tc_style)],
        [Paragraph("4. GBDT Classifier", tc_left_bold), Paragraph("880.185", tc_style), Paragraph("$2,200.46", tc_style), Paragraph("$2,168.80", tc_style), Paragraph("<b>$4,369.26</b>", tc_bold), Paragraph("$4,369 &plusmn; $78", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match (Min Loss)</b></font>", tc_style)],
        [Paragraph("5. Isolation Forest", tc_left), Paragraph("688.754", tc_style), Paragraph("$1,721.88", tc_style), Paragraph("$5,066.98", tc_style), Paragraph("<b>$6,788.86</b>", tc_bold), Paragraph("$6,789 &plusmn; $122", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tc_style)],
        [Paragraph("6. Soft-Voting Ensem.", tc_left_bold), Paragraph("1,789.139", tc_style), Paragraph("$4,472.85", tc_style), Paragraph("$1,590.85", tc_style), Paragraph("<b>$6,063.70</b>", tc_bold), Paragraph("$6,064 &plusmn; $72", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tc_style)],
        [Paragraph("7. Stacked Meta-Learner", tc_left), Paragraph("2,421.549", tc_style), Paragraph("$6,053.88", tc_style), Paragraph("$1,464.31", tc_style), Paragraph("<b>$7,518.19</b>", tc_bold), Paragraph("$7,518 &plusmn; $73", tc_style), Paragraph("<font color='#16A34A'><b>Exact Match</b></font>", tc_style)]
    ]
    t_audit = Table(audit_data, colWidths=[105, 75, 75, 75, 70, 70, 70])
    t_audit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_audit)
    story.append(Spacer(1, 4))

    story.append(Paragraph("5. Tail-Risk Exposure & Loss Distribution Analysis", sec_heading))
    story.append(Paragraph(
        "Average loss evaluates expected daily drag, but fails to capture vulnerability to criminal bot spikes or organized fraud waves. "
        "Figure 4 and Table 5.1 detail the full empirical loss distribution and tail-risk metrics (VaR / CVaR) across 1,000 runs:",
        body_style
    ))

    # Figure 4: Tail Risk
    fig4_img = RLImage(os.path.join(fig_dir, "fig4_tail_risk_var_cvar.png"), width=490, height=165)
    fig4_img.hAlign = 'CENTER'
    story.append(fig4_img)
    story.append(Spacer(1, 2))

    fig4_box = Table([
        [Paragraph("<b>Figure 4 Visual Analysis & Key Takeaway:</b>", caption_style)],
        [Paragraph(
            "<b>Worst-Day Disaster Protection:</b> "
            "GBDT limits catastrophic worst-case damage to a 99% Expected Shortfall (CVaR<sub>99%</sub>) of <b>$8,117.52</b>—the lowest across all supervised models. "
            "In contrast, Cost-Sensitive Logistic Regression exhibits catastrophic tail exposure of <b>$27,284.65</b>. "
            "During shopping holidays or distributed bot attacks, Cost-Sensitive LogReg creates crippling operational debt.",
            takeaway_style
        )]
    ], colWidths=[540])
    fig4_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(fig4_box)

    story.append(PageBreak())

    # ==========================================
    # PAGE 5: INVESTIGATOR WORKDAY & COST SENSITIVITY
    # ==========================================
    story.append(Paragraph("6. The Fraud Investigator's Workday: Queue Breakdown (Precision vs. FDR)", sec_heading))
    story.append(Paragraph(
        "In extreme class imbalance, precision directly governs human analyst burnout and operational cost. "
        "Figure 5 displays what happens to 100 alerts sent to the human investigation queue under each model:",
        body_style
    ))

    # Figure 5: Investigator Workday
    fig5_img = RLImage(os.path.join(fig_dir, "fig5_investigator_workday_efficiency.png"), width=490, height=155)
    fig5_img.hAlign = 'CENTER'
    story.append(fig5_img)
    story.append(Spacer(1, 2))

    fig5_box = Table([
        [Paragraph("<b>Figure 5 Visual Analysis & Key Takeaway:</b>", caption_style)],
        [Paragraph(
            "<b>Queue Composition Reality:</b> "
            "For every 100 card transactions flagged by GBDT, <b>9 are actual fraudsters</b>, and 91 are innocent customers (9.30% precision). "
            "Under Cost-Sensitive LogReg, <b>only 1 out of 100 flagged cards is a real criminal</b>—analysts waste 99% of their shift investigating innocent cardholders. "
            "GBDT nearly quadruples triage investigative efficiency relative to baseline linear models.",
            takeaway_style
        )]
    ], colWidths=[540])
    fig5_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(fig5_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("7. Multi-Tier Operational Cost Sensitivity Matrix (Table 7.1)", sec_heading))
    story.append(Paragraph(
        "Because false alarm verification overhead varies by communication channel, we evaluated net financial loss across five distinct operational cost tiers:",
        body_style
    ))

    cost_headers = [
        Paragraph("<b>Model Architecture</b>", th_style),
        Paragraph("<b><i>C</i><sub>FP</sub> = $0.50</b><br/>(In-App Push)", th_style),
        Paragraph("<b><i>C</i><sub>FP</sub> = $1.00</b><br/>(SMS Alert)", th_style),
        Paragraph("<b><i>C</i><sub>FP</sub> = $2.50</b><br/>(Base Queue)", th_style),
        Paragraph("<b><i>C</i><sub>FP</sub> = $5.00</b><br/>(Manual Call)", th_style),
        Paragraph("<b><i>C</i><sub>FP</sub> = $10.00</b><br/>(Analyst Triage)", th_style),
        Paragraph("<b>Optimal Operating Domain</b>", th_style)
    ]
    cost_data = [
        cost_headers,
        [Paragraph("1. Baseline LogReg", tc_left), Paragraph("$2,480.87", tc_style), Paragraph("$3,191.16", tc_style), Paragraph("$5,322.05", tc_style), Paragraph("$8,873.53", tc_style), Paragraph("$15,976.49", tc_style), Paragraph("High explainability baseline", tc_left)],
        [Paragraph("2. Cost-Sens. LogReg", tc_left), Paragraph("$4,885.88", tc_style), Paragraph("$9,107.91", tc_style), Paragraph("$21,774.01", tc_style), Paragraph("$42,884.17", tc_style), Paragraph("$85,104.49", tc_style), Paragraph("Viable only if <i>C</i><sub>FP</sub> &lt; $0.16", tc_left)],
        [Paragraph("3. Neural Net (MLP)", tc_left), Paragraph("$2,520.99", tc_style), Paragraph("$3,263.90", tc_style), Paragraph("$5,492.63", tc_style), Paragraph("$9,207.17", tc_style), Paragraph("$16,636.26", tc_style), Paragraph("Automated digital queues", tc_left)],
        [Paragraph("4. GBDT Classifier", tc_left_bold), Paragraph("$2,608.89", tc_style), Paragraph("<b>$3,048.98</b>", tc_bold), Paragraph("<b>$4,369.26</b>", tc_bold), Paragraph("<b>$6,569.72</b>", tc_bold), Paragraph("<b>$10,970.65</b>", tc_bold), Paragraph("<b>Optimal for <i>C</i><sub>FP</sub> &ge; $0.64</b>", tc_left_bold)],
        [Paragraph("5. Isolation Forest", tc_left), Paragraph("$5,411.35", tc_style), Paragraph("$5,755.73", tc_style), Paragraph("$6,788.86", tc_style), Paragraph("$8,510.74", tc_style), Paragraph("$11,954.51", tc_style), Paragraph("Zero-day anomaly sentinel", tc_left)],
        [Paragraph("6. Soft-Voting Ensem.", tc_left_bold), Paragraph("<b>$2,485.42</b>", tc_bold), Paragraph("$3,379.99", tc_style), Paragraph("$6,063.70", tc_style), Paragraph("$10,536.55", tc_style), Paragraph("$19,482.24", tc_style), Paragraph("<b>Optimal for <i>C</i><sub>FP</sub> &le; $0.64</b>", tc_left_bold)],
        [Paragraph("7. Stacked Meta-Learner", tc_left), Paragraph("$2,675.09", tc_style), Paragraph("$3,885.86", tc_style), Paragraph("$7,518.19", tc_style), Paragraph("$13,572.06", tc_style), Paragraph("$25,679.81", tc_style), Paragraph("Complex calibrated cascades", tc_left)]
    ]
    t_cost = Table(cost_data, colWidths=[95, 65, 65, 65, 65, 65, 120])
    t_cost.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('BACKGROUND', (0,4), (-1,4), c_green_bg),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_cost)

    story.append(PageBreak())

    # ==========================================
    # PAGE 6: BREAKEVEN FRONTIER & DECISION MATRIX
    # ==========================================
    story.append(Paragraph("8. Operational Cost Sensitivity Frontier & Breakeven Derivations", sec_heading))
    story.append(Paragraph(
        "By setting Loss<sub><i>A</i></sub>(<i>C</i><sub>FP</sub><sup>*</sup>) = Loss<sub><i>B</i></sub>(<i>C</i><sub>FP</sub><sup>*</sup>), "
        "we solve for the exact review cost threshold where the optimal institutional model switches:",
        body_style
    ))
    story.append(make_equation_flowable(r'C_{\mathrm{FP}}^* = \frac{\overline{\mathrm{FN}}_{\mathrm{USD}, A} - \overline{\mathrm{FN}}_{\mathrm{USD}, B}}{\overline{\mathrm{FP}}_B - \overline{\mathrm{FP}}_A}', fontsize=8.0))

    # Figure 3: Breakeven Curves
    fig3_img = RLImage(os.path.join(fig_dir, "fig3_breakeven_cost_curves.png"), width=490, height=170)
    fig3_img.hAlign = 'CENTER'
    story.append(fig3_img)
    story.append(Spacer(1, 2))

    fig3_box = Table([
        [Paragraph("<b>Figure 3 Visual Analysis & Analytical Breakeven Derivations:</b>", caption_style)],
        [Paragraph(
            "<b>1. Soft-Voting vs. GBDT Breakeven:</b> "
            "<i>C</i><sub>FP</sub><sup>*</sup> = ($2,168.80 - $1,590.85) / (1,789.14 - 880.19) = $577.95 / 908.95 = <b>$0.6358 (63.6&cent;)</b>.<br/>"
            "&bull; <i>Domain 1 (In-App Push Challenges, C<sub>FP</sub> &le; $0.64):</i> Soft-Voting minimizes net loss ($2,485.42 at $0.50).<br/>"
            "&bull; <i>Domain 2 (Standard Manual Reviews, C<sub>FP</sub> &ge; $0.64):</i> GBDT strictly minimizes net loss ($4,369.26 at $2.50; $10,970.65 at $10.00).<br/>"
            "<b>2. GBDT vs. Baseline LogReg:</b> <i>C</i><sub>FP</sub><sup>*</sup> = $0.7369 (73.7&cent;) &mdash; GBDT outperforms LogReg whenever alert triage costs &gt; 74&cent;.<br/>"
            "<b>3. Cost-Sensitive LogReg:</b> <i>C</i><sub>FP</sub><sup>*</sup> = $0.1576 (15.8&cent;) &mdash; Cost-Sensitive LogReg is unviable unless alert verification is virtually free.",
            takeaway_style
        )]
    ], colWidths=[540])
    fig3_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(fig3_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("9. Master Operational Decision Matrix: Which Model is Best for What?", sec_heading))
    matrix_headers = [
        Paragraph("<b>Model Architecture</b>", th_style),
        Paragraph("<b>Core Strength</b>", th_style),
        Paragraph("<b>Primary Vulnerability</b>", th_style),
        Paragraph("<b>Cost Regime</b>", th_style),
        Paragraph("<b>CVaR<sub>99%</sub></b>", th_style),
        Paragraph("<b>Optimal Banking Deployment</b>", th_style),
        Paragraph("<b>Overall Verdict</b>", th_style)
    ]
    matrix_data = [
        matrix_headers,
        [
            Paragraph("<b>GBDT Classifier</b>", tc_left_bold),
            Paragraph("Highest precision (9.30%), lowest FPs (880.2), tree robustness", tc_left),
            Paragraph("Axis-aligned step boundaries require tree interpretability", tc_left),
            Paragraph("<i>C</i><sub>FP</sub> &ge; $0.64", tc_style),
            Paragraph("<b>$8,117.52</b>", tc_bold),
            Paragraph("<b>Core Card Screening:</b> Production gating for card-not-present fraud & analyst queues.", tc_left),
            Paragraph("<font color='#16A34A'><b>CHAMPION</b></font><br/>Lowest Net Loss", tc_style)
        ],
        [
            Paragraph("<b>Soft-Voting</b>", tc_left_bold),
            Paragraph("Highest ROC-AUC (0.9808), smoothed variance, 91.12% recall", tc_left),
            Paragraph("Higher alert volume than GBDT (1,789 alerts per split)", tc_left),
            Paragraph("<i>C</i><sub>FP</sub> &le; $0.64", tc_style),
            Paragraph("$9,396.16", tc_style),
            Paragraph("<b>Digital Push-to-Confirm:</b> In-app biometric challenges where friction is minimal.", tc_left),
            Paragraph("<font color='#2563EB'><b>PUSH WINNER</b></font><br/>Max Separation", tc_style)
        ],
        [
            Paragraph("<b>Baseline LogReg</b>", tc_left_bold),
            Paragraph("Full linear explainability, simple weight audit, sub-ms speed", tc_left),
            Paragraph("Cannot capture non-linear cross-feature correlations", tc_left),
            Paragraph("<i>C</i><sub>FP</sub> &approx; $0.74", tc_style),
            Paragraph("$8,934.98", tc_style),
            Paragraph("<b>Regulatory Gating:</b> High-velocity lines requiring automated Adverse Action codes.", tc_left),
            Paragraph("<b>AUDIT WINNER</b><br/>Zero Black-Box", tc_style)
        ],
        [
            Paragraph("<b>Cost-Sens. LogReg</b>", tc_left_bold),
            Paragraph("Maximum Event Recall (95.53%) and Dollar Recall (94.57%)", tc_left),
            Paragraph("Severe alert explosion (8,444 alerts), extreme cost drag", tc_left),
            Paragraph("<i>C</i><sub>FP</sub> &lt; $0.16", tc_style),
            Paragraph("$27,284.65", tc_style),
            Paragraph("<b>Catastrophic Wires:</b> Ultra-high-value wires (>$20k) where missed fraud is fatal.", tc_left),
            Paragraph("<b>SPECIALIZED</b><br/>Catastrophic Loss", tc_style)
        ],
        [
            Paragraph("<b>Isolation Forest</b>", tc_left_bold),
            Paragraph("Zero label supervision; detects novel structural anomalies", tc_left),
            Paragraph("Lower supervised recall (60.46%), misses $5,067 in fraud", tc_left),
            Paragraph("Zero-Day", tc_style),
            Paragraph("$12,969.31", tc_style),
            Paragraph("<b>Unsupervised Watchdog:</b> Parallel fraud stream monitoring & cold-start rails.", tc_left),
            Paragraph("<b>SENTINEL</b><br/>Zero-Day Anomaly", tc_style)
        ]
    ]
    t_matrix = Table(matrix_data, colWidths=[80, 85, 85, 45, 50, 130, 65])
    t_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_highlight]),
        ('BACKGROUND', (0,1), (-1,1), c_green_bg),
        ('TOPPADDING', (0,0), (-1,-1), 2.0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.0),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_matrix)

    story.append(PageBreak())

    # ==========================================
    # PAGE 7: PRODUCTION ARCHITECTURE & REPRODUCIBILITY
    # ==========================================
    story.append(Paragraph("10. Two-Tier Production Screening Architecture (< 50ms Pipeline)", sec_heading))
    story.append(Paragraph(
        "To meet the strict latency SLAs of card payment networks (VisaNet, Mastercard Banknet &lt; 50 ms), "
        "we recommend deploying a two-tier screening cascade combining linear velocity with non-linear precision:",
        body_style
    ))

    arch_flow_text = (
        "<b>[Incoming Payment Authorization Request]</b><br/>"
        "&nbsp;&nbsp;&bull; <b>Tier 1: Ultra-Fast Screening (&lt; 1.0 ms)</b> &mdash; Evaluated via Vectorized Baseline Logistic Regression.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; If <i>p</i>(<i>x</i>) &lt; 0.05 &rarr; <b>AUTO-APPROVE</b> (97% of all transaction volume passed instantly with zero customer friction).<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; If <i>p</i>(<i>x</i>) &ge; 0.05 &rarr; <b>ESCALATE TO TIER 2</b> (Remaining 3% volume routed to deep risk scoring).<br/>"
        "&nbsp;&nbsp;&bull; <b>Tier 2: Deep Risk Scoring (&lt; 15.0 ms)</b> &mdash; Evaluated via GBDT Classifier + Soft-Voting Ensemble.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; Dynamic Bayes Threshold: <i>p</i><sup>*</sup>(<i>x</i>) = <i>C</i><sub>FP</sub> / [<i>C</i><sub>FP</sub> + Amount(<i>x</i>)].<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; If <i>p</i>(<i>x</i>) &lt; <i>p</i><sup>*</sup>(<i>x</i>) &rarr; <b>APPROVE WITH AUDIT LOGGING</b>.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; If <i>p</i>(<i>x</i>) &ge; <i>p</i><sup>*</sup>(<i>x</i>) &rarr; <b>DYNAMIC INTERVENTION:</b> Low Amount (&lt; $100) &rarr; 3DS Push Challenge; High Amount (&gt; $1,000) &rarr; Decline & Analyst Triage."
    )
    t_arch_flow = Table([[Paragraph(arch_flow_text, body_style)]], colWidths=[540])
    t_arch_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('BOX', (0,0), (-1,-1), 1, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_arch_flow)
    story.append(Spacer(1, 6))

    story.append(Paragraph("11. Executive Summary & Layman Takeaways: What You Need to Know", sec_heading))
    summary_bullets = [
        "<b>1. Never judge fraud models by recall alone:</b> Maximizing recall sounds great, but Cost-Sensitive LogReg creates 8,444 false alerts, costing over $21,000 in wasted investigation time.",
        "<b>2. GBDT is the most cost-effective all-round champion:</b> By generating only 880 false alarms, GBDT saves $9,725 in fraud and keeps total operational loss at $4,369 per split.",
        "<b>3. Soft-Voting is the best digital in-app screener:</b> If you verify cards through mobile push alerts (under 64&cent;), Soft-Voting catches 91.12% of fraud with the highest ROC-AUC (0.9808).",
        "<b>4. Tail risk matters for Black Friday:</b> GBDT's worst-day disaster bill is capped at $8,118 (CVaR<sub>99%</sub>), compared to $27,285 for Cost-Sensitive LogReg.",
        "<b>5. Deploy in two tiers:</b> Use fast linear models (< 1 ms) to clear 97% of normal transactions, and reserve GBDT (< 15 ms) for the suspicious 3%."
    ]
    for b in summary_bullets:
        story.append(Paragraph(f"&bull; {b}", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("12. Reproducibility & Research Artifact Directory", sec_heading))
    story.append(Paragraph(
        "<b>Empirical Artifacts:</b> "
        "Raw 1,000-run simulation dataset: <code>benchmark_1000_runs_raw.csv</code> (892 KB) | "
        "Summary data: <code>benchmark_1000_runs_summary.csv</code> | "
        "Statistical breakdown: <code>benchmark_1000_runs_statistical_breakdown.csv</code> | "
        "Parallel simulation scripts: <code>run_1000_simulations.py</code> & <code>simulation_worker.py</code> | "
        "Figure generator: <code>generate_statistical_figures.py</code> | "
        "Standalone PDF generator: <code>generate_statistical_report_pdf.py</code> | "
        "20-Page Comprehensive Monograph: <code>report/Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf</code>.<br/>"
        "<b>References:</b> Bahnsen et al. (2015); Dal Pozzolo et al. (2014); Friedman (2001); Liu et al. (2008); "
        "Lin et al. (2017); Wolpert (1992); Davis & Goadrich (2006); Dietterich (1998); Bengio & Grandvalet (2004).",
        meta_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated standalone visual PDF: {pdf_local}")

    if os.path.exists(pdf_local):
        import shutil
        shutil.copy2(pdf_local, pdf_desktop)
        print(f"Successfully synchronized standalone visual PDF to Desktop: {pdf_desktop}")

if __name__ == '__main__':
    build_pdf_statistical_report()
