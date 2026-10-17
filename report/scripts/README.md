# Report & Figure Generation Scripts

This directory houses the automated generation pipelines responsible for compiling the publication PDFs and high-resolution statistical charts.

---

## 1. Directory Manifest

| Script | Output Artifact | Description |
| :--- | :--- | :--- |
| `generate_pdf_report.py` | `report/Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf` | Compiles the 20+ page master capstone research guide using ReportLab Flowables, mathematical equations, and custom layouts. |
| `generate_statistical_report_pdf.py` | `report/Statistical_Report_1000_Simulations.pdf` | Compiles the standalone 12-section publication report covering the 1,000-run Monte Carlo results, Value-at-Risk, and banking trade-offs. |
| `generate_statistical_figures.py` | `report/figures_statistical/` (5 figures) | Ingests `simulations/benchmark_1000_runs_raw.csv` and renders publication-quality 300 DPI vector charts. |

---

## 2. Generated Visual Figures (`figures_statistical/`)

1. **`fig1_loss_distribution_boxplots.png`**: Net financial operational loss boxplots with diamond mean markers across all 7 models.
2. **`fig2_recall_vs_false_alarms.png`**: The "Recall Fallacy" hockey-stick curve contrasting GBDT's efficient frontier with Cost-Sensitive LogReg's false alarm explosion.
3. **`fig3_empirical_tail_distributions.png`**: KDE and empirical tail loss distributions detailing 95% and 99% Value-at-Risk ($VaR$) thresholds.
4. **`fig4_capital_protected_vs_risk.png`**: Risk-return frontier plotting protected fraud dollars against operational error volatility.
5. **`fig5_production_two_tier_pipeline.png`**: High-throughput two-tier architectural schematic (Sub-millisecond screener + GBDT deep inspector).

---

## 3. How to Execute

### Re-render Statistical Figures
```bash
python report/scripts/generate_statistical_figures.py
```

### Re-compile 1,000-Run Statistical PDF Report
```bash
python report/scripts/generate_statistical_report_pdf.py
```

### Re-compile 20+ Page Comprehensive Research Monograph
```bash
python report/scripts/generate_pdf_report.py
```
*Note: Scripts automatically resolve asset paths whether executed from root or inside `report/scripts/`.*
