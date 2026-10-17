# Course Rubrics & Academic Requirements

This directory stores the official course rubrics for the **Python for Data Science (PDS)** capstone project at IIIT Sri City.

---

## 1. Grade Weight & Timeline

- **Project Contribution:** 50 Marks (20% of final course grade).
- **Final Submission Deadline:** November 15, 2026.
- **Deliverables:**
  - `report.pdf`: Academic project report submitted via Turnitin (plagiarism check).
  - `Project_XXX.zip`: Clean zip archive containing code, notebooks, dataset instructions, and report.

---

## 2. Rubric Evaluation Matrix (50 Marks)

| Dimension | Weight | Course Requirement | Our Project Implementation |
| :--- | :--- | :--- | :--- |
| **Problem Statement & Theory** | 10% (5 Marks) | Rigorous mathematical and theoretical foundation of fraud detection, imbalanced data, and loss metrics. | Complete mathematical derivation in `report/report.tex`, `report/project_walkthrough_guide.md`, and notebooks. |
| **From-Scratch Implementation** | 20% (10 Marks) | Strict rule: core models, loss functions, and optimization built directly from first principles in NumPy without ML libraries. | All 7 models (Logistic Regression, Cost-Sensitive LogReg, Deep MLP, GBDT, Isolation Forest, Soft-Voting, Stacking) implemented in pure NumPy. |
| **Experimental Analysis & Results** | 10% (5 Marks) | Comprehensive empirical evaluation, confusion matrices, ROC-AUC, thresholding, and comparative tables. | Single-split, 100-run, and 1,000-run Monte Carlo simulations with exact empirical distributions, CI95 margins, and Value-at-Risk ($VaR_{99\%}$). |
| **LaTeX Report & Formatting** | 10% (5 Marks) | Professional LaTeX document with title page, student contributions table, formal math formatting, and citations. | Complete LaTeX report in `report/report.tex` following IEEE/ACM style with formal equations and contribution table. |
| **Oral Defense / Viva & Demo** | 50% (25 Marks) | Clear explanation of architectural trade-offs, code walkthrough, and practical insights. | Ready-to-demo Jupyter notebooks (`notebooks/fraud_detection.ipynb`, `notebooks/fraud_detection_master_capstone.ipynb`) and empirical charts. |

---

## 3. Rubric Files

- `PDS_Project Rubrics.pdf`: Official PDF rubric document detailing marking guidelines.
- `PDS_Project_Rubrics.pdf`: Exact copy retained for redundancy.
