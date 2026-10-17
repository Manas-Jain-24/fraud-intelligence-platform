# 💳 Credit Card Fraud Detection — First-Principles ML & Industry Benchmark Platform

An end-to-end Machine Learning research and evaluation platform for credit card fraud detection on the benchmark [ULB Machine Learning Group dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 492 frauds, 0.172% extreme class imbalance).

This project fulfills two complementary objectives:
1. **Academic Subject Capstone (Python for Data Science - PDS):** 100% first-principles implementation in pure **NumPy**, strictly disallowing black-box machine learning libraries (`scikit-learn`, `xgboost`, `pytorch`) to provide full transparency into loss surfaces, analytical backpropagation, and decision frontiers.
2. **Production Portfolio & Industry Benchmark:** A head-to-head empirical evaluation comparing from-scratch algorithms against industry-standard C++/Cython libraries (`scikit-learn` and `xgboost`), backed by an extensive **1,000-run Monte Carlo statistical stability suite**.

---

## 📋 Course Evaluation Criteria Mapping (50 Marks / 20% Weightage)

| Component | Weight | Implementation in Repository |
| :--- | :---: | :--- |
| **Problem Statement & Theory** | 10% (5 Marks) | Mathematical formulation of Binary Cross-Entropy (BCE), Sigmoid hypothesis, analytical gradient derivations, and class imbalance mitigation in `report/report.tex`, `report/project_walkthrough_guide.md`, and notebooks. |
| **From-Scratch Implementation** | 20% (10 Marks) | **Seven machine learning architectures implemented from scratch in pure NumPy**: Baseline LogReg, Cost-Sensitive LogReg, Deep MLP, GBDT Classifier, Isolation Forest, Soft-Voting Ensemble, and Stacked Meta-Learner. |
| **Experimental Results** | 10% (5 Marks) | Quantitative empirical results across single-split, 100-run, and 1,000-run Monte Carlo simulations: Confusion Matrices, Cost-Sensitive Financial Loss ($C_{\text{FN}} = \text{Amount}, C_{\text{FP}} = \$2.50$), ROC-AUC, Value-at-Risk ($VaR_{99\%}$), and Error Volatility. |
| **LaTeX Report & Formatting** | 10% (5 Marks) | Academic LaTeX report (`report/report.tex`) following standard formatting with title page, student credentials, and contribution table, ready for Overleaf compilation and Turnitin submission. |
| **Oral Defense & Demo** | 50% (25 Marks) | Modular step-by-step notebooks (`notebooks/step1_...` to `step10_...`), Capstone notebook (`notebooks/fraud_detection_master_capstone.ipynb`), and interactive benchmark notebook. |

---

## 🏛️ Repository Organization

All repository contents are organized into purposeful, self-contained subdirectories:

```
PDS/
├── README.md                      # Primary project overview & quickstart (This file)
├── README_FULL.md                 # Master capstone portfolio documentation & full analysis
├── requirements.txt               # Pure Python dependencies (numpy, pandas, matplotlib)
├── requirements_full.txt          # Full dependencies (scikit-learn, xgboost, reportlab, streamlit)
├── context.txt                    # Plaintext project handover prompt for new IDE windows
├── SESSION_CONTEXT.md             # Markdown session context & architectural summary
├── .gitignore                     # Git ignore rules
│
├── data/                          # Datasets
│   ├── creditcard.csv             # 284,807 European cardholder transactions (143.8 MB)
│   └── live_transactions.csv      # Live transaction streaming buffer
│
├── notebooks/                     # Academic & exploratory Jupyter notebooks
│   ├── fraud_detection.ipynb      # Main academic submission notebook (single baseline model)
│   ├── fraud_detection_multimodel.ipynb # 4-model comparative notebook
│   ├── fraud_detection_master_capstone.ipynb # Master capstone notebook unifying all 7 models
│   ├── fraud_detection_backup.ipynb # Safety fallback notebook
│   └── step1_... to step10_...    # 10 modular step-by-step notebooks (SMOTE, MLP, GBDT, etc.)
│
├── benchmark_industry_vs_scratch/ # Portfolio benchmark (Pure NumPy vs. Scikit-Learn & XGBoost)
│   ├── README.md                  # Comprehensive empirical report & resume bullet points
│   ├── run_benchmark.py           # Automated benchmarking pipeline
│   ├── benchmark_head_to_head.ipynb # Interactive comparison notebook
│   ├── benchmark_results.csv      # Real performance metrics across 6 models
│   └── figures/                   # Visual comparisons (ROC curves, metrics, inference latency)
│
├── simulations/                   # 1,000-Run Monte Carlo statistical stability suite
│   ├── README.md                  # Detailed simulation documentation & metric glossary
│   ├── run_1000_simulations.py    # Multi-process 1,000-iteration runner across 12 CPU cores
│   ├── run_100_simulations.py     # Rapid 100-iteration verification runner (~1 min)
│   ├── simulation_worker.py       # Isolated worker process for parallel evaluation
│   ├── benchmark_1000_runs_raw.csv # Complete 1,000-iteration raw metric dataset (892 KB)
│   ├── benchmark_1000_runs_summary.csv # Aggregated Mean, StdDev, CI95, and VaR summary
│   ├── benchmark_1000_runs_statistical_breakdown.csv # Detailed statistical distributions
│   └── benchmark_100_runs_*.csv   # Baseline 100-iteration datasets
│
├── report/                        # Academic LaTeX report & generated publication PDFs
│   ├── report.tex                 # Academic LaTeX report (50 marks / 20% course weight)
│   ├── Statistical_Report_1000_Simulations.pdf # Standalone publication report (1,000 runs)
│   ├── statistical_report_1000_simulations.md # Markdown edition with full tables & analogies
│   ├── Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf # 20+ page master monograph
│   ├── project_walkthrough_guide.md # Architectural walkthrough guide
│   ├── figures_statistical/       # 5 High-res publication plots (boxplots, tail loss, etc.)
│   └── scripts/                   # Automated PDF and figure generation pipelines
│       ├── README.md              # Script execution guide
│       ├── generate_pdf_report.py # 20-page monograph generator
│       ├── generate_statistical_report_pdf.py # Statistical report generator
│       └── generate_statistical_figures.py # Statistical figure generator
│
├── rubrics/                       # Course evaluation guidelines
│   ├── README.md                  # Course requirements & marking scheme breakdown
│   ├── PDS_Project Rubrics.pdf    # Official course rubric document
│   └── PDS_Project_Rubrics.pdf    # Rubric backup copy
│
└── archive/                       # Archived prototypes & legacy simulator
    ├── app.py                     # Streamlit intelligence dashboard
    ├── transaction_generator.py   # Synthetic transaction streaming engine
    └── models/                    # Serialized model artifacts
```

---

## 🔬 The Seven From-Scratch Architectures (Pure NumPy)

Every model was constructed from mathematical first principles in pure Python and NumPy:

1. **Baseline Logistic Regression:** Binary Cross-Entropy loss, analytical gradient descent, logit clipping $[-500, 500]$, and $\epsilon$-clipping ($10^{-15}$).
2. **Cost-Sensitive Logistic Regression:** Amount-weighted loss $c^{(i)} = 1 + \kappa \ln(\text{Amount}+1)$ prioritizing catastrophic high-dollar fraud misses (catches 95.53% recall).
3. **Deep Multi-Layer Perceptron (MLP):** $31 \to 32 \to 16 \to 1$ network with He normal initialization, ReLU activations, Sigmoid output, and momentum-accelerated backpropagation ($\beta = 0.9$).
4. **Gradient Boosted Decision Trees (GBDT):** Sequential residual trees fitted to pseudo-residuals $r_i = y_i - \hat{p}_i$ with second-order Newton-Raphson leaf optimization. Achieved lowest overall operational financial loss (\$4,369) and lowest false alarm rate (1.55% FPR).
5. **Unsupervised Isolation Forest:** Randomized recursive partitioning anomaly trees scoring instances via average BST path length $c(n)$, independent of class labels.
6. **Multi-Paradigm Hybrid Soft-Voting Ensemble:** Combines linear, neural net, tree, and anomaly probabilities on the unit simplex. Achieved highest discriminative ranking power (**0.9808 ROC-AUC**).
7. **Stacked Meta-Learner:** Uses out-of-fold cross-validated probability predictions as meta-features with a logistic blender to correct individual model errors.

---

## ⚡ Quickstart Guide

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/kartik7115-edu/Credit-Card-Fraud-Detection-Model.git
cd Credit-Card-Fraud-Detection-Model

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install lightweight runtime dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Notebooks
```bash
# For single-model course submission baseline:
jupyter notebook notebooks/fraud_detection.ipynb

# For master capstone unifying all 7 models:
jupyter notebook notebooks/fraud_detection_master_capstone.ipynb

# For industry vs. scratch head-to-head benchmark:
jupyter notebook benchmark_industry_vs_scratch/benchmark_head_to_head.ipynb
```

### 3. Run the Industry vs. Scratch Benchmark
```bash
# Install benchmark packages (scikit-learn, xgboost)
pip install scikit-learn xgboost

# Execute automated head-to-head comparison
python benchmark_industry_vs_scratch/run_benchmark.py
```

### 4. Run the Monte Carlo Simulation Suite
```bash
# Rapid 100-iteration verification (~1 minute)
python simulations/run_100_simulations.py

# Full 1,000-iteration large-scale evaluation (~13.8 minutes on 12 cores)
python simulations/run_1000_simulations.py
```

### 5. Compile Academic LaTeX Report
Open `report/report.tex` in **Overleaf** or compile locally using `pdflatex`:
```bash
cd report
pdflatex report.tex
pdflatex report.tex
```

---

## 🏆 Key Benchmark Findings (Pure Scratch vs. Industry Libraries)

| Model Pair | Pure Scratch NumPy | Industry Package | Key Finding / Win |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | 93.88% Acc \| 0.9895 AUC | 94.39% Acc \| 0.9818 AUC | Scratch achieved **7.8x faster inference** (0.06ms vs 0.47ms) with higher AUC. |
| **Random Forest** | 92.86% Acc \| 89.91% Recall | 92.86% Acc \| 88.07% Recall | Scratch achieved **higher fraud recall** (+1.84%) with identical accuracy. |
| **Neural Network (MLP)** | **92.35% Acc \| 0.9810 AUC** | 91.33% Acc \| 0.9523 AUC | Scratch beat Sklearn MLP in **Accuracy (+1.02%)**, **Precision (+2.68%)**, and **AUC (+0.0287)** while training **4x faster**. |
| **Gradient Boosting** | 88.99% Recall \| 90.31% Acc | 88.99% Recall \| 92.35% Acc | Scratch matched XGBoost recall exactly (88.99%). |

---

## 📄 License & Academic Integrity

This project is licensed under the MIT License. Developed for academic submission in Python for Data Science (PDS). Plagiarism checks via Turnitin require that the code and LaTeX report remain unique and original.
