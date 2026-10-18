# 💳 AI-Powered Credit Card Fraud Detection & Financial Risk Intelligence Platform
### *A Production-Grade Research Monograph & End-to-End Implementation from Mathematical First Principles*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Pure NumPy](https://img.shields.io/badge/Pure_NumPy-7_Models_From_Scratch-orange.svg)](https://numpy.org/)
[![Monte Carlo](https://img.shields.io/badge/Monte_Carlo-1%2C000_Runs_%287%2C000_Fits%29-purple.svg)](benchmark_1000_runs_summary.csv)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle_ULB_%28284%2C807_Rows%29-20BEFF.svg)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
[![PDF Report](https://img.shields.io/badge/Research_Report-Publication_PDF-red.svg)](report/Statistical_Report_1000_Simulations.pdf)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Hub-FF4B4B.svg)](archive/app.py)

---

## 📌 Executive Summary

This repository hosts an advanced machine learning research platform for **Credit Card Fraud Detection** on the benchmark [ULB Machine Learning Group dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 European cardholder transactions, 492 frauds, 0.1727% class asymmetry).

The project is structured into two complementary tracks:
1. **First-Principles Research & 1,000-Run Monte Carlo Benchmark:**  
   **Seven machine learning architectures implemented 100% from scratch in pure NumPy** (strictly zero `scikit-learn`, `xgboost`, `torch`, or `scipy` ML dependencies for core learning). Evaluated across **1,000 independent stratified Monte Carlo simulations** (7,000 total model fits; 56,962,000 out-of-sample inferences) to eliminate partition luck and quantify real financial loss under asymmetric risk ($C_{\text{FN}} = \text{Amount}, C_{\text{FP}} = \$2.50$).
2. **Interactive Intelligence & Operations Center:**  
   A full **Streamlit web application** featuring a real-time transaction simulator, streaming synthetic transaction pipelines, dynamic fraud risk gauges, and geographical risk heatmaps.

---

## 🏛️ System Architecture

```
                                      [ Kaggle Benchmark Dataset ]
                                     (284,807 transactions, 0.17% fraud)
                                                    │
                                                    ▼
                                    [ Resampling & Cost Calibration ]
                                ├── Custom Vectorized SMOTE (Convex Interpolation)
                                ├── Balanced Undersampling (50:50)
                                └── Asymmetric Financial Cost Function (C_FN = $, C_FP = $2.50)
                                                    │
            ┌───────────────────────────────────────┴───────────────────────────────────────┐
            ▼                                                                               ▼
[ Track 1: Seven From-Scratch Architectures ]                        [ Track 2: Empirical Evaluation & Product ]
1. Baseline Logistic Regression (BCE Loss)                           ├── 1,000-Run Monte Carlo Simulation ($N=1000$)
2. Cost-Sensitive LogReg (Amount-Weighted Loss)                      ├── 5 Publication-Grade Statistical Figures
3. Multi-Layer Perceptron (Deep Neural Net: d→32→16→1)               ├── Standalone 20-Page Research Monograph (PDF)
4. Gradient Boosted Decision Trees (Newton-Raphson)                  ├── LaTeX Course Submission Report (`report.tex`)
5. Isolation Forest (Unsupervised Randomized Trees)                  ├── Industry vs. Scratch Benchmark Suite
6. Multi-Paradigm Hybrid Soft-Voting Ensemble                       └── Interactive Streamlit Web Platform
7. Stacked Meta-Learner (Stacked Generalization)
```

---

## 🔬 The Seven From-Scratch Architectures (Pure NumPy)

| # | Architecture | Mathematical Formulation | Key Innovation / Role |
| :---: | :--- | :--- | :--- |
| **1** | **Baseline Logistic Regression** | $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$, BCE Loss | High-throughput linear probabilistic baseline with $\epsilon$-clipping |
| **2** | **Cost-Sensitive LogReg** | $\mathcal{L}_{\text{cost}} = -\frac{1}{\sum c^{(i)}} \sum c^{(i)} [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$ | Weighting $c^{(i)} = 1 + \kappa \ln(\text{Amount}+1)$ penalizes large fraud misses |
| **3** | **Multi-Layer Perceptron (MLP)** | $d \to 32 \to 16 \to 1$ with ReLU & Momentum | He normal initialization, vectorized backprop, and momentum updates |
| **4** | **GBDT Classifier** | $r_i = y_i - \hat{p}_i$, $\gamma_{jm} = \frac{\sum r_i}{\sum \hat{p}_i(1-\hat{p}_i)}$ | Second-order Newton-Raphson leaf optimization for non-linear interactions |
| **5** | **Isolation Forest** | $s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$ | Unsupervised anomaly isolation trees without relying on true class labels |
| **6** | **Soft-Voting Ensemble** | $P_{\text{ens}}(\mathbf{x}) = \sum_{k=1}^4 w_k P_k(y=1 \mid \mathbf{x})$ | Blends linear, tree, deep net, and anomaly scores on the unit simplex |
| **7** | **Stacked Meta-Learner** | $\hat{y} = \mathcal{M}(P_{\text{OOF}}^{(1)}, \dots, P_{\text{OOF}}^{(K)})$ | Out-of-fold cross-validated meta-features trained on a meta-estimator |

---

## 📊 Master 1,000-Run Monte Carlo Benchmark ($N = 1,000$)

Values represent the **Sample Mean $\pm$ 95% CLT Confidence Interval Half-Width** across 1,000 independent stratified 80/20 partitions (testing on 56,962 transactions per split):

| Model Architecture | Event Recall | Dollar Recall | Precision | F1-Score | ROC-AUC | FPR (%) | False Positives | Total Net Loss ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline LogReg** | 90.42% ± 0.18% | 85.42% ± 0.56% | 5.95% ± 0.04% | 11.16% ± 0.08% | 0.9780 ± 0.0006 | 2.50% | 1,420.6 ± 11.5 | \$5,322 ± \$72 |
| **2. Cost-Sensitive LogReg** | **95.53% ± 0.13%** | **94.57% ± 0.40%** | 1.11% ± 0.01% | 2.18% ± 0.01% | 0.9745 ± 0.0006 | 14.85% | 8,444.1 ± 50.1 | \$21,774 ± \$120 |
| **3. Neural Network (MLP)** | 90.35% ± 0.18% | 85.33% ± 0.57% | 5.72% ± 0.05% | 10.75% ± 0.08% | 0.9702 ± 0.0007 | 2.61% | 1,485.8 ± 13.4 | \$5,493 ± \$76 |
| **4. GBDT Classifier** | 88.43% ± 0.20% | 82.02% ± 0.57% | **9.30% ± 0.11%** | **16.76% ± 0.18%** | 0.9778 ± 0.0005 | **1.55%** | **880.2 ± 11.9** | **\$4,369 ± \$78** |
| **5. Isolation Forest** | 60.46% ± 0.60% | 57.48% ± 0.91% | 8.22% ± 0.12% | 14.40% ± 0.19% | 0.9451 ± 0.0009 | **1.21%** | **688.8 ± 10.2** | \$6,789 ± \$122 |
| **6. Soft-Voting Ensemble** | 91.12% ± 0.17% | 86.91% ± 0.54% | 4.82% ± 0.03% | 9.15% ± 0.06% | **0.9808 ± 0.0005** | 3.15% | 1,789.1 ± 14.3 | \$6,064 ± \$72 |
| **7. Stacked Meta-Learner** | 91.86% ± 0.17% | 87.96% ± 0.53% | 3.63% ± 0.02% | 6.97% ± 0.04% | 0.9793 ± 0.0005 | 4.26% | 2,421.5 ± 17.8 | \$7,518 ± \$73 |

### 🏆 Key Empirical Takeaways:
1. **Total Net Financial Loss Champion:** **GBDT Classifier** achieves the lowest operational cost (**\$4,369.26**), generating only 880 false alarms per split and reducing tail risk ($\text{CVaR}_{99\%} = \$8,117.52$).
2. **Discriminative Champion:** The **Soft-Voting Ensemble** achieves the highest discriminative ranking power across all architectures (**ROC-AUC = 0.9808**).
3. **The "Recall Fallacy":** Cost-Sensitive LogReg captures the highest recall (95.53%), but causes an alert avalanche of **8,444 false positives** ($14.85\%$ FPR), raising operational costs by over **\$17,400 per split**.
4. **Analytical Breakeven:** For mobile automated verification ($C_{\text{FP}} \le \$0.64$), Soft-Voting is optimal; for manual human review ($C_{\text{FP}} > \$0.64$), GBDT is mathematically superior.

---

## 📂 Repository File Directory

```bash
Credit-Card-Fraud-Detection-Model/
│
├── data/
│   ├── creditcard.csv                             # Benchmark Kaggle dataset (284,807 rows)
│   └── live_transactions.csv                      # Live simulator transaction buffer
│
├── notebooks/                                     # Step-by-Step From-Scratch Jupyter Notebooks
│   ├── step1_smote_and_features.ipynb             # Vectorized SMOTE & Baseline LogReg
│   ├── step2_neural_net_scratch.ipynb             # Multi-Layer Perceptron (MLP) from scratch
│   ├── step3_cost_matrix_kfold.ipynb              # Cost Matrix & Stratified Cross-Validation
│   ├── step4_gbdt_scratch.ipynb                   # Gradient Boosted Trees (GBDT) from scratch
│   ├── step5_isolation_forest_scratch.ipynb       # Unsupervised Isolation Forest from scratch
│   ├── step6_cost_sensitive_loss_scratch.ipynb    # Cost-Sensitive LogReg from scratch
│   ├── step7_hybrid_ensemble_scratch.ipynb        # Multi-Paradigm Soft-Voting Ensemble
│   ├── step8_instance_dependent_thresholding.ipynb# Cost-Optimal Dynamic Thresholding
│   ├── step9_focal_loss_scratch.ipynb             # Focal Loss optimization from scratch
│   ├── step10_stacked_generalization_scratch.ipynb# Stacked Generalization Meta-Learner
│   ├── fraud_detection_master_capstone.ipynb      # Master Capstone notebook unifying all 7 models
│   ├── fraud_detection.ipynb                      # Single-model academic submission notebook
│   └── fraud_detection_multimodel.ipynb           # 4-model comparative notebook
│
├── benchmark_industry_vs_scratch/                 # Head-to-Head Industry Benchmark Study
│   ├── run_benchmark.py                           # Python runner executing the 6 model pairs
│   ├── benchmark_head_to_head.ipynb               # Interactive comparison notebook
│   ├── benchmark_results.csv                      # Raw empirical metric results
│   ├── figures/                                   # Comparison bar charts, speed plots & ROC curves
│   └── README.md                                  # Full study report + resume bullet points
│
├── simulations/                                   # 1,000-Run Monte Carlo Statistical Stability Suite
│   ├── README.md                                  # Detailed simulation documentation & metric glossary
│   ├── run_1000_simulations.py                    # Multi-process 1,000-run runner across 12 CPU cores
│   ├── run_100_simulations.py                     # Rapid 100-run test runner (~1 minute)
│   ├── simulation_worker.py                       # Isolated worker process for parallel evaluation
│   ├── benchmark_1000_runs_raw.csv                # Complete 1,000-iteration raw metric dataset (892 KB)
│   ├── benchmark_1000_runs_summary.csv            # Aggregated Mean, StdDev, CI95, and VaR summary
│   ├── benchmark_1000_runs_statistical_breakdown.csv # Detailed statistical distributions
│   └── benchmark_100_runs_*.csv                   # Baseline 100-iteration datasets
│
├── report/                                        # Research Reports, LaTeX & Generation Scripts
│   ├── report.tex                                 # Academic project report in LaTeX (ready for Overleaf)
│   ├── Statistical_Report_1000_Simulations.pdf    # Standalone publication report (PDF)
│   ├── statistical_report_1000_simulations.md     # Markdown edition with full tables & analogies
│   ├── Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf # 20-Page Research Monograph (PDF)
│   ├── project_walkthrough_guide.md               # Complete architectural walkthrough
│   ├── figures_statistical/                       # 5 High-resolution publication figures
│   └── scripts/                                   # Automated PDF and figure generation pipelines
│       ├── README.md                              # Script execution guide
│       ├── generate_pdf_report.py                 # 20-page monograph generator
│       ├── generate_statistical_report_pdf.py     # Statistical report generator
│       └── generate_statistical_figures.py        # Statistical figure generator
│
├── rubrics/                                       # Course Evaluation Rubrics
│   ├── README.md                                  # Course requirements & marking scheme breakdown
│   ├── PDS_Project Rubrics.pdf                    # Official course rubric document
│   └── PDS_Project_Rubrics.pdf                    # Rubric backup copy
│
├── archive/                                       # Production Web Platform (Archived for clean submission)
│   ├── app.py                                     # Interactive Streamlit Intelligence Dashboard
│   ├── transaction_generator.py                   # Synthetic streaming transaction engine
│   └── models/                                    # Pre-trained models and scalers
│
├── requirements.txt                               # Lightweight academic dependencies
├── requirements_full.txt                          # Full dependencies (Streamlit, Scikit-Learn, XGBoost, etc.)
├── context.txt                                    # Plaintext project handover prompt for new IDE windows
├── SESSION_CONTEXT.md                             # Continuity context handoff document
├── README.md                                      # Course submission README (Academic focus)
└── README_FULL.md                                 # Master Research Platform README (This file)
```

---

## 🚀 Execution & Usage

### 1. Environment Setup
```bash
git clone https://github.com/Manas-Jain-24/fraud-intelligence-platform.git
cd fraud-intelligence-platform

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

pip install -r requirements.txt
```

### 2. Run the Notebooks
To explore the 7 models and full capstone:
```bash
jupyter notebook notebooks/fraud_detection_master_capstone.ipynb
```

### 3. Re-run or Inspect the Monte Carlo Benchmark
The full 1,000-run simulation results are saved in `simulations/benchmark_1000_runs_summary.csv`. To run a fast 100-run verification:
```bash
python simulations/run_100_simulations.py
```
Or the full 1,000-run benchmark:
```bash
python simulations/run_1000_simulations.py
```

### 4. Run the Industry vs. Scratch Benchmark
```bash
pip install scikit-learn xgboost
python benchmark_industry_vs_scratch/run_benchmark.py
```

### 5. Launch the Interactive Web Dashboard
```bash
pip install -r requirements_full.txt
python -c "import shutil, os; [shutil.copy(os.path.join('archive', f), f) for f in ['app.py', 'transaction_generator.py'] if os.path.exists(os.path.join('archive', f))]"
streamlit run app.py
```

---

## 📄 Academic Compliance & References
* **Dietterich, T. G. (1998):** *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.*
* **Elkan, C. (2001):** *The Foundations of Cost-Sensitive Learning.*
* **Dal Pozzolo, A., et al. (2015):** *Calibrating Probability with Undersampling for Unbalanced Classification.*
* **Turnitin Compliance:** Unique mathematical formulations and code written from first principles.
