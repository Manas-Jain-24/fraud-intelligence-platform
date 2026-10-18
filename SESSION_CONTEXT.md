# 📋 PROJECT CONTINUITY & CONTEXT HANDOFF DOCUMENT
**Project:** Credit Card Fraud Detection (Dual-Track: Academic Capstone & Industry Portfolio)  
**Target Workspace Directory:** `C:\Users\HP\Desktop\PDS`  
**Last Updated:** October 17, 2026  
**Author / Student:** Kartik Kumthe

---

## 📌 1. EXECUTIVE MISSION & PROJECT OBJECTIVES

This workspace hosts two complementary tracks for credit card fraud detection on the benchmark **ULB Machine Learning Group Kaggle dataset** (284,807 European cardholder transactions, 492 fraud cases, 0.1727% extreme class asymmetry):

1. **Academic Subject Project (Python for Data Science - PDS):**
   * **Total Marks:** 50 marks (20% weightage toward final grade).
   * **Submission Deadline:** November 15, 2026 at 18:00 hrs.
   * **Submission Format:** A single zip file named `Project_XXX.zip` containing:
     1. Python notebook with implementation
     2. LaTeX report (`report.pdf`) checked via Turnitin.
   * **Strict Professor Constraint (20% Marks):** Code must be implemented **from scratch without using existing ML library packages** (strictly zero `scikit-learn`, `xgboost`, `torch` for the core model).
2. **Personal Portfolio & CV/Interview Project:**
   * A head-to-head empirical benchmark comparing **from-scratch NumPy algorithms** against **industry-standard C++/Cython libraries** (`scikit-learn` and `xgboost`).
   * A full interactive **Streamlit web application** featuring real-time risk gauges, custom transaction simulator, streaming synthetic transaction pipelines, and country fraud heatmaps.

---

## 🏛️ 2. THE SEVEN FROM-SCRATCH MODELS IMPLEMENTED (PURE NUMPY)

All algorithms were built from mathematical first principles in pure NumPy/Pandas without black-box ML packages:

1. **Baseline Logistic Regression:**
   * Linear hypothesis $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$ with Binary Cross-Entropy (BCE) loss.
   * Vectorized analytical gradient updates $\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$ with logit clipping $[-500, 500]$ and epsilon clipping $\epsilon = 10^{-15}$.
   * Notebook: `notebooks/step1_smote_and_features.ipynb`
2. **Cost-Sensitive Logistic Regression:**
   * Amount-weighted cross-entropy loss where fraud weight $c^{(i)} = 1 + \kappa \ln(\text{Amount}^{(i)} + 1)$.
   * Penalizes high-dollar fraud misses heavily, capturing 95.53% recall.
   * Notebook: `notebooks/step6_cost_sensitive_loss_scratch.ipynb`
3. **Multi-Layer Perceptron (Deep Neural Network - MLP):**
   * 2-hidden layer deep architecture ($d \to 32 \to 16 \to 1$) parameterized with He normal initialization.
   * Vectorized forward pass, ReLU activations, Sigmoid output, and momentum-accelerated backpropagation ($\beta = 0.9$).
   * Notebook: `notebooks/step2_neural_net_scratch.ipynb`
4. **Gradient Boosted Decision Trees (GBDT):**
   * Fits regression trees sequentially to negative gradient pseudo-residuals $r_i = y_i - \hat{p}_i$.
   * Optimal leaf values computed via second-order Newton-Raphson approximation.
   * Lowest overall net operational financial loss ($4,369) across 1,000 Monte Carlo runs.
   * Notebook: `notebooks/step4_gbdt_scratch.ipynb`
5. **Unsupervised Isolation Forest:**
   * Constructs randomized recursive partitioning isolation trees without using ground-truth fraud labels.
   * Anomaly score calibrated using average BST search path length $c(n) = 2(\ln(n - 1) + 0.5772) - \frac{2(n - 1)}{n}$.
   * Notebook: `notebooks/step5_isolation_forest_scratch.ipynb`
6. **Multi-Paradigm Hybrid Soft-Voting Ensemble:**
   * Blends predictions across linear, tree, neural net, and anomaly paradigms on the unit simplex.
   * Achieved the highest discriminative ranking power across 1,000 runs (**ROC-AUC = 0.9808**).
   * Notebook: `notebooks/step7_hybrid_ensemble_scratch.ipynb`
7. **Stacked Meta-Learner (Stacked Generalization):**
   * Trains a meta-estimator on out-of-fold cross-validation predictions to correct individual base-model error correlations.
   * Notebook: `notebooks/step10_stacked_generalization_scratch.ipynb`

* **Master Notebook Uniting All 7 Models:**  
  `notebooks/fraud_detection_master_capstone.ipynb`

---

## 📊 3. THE 1,000-RUN MONTE CARLO SIMULATION BENCHMARK ($N = 1,000$)

* **Protocol:** 1,000 independent stratified 80/20 train/test splits (seeds 1000 to 1999) $\rightarrow$ 7,000 independent model fits $\rightarrow$ 56,962,000 out-of-sample predictions.
* **Master Summary CSV:** `benchmark_1000_runs_summary.csv`
* **Publication PDF:** `report/Statistical_Report_1000_Simulations.pdf` & `report/statistical_report_1000_simulations.md`
* **Key Findings:**
  * **Financial Champion:** GBDT achieves the lowest net operational loss (**$4,369.26 ± $77.66** per split) with only 880 false alarms (1.55% FPR) and lowest tail risk ($\text{CVaR}_{99\%} = \$8,117.52$).
  * **Discriminative Champion:** Soft-Voting Ensemble achieves highest ROC-AUC (**0.9808 ± 0.0005**).
  * **The Recall Fallacy:** Cost-Sensitive LogReg achieves 95.53% recall, but causes 8,444 false positives per split, driving operational investigation costs to $21,774.
  * **Breakeven Threshold:** For alert verification costs under **$0.64** (mobile biometric push challenge), Soft-Voting wins; above $0.64$ (manual call center review), GBDT wins.

---

## 🔬 4. HEAD-TO-HEAD BENCHMARK: SCRATCH (NUMPY) VS. INDUSTRY LIBRARIES

Located in dedicated folder: `benchmark_industry_vs_scratch/`

* **Models Evaluated:**
  * Logistic Regression (NumPy Scratch vs. Scikit-Learn)
  * Decision Tree (NumPy Scratch vs. Scikit-Learn)
  * Random Forest (NumPy Scratch vs. Scikit-Learn)
  * Neural Network / MLP (NumPy Scratch vs. Scikit-Learn `MLPClassifier`)
  * Gradient Boosting (NumPy Scratch GBDT vs. XGBoost)
  * Isolation Forest (NumPy Scratch vs. Scikit-Learn)
* **Results Highlights (from `benchmark_results.csv` on 284,807 transactions):**
  * **LogReg:** Scratch ROC-AUC 0.9895 vs. Sklearn 0.9818 (Scratch is 7.8x faster on raw inference: 0.06ms vs 0.47ms).
  * **Random Forest:** Scratch matches Sklearn accuracy identically (92.86%) and catches more frauds (89.91% recall vs 88.07%).
  * **Neural Network:** Scratch MLP beat Sklearn MLP in Accuracy (92.35% vs 91.33%), Precision (96.08% vs 93.40%), ROC-AUC (0.9810 vs 0.9523), and trained 4x faster (0.21s vs 0.88s).
  * **XGBoost:** Both Scratch GBDT and XGBoost achieved identical fraud recall (88.99%).

---

## 📂 5. COMPLETE REPOSITORY FILE DIRECTORY (`C:\Users\HP\Desktop\PDS`)

```bash
C:\Users\HP\Desktop\PDS\
├── app.py                                         # Next-Gen Live Fraud Intelligence Dashboard (Streamlit)
├── models_engine.py                               # Unified From-Scratch NumPy & Industry ML Engine (XAI & Inference)
├── data_generator.py                              # Real-time streaming transaction generator with PCA fidelity
├── README.md                                      # Primary project overview & quickstart (Academic focus)
├── README_FULL.md                                 # Master capstone documentation & full portfolio analysis
├── requirements.txt                               # Lightweight academic dependencies (numpy, pandas, matplotlib)
├── requirements_full.txt                          # Full dependencies (scikit-learn, xgboost, reportlab, streamlit)
├── context.txt                                    # Plaintext project handover prompt for new IDE windows
├── SESSION_CONTEXT.md                             # Continuity context handoff document (This file)
├── .gitignore                                     # Git ignore rules
│
├── data/
│   ├── creditcard.csv                             # Benchmark Kaggle dataset (284,807 rows, 143.8 MB)
│   ├── model_weights.npz                          # Pre-trained weights for all 6 Scratch algorithms & scalers
│   ├── sample_cases.json                          # 20 verified Kaggle test cases for interactive audit
│   └── live_transactions.csv                      # Live simulator transaction buffer
│
├── benchmark_industry_vs_scratch/                 # Head-to-Head Industry Benchmark Study
│   ├── run_benchmark.py                           # Python runner executing the 6 model pairs
│   ├── benchmark_head_to_head.ipynb               # Interactive comparison notebook
│   ├── benchmark_results.csv                      # Raw empirical metric results across all 6 models
│   ├── figures/                                   # Comparison bar charts, speed plots & ROC curves
│   └── README.md                                  # Full study report + resume bullet points
│
├── simulations/                                   # 1,000-Run Monte Carlo Statistical Stability Suite
│   ├── README.md                                  # Detailed simulation documentation & metric glossary
│   ├── run_1000_simulations.py                    # Multi-process 1,000-run runner across 12 CPU cores
│   ├── run_100_simulations.py                     # Rapid 100-run test runner (~1 minute)
│   ├── simulation_worker.py                       # Isolated worker process for parallel evaluation
│   ├── benchmark_12_runs.py                       # Prototype 12-run benchmarking script
│   ├── test_multi_runs.py                         # Multi-run test script
│   ├── benchmark_1000_runs_raw.csv                # Complete 1,000-iteration raw metric dataset (892 KB)
│   ├── benchmark_1000_runs_summary.csv            # Aggregated Mean, StdDev, CI95, and VaR summary
│   ├── benchmark_1000_runs_statistical_breakdown.csv # Detailed statistical distributions
│   ├── benchmark_100_runs_*.csv                   # Baseline 100-iteration datasets
│   └── benchmark_12_runs_summary.csv              # Preliminary 12-run baseline output
│
├── notebooks/                                     # From-Scratch Jupyter Notebooks
│   ├── fraud_detection.ipynb                      # Single-model academic submission notebook
│   ├── fraud_detection_multimodel.ipynb           # 4-model comparative notebook
│   ├── fraud_detection_master_capstone.ipynb      # Master Capstone notebook unifying all 7 models
│   ├── fraud_detection_backup.ipynb               # Safety fallback notebook
│   ├── step1_smote_and_features.ipynb             # Vectorized SMOTE & Baseline LogReg
│   ├── step2_neural_net_scratch.ipynb             # Multi-Layer Perceptron (MLP) from scratch
│   ├── step3_cost_matrix_kfold.ipynb              # Cost Matrix & Stratified Cross-Validation
│   ├── step4_gbdt_scratch.ipynb                   # Gradient Boosted Trees (GBDT) from scratch
│   ├── step5_isolation_forest_scratch.ipynb       # Unsupervised Isolation Forest from scratch
│   ├── step6_cost_sensitive_loss_scratch.ipynb    # Cost-Sensitive LogReg from scratch
│   ├── step7_hybrid_ensemble_scratch.ipynb        # Multi-Paradigm Soft-Voting Ensemble
│   ├── step8_instance_dependent_thresholding.ipynb# Cost-Optimal Dynamic Thresholding
│   ├── step9_focal_loss_scratch.ipynb             # Focal Loss optimization from scratch
│   └── step10_stacked_generalization_scratch.ipynb# Stacked Generalization Meta-Learner
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
├── tests/                                         # Mathematical Unit Tests & Numerical Gradient Verification
│   ├── run_tests.py                               # Standalone colored test runner
│   ├── test_gradient_checking.py                  # Finite-difference analytical gradient tests (<1e-5 error)
│   ├── test_numerical_stability.py                # Logit overflow, extreme amounts & zero-variance guards
│   ├── test_mathematical_invariants.py            # He normal variance & loss monotonicity invariants
│   └── README.md                                  # Complete testing guide & execution instructions
│
├── adversarial_security/                          # Fintech Adversarial Evasion & Red-Teaming Suite
│   ├── adversarial_engine.py                      # FGSM perturbation & boundary search engine
│   ├── run_red_team_audit.py                      # Multi-architecture penetration test runner
│   ├── red_team_results.csv                       # Empirical evasion rates across epsilon budgets
│   ├── figures/                                   # High-resolution adversarial robustness curves
│   └── README.md                                  # Threat model, security findings & resume bullets
│
└── archive/                                       # Production Web Platform (Archived for clean submission)
    ├── app.py                                     # Interactive Streamlit Intelligence Dashboard
    ├── transaction_generator.py                   # Synthetic streaming transaction engine
    └── models/                                    # Pre-trained models and scalers
```

---

## 🎯 6. IMMEDIATE NEXT ACTIONS FOR THE USER

1. **For University Subject Submission:**
   * Open `report/report.tex` (lines 53–75) and replace placeholder Student Names, Roll Numbers, and `Project_XXX` with actual group credentials.
   * Compile `report/report.tex` on Overleaf to generate `report.pdf`.
   * Create `Project_XXX.zip` containing `notebooks/fraud_detection.ipynb` (or `fraud_detection_master_capstone.ipynb`) + `report.pdf`, and submit to Turnitin before November 15, 2026.
2. **For Resume / CV / GitHub Portfolio:**
   * Use the ready-to-copy bullet points in `benchmark_industry_vs_scratch/README.md`.
   * If showcasing as an all-in-one portfolio repo on GitHub, replace `README.md` with `README_FULL.md`.
