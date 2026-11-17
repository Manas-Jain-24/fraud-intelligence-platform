# 📋 PROJECT CONTINUITY & CONTEXT HANDOFF DOCUMENT
**Project:** Credit Card & Mobile Money Fraud Intelligence Platform (Dual-Track: Academic Capstone & Enterprise Portfolio)  
**Target Workspace Directory:** `C:\Users\HP\Desktop\PDS`  
**GitHub Repository:** https://github.com/Manas-Jain-24/fraud-intelligence-platform  
**Last Updated:** October 27, 2026  
**Author / Student:** Manas Jain

---

## 📌 1. EXECUTIVE MISSION & PROJECT OBJECTIVES

This workspace hosts an enterprise-grade, dual-track machine learning research and financial risk platform evaluated across two critical financial crime domains:

1. **Card-Present POS Intelligence (ULB European Dataset):**
   * **Dataset:** 284,807 European cardholder transactions, 492 fraud cases (0.1727% extreme class asymmetry), 28 PCA features + Amount.
   * **Academic Subject Track (Python for Data Science - PDS):**
     * **Total Marks:** 50 marks (20% weightage toward final grade).
     * **Submission Deadline:** November 15, 2026 at 18:00 hrs.
     * **Submission Format:** A single zip file named `Project_XXX.zip` containing:
       1. Python notebook with implementation
       2. LaTeX report (`report.pdf`) checked via Turnitin.
     * **Strict Professor Constraint (20% Marks):** Code must be implemented **from scratch without using existing ML library packages** (strictly zero `scikit-learn`, `xgboost`, `torch` for the core model).
   * **Automated Submission Builder:** Run `python scripts/generate_academic_repo.py` to auto-package the clean, isolated academic submission repository into `C:\Users\HP\Desktop\PDS_Academic_Project`.

2. **Mobile Money & Wire Transfer Intelligence (PaySim Dataset):**
   * **Dataset:** 100,000 curated benchmark transactions with all 8,213 fraud instances.
   * **Governing Mechanism:** Ledger balance conservation laws ($\text{errorBalanceOrig}$, $\text{errorBalanceDest}$, $\text{balanceDrainRatio}$).
   * **Cross-Domain Generalization:** Proves algorithmic robustness beyond PCA latent vectors to deterministic balance discrepancies.

3. **Enterprise Production & Microservice Architecture:**
   * **FastAPI Microservice (`api.py`):** Sub-millisecond inference SLA (0.81ms average round-trip) exposing health probes, metadata, dual-track scoring, and narrative XAI diagnostics.
   * **Operations Dashboard (`app.py`):** Interactive Streamlit 1.66 dashboard featuring live transaction ticker, dynamic risk gauge, transaction simulator, and Bayes-optimal cost matrix tuning ($\tau^*$).
   * **Containerization & CI/CD:** Multi-target `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow (`.github/workflows/ci.yml`) validating Python 3.11–3.13 matrix builds and 27 Pytest unit tests.

---

## 🏛️ 2. THE SEVEN FROM-SCRATCH CREDIT CARD ALGORITHMS (PURE NUMPY)

All algorithms were built from mathematical first principles in pure NumPy/Pandas without black-box ML packages:

1. **Baseline Logistic Regression:**
   * Hypothesis $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$ with BCE loss.
   * Vectorized analytical updates $\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$ with logit clipping $[-500, 500]$ and $\epsilon = 10^{-15}$.
   * Notebook: `notebooks/step1_smote_and_features.ipynb`
2. **Cost-Sensitive Logistic Regression:**
   * Amount-weighted cross-entropy loss where fraud mistake weight $c^{(i)} = 1 + \kappa \ln(\text{Amount}^{(i)} + 1)$.
   * Notebook: `notebooks/step6_cost_sensitive_loss_scratch.ipynb`
3. **Multi-Layer Perceptron (Deep Neural Network - MLP):**
   * 2-hidden layer deep architecture ($d \to 32 \to 16 \to 1$) parameterized with He normal initialization.
   * Vectorized forward pass, ReLU activations, Sigmoid output, and momentum-accelerated backpropagation ($\beta = 0.9$).
   * Notebook: `notebooks/step2_neural_net_scratch.ipynb`
4. **Gradient Boosted Decision Trees (GBDT):**
   * Fits regression trees sequentially to negative gradient pseudo-residuals $r_i = y_i - \hat{p}_i$.
   * Optimal leaf values computed via second-order Newton-Raphson approximation.
   * Notebook: `notebooks/step4_gbdt_scratch.ipynb`
5. **Unsupervised Isolation Forest:**
   * Constructs randomized recursive partitioning isolation trees without using ground-truth fraud labels.
   * Anomaly score calibrated using average BST search path length $c(n) = 2(\ln(n - 1) + 0.5772) - \frac{2(n - 1)}{n}$.
   * Notebook: `notebooks/step5_isolation_forest_scratch.ipynb`
6. **Multi-Paradigm Hybrid Soft-Voting Ensemble:**
   * Blends predictions across linear, tree, neural net, and anomaly paradigms on the unit simplex.
   * Achieved highest discriminative ranking power (**ROC-AUC = 0.9808**).
   * Notebook: `notebooks/step7_hybrid_ensemble_scratch.ipynb`
7. **Stacked Meta-Learner (Stacked Generalization):**
   * Trains a meta-estimator on out-of-fold cross-validation predictions to correct base-model error correlations.
   * Notebook: `notebooks/step10_stacked_generalization_scratch.ipynb`

* **Master Notebook Uniting All 7 Models:** `notebooks/fraud_detection_master_capstone.ipynb`

---

## 📱 3. PAYSIM MOBILE MONEY & WIRE TRANSFER ENGINE (`paysim_engine.py`)

* **Inference Speed:** Pure NumPy Cost-Sensitive Logistic Regression achieves **6.6 μs** inference latency with **0.9519 ROC-AUC** and **86.4% precision**.
* **2,800x Throughput Advantage:** Outperforms Scikit-Learn Random Forest (6.6 μs vs 18.7 ms) allowing line-rate wire verification at 150,000 transactions/second.
* **Deterministic Rules Engine:** Captures ledger drain anomalies before passing to ML classifier, ensuring zero false-negative bypass on classic account drain patterns.

---

## 📊 4. EMPIRICAL BENCHMARKS & MONTE CARLO SIMULATIONS

1. **1,000-Run Monte Carlo Simulation ($N = 1,000$):**
   * 1,000 stratified 80/20 train/test splits $\rightarrow$ 7,000 independent fits $\rightarrow$ 56.9M out-of-sample inferences.
   * **Financial Champion:** GBDT achieves lowest net operational loss (**$4,369.26 ± $77.66** per split) and lowest tail risk ($\text{CVaR}_{99\%} = \$8,117.52$).
   * **Discriminative Champion:** Soft-Voting Ensemble delivers highest ROC-AUC (**0.9808 ± 0.0005**).
   * **Artifacts:** `simulations/benchmark_1000_runs_summary.csv`, `report/Statistical_Report_1000_Simulations.pdf`.
2. **Head-to-Head Industry Benchmark (NumPy vs. Sklearn & XGBoost):**
   * **LogReg:** Scratch ROC-AUC 0.9895 vs. Sklearn 0.9818 (Scratch is 7.8x faster on raw inference: 0.06ms vs 0.47ms).
   * **Random Forest:** Scratch matches Sklearn accuracy identically (92.86%) with higher fraud recall (89.91% vs 88.07%).
   * **Neural Network:** Scratch MLP beat Sklearn MLP in Accuracy (92.35% vs 91.33%), Precision (96.08% vs 93.40%), ROC-AUC (0.9810 vs 0.9523), and trained 4x faster (0.21s vs 0.88s).
   * **Artifacts:** `benchmark_industry_vs_scratch/benchmark_results.csv`.

---

## 📂 5. COMPLETE REPOSITORY FILE DIRECTORY (`C:\Users\HP\Desktop\PDS`)

```bash
C:\Users\HP\Desktop\PDS\
├── api.py                             # High-Throughput FastAPI REST Microservice (0.81ms SLA)
├── app.py                             # Dual-Rail Fraud Operations Dashboard (Streamlit 1.66)
├── models_engine.py                   # Pure NumPy Credit Card ML Engine (XAI & 7 Scratch Models)
├── paysim_engine.py                   # Mobile Money & Wire Ledger Discrepancy Engine
├── data_generator.py                  # Real-Time Streaming Transaction Generator (PCA-calibrated)
├── Dockerfile                         # Multi-target Production Container (FastAPI + Streamlit)
├── docker-compose.yml                 # Orchestration for API Gateway & Analytics Dashboard
├── .dockerignore                      # Build context exclusion rules
├── requirements.txt                   # Minimal academic dependencies (NumPy, Pandas, Matplotlib)
├── requirements_full.txt              # Full enterprise stack (FastAPI, Streamlit, XGBoost, etc.)
├── .gitignore                         # Git exclusion rules (large datasets, zips, logs)
├── LICENSE                            # MIT Open-Source License
├── README.md                          # Primary project documentation
├── SESSION_CONTEXT.md                 # Master context handoff & continuity document (This file)
│
├── .github/workflows/                 # CI/CD Automation
│   └── ci.yml                         # Multi-version matrix test (Python 3.11-3.13) & Docker build
│
├── data/                              # Datasets & Serialized Model Weights
│   ├── creditcard.csv                 # Kaggle ULB Benchmark Dataset (284k transactions)
│   ├── paysim_benchmark.csv           # PaySim Mobile Money Benchmark (100k transactions)
│   ├── model_weights.npz              # Pre-trained weights for Credit Card NumPy models
│   ├── paysim_weights.npz             # Pre-trained weights for PaySim Mobile Money models
│   ├── sample_cases.json              # 20 verified Credit Card test cases for XAI audit
│   ├── paysim_samples.json            # Curated wire transfer / account drain test cases
│   └── live_transactions.csv          # Real-time transaction buffer for live ticker
│
├── tests/                             # Mathematical & Enterprise Test Suite (27 Pytests)
│   ├── run_tests.py                   # Colored test runner with microsecond latency reports
│   ├── test_api.py                    # FastAPI endpoint contracts, status codes & latency SLA
│   ├── test_paysim_engine.py          # Mobile money ledger balance invariants & inference
│   ├── test_gradient_checking.py      # Finite-difference analytical gradient checks (<1e-5 error)
│   ├── test_numerical_stability.py    # Saturated logit overflow & zero-variance guards
│   ├── test_mathematical_invariants.py# He normal variance & loss monotonicity invariants
│   └── README.md                      # Complete testing documentation & execution guide
│
├── paysim_benchmark/                  # Cross-Domain Mobile Money Empirical Benchmark
│   ├── benchmark_paysim.py            # Head-to-head evaluation runner (NumPy vs Sklearn vs XGBoost)
│   ├── benchmark_results.csv          # Empirical metrics (ROC-AUC, PR-AUC, F1, Latency)
│   ├── benchmark_comparison.png       # High-resolution benchmark comparison chart
│   └── README.md                      # Comprehensive PaySim benchmark report & findings
│
├── benchmark_industry_vs_scratch/     # Credit Card Benchmark (NumPy vs. Sklearn & XGBoost)
│   ├── run_benchmark.py               # Automated benchmarking pipeline across 6 model pairs
│   ├── benchmark_head_to_head.ipynb   # Interactive comparison notebook
│   ├── benchmark_results.csv          # Empirical metric results across all 6 models
│   ├── figures/                       # Comparison bar charts, speed plots & ROC curves
│   └── README.md                      # Full benchmark study report & resume bullet points
│
├── simulations/                       # 1,000-Run Monte Carlo Statistical Stability Suite
│   ├── run_1000_simulations.py        # Multi-process 1,000-iteration runner across 12 CPU cores
│   ├── run_100_simulations.py         # Rapid 100-run test runner (~1 minute)
│   ├── run_paysim_simulation.py       # 50-run Monte Carlo simulation for PaySim mobile wire fraud
│   ├── simulation_worker.py           # Isolated multiprocessing worker for parallel card evaluation
│   ├── paysim_simulation_worker.py    # Isolated multiprocessing worker for PaySim evaluation
│   ├── benchmark_1000_runs_summary.csv# Aggregated Mean, StdDev, CI95, and VaR summary
│   ├── benchmark_1000_runs_raw.csv    # Complete 1,000-iteration raw metric dataset (892 KB)
│   └── README.md                      # Detailed simulation methodology & metric glossary
│
├── adversarial_security/              # Fintech Adversarial Evasion & Red-Teaming Suite
│   ├── adversarial_engine.py          # FGSM perturbation & boundary search engine
│   ├── run_red_team_audit.py          # Multi-architecture penetration test runner
│   ├── red_team_results.csv           # Empirical evasion rates across epsilon budgets
│   ├── figures/                       # High-resolution adversarial robustness curves
│   └── README.md                      # Threat model, security findings & resume bullets
│
├── scripts/                           # Automation & Utility Scripts
│   ├── generate_academic_repo.py      # Automated generator for clean academic submission repo
│   └── prepare_paysim.py              # PaySim dataset curation and feature preprocessing
│
├── notebooks/                         # 14 From-Scratch Jupyter Research Notebooks
│   ├── fraud_detection_master_capstone.ipynb # Master Capstone notebook unifying all 7 models
│   ├── fraud_detection.ipynb          # Single baseline model notebook
│   ├── fraud_detection_multimodel.ipynb# 4-model comparative notebook
│   └── step1_... to step10_...        # 10 modular step-by-step notebooks (SMOTE, MLP, GBDT, etc.)
│
├── report/                            # Research Reports, LaTeX & Generation Pipelines
│   ├── report.tex                     # Academic project report in LaTeX (ready for Overleaf)
│   ├── Statistical_Report_1000_Simulations.pdf # Standalone publication report (1,000 runs)
│   ├── statistical_report_1000_simulations.md # Markdown edition with full tables & analogies
│   ├── Credit_Card_Fraud_Detection_Comprehensive_Guide.pdf # 20-Page Research Monograph (PDF)
│   ├── project_walkthrough_guide.md   # Step-by-step code and architectural walkthrough
│   └── figures_statistical/           # 5 High-resolution publication figures
│
├── rubrics/                           # Course Evaluation Rubrics
│   ├── PDS_Project Rubrics.pdf        # Official course rubric document
│   └── README.md                      # Course requirements & marking scheme breakdown
│
└── archive/                           # Archived Prototypes & Historical Artifacts
    ├── README.md                      # Inventory of archived assets and active counterparts
    ├── app.py                         # Legacy single-rail dashboard prototype
    ├── transaction_generator.py       # Legacy transaction generator
    ├── README_LEGACY_CAPSTONE.md      # Previous iteration of project documentation
    └── fraud-intelligence-platform.zip# Standalone repository backup bundle (ignored by git)
```

---

## 🎯 6. OPERATIONAL WORKFLOWS & COMMON COMMANDS

1. **Run the 27 Unit Tests (Mathematical & API Verification):**
   ```bash
   .venv\Scripts\python.exe tests/run_tests.py
   ```
2. **Launch the FastAPI Microservice:**
   ```bash
   .venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
   # Interactive Swagger Docs: http://localhost:8000/docs
   ```
3. **Launch the Interactive Streamlit Operations Dashboard:**
   ```bash
   .venv\Scripts\python.exe -m streamlit run app.py
   ```
4. **Build & Spin Up Docker Multi-Container Platform:**
   ```bash
   docker-compose up --build
   ```
5. **Generate the Dedicated Academic Submission Repository:**
   ```bash
   .venv\Scripts\python.exe scripts/generate_academic_repo.py
   # Generates clean submission folder at C:\Users\HP\Desktop\PDS_Academic_Project
   ```
6. **Academic LaTeX Report Compilation:**
   * Open `report/report.tex` (lines 53–75) and verify student name (`Manas Jain`) and group credentials.
   * Compile on Overleaf or local TeX Live to produce `report.pdf`.
