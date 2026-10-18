# Monte Carlo Statistical Stability & Value-at-Risk Simulation Suite

This directory contains the large-scale Monte Carlo resampling framework developed to evaluate the statistical stability, variance, and financial tail-risk of all **7 from-scratch fraud detection models** across $N = 1,000$ independent pseudo-random train/test partitions.

---

## 1. Why Monte Carlo Simulations?

In imbalanced datasets (e.g., credit card fraud where positive events account for only 0.172% of transactions), single-split performance metrics are susceptible to severe sampling noise:
1. **Sampling Luck:** A single lucky test split may make an unstable model appear state-of-the-art.
2. **Variance Quantification:** Real-world banking deployment demands knowing not just mean recall, but the **95% confidence interval** and **worst-case drawdown (Value-at-Risk / Conditional Value-at-Risk)**.
3. **Operational Robustness:** Demonstrates that pure NumPy models generalize reliably without overfitting.

---

## 2. Directory Manifest

| File | Type | Description |
| :--- | :--- | :--- |
| `run_1000_simulations.py` | Python Script | Primary European Credit Card production runner executing $N = 1,000$ parallel simulations. |
| `run_100_simulations.py` | Python Script | Rapid European Credit Card runner executing $N = 100$ parallel iterations. |
| `simulation_worker.py` | Python Module | Isolated worker unit executing Credit Card stratified subsampling, 7-model training, and logging. |
| `run_paysim_simulation.py` | Python Script | PaySim Mobile Money multi-run Monte Carlo runner executing $N = 50$ parallel iterations. |
| `paysim_simulation_worker.py` | Python Module | Isolated worker unit executing PaySim balance-drain feature normalization, model training, and financial loss evaluation. |
| `paysim_simulation_raw.csv` | Dataset | Complete run-by-run dataset capturing Recall, Dollar Recall, Precision, F1, ROC-AUC, Dollars Protected, and Net Loss. |
| `paysim_simulation_summary.csv` | Dataset | Statistical summary table with Mean ± 95% CI across all 6 models. |
| `paysim_simulation_breakdown.csv` | Dataset | Comprehensive distribution metrics including VaR (95%) and CVaR (99%) expected shortfalls. |
| `paysim_simulation_distribution.png` | Visualization | Boxplot distribution of Net Financial Losses and ROC-AUC stability across all 50 iterations. |
| `benchmark_1000_runs_raw.csv` | Dataset (892 KB) | Complete iteration-by-iteration dataset: 1,000 rows capturing every metric across all 7 models. |
| `benchmark_1000_runs_summary.csv` | Dataset (18 KB) | High-level summary capturing Mean, StdDev, 95% CI Margin, Median, and Tail-Risk metrics per model. |
| `benchmark_1000_runs_statistical_breakdown.csv` | Dataset (18 KB) | Comprehensive statistical distribution metrics: Mean, Std, CI, Median, IQR, P1, P5, P95, P99, Min, Max, Skewness, and Excess Kurtosis. |
| `benchmark_100_runs_*.csv` | Datasets | Corresponding CSV results from the 100-run simulation baseline. |
| `benchmark_12_runs_summary.csv` | Dataset | Preliminary 12-run baseline output. |

---

## 3. Evaluated Models (All Built from First-Principles NumPy)

1. **Baseline Logistic Regression** ($\lambda = 0.01$, $\eta = 0.05$, 500 epochs)
2. **Cost-Sensitive Logistic Regression** (Loss weighting $w_1 = 5.0, w_0 = 1.0$)
3. **Deep Multi-Layer Perceptron (MLP)** ($31 \to 64 \to 32 \to 1$ with ReLU, Dropout, and He initialization)
4. **Gradient Boosted Decision Trees (GBDT)** (30 residual trees, learning rate $\eta = 0.1$, max depth 3)
5. **Isolation Forest** (50 completely random isolation trees, anomaly depth threshold)
6. **Soft-Voting Ensemble** (Weighted probability combination of LogReg, MLP, and GBDT)
7. **Stacked Meta-Learner** (Out-of-fold meta-feature generation with Logistic Regression blender)

---

## 4. How to Run

### Run Quick 100-Iteration Benchmark (~1 minute)
```bash
python simulations/run_100_simulations.py
```

### Run Full 1,000-Iteration Benchmark (~13.8 minutes on 12 CPU cores)
```bash
python simulations/run_1000_simulations.py
```
*Note: Both scripts automatically resolve `data/creditcard.csv` whether executed from repository root or inside the `simulations/` directory.*
