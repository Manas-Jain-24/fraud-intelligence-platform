"""
run_paysim_simulation.py
========================
Large-Scale Monte Carlo Simulation & Financial Risk Benchmark on PaySim Mobile Money Dataset.
Evaluates statistical stability across N = 50 independent randomized 80/20 train/test splits.
Computes 95% Confidence Intervals, Value-at-Risk (VaR), and Expected Shortfall (CVaR).
Author: Manas Jain
"""

import os
import sys

# Windows OpenBLAS process limit guard
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simulations.paysim_simulation_worker import run_single_paysim_simulation
from paysim_engine import PaySimFraudEngine

DATA_PATH = os.path.join(BASE_DIR, "data", "paysim_benchmark.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_OUT = os.path.join(OUT_DIR, "paysim_simulation_raw.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "paysim_simulation_summary.csv")
BREAKDOWN_OUT = os.path.join(OUT_DIR, "paysim_simulation_breakdown.csv")
PNG_OUT = os.path.join(OUT_DIR, "paysim_simulation_distribution.png")


def main(n_runs=50):
    print("=" * 85)
    print("  PAYSIM MOBILE MONEY MULTI-RUN MONTE CARLO STABILITY & FINANCIAL RISK BENCHMARK")
    print(f"  Evaluating Statistical Invariance across N = {n_runs} Independent Stratified Runs")
    print("=" * 85)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    t0_load = time.time()
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded in {time.time() - t0_load:.2f}s: {len(df):,} records ({df['isFraud'].sum():,} frauds).\n")

    features = PaySimFraudEngine.FEATURE_NAMES
    X_all = df[features].to_numpy()
    y_all = df['isFraud'].to_numpy()
    amounts_all = df['amount'].to_numpy()

    seeds = [2000 + i for i in range(n_runs)]
    n_workers = 3
    print(f"Distributing {n_runs} simulations across {n_workers} CPU worker processes...")

    t0_runs = time.time()
    task_args = [(s, X_all, y_all, amounts_all) for s in seeds]

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, result in enumerate(executor.map(run_single_paysim_simulation, task_args), 1):
            all_results.append(result)
            if idx % 10 == 0 or idx == n_runs:
                elapsed = time.time() - t0_runs
                rate = idx / elapsed
                remaining = (n_runs - idx) / rate if rate > 0 else 0
                print(f"  Progress: {idx:3d}/{n_runs} completed [{idx/n_runs*100:5.1f}%] | Elapsed: {elapsed:5.1f}s | Est. Remaining: {remaining:4.1f}s")

    total_time = time.time() - t0_runs
    print(f"\nAll {n_runs} Monte Carlo runs completed in {total_time:.2f}s ({total_time/n_runs:.2f}s/run average)!\n")

    # Save raw outputs
    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(RAW_OUT, index=False)
    print(f"Raw simulation results saved to: {RAW_OUT}")

    # Models & Metrics
    models = [
        "1. Scratch Baseline LogReg",
        "2. Scratch Cost-Sensitive LogReg",
        "3. Scratch Neural Net (MLP)",
        "4. Scratch Stump GBDT",
        "5. Scratch Soft-Voting Ensemble",
        "6. Industry XGBoost"
    ]

    metrics = ["Recall", "Dollar_Recall", "Precision", "F1", "ROC_AUC", "Dollars_Protected", "Dollars_Lost", "Net_Loss", "FP", "FN"]

    summary_rows = []
    breakdown_rows = []

    for m in models:
        m_row = {"Model": m}
        for met in metrics:
            col_name = f"{m}__{met}"
            vals = df_raw[col_name].values

            is_pct = met in ["Recall", "Dollar_Recall", "Precision", "F1"]
            mult = 100.0 if is_pct else 1.0
            vals_disp = vals * mult

            mu = float(np.mean(vals_disp))
            sigma = float(np.std(vals_disp, ddof=1))
            margin_err = float(1.96 * (sigma / np.sqrt(n_runs)))
            ci_low = mu - margin_err
            ci_high = mu + margin_err
            median = float(np.median(vals_disp))

            # Financial Risk: VaR 95% and CVaR 99%
            if met in ["Net_Loss", "Dollars_Lost"]:
                var_95 = float(np.percentile(vals_disp, 95))
                tail_99 = vals_disp[vals_disp >= np.percentile(vals_disp, 99)]
                cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_95
            else:
                var_95 = np.nan
                cvar_99 = np.nan

            suffix = "%" if is_pct else ""
            m_row[f"{met}_Mean"] = f"{mu:.2f}{suffix} ± {margin_err:.2f}{suffix}"

            breakdown_rows.append({
                "Model": m,
                "Metric": met,
                "Mean": mu,
                "Std": sigma,
                "CI_95_Low": ci_low,
                "CI_95_High": ci_high,
                "Median": median,
                "VaR_95": var_95,
                "CVaR_99": cvar_99
            })

        summary_rows.append(m_row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Summary table saved to: {SUMMARY_OUT}")

    df_breakdown = pd.DataFrame(breakdown_rows)
    df_breakdown.to_csv(BREAKDOWN_OUT, index=False)
    print(f"Detailed statistical breakdown saved to: {BREAKDOWN_OUT}")

    # Plotting Monte Carlo Loss and ROC-AUC Distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Plot 1: Net Financial Loss Distribution
    loss_data = [df_raw[f"{m}__Net_Loss"].values / 1e6 for m in models]
    bplot1 = axes[0].boxplot(loss_data, patch_artist=True)
    axes[0].set_xticks(range(1, len(models) + 1))
    axes[0].set_xticklabels([m.split('. ')[1] for m in models], rotation=20, fontsize=8)
    colors = ['#38bdf8', '#38bdf8', '#38bdf8', '#38bdf8', '#10b981', '#f59e0b']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title(f"Expected Net Financial Loss Distribution ($ Millions, N={n_runs})", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Net Financial Loss ($ Millions)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot 2: ROC-AUC Distribution
    auc_data = [df_raw[f"{m}__ROC_AUC"].values for m in models]
    bplot2 = axes[1].boxplot(auc_data, patch_artist=True)
    axes[1].set_xticks(range(1, len(models) + 1))
    axes[1].set_xticklabels([m.split('. ')[1] for m in models], rotation=20, fontsize=8)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title(f"ROC-AUC Stability Across Random Train/Test Splits (N={n_runs})", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("PaySim Mobile Money: 50-Run Monte Carlo Statistical Invariance", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {PNG_OUT}")

    # Display clean executive table
    disp_cols = ["Model", "ROC_AUC_Mean", "Recall_Mean", "Dollar_Recall_Mean", "Precision_Mean", "F1_Mean", "Net_Loss_Mean"]
    print("\nExecutive Statistical Summary:")
    print(df_summary[disp_cols].to_string(index=False))


if __name__ == "__main__":
    runs_to_execute = 50
    if len(sys.argv) > 1:
        try:
            runs_to_execute = int(sys.argv[1])
        except ValueError:
            pass
    main(n_runs=runs_to_execute)
