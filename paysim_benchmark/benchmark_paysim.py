"""
benchmark_paysim.py
===================
Rigorous Head-to-Head Benchmark: From-Scratch Pure NumPy Models vs. Industry Standard (XGBoost / Scikit-Learn)
Dataset: PaySim Mobile Money Transfer Fraud (100,000 Curated Benchmark Sample)
Author: Manas Jain
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from paysim_engine import (
    ScratchLogReg,
    ScratchCostSensitiveLogReg,
    ScratchMLP,
    ScratchStumpGBDT,
    ScratchAnomalyDetector,
    PaySimFraudEngine
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "paysim_benchmark.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(OUT_DIR, "benchmark_results.csv")
PNG_OUT = os.path.join(OUT_DIR, "benchmark_comparison.png")


def run_benchmark():
    print(f"Loading PaySim benchmark dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    features = PaySimFraudEngine.FEATURE_NAMES
    X = df[features].to_numpy()
    y = df['isFraud'].to_numpy()

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train):,} samples ({np.sum(y_train):,} fraud)")
    print(f"Test set : {len(X_test):,} samples ({np.sum(y_test):,} fraud)")

    # Standard scaling
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Measure latency helper
    def measure_latency(model_fn, x_sample, runs=1000):
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model_fn(x_sample)
        t1 = time.perf_counter()
        return ((t1 - t0) / runs) * 1e6  # microseconds

    results = []
    x_single_norm = X_test_norm[0:1]
    x_single_raw = X_test[0:1]

    # ---------------------------------------------------------
    # 1. Scratch Baseline LogReg (NumPy)
    # ---------------------------------------------------------
    print("Training 1. Scratch Baseline LogReg (NumPy)...")
    s_lr = ScratchLogReg(lr=0.1, n_iter=800).fit(X_train_norm, y_train)
    p_s_lr = s_lr.predict_proba(X_test_norm)
    lat_s_lr = measure_latency(s_lr.predict_proba, x_single_norm)
    results.append({
        "Model": "Scratch Baseline LogReg (NumPy)",
        "Type": "From-Scratch (Pure NumPy)",
        "ROC-AUC": roc_auc_score(y_test, p_s_lr),
        "PR-AUC": average_precision_score(y_test, p_s_lr),
        "F1-Score": f1_score(y_test, (p_s_lr >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_s_lr >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_s_lr >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_s_lr >= 0.5).astype(int)),
        "Latency (us)": lat_s_lr
    })

    # ---------------------------------------------------------
    # 2. Scratch Cost-Sensitive LogReg (NumPy)
    # ---------------------------------------------------------
    print("Training 2. Scratch Cost-Sensitive LogReg (NumPy)...")
    s_cslr = ScratchCostSensitiveLogReg(lr=0.08, n_iter=800, fraud_weight=4.0).fit(X_train_norm, y_train)
    p_s_cslr = s_cslr.predict_proba(X_test_norm)
    lat_s_cslr = measure_latency(s_cslr.predict_proba, x_single_norm)
    results.append({
        "Model": "Scratch Cost-Sensitive LogReg (NumPy)",
        "Type": "From-Scratch (Pure NumPy)",
        "ROC-AUC": roc_auc_score(y_test, p_s_cslr),
        "PR-AUC": average_precision_score(y_test, p_s_cslr),
        "F1-Score": f1_score(y_test, (p_s_cslr >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_s_cslr >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_s_cslr >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_s_cslr >= 0.5).astype(int)),
        "Latency (us)": lat_s_cslr
    })

    # ---------------------------------------------------------
    # 3. Scratch MLP (NumPy Neural Net)
    # ---------------------------------------------------------
    print("Training 3. Scratch MLP Neural Net (NumPy)...")
    s_mlp = ScratchMLP(hidden=(32, 16), lr=0.03, epochs=300).fit(X_train_norm, y_train)
    p_s_mlp = s_mlp.predict_proba(X_test_norm)
    lat_s_mlp = measure_latency(s_mlp.predict_proba, x_single_norm)
    results.append({
        "Model": "Scratch MLP Neural Net (NumPy)",
        "Type": "From-Scratch (Pure NumPy)",
        "ROC-AUC": roc_auc_score(y_test, p_s_mlp),
        "PR-AUC": average_precision_score(y_test, p_s_mlp),
        "F1-Score": f1_score(y_test, (p_s_mlp >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_s_mlp >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_s_mlp >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_s_mlp >= 0.5).astype(int)),
        "Latency (us)": lat_s_mlp
    })

    # ---------------------------------------------------------
    # 4. Scratch Stump GBDT (NumPy)
    # ---------------------------------------------------------
    print("Training 4. Scratch Stump GBDT (NumPy)...")
    s_gbdt = ScratchStumpGBDT(n_estimators=30, lr=0.15).fit(X_train_norm, y_train)
    p_s_gbdt = s_gbdt.predict_proba(X_test_norm)
    lat_s_gbdt = measure_latency(s_gbdt.predict_proba, x_single_norm)
    results.append({
        "Model": "Scratch Stump GBDT (NumPy)",
        "Type": "From-Scratch (Pure NumPy)",
        "ROC-AUC": roc_auc_score(y_test, p_s_gbdt),
        "PR-AUC": average_precision_score(y_test, p_s_gbdt),
        "F1-Score": f1_score(y_test, (p_s_gbdt >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_s_gbdt >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_s_gbdt >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_s_gbdt >= 0.5).astype(int)),
        "Latency (us)": lat_s_gbdt
    })

    # ---------------------------------------------------------
    # 5. Scratch Soft-Voting Ensemble (Champion)
    # ---------------------------------------------------------
    print("Evaluating 5. Scratch Soft-Voting Ensemble (Champion)...")
    ad = ScratchAnomalyDetector().fit(X_train_norm[y_train == 0])
    p_ad = ad.anomaly_score(X_test_norm)

    p_ens = (0.35 * p_s_mlp + 0.35 * p_s_gbdt + 0.20 * p_s_cslr + 0.10 * p_ad)
    lat_ens = lat_s_mlp + lat_s_gbdt + lat_s_cslr
    results.append({
        "Model": "Scratch Soft-Voting Ensemble (Champion)",
        "Type": "From-Scratch (Pure NumPy)",
        "ROC-AUC": roc_auc_score(y_test, p_ens),
        "PR-AUC": average_precision_score(y_test, p_ens),
        "F1-Score": f1_score(y_test, (p_ens >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_ens >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_ens >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_ens >= 0.5).astype(int)),
        "Latency (us)": lat_ens
    })

    # ---------------------------------------------------------
    # 6. Industry Benchmark: Scikit-Learn Logistic Regression
    # ---------------------------------------------------------
    print("Training 6. Scikit-Learn Logistic Regression...")
    sk_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    sk_lr.fit(X_train_norm, y_train)
    p_sk_lr = sk_lr.predict_proba(X_test_norm)[:, 1]
    lat_sk_lr = measure_latency(lambda x: sk_lr.predict_proba(x), x_single_norm)
    results.append({
        "Model": "Scikit-Learn LogReg (Balanced)",
        "Type": "Industry Standard",
        "ROC-AUC": roc_auc_score(y_test, p_sk_lr),
        "PR-AUC": average_precision_score(y_test, p_sk_lr),
        "F1-Score": f1_score(y_test, (p_sk_lr >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_sk_lr >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_sk_lr >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_sk_lr >= 0.5).astype(int)),
        "Latency (us)": lat_sk_lr
    })

    # ---------------------------------------------------------
    # 7. Industry Benchmark: Scikit-Learn Random Forest
    # ---------------------------------------------------------
    print("Training 7. Scikit-Learn Random Forest...")
    sk_rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    sk_rf.fit(X_train, y_train)
    p_sk_rf = sk_rf.predict_proba(X_test)[:, 1]
    lat_sk_rf = measure_latency(lambda x: sk_rf.predict_proba(x), x_single_raw)
    results.append({
        "Model": "Scikit-Learn Random Forest",
        "Type": "Industry Standard",
        "ROC-AUC": roc_auc_score(y_test, p_sk_rf),
        "PR-AUC": average_precision_score(y_test, p_sk_rf),
        "F1-Score": f1_score(y_test, (p_sk_rf >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_sk_rf >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_sk_rf >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_sk_rf >= 0.5).astype(int)),
        "Latency (us)": lat_sk_rf
    })

    # ---------------------------------------------------------
    # 8. Industry Benchmark: XGBoost Classifier
    # ---------------------------------------------------------
    print("Training 8. XGBoost Classifier (Industry Standard)...")
    xgb = XGBClassifier(
        n_estimators=60,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    lat_xgb = measure_latency(lambda x: xgb.predict_proba(x), x_single_raw)
    results.append({
        "Model": "XGBoost Classifier (Industry Standard)",
        "Type": "Industry Standard",
        "ROC-AUC": roc_auc_score(y_test, p_xgb),
        "PR-AUC": average_precision_score(y_test, p_xgb),
        "F1-Score": f1_score(y_test, (p_xgb >= 0.5).astype(int)),
        "Recall": recall_score(y_test, (p_xgb >= 0.5).astype(int)),
        "Precision": precision_score(y_test, (p_xgb >= 0.5).astype(int), zero_division=0),
        "Accuracy": accuracy_score(y_test, (p_xgb >= 0.5).astype(int)),
        "Latency (us)": lat_xgb
    })

    res_df = pd.DataFrame(results)
    res_df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved benchmark results to {CSV_OUT}")

    # Plotting comparison with pure matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ['#2563eb' if t == "From-Scratch (Pure NumPy)" else '#059669' for t in res_df['Type']]

    y_pos = np.arange(len(res_df))

    # Plot 1: ROC-AUC
    axes[0].barh(y_pos, res_df['ROC-AUC'], color=colors, height=0.65)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(res_df['Model'], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("ROC-AUC Comparison (PaySim Mobile Money)", fontsize=11, fontweight='bold')
    axes[0].set_xlim(0.85, 1.0)
    for i, v in enumerate(res_df['ROC-AUC']):
        axes[0].text(v - 0.02, i, f"{v:.4f}", color='white', va='center', ha='center', fontweight='bold', fontsize=9)

    # Plot 2: PR-AUC
    axes[1].barh(y_pos, res_df['PR-AUC'], color=colors, height=0.65)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_title("PR-AUC (Precision-Recall Area)", fontsize=11, fontweight='bold')
    axes[1].set_xlim(0.70, 1.0)
    for i, v in enumerate(res_df['PR-AUC']):
        axes[1].text(v - 0.03, i, f"{v:.4f}", color='white', va='center', ha='center', fontweight='bold', fontsize=9)

    # Plot 3: Latency
    axes[2].barh(y_pos, res_df['Latency (us)'], color=colors, height=0.65)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_title("Inference Latency (Microseconds / Tx)", fontsize=11, fontweight='bold')
    for i, v in enumerate(res_df['Latency (us)']):
        axes[2].text(v + 5, i, f"{v:.1f} us", color='#1e293b', va='center', fontweight='bold', fontsize=9)

    plt.suptitle("Head-to-Head Benchmark: From-Scratch NumPy vs. Industry Standard (PaySim Mobile Money)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved benchmark figure to {PNG_OUT}")

    print("\nBenchmark Summary Table:")
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
