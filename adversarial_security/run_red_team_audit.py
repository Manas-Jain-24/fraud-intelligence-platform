"""
adversarial_security/run_red_team_audit.py
==========================================
Automated Red-Teaming & Adversarial Robustness Benchmark Runner.
Executes systematic cyber attacks against all 6 machine learning architectures:
1. Fast Gradient Sign Perturbation (FGSM) across varying budgets (epsilon in [0.05, 2.5]).
2. Multi-Architecture Evasion Resilience Comparison (Linear vs Tree vs MLP vs Ensemble).
3. Smurfing / Micro-Transaction Splitting Attack Simulation.
Saves metrics to CSV and generates publication-grade robustness curves.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from models_engine import FraudIntelligenceEngine, engine
from adversarial_security.adversarial_engine import AdversarialRedTeamEngine, red_team

FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_OUT = os.path.join(SCRIPT_DIR, "red_team_results.csv")


def run_adversarial_campaign():
    print("=" * 80)
    print("  FINTECH ADVERSARIAL RED-TEAMING & ROBUSTNESS BENCHMARK")
    print("=" * 80)

    if not engine.sample_cases:
        engine.load()

    # Filter to real fraudulent sample cases from Kaggle
    fraud_samples = [s for s in engine.sample_cases if s['is_fraud'] == 1]
    if not fraud_samples:
        print("Warning: Generating synthetic fraud vectors for red-teaming...")
        from data_generator import generator
        fraud_samples = [generator.generate_single_transaction(force_fraud=True) for _ in range(10)]

    print(f"Loaded {len(fraud_samples)} verified ground-truth fraud vectors for red-team testing.\n")

    # Architectures to benchmark
    models_to_test = [
        "Baseline Logistic Regression (NumPy)",
        "Cost-Sensitive LogReg (NumPy)",
        "Multi-Layer Perceptron (NumPy MLP)",
        "Gradient Boosted Trees (NumPy GBDT)",
        "Unsupervised Isolation Profiler (NumPy)",
        "Soft-Voting Ensemble (Scratch Champion)"
    ]

    epsilon_budgets = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    results_records = []

    print("--> [1/3] Benchmarking FGSM Evasion Success Rates vs Perturbation Budget (epsilon)...")
    evasion_curves = {m: [] for m in models_to_test}

    for eps in epsilon_budgets:
        for model_name in models_to_test:
            evaded_count = 0
            total_tested = 0
            risk_deltas = []

            for case in fraud_samples:
                # Build raw vector
                if "features" in case:
                    x = np.zeros(29)
                    for i in range(1, 29):
                        x[i-1] = case['features'][f"V{i}"]
                    x[28] = case['amount']
                else:
                    x = case['raw_features']

                init_prob = engine.predict_all(x)[model_name]
                # Only test on cases initially caught by the model
                if init_prob >= 0.50:
                    total_tested += 1
                    x_adv = red_team.fgsm_perturbation(x, epsilon=eps, mutable_only=True)
                    adv_prob = engine.predict_all(x_adv)[model_name]
                    risk_deltas.append(init_prob - adv_prob)
                    if adv_prob < 0.40:  # Dropped from Blocked/Flagged to Approved
                        evaded_count += 1

            evasion_rate = (evaded_count / total_tested) if total_tested > 0 else 0.0
            avg_delta = float(np.mean(risk_deltas)) if risk_deltas else 0.0
            evasion_curves[model_name].append(evasion_rate * 100)

            results_records.append({
                "Epsilon_Budget": eps,
                "Model_Architecture": model_name,
                "Evasion_Success_Rate_%": round(evasion_rate * 100, 2),
                "Avg_Risk_Reduction_%": round(avg_delta * 100, 2),
                "Tested_Attacks": total_tested
            })

    # Save to CSV
    df_res = pd.DataFrame(results_records)
    df_res.to_csv(CSV_OUT, index=False)
    print(f"Adversarial red-team benchmark metrics saved to: {CSV_OUT}")

    # Plot Robustness Curves
    print("\n--> [2/3] Generating High-Resolution Adversarial Robustness Plot...")
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('dark_background')

    palette = {
        "Baseline Logistic Regression (NumPy)": "#EF4444",
        "Cost-Sensitive LogReg (NumPy)": "#F59E0B",
        "Multi-Layer Perceptron (NumPy MLP)": "#8B5CF6",
        "Gradient Boosted Trees (NumPy GBDT)": "#10B981",
        "Unsupervised Isolation Profiler (NumPy)": "#6B7280",
        "Soft-Voting Ensemble (Scratch Champion)": "#06B6D4"
    }

    for model_name, curve in evasion_curves.items():
        plt.plot(
            epsilon_budgets,
            curve,
            marker='o',
            linewidth=2.5,
            label=model_name,
            color=palette.get(model_name, "#FFFFFF")
        )

    plt.title("Adversarial Evasion Success Rate vs Perturbation Budget (FGSM)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Attacker Perturbation Budget (epsilon)", fontsize=11)
    plt.ylabel("Evasion Success Rate (% Bypassed)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.25)
    plt.legend(frameon=True, facecolor="#111827", edgecolor="#374151", fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(FIG_DIR, "adversarial_robustness_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Robustness curves saved to: {plot_path}")

    # Smurfing Attack Simulation
    print("\n--> [3/3] Simulating Structured Transaction Splitting ('Smurfing') Attacks...")
    smurf_results = []
    for split_k in [2, 3, 5, 10]:
        rates = []
        for case in fraud_samples:
            if "features" in case:
                x = np.zeros(29)
                for i in range(1, 29):
                    x[i-1] = case['features'][f"V{i}"]
                x[28] = max(1000.0, float(case['amount']))
            else:
                x = case['raw_features']
                x[28] = max(1000.0, float(x[28]))

            res = red_team.simulate_smurfing_attack(x, split_count=split_k)
            rates.append(res['evasion_success_rate'])

        avg_smurf_evasion = float(np.mean(rates)) * 100
        smurf_results.append({
            "Split_Count (N)": split_k,
            "Smurfing_Evasion_Rate_%": round(avg_smurf_evasion, 1),
            "Detection_Status": "PROTECTED" if avg_smurf_evasion < 40 else "VULNERABLE"
        })
        print(f"  Split into {split_k:2d} parts: Evasion Rate = {avg_smurf_evasion:5.1f}% [{smurf_results[-1]['Detection_Status']}]")

    print("\n" + "=" * 80)
    print("  [SUCCESS] ADVERSARIAL RED-TEAM CAMPAIGN COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_adversarial_campaign()
