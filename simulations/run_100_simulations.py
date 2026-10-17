import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from simulation_worker import run_single_simulation

def main():
    print("=" * 85)
    print("100-RUN LARGE-SCALE MONTE CARLO STABILITY & CONFIDENCE INTERVAL BENCHMARK")
    print("Eliminating the Luck Factor: Rigorous Statistical Significance across N = 100 Runs")
    print("=" * 85)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Resolve dataset path
    data_paths = [
        os.path.join(repo_root, "data", "creditcard.csv"),
        os.path.join("data", "creditcard.csv"),
        os.path.join("..", "data", "creditcard.csv")
    ]
    data_path = next((p for p in data_paths if os.path.exists(p)), "data/creditcard.csv")

    t0_load = time.time()
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded in {time.time() - t0_load:.2f}s: {len(df):,} transactions, {df['Class'].sum():,} frauds.\n")
    
    seconds_in_day = 86400.0
    time_sin = np.sin(2 * np.pi * df['Time'].values / seconds_in_day)
    time_cos = np.cos(2 * np.pi * df['Time'].values / seconds_in_day)
    log_amount = np.log1p(df['Amount'].values)
    v_cols = [f'V{i}' for i in range(1, 29)]
    X_raw = np.column_stack([time_sin, time_cos, log_amount, df[v_cols].values])
    y_raw = df['Class'].values.astype(np.int64)
    amounts_raw = df['Amount'].values
    
    N_RUNS = 100
    seeds = [1000 + i for i in range(N_RUNS)]
    
    n_workers = min(12, os.cpu_count() or 4)
    print(f"Distributing {N_RUNS} runs across {n_workers} CPU worker processes in parallel...")
    
    t0_runs = time.time()
    task_args = [(s, X_raw, y_raw, amounts_raw) for s in seeds]
    
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, result in enumerate(executor.map(run_single_simulation, task_args), 1):
            all_results.append(result)
            if idx % 10 == 0 or idx == N_RUNS:
                elapsed = time.time() - t0_runs
                rate = idx / elapsed
                remaining = (N_RUNS - idx) / rate if rate > 0 else 0
                print(f"Progress: {idx:3d}/{N_RUNS} runs completed [{idx/N_RUNS*100:5.1f}%] | Elapsed: {elapsed:5.1f}s | Est. Remaining: {remaining:4.1f}s")
                
    total_time = time.time() - t0_runs
    print(f"\nAll {N_RUNS} Monte Carlo Simulations Completed in {total_time:.2f}s ({total_time/N_RUNS:.2f}s/run average)!\n")
    
    out_dir = script_dir if os.path.basename(os.getcwd()) != 'simulations' else '.'
    raw_path = os.path.join(out_dir, "benchmark_100_runs_raw.csv")
    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw run-by-run data saved to: {raw_path}")
    
    # Statistical Aggregation per Model
    models = [
        '1. Baseline Logistic Regression',
        '2. Cost-Sensitive LogReg',
        '3. Neural Network (MLP)',
        '4. GBDT Classifier',
        '5. Isolation Forest (Unsupervised)',
        '6. Soft-Voting Ensemble',
        '7. Stacked Meta-Learner'
    ]
    
    metrics = ['Recall', 'Dollar_Recall', 'Precision', 'F1', 'ROC_AUC', 'Dollars_Protected', 'Net_Loss', 'FP', 'FN']
    
    summary_rows = []
    detailed_stat_rows = []
    
    for m in models:
        m_row = {'Model': m}
        for met in metrics:
            col_name = f"{m}__{met}"
            vals = df_raw[col_name].values
            
            # Multiply percentage metrics by 100 for display
            is_pct = met in ['Recall', 'Dollar_Recall', 'Precision', 'F1']
            mult = 100.0 if is_pct else 1.0
            
            vals_disp = vals * mult
            mu = np.mean(vals_disp)
            sigma = np.std(vals_disp, ddof=1)
            # 95% Confidence Interval: margin of error = 1.96 * sigma / sqrt(N)
            margin_err = 1.96 * (sigma / np.sqrt(N_RUNS))
            ci_low = mu - margin_err
            ci_high = mu + margin_err
            median = np.median(vals_disp)
            p25 = np.percentile(vals_disp, 25)
            p75 = np.percentile(vals_disp, 75)
            iqr = p75 - p25
            v_min = np.min(vals_disp)
            v_max = np.max(vals_disp)
            
            m_row[f"{met}_Mean"] = mu
            m_row[f"{met}_Std"] = sigma
            m_row[f"{met}_CI95_Margin"] = margin_err
            m_row[f"{met}_CI95_Low"] = ci_low
            m_row[f"{met}_CI95_High"] = ci_high
            m_row[f"{met}_Median"] = median
            m_row[f"{met}_IQR"] = iqr
            m_row[f"{met}_Min"] = v_min
            m_row[f"{met}_Max"] = v_max
            
            detailed_stat_rows.append({
                'Model': m,
                'Metric': met,
                'Mean': mu,
                'StdDev': sigma,
                'CI_95_Margin': margin_err,
                'CI_95_Lower': ci_low,
                'CI_95_Upper': ci_high,
                'Median': median,
                'IQR': iqr,
                'Min': v_min,
                'Max': v_max
            })
            
        summary_rows.append(m_row)
        
    summary_path = os.path.join(out_dir, "benchmark_100_runs_summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary benchmarks saved to: {summary_path}")

    breakdown_path = os.path.join(out_dir, "benchmark_100_runs_statistical_breakdown.csv")
    df_detailed = pd.DataFrame(detailed_stat_rows)
    df_detailed.to_csv(breakdown_path, index=False)
    print(f"Detailed statistical breakdown saved to: {breakdown_path}")
    
    print("=" * 110)
    print("100-RUN STATISTICAL SIGNIFICANCE SUMMARY (MEAN ± 95% CONFIDENCE INTERVAL)")
    print("=" * 110)
    for _, r in df_summary.iterrows():
        print(f"\n>>> {r['Model']} <<<")
        print(f"  • Event Recall       : {r['Recall_Mean']:5.2f}% ± {r['Recall_CI95_Margin']:4.2f}%  (95% CI: [{r['Recall_CI95_Low']:5.2f}%, {r['Recall_CI95_High']:5.2f}%], Std: {r['Recall_Std']:4.2f}%)")
        print(f"  • Dollar Value Recall: {r['Dollar_Recall_Mean']:5.2f}% ± {r['Dollar_Recall_CI95_Margin']:4.2f}%  (95% CI: [{r['Dollar_Recall_CI95_Low']:5.2f}%, {r['Dollar_Recall_CI95_High']:5.2f}%], Std: {r['Dollar_Recall_Std']:4.2f}%)")
        print(f"  • Precision          : {r['Precision_Mean']:5.2f}% ± {r['Precision_CI95_Margin']:4.2f}%  (95% CI: [{r['Precision_CI95_Low']:5.2f}%, {r['Precision_CI95_High']:5.2f}%], Std: {r['Precision_Std']:4.2f}%)")
        print(f"  • F1-Score           : {r['F1_Mean']:5.2f}% ± {r['F1_CI95_Margin']:4.2f}%  (95% CI: [{r['F1_CI95_Low']:5.2f}%, {r['F1_CI95_High']:5.2f}%], Std: {r['F1_Std']:4.2f}%)")
        print(f"  • ROC-AUC            : {r['ROC_AUC_Mean']:6.4f} ± {r['ROC_AUC_CI95_Margin']:6.4f}  (95% CI: [{r['ROC_AUC_CI95_Low']:6.4f}, {r['ROC_AUC_CI95_High']:6.4f}], Std: {r['ROC_AUC_Std']:6.4f})")
        print(f"  • Capital Protected  : ${r['Dollars_Protected_Mean']:,.2f} / run  (Range: [${r['Dollars_Protected_Min']:,.2f}, ${r['Dollars_Protected_Max']:,.2f}])")
        print(f"  • Net Financial Loss : ${r['Net_Loss_Mean']:,.2f} / run  (Investigation Fees + Unrecovered Fraud)")
        print(f"  • Errors (Mean)      : {r['FP_Mean']:6.1f} False Alarms, {r['FN_Mean']:4.1f} Missed Frauds")

if __name__ == '__main__':
    main()
