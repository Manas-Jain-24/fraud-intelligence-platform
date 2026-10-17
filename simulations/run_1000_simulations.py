import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from simulation_worker import run_single_simulation

def compute_skewness(x):
    n = len(x)
    if n < 3: return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0: return 0.0
    return float((np.sum((x - m) ** 3) * n) / ((n - 1) * (n - 2) * (s ** 3)))

def compute_kurtosis(x):
    n = len(x)
    if n < 4: return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0: return 0.0
    term1 = (n * (n + 1) * np.sum((x - m) ** 4)) / ((n - 1) * (n - 2) * (n - 3) * (s ** 4))
    term2 = (3.0 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))
    return float(term1 - term2) # Excess kurtosis

def main():
    print("=" * 90)
    print("1,000-RUN LARGE-SCALE MONTE CARLO STABILITY & TAIL-RISK BENCHMARK")
    print("Full Statistical Significance, Empirical Density Estimation & Value-at-Risk across N = 1,000 Runs")
    print("=" * 90)
    
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
    
    N_RUNS = 1000
    seeds = [1000 + i for i in range(N_RUNS)]
    
    n_workers = min(12, os.cpu_count() or 4)
    print(f"Distributing {N_RUNS:,} runs across {n_workers} CPU worker processes in parallel...")
    print(f"Total model fits to execute: {N_RUNS * 7:,} (7 models per run)")
    print(f"Total transactions to classify: {N_RUNS * 56962:,}\n")
    
    t0_runs = time.time()
    task_args = [(s, X_raw, y_raw, amounts_raw) for s in seeds]
    
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, result in enumerate(executor.map(run_single_simulation, task_args), 1):
            all_results.append(result)
            if idx % 50 == 0 or idx == N_RUNS:
                elapsed = time.time() - t0_runs
                rate = idx / elapsed
                remaining = (N_RUNS - idx) / rate if rate > 0 else 0
                pct = (idx / N_RUNS) * 100
                print(f"Progress: [{idx:4d}/{N_RUNS}] ({pct:5.1f}%) | Speed: {rate:4.1f} runs/sec | Elapsed: {elapsed:5.1f}s | Est. Remaining: {remaining:5.1f}s")
                sys.stdout.flush()
                
    total_time = time.time() - t0_runs
    print(f"\nAll {N_RUNS:,} Monte Carlo Simulations Completed in {total_time:.2f}s ({total_time/60.0:.2f} minutes; {total_time/N_RUNS:.3f}s/run average)!\n")
    
    out_dir = script_dir if os.path.basename(os.getcwd()) != 'simulations' else '.'
    raw_path = os.path.join(out_dir, "benchmark_1000_runs_raw.csv")
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
            mu = float(np.mean(vals_disp))
            sigma = float(np.std(vals_disp, ddof=1))
            margin_err = float(1.96 * (sigma / np.sqrt(N_RUNS)))
            ci_low = mu - margin_err
            ci_high = mu + margin_err
            median = float(np.median(vals_disp))
            p1 = float(np.percentile(vals_disp, 1))
            p5 = float(np.percentile(vals_disp, 5))
            p25 = float(np.percentile(vals_disp, 25))
            p75 = float(np.percentile(vals_disp, 75))
            p95 = float(np.percentile(vals_disp, 95))
            p99 = float(np.percentile(vals_disp, 99))
            iqr = p75 - p25
            v_min = float(np.min(vals_disp))
            v_max = float(np.max(vals_disp))
            skew = compute_skewness(vals_disp)
            kurt = compute_kurtosis(vals_disp)
            
            m_row[f"{met}_Mean"] = mu
            m_row[f"{met}_Std"] = sigma
            m_row[f"{met}_CI95_Margin"] = margin_err
            m_row[f"{met}_CI95_Low"] = ci_low
            m_row[f"{met}_CI95_High"] = ci_high
            m_row[f"{met}_Median"] = median
            m_row[f"{met}_IQR"] = iqr
            m_row[f"{met}_P1"] = p1
            m_row[f"{met}_P5"] = p5
            m_row[f"{met}_P95"] = p95
            m_row[f"{met}_P99"] = p99
            m_row[f"{met}_Min"] = v_min
            m_row[f"{met}_Max"] = v_max
            m_row[f"{met}_Skewness"] = skew
            m_row[f"{met}_Kurtosis"] = kurt
            
            # Tail risk metrics for loss
            if met == 'Net_Loss':
                # VaR 95% is 95th percentile of loss
                m_row['Net_Loss_VaR95'] = p95
                m_row['Net_Loss_VaR99'] = p99
                # CVaR 99% is expected loss exceeding VaR99
                worst_1pct = vals_disp[vals_disp >= p99]
                m_row['Net_Loss_CVaR99'] = float(np.mean(worst_1pct)) if len(worst_1pct) > 0 else p99
            
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
                'P1': p1,
                'P5': p5,
                'P95': p95,
                'P99': p99,
                'Min': v_min,
                'Max': v_max,
                'Skewness': skew,
                'Kurtosis': kurt
            })
            
        summary_rows.append(m_row)
        
    summary_path = os.path.join(out_dir, "benchmark_1000_runs_summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary benchmarks saved to: {summary_path}")
    
    breakdown_path = os.path.join(out_dir, "benchmark_1000_runs_statistical_breakdown.csv")
    df_detailed = pd.DataFrame(detailed_stat_rows)
    df_detailed.to_csv(breakdown_path, index=False)
    print(f"Detailed statistical breakdown saved to: {breakdown_path}")
    
    print("\n" + "=" * 115)
    print("1,000-RUN STATISTICAL BENCHMARK SUMMARY (MEAN ± 95% NOMINAL MARGIN, MEDIAN, AND TAIL LOSS)")
    print("=" * 115)
    for _, r in df_summary.iterrows():
        print(f"\n>>> {r['Model']} <<<")
        print(f"  • Event Recall       : {r['Recall_Mean']:5.2f}% ± {r['Recall_CI95_Margin']:4.2f}% (Median: {r['Recall_Median']:5.2f}%, Range: [{r['Recall_Min']:5.2f}%, {r['Recall_Max']:5.2f}%], Std: {r['Recall_Std']:4.2f}%)")
        print(f"  • Dollar Value Recall: {r['Dollar_Recall_Mean']:5.2f}% ± {r['Dollar_Recall_CI95_Margin']:4.2f}% (Median: {r['Dollar_Recall_Median']:5.2f}%, Range: [{r['Dollar_Recall_Min']:5.2f}%, {r['Dollar_Recall_Max']:5.2f}%], Std: {r['Dollar_Recall_Std']:4.2f}%)")
        print(f"  • Precision          : {r['Precision_Mean']:5.2f}% ± {r['Precision_CI95_Margin']:4.2f}% (Median: {r['Precision_Median']:5.2f}%, Range: [{r['Precision_Min']:5.2f}%, {r['Precision_Max']:5.2f}%], Std: {r['Precision_Std']:4.2f}%)")
        print(f"  • F1-Score           : {r['F1_Mean']:5.2f}% ± {r['F1_CI95_Margin']:4.2f}% (Median: {r['F1_Median']:5.2f}%, Range: [{r['F1_Min']:5.2f}%, {r['F1_Max']:5.2f}%], Std: {r['F1_Std']:4.2f}%)")
        print(f"  • ROC-AUC            : {r['ROC_AUC_Mean']:6.4f} ± {r['ROC_AUC_CI95_Margin']:6.4f} (Median: {r['ROC_AUC_Median']:6.4f}, Range: [{r['ROC_AUC_Min']:6.4f}, {r['ROC_AUC_Max']:6.4f}], Std: {r['ROC_AUC_Std']:6.4f})")
        print(f"  • Capital Protected  : ${r['Dollars_Protected_Mean']:,.2f} / run (Median: ${r['Dollars_Protected_Median']:,.2f}, Range: [${r['Dollars_Protected_Min']:,.2f}, ${r['Dollars_Protected_Max']:,.2f}])")
        print(f"  • Net Financial Loss : ${r['Net_Loss_Mean']:,.2f} / run (Median: ${r['Net_Loss_Median']:,.2f}, Std: ${r['Net_Loss_Std']:,.2f})")
        print(f"    - VaR 95% (Worst 5% Loss): ${r['Net_Loss_VaR95']:,.2f}")
        print(f"    - VaR 99% (Worst 1% Loss): ${r['Net_Loss_VaR99']:,.2f}")
        print(f"    - CVaR 99% (Expected Shortfall): ${r['Net_Loss_CVaR99']:,.2f}")
        print(f"  • Operating Errors   : {r['FP_Mean']:6.1f} False Alarms, {r['FN_Mean']:4.1f} Missed Frauds")

if __name__ == '__main__':
    main()
