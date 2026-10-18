import pandas as pd
import numpy as np
import os
import sys

raw_path = r'C:\Users\HP\.cache\kagglehub\datasets\ealaxi\paysim1\versions\2\PS_20174392719_1491204439457_log.csv'
if not os.path.exists(raw_path):
    import kagglehub
    print("Downloading PaySim from Kaggle via kagglehub...", flush=True)
    path = kagglehub.dataset_download("ealaxi/paysim1")
    for f in os.listdir(path):
        if f.endswith('.csv'):
            raw_path = os.path.join(path, f)
            break

frauds = []
legit_samples = []

print("Reading PaySim dataset stream...", flush=True)
cols = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud']

chunk_iter = pd.read_csv(raw_path, usecols=cols, chunksize=500000)
for i, chunk in enumerate(chunk_iter):
    filtered = chunk[chunk['type'].isin(['TRANSFER', 'CASH_OUT'])]
    f = filtered[filtered['isFraud'] == 1]
    if len(f) > 0:
        frauds.append(f)
    
    nf = filtered[filtered['isFraud'] == 0]
    sample_nf = nf.sample(frac=0.035, random_state=42 + i)
    legit_samples.append(sample_nf)
    print(f"Processed chunk {i+1} (500k rows)...", flush=True)

df_fraud = pd.concat(frauds, ignore_index=True)
df_legit = pd.concat(legit_samples, ignore_index=True)
print(f"Frauds collected: {len(df_fraud):,}, Legit sample collected: {len(df_legit):,}", flush=True)

target_legit = 100000 - len(df_fraud)
if len(df_legit) > target_legit:
    df_legit = df_legit.sample(n=target_legit, random_state=42)

df_benchmark = pd.concat([df_fraud, df_legit], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Domain feature engineering: balance discrepancies reveal fraud in mobile money transfers
df_benchmark['errorBalanceOrig'] = df_benchmark['newbalanceOrig'] + df_benchmark['amount'] - df_benchmark['oldbalanceOrg']
df_benchmark['errorBalanceDest'] = df_benchmark['oldbalanceDest'] + df_benchmark['amount'] - df_benchmark['newbalanceDest']
df_benchmark['balanceDrainRatio'] = df_benchmark['amount'] / (df_benchmark['oldbalanceOrg'] + 1.0)
df_benchmark['isTransfer'] = (df_benchmark['type'] == 'TRANSFER').astype(int)

out_file = 'data/paysim_benchmark.csv'
df_benchmark.to_csv(out_file, index=False)
sz_mb = os.path.getsize(out_file) / (1024 * 1024)
print(f"SUCCESS: Saved {len(df_benchmark):,} transactions ({len(df_fraud):,} fraud) to {out_file} ({sz_mb:.2f} MB)", flush=True)
