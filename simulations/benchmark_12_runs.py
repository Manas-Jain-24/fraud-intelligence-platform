import time
import numpy as np
import pandas as pd

print("=" * 80)
print("12-RUN MONTE CARLO STABILITY & HEAD-TO-HEAD BENCHMARK SUITE")
print("Evaluating 6 First-Principles ML Models on Kaggle Credit Card Fraud Dataset")
print("100% Pure NumPy Implementation (Zero Scikit-Learn / ML Library Packages)")
print("=" * 80)

# Load dataset once
t0_load = time.time()
df = pd.read_csv("data/creditcard.csv")
print(f"Dataset Loaded in {time.time() - t0_load:.2f}s: {len(df):,} transactions, {df['Class'].sum():,} frauds.\n")

seconds_in_day = 86400.0
time_sin = np.sin(2 * np.pi * df['Time'].values / seconds_in_day)
time_cos = np.cos(2 * np.pi * df['Time'].values / seconds_in_day)
log_amount = np.log1p(df['Amount'].values)
v_cols = [f'V{i}' for i in range(1, 29)]
X_raw = np.column_stack([time_sin, time_cos, log_amount, df[v_cols].values])
y_raw = df['Class'].values.astype(np.int64)
amounts_raw = df['Amount'].values

# First-principles quadrature for ROC-AUC
def trapezoid_scratch(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))

def compute_metrics(y_true, amounts, prob, threshold=0.5):
    y_pred = (prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    tot_dollars = np.sum(amounts[y_true == 1])
    caught_dollars = np.sum(amounts[(y_true == 1) & (y_pred == 1)])
    missed_dollars = np.sum(amounts[(y_true == 1) & (y_pred == 0)])
    dollar_rec = caught_dollars / tot_dollars if tot_dollars > 0 else 0.0
    bank_loss = missed_dollars + fp * 2.50
    
    # 41 threshold points for fast & accurate ROC-AUC
    th_s = np.linspace(0, 1, 41)
    tprs, fprs = [], []
    P, N = np.sum(y_true == 1), np.sum(y_true == 0)
    for th in th_s:
        yp = (prob >= th).astype(int)
        tprs.append(np.sum((y_true == 1) & (yp == 1)) / P)
        fprs.append(np.sum((y_true == 0) & (yp == 1)) / N)
    fpr_arr = np.array(fprs)
    tpr_arr = np.array(tprs)
    si = np.argsort(fpr_arr)
    auc = trapezoid_scratch(tpr_arr[si], fpr_arr[si])
    
    return {
        'Precision': prec,
        'Recall': rec,
        'Dollar_Recall': dollar_rec,
        'F1': f1,
        'ROC_AUC': auc,
        'Dollars_Protected': caught_dollars,
        'Bank_Loss': bank_loss,
        'FP': int(fp),
        'FN': int(fn)
    }

# Vectorized SMOTE
def smote_scratch(X_min, n_synthetic, k=5, seed=42):
    np.random.seed(seed)
    n_min, _ = X_min.shape
    k = min(k, n_min - 1)
    dot_prods = np.dot(X_min, X_min.T)
    sq_norms = np.diag(dot_prods)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot_prods
    np.maximum(dist_sq, 0.0, out=dist_sq)
    np.fill_diagonal(dist_sq, np.inf)
    knn_idx = np.argpartition(dist_sq, kth=k, axis=1)[:, :k]
    base_idx = np.random.choice(n_min, size=n_synthetic, replace=True)
    picks = np.random.randint(0, k, size=n_synthetic)
    chosen = knn_idx[base_idx, picks]
    lambdas = np.random.uniform(0.0, 1.0, size=(n_synthetic, 1))
    return X_min[base_idx] + lambdas * (X_min[chosen] - X_min[base_idx])

# 1. Baseline Logistic Regression
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=600, l2_reg=0.01):
        self.lr, self.n_iter, self.l2_reg = lr, n_iter, l2_reg
    def fit(self, X, y):
        m, d = X.shape
        self.w, self.b = np.zeros(d, dtype=np.float64), 0.0
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
            y_hat = 1.0 / (1.0 + np.exp(-z))
            err = y_hat - y
            dw = (1.0 / m) * np.dot(X.T, err) + (self.l2_reg / m) * self.w
            db = (1.0 / m) * np.sum(err)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self
    def predict_proba(self, X):
        z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

# 2. Cost-Sensitive Logistic Regression
class CostSensitiveLogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=600, l2_reg=0.01, kappa=1.5):
        self.lr, self.n_iter, self.l2_reg, self.kappa = lr, n_iter, l2_reg, kappa
    def fit(self, X, y, amounts):
        m, d = X.shape
        self.w, self.b = np.zeros(d, dtype=np.float64), 0.0
        c = np.ones_like(y, dtype=np.float64)
        c[y == 1] += self.kappa * np.log1p(amounts[y == 1])
        c_sum = np.sum(c)
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
            y_hat = 1.0 / (1.0 + np.exp(-z))
            err = c * (y_hat - y)
            dw = (1.0 / c_sum) * np.dot(X.T, err) + (self.l2_reg / m) * self.w
            db = (1.0 / c_sum) * np.sum(err)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self
    def predict_proba(self, X):
        z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

# 3. Multi-Layer Perceptron (Neural Network)
class NeuralNetworkScratch:
    def __init__(self, hidden=(32, 16), lr=0.05, epochs=180, momentum=0.9, seed=42):
        self.hidden, self.lr, self.epochs, self.momentum, self.seed = hidden, lr, epochs, momentum, seed
    def fit(self, X, y):
        np.random.seed(self.seed)
        m, d = X.shape
        h1, h2 = self.hidden
        self.W1 = np.random.randn(d, h1) * np.sqrt(2.0 / d)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, 1))
        vW1, vb1, vW2, vb2, vW3, vb3 = np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2), np.zeros_like(self.W3), np.zeros_like(self.b3)
        y_mat = y.reshape(-1, 1)
        for _ in range(self.epochs):
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = np.maximum(0.0, Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = np.maximum(0.0, Z2)
            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = 1.0 / (1.0 + np.exp(-np.clip(Z3, -500.0, 500.0)))
            dZ3 = A3 - y_mat
            dW3 = np.dot(A2.T, dZ3) / m
            db3 = np.sum(dZ3, axis=0, keepdims=True) / m
            dA2 = np.dot(dZ3, self.W3.T)
            dZ2 = dA2 * (Z2 > 0)
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * (Z1 > 0)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            vW3 = self.momentum * vW3 + self.lr * dW3
            vb3 = self.momentum * vb3 + self.lr * db3
            vW2 = self.momentum * vW2 + self.lr * dW2
            vb2 = self.momentum * vb2 + self.lr * db2
            vW1 = self.momentum * vW1 + self.lr * dW1
            vb1 = self.momentum * vb1 + self.lr * db1
            self.W3 -= vW3; self.b3 -= vb3
            self.W2 -= vW2; self.b2 -= vb2
            self.W1 -= vW1; self.b1 -= vb1
        return self
    def predict_proba(self, X):
        A1 = np.maximum(0.0, np.dot(X, self.W1) + self.b1)
        A2 = np.maximum(0.0, np.dot(A1, self.W2) + self.b2)
        return (1.0 / (1.0 + np.exp(-np.clip(np.dot(A2, self.W3) + self.b3, -500.0, 500.0)))).flatten()

# 4. Gradient Boosted Decision Trees (GBDT)
class DecisionTreeRegressorNode:
    def __init__(self, depth=0, max_depth=3):
        self.depth, self.max_depth = depth, max_depth
        self.feature, self.threshold, self.val = None, None, None
        self.left, self.right = None, None
    def fit(self, X, r, p):
        if self.depth >= self.max_depth or len(r) <= 10:
            self.val = np.sum(r) / np.sum(p * (1.0 - p) + 1e-10)
            return self
        m, d = X.shape
        best_feat, best_th, min_sse = None, None, np.inf
        current_sse = np.sum((r - np.mean(r)) ** 2)
        sub_feats = np.random.choice(d, size=min(6, d), replace=False)
        for f in sub_feats:
            vals = np.percentile(X[:, f], [25, 50, 75])
            for th in vals:
                left_mask = X[:, f] <= th
                if np.sum(left_mask) < 5 or np.sum(~left_mask) < 5:
                    continue
                sse = np.sum((r[left_mask] - np.mean(r[left_mask]))**2) + np.sum((r[~left_mask] - np.mean(r[~left_mask]))**2)
                if sse < min_sse:
                    min_sse, best_feat, best_th = sse, f, th
        if best_feat is None or (current_sse - min_sse) < 1e-4:
            self.val = np.sum(r) / np.sum(p * (1.0 - p) + 1e-10)
            return self
        self.feature, self.threshold = best_feat, best_th
        mask = X[:, best_feat] <= best_th
        self.left = DecisionTreeRegressorNode(self.depth+1, self.max_depth).fit(X[mask], r[mask], p[mask])
        self.right = DecisionTreeRegressorNode(self.depth+1, self.max_depth).fit(X[~mask], r[~mask], p[~mask])
        return self
    def predict(self, X):
        if self.val is not None: return np.full(len(X), self.val)
        mask = X[:, self.feature] <= self.threshold
        out = np.zeros(len(X))
        out[mask] = self.left.predict(X[mask])
        out[~mask] = self.right.predict(X[~mask])
        return out

class GBDTClassifierScratch:
    def __init__(self, n_trees=20, lr=0.15, max_depth=3, seed=42):
        self.n_trees, self.lr, self.max_depth, self.seed = n_trees, lr, max_depth, seed
        self.trees, self.f0 = [], 0.0
    def fit(self, X, y):
        np.random.seed(self.seed)
        p_pos = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5)
        self.f0 = np.log(p_pos / (1.0 - p_pos))
        F = np.full(len(X), self.f0)
        self.trees = []
        for _ in range(self.n_trees):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500.0, 500.0)))
            r = y - p
            tree = DecisionTreeRegressorNode(max_depth=self.max_depth).fit(X, r, p)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self
    def predict_proba(self, X):
        F = np.full(len(X), self.f0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(F, -500.0, 500.0)))

# 5. Isolation Forest
class IsolationTreeNode:
    def __init__(self, depth=0, max_depth=8):
        self.depth, self.max_depth = depth, max_depth
        self.split_feat, self.split_val = None, None
        self.left, self.right, self.size = None, None, 0
    def fit(self, X):
        n, d = X.shape
        self.size = n
        if self.depth >= self.max_depth or n <= 1: return self
        self.split_feat = np.random.randint(0, d)
        col = X[:, self.split_feat]
        mi, ma = np.min(col), np.max(col)
        if mi == ma: return self
        self.split_val = np.random.uniform(mi, ma)
        left_mask = col < self.split_val
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0: return self
        self.left = IsolationTreeNode(self.depth + 1, self.max_depth).fit(X[left_mask])
        self.right = IsolationTreeNode(self.depth + 1, self.max_depth).fit(X[~left_mask])
        return self
    def path_length(self, X):
        if self.split_feat is None or self.left is None or self.right is None or len(X) == 0:
            return np.full(len(X), self.depth + self._c(self.size))
        mask = X[:, self.split_feat] < self.split_val
        res = np.zeros(len(X))
        res[mask] = self.left.path_length(X[mask])
        res[~mask] = self.right.path_length(X[~mask])
        return res
    def _c(self, n):
        if n <= 1: return 0.0
        if n == 2: return 1.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

class IsolationForestScratch:
    def __init__(self, n_trees=35, max_samples=256, seed=42):
        self.n_trees, self.max_samples, self.seed = n_trees, max_samples, seed
        self.trees = []
    def fit(self, X):
        np.random.seed(self.seed)
        n = len(X)
        sub_size = min(self.max_samples, n)
        max_depth = int(np.ceil(np.log2(max(sub_size, 2))))
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, size=sub_size, replace=False)
            self.trees.append(IsolationTreeNode(max_depth=max_depth).fit(X[idx]))
        return self
    def _c(self, n):
        if n <= 1: return 0.0
        if n == 2: return 1.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
    def predict_proba(self, X):
        paths = np.array([tree.path_length(X) for tree in self.trees])
        mean_paths = np.mean(paths, axis=0)
        scores = 2.0 ** (-mean_paths / self._c(min(self.max_samples, len(X))))
        s_min, s_max = np.min(scores), np.max(scores)
        return (scores - s_min) / (s_max - s_min + 1e-10)

# 12 Independent Random Seeds
SEEDS = [42, 101, 7, 999, 2026, 13, 88, 555, 1234, 777, 314, 99]

models_to_test = [
    'Baseline Logistic Regression',
    'Cost-Sensitive LogReg',
    'Multi-Layer Perceptron (NN)',
    'GBDT Classifier',
    'Isolation Forest (Unsupervised)',
    'Hybrid Soft-Voting Ensemble'
]

# Tracking container
records = {m: [] for m in models_to_test}

print(f"Beginning 12 Monte Carlo Runs across {len(SEEDS)} seeds...")
t_start = time.time()

for r_idx, s in enumerate(SEEDS, 1):
    t_run = time.time()
    np.random.seed(s)
    
    # 1. Stratified 80/20 train/test split
    idx_0 = np.where(y_raw == 0)[0]
    idx_1 = np.where(y_raw == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    n_te_0 = int(len(idx_0) * 0.2)
    n_te_1 = int(len(idx_1) * 0.2)
    test_idx = np.concatenate([idx_0[:n_te_0], idx_1[:n_te_1]])
    train_idx = np.concatenate([idx_0[n_te_0:], idx_1[n_te_1:]])
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)
    
    X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
    y_tr, y_te = y_raw[train_idx], y_raw[test_idx]
    amt_tr, amt_te = amounts_raw[train_idx], amounts_raw[test_idx]
    
    # 2. Vectorized SMOTE & Class Rebalancing
    X_min_tr = X_tr[y_tr == 1]
    amt_min_tr = amt_tr[y_tr == 1]
    n_syn = len(X_min_tr) * 3
    X_syn = smote_scratch(X_min_tr, n_synthetic=n_syn, k=5, seed=s)
    amt_syn = np.random.choice(amt_min_tr, size=n_syn, replace=True)
    
    X_fraud_aug = np.vstack([X_min_tr, X_syn])
    amt_fraud_aug = np.concatenate([amt_min_tr, amt_syn])
    idx_0_tr = np.where(y_tr == 0)[0]
    sampled_0 = np.random.choice(idx_0_tr, size=len(X_fraud_aug), replace=False)
    
    X_bal = np.vstack([X_fraud_aug, X_tr[sampled_0]])
    y_bal = np.concatenate([np.ones(len(X_fraud_aug), dtype=np.int64), np.zeros(len(sampled_0), dtype=np.int64)])
    amt_bal = np.concatenate([amt_fraud_aug, amt_tr[sampled_0]])
    
    perm = np.random.permutation(len(y_bal))
    X_bal, y_bal, amt_bal = X_bal[perm], y_bal[perm], amt_bal[perm]
    
    # Standardize
    mean = np.mean(X_bal, axis=0)
    std = np.std(X_bal, axis=0)
    std[std == 0.0] = 1.0
    X_tr_n = (X_bal - mean) / std
    X_te_n = (X_te - mean) / std
    
    # Fit Models
    m1 = LogisticRegressionScratch(lr=0.1, n_iter=500).fit(X_tr_n, y_bal)
    p1 = m1.predict_proba(X_te_n)
    
    m2 = CostSensitiveLogisticRegressionScratch(lr=0.1, n_iter=500, kappa=1.5).fit(X_tr_n, y_bal, amt_bal)
    p2 = m2.predict_proba(X_te_n)
    
    m3 = NeuralNetworkScratch(hidden=(32, 16), lr=0.05, epochs=160, seed=s).fit(X_tr_n, y_bal)
    p3 = m3.predict_proba(X_te_n)
    
    m4 = GBDTClassifierScratch(n_trees=20, lr=0.15, max_depth=3, seed=s).fit(X_tr_n, y_bal)
    p4 = m4.predict_proba(X_te_n)
    
    m5 = IsolationForestScratch(n_trees=35, max_samples=256, seed=s).fit(X_tr[y_tr == 0])
    p5 = m5.predict_proba(X_te)
    
    # Hybrid Soft-Voting Ensemble
    p6 = 0.35 * p2 + 0.30 * p3 + 0.25 * p4 + 0.10 * p5
    
    probs = [p1, p2, p3, p4, p5, p6]
    for m_name, p_val in zip(models_to_test, probs):
        m_dict = compute_metrics(y_te, amt_te, p_val)
        records[m_name].append(m_dict)
        
    print(f"Run {r_idx:2d}/12 (Seed={s:4d}) Finished in {time.time() - t_run:.2f}s | Test Fraud: ${amt_te[y_te==1].sum():,.2f}")

tot_time = time.time() - t_start
print(f"\nAll 12 Runs Completed Successfully in {tot_time:.2f}s (~{tot_time/12:.2f}s per run)!\n")

# Summary Analysis
summary_list = []
for m_name in models_to_test:
    df_m = pd.DataFrame(records[m_name])
    summary_list.append({
        'Model': m_name,
        'Precision (Mean)': df_m['Precision'].mean() * 100,
        'Precision (Std)': df_m['Precision'].std() * 100,
        'Recall (Mean)': df_m['Recall'].mean() * 100,
        'Recall (Std)': df_m['Recall'].std() * 100,
        'Dollar Recall (Mean)': df_m['Dollar_Recall'].mean() * 100,
        'Dollar Recall (Std)': df_m['Dollar_Recall'].std() * 100,
        'F1-Score (Mean)': df_m['F1'].mean() * 100,
        'F1-Score (Std)': df_m['F1'].std() * 100,
        'ROC-AUC (Mean)': df_m['ROC_AUC'].mean(),
        'ROC-AUC (Std)': df_m['ROC_AUC'].std(),
        'Avg Dollars Saved': df_m['Dollars_Protected'].mean(),
        'Avg Bank Loss': df_m['Bank_Loss'].mean(),
        'Avg FP': df_m['FP'].mean(),
        'Avg FN': df_m['FN'].mean()
    })

df_summary = pd.DataFrame(summary_list)

print("=" * 115)
print("12-RUN MONTE CARLO STABILITY & COMPARATIVE SUMMARY TABLE (MEAN ± STD)")
print("=" * 115)
format_cols = []
for _, row in df_summary.iterrows():
    print(f"\n>>> {row['Model']} <<<")
    print(f"  • Precision          : {row['Precision (Mean)']:.2f}% ± {row['Precision (Std)']:.2f}%")
    print(f"  • Event Recall       : {row['Recall (Mean)']:.2f}% ± {row['Recall (Std)']:.2f}%")
    print(f"  • Dollar Value Recall: {row['Dollar Recall (Mean)']:.2f}% ± {row['Dollar Recall (Std)']:.2f}%  <-- Real Capital Saved")
    print(f"  • F1-Score           : {row['F1-Score (Mean)']:.2f}% ± {row['F1-Score (Std)']:.2f}%")
    print(f"  • ROC-AUC            : {row['ROC-AUC (Mean)']:.4f} ± {row['ROC-AUC (Std)']:.4f}")
    print(f"  • Financial Recovery : ${row['Avg Dollars Saved']:,.2f} protected per run")
    print(f"  • Residual Loss      : ${row['Avg Bank Loss']:,.2f} total net cost (FN + FP)")
    print(f"  • Error Profile      : {row['Avg FP']:.1f} False Alarms, {row['Avg FN']:.1f} Missed Frauds")

# Save summary to CSV for user reference
df_summary.to_csv("benchmark_12_runs_summary.csv", index=False)
print("\nDetailed summary table saved to benchmark_12_runs_summary.csv!")
