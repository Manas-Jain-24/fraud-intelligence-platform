import time
import numpy as np
import pandas as pd

print("=" * 85)
print("STARTING 5-RUN MONTE CARLO STABILITY BENCHMARK ACROSS 5 INDEPENDENT SEEDS")
print("Dataset: 284,807 transactions (Credit Card Fraud Detection)")
print("Permitted: Pure NumPy from scratch (Zero external ML libraries)")
print("=" * 85)

t_grand0 = time.time()

# 1. Load data
df = pd.read_csv('C:/Users/HP/Desktop/PDS/data/creditcard.csv')

# 2. Feature engineering
hours = (df['Time'] / 3600.0) % 24.0
df['Time_sin'] = np.sin(2.0 * np.pi * hours / 24.0)
df['Time_cos'] = np.cos(2.0 * np.pi * hours / 24.0)
df['Amount_log'] = np.log1p(df['Amount'])

feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time_sin', 'Time_cos', 'Amount_log']
X_all = df[feature_cols].to_numpy(dtype=np.float64)
y_all = df['Class'].to_numpy(dtype=np.int64)

def trapezoid_scratch(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))

def smote_scratch(X_min, n_synthetic, k=5, seed=42):
    np.random.seed(seed)
    n_min, n_features = X_min.shape
    k = min(k, n_min - 1)
    dot_prods = np.dot(X_min, X_min.T)
    sq_norms = np.diag(dot_prods)
    dist_matrix_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * dot_prods
    np.maximum(dist_matrix_sq, 0.0, out=dist_matrix_sq)
    np.fill_diagonal(dist_matrix_sq, np.inf)
    knn_indices = np.argpartition(dist_matrix_sq, kth=k, axis=1)[:, :k]
    base_indices = np.random.choice(n_min, size=n_synthetic, replace=True)
    neighbor_col_picks = np.random.randint(0, k, size=n_synthetic)
    chosen_neighbors = knn_indices[base_indices, neighbor_col_picks]
    lambdas = np.random.uniform(0.0, 1.0, size=(n_synthetic, 1))
    diffs = X_min[chosen_neighbors] - X_min[base_indices]
    return X_min[base_indices] + lambdas * diffs

# Model 1: Logistic Regression
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=1000, l2=0.01):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
    def fit(self, X, y):
        m, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            self.w -= self.lr * ((1.0 / m) * np.dot(X.T, (p - y)) + (self.l2 / m) * self.w)
            self.b -= self.lr * ((1.0 / m) * np.sum(p - y))
        return self
    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.w) + self.b, -500, 500)))

# Model 2: Deep Neural Network (MLP)
class NeuralNetworkScratch:
    def __init__(self, d, h1=32, h2=16, lr=0.03, beta=0.9, l2=0.001, seed=42):
        np.random.seed(seed)
        self.lr, self.beta, self.l2 = lr, beta, l2
        self.W1 = np.random.randn(d, h1) * np.sqrt(2.0 / d)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, 1))
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.vW3, self.vb3 = np.zeros_like(self.W3), np.zeros_like(self.b3)
    def relu(self, z): return np.maximum(0.0, z)
    def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.sigmoid(z3)
        return a3, (X, z1, a1, z2, a2, z3, a3)
    def backward(self, cache, y_b):
        X, z1, a1, z2, a2, z3, a3 = cache
        m = len(X)
        d3 = (a3 - y_b) / m
        dW3 = np.dot(a2.T, d3) + (self.l2 / m) * self.W3
        db3 = np.sum(d3, axis=0, keepdims=True)
        d2 = np.dot(d3, self.W3.T) * (z2 > 0)
        dW2 = np.dot(a1.T, d2) + (self.l2 / m) * self.W2
        db2 = np.sum(d2, axis=0, keepdims=True)
        d1 = np.dot(d2, self.W2.T) * (z1 > 0)
        dW1 = np.dot(X.T, d1) + (self.l2 / m) * self.W1
        db1 = np.sum(d1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2, dW3, db3
    def update(self, dW1, db1, dW2, db2, dW3, db3):
        self.vW1 = self.beta * self.vW1 + (1 - self.beta) * dW1
        self.vb1 = self.beta * self.vb1 + (1 - self.beta) * db1
        self.vW2 = self.beta * self.vW2 + (1 - self.beta) * dW2
        self.vb2 = self.beta * self.vb2 + (1 - self.beta) * db2
        self.vW3 = self.beta * self.vW3 + (1 - self.beta) * dW3
        self.vb3 = self.beta * self.vb3 + (1 - self.beta) * db3
        self.W1 -= self.lr * self.vW1
        self.b1 -= self.lr * self.vb1
        self.W2 -= self.lr * self.vW2
        self.b2 -= self.lr * self.vb2
        self.W3 -= self.lr * self.vW3
        self.b3 -= self.lr * self.vb3
    def fit(self, X, y, epochs=80, batch_size=64):
        m = len(X)
        y_col = y.reshape(-1, 1)
        for _ in range(epochs):
            perm = np.random.permutation(m)
            X_s, y_s = X[perm], y_col[perm]
            for i in range(0, m, batch_size):
                X_b, y_b = X_s[i:i+batch_size], y_s[i:i+batch_size]
                a3, cache = self.forward(X_b)
                grads = self.backward(cache, y_b)
                self.update(*grads)
        return self
    def predict_proba(self, X):
        a3, _ = self.forward(X)
        return a3.ravel()

# Model 3: Gradient Boosted Trees
class TreeNode:
    def __init__(self, depth=0, max_depth=3):
        self.depth, self.max_depth = depth, max_depth
        self.feature_idx, self.threshold = None, None
        self.left, self.right = None, None
        self.value = 0.0
        self.is_leaf = False

class RegressionTreeScratch:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None
    def fit(self, X, r, p):
        self.root = self._build(X, r, p, 0)
        return self
    def _build(self, X, r, p, depth):
        node = TreeNode(depth=depth, max_depth=self.max_depth)
        node.value = np.sum(r) / (np.sum(p * (1.0 - p)) + 1e-10)
        if depth >= self.max_depth or len(r) < 10:
            node.is_leaf = True
            return node
        best_gain, best_feat, best_th = 0.0, None, None
        base_var = np.var(r) * len(r)
        for f in range(X.shape[1]):
            col = X[:, f]
            for th in np.unique(np.percentile(col, q=np.linspace(15, 85, 7))):
                lm = col <= th
                if np.sum(lm) < 5 or np.sum(~lm) < 5: continue
                gain = base_var - (np.var(r[lm]) * np.sum(lm) + np.var(r[~lm]) * np.sum(~lm))
                if gain > best_gain:
                    best_gain, best_feat, best_th = gain, f, th
        if best_gain <= 0.0 or best_feat is None:
            node.is_leaf = True
            return node
        node.feature_idx, node.threshold = best_feat, best_th
        lm = X[:, best_feat] <= best_th
        node.left = self._build(X[lm], r[lm], p[lm], depth + 1)
        node.right = self._build(X[~lm], r[~lm], p[~lm], depth + 1)
        return node
    def _pred_row(self, node, x):
        if node.is_leaf: return node.value
        return self._pred_row(node.left if x[node.feature_idx] <= node.threshold else node.right, x)
    def predict(self, X):
        return np.array([self._pred_row(self.root, row) for row in X])

class GBDTScratch:
    def __init__(self, n_est=25, lr=0.1, max_depth=3, seed=42):
        self.n_est, self.lr, self.max_depth, self.seed = n_est, lr, max_depth, seed
        self.trees, self.F0 = [], 0.0
    def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        np.random.seed(self.seed)
        p0 = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5)
        self.F0 = float(np.log(p0 / (1.0 - p0)))
        F = np.full(len(X), self.F0)
        self.trees = []
        for _ in range(self.n_est):
            p = self.sigmoid(F)
            r = y - p
            tree = RegressionTreeScratch(max_depth=self.max_depth).fit(X, r, p)
            self.trees.append(tree)
            F += self.lr * tree.predict(X)
        return self
    def predict_proba(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return self.sigmoid(F)

# Model 4: Isolation Forest
class IsolationForestScratch:
    def __init__(self, n_est=60, max_samples=256, seed=42):
        self.n_est, self.max_samples, self.seed = n_est, max_samples, seed
    def _c(self, n):
        if n <= 1: return 0.0
        if n == 2: return 1.0
        return 2.0 * (np.log(n - 1.0) + 0.5772156649) - (2.0 * (n - 1.0) / n)
    def fit(self, X):
        np.random.seed(self.seed)
        psi = min(self.max_samples, len(X))
        self.c_psi = self._c(psi)
        self.max_d = int(np.ceil(np.log2(max(psi, 2))))
        self.trees = []
        for _ in range(self.n_est):
            X_sub = X[np.random.choice(len(X), size=psi, replace=False)]
            self.trees.append(self._build(X_sub, 0))
        return self
    def _build(self, X, d):
        if d >= self.max_d or len(X) <= 1:
            return {'leaf': True, 'depth': d, 'size': len(X)}
        f = np.random.randint(0, X.shape[1])
        col = X[:, f]
        mn, mx = np.min(col), np.max(col)
        if mn == mx: return {'leaf': True, 'depth': d, 'size': len(X)}
        p = np.random.uniform(mn, mx)
        lm = col < p
        return {'leaf': False, 'f': f, 'p': p, 'l': self._build(X[lm], d+1), 'r': self._build(X[~lm], d+1)}
    def _path(self, x, node):
        if node['leaf']: return node['depth'] + self._c(node['size'])
        return self._path(x, node['l'] if x[node['f']] < node['p'] else node['r'])
    def anomaly_score(self, X):
        paths = np.mean([[self._path(row, t) for row in X] for t in self.trees], axis=0)
        return 2.0 ** (-paths / self.c_psi)

def calc_metrics(y_true, prob, th=0.5):
    yp = (prob >= th).astype(int)
    tp = np.sum((y_true == 1) & (yp == 1))
    fp = np.sum((y_true == 0) & (yp == 1))
    fn = np.sum((y_true == 1) & (yp == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    th_s = np.linspace(0, 1, 101)
    tprs, fprs = [], []
    P, N = np.sum(y_true == 1), np.sum(y_true == 0)
    for t in th_s:
        y_c = (prob >= t).astype(int)
        tprs.append(np.sum((y_true == 1) & (y_c == 1)) / P)
        fprs.append(np.sum((y_true == 0) & (y_c == 1)) / N)
    auc = trapezoid_scratch(np.array(tprs)[np.argsort(fprs)], np.sort(fprs))
    return {'Prec': prec, 'Rec': rec, 'F1': f1, 'AUC': auc, 'TP': tp, 'FP': fp, 'FN': fn}

SEEDS = [42, 101, 7, 999, 2026]
results_lr, results_nn, results_gbdt, results_iforest = [], [], [], []

for run_i, seed in enumerate(SEEDS, 1):
    t_run0 = time.time()
    print(f"\n>>> Executing Monte-Carlo Trial {run_i}/5 (Random Seed = {seed}) ...")
    np.random.seed(seed)
    
    # 80/20 Stratified Split
    idx_0 = np.where(y_all == 0)[0]
    idx_1 = np.where(y_all == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    n_te_0, n_te_1 = int(len(idx_0) * 0.2), int(len(idx_1) * 0.2)
    test_idx = np.concatenate([idx_0[:n_te_0], idx_1[:n_te_1]])
    train_idx = np.concatenate([idx_0[n_te_0:], idx_1[n_te_1:]])
    
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]
    
    # SMOTE on train
    X_min = X_tr[y_tr == 1]
    X_syn = smote_scratch(X_min, n_synthetic=len(X_min) * 3, k=5, seed=seed)
    X_f_aug = np.vstack([X_min, X_syn])
    idx_0_tr = np.where(y_tr == 0)[0]
    samp_0 = np.random.choice(idx_0_tr, size=len(X_f_aug), replace=False)
    X_bal = np.vstack([X_f_aug, X_tr[samp_0]])
    y_bal = np.concatenate([np.ones(len(X_f_aug)), np.zeros(len(samp_0))])
    
    # Standardize strictly using unaugmented training partition statistics
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std[std == 0.0] = 1.0
    X_bal_norm = (X_bal - mean) / std
    X_te_norm = (X_te - mean) / std
    
    # 1. Logistic Regression
    m_lr = LogisticRegressionScratch(lr=0.1, n_iter=1000).fit(X_bal_norm, y_bal)
    p_lr = m_lr.predict_proba(X_te_norm)
    res_lr = calc_metrics(y_te, p_lr)
    results_lr.append(res_lr)
    print(f"   [1. Logistic Regression] Recall: {res_lr['Rec']*100:.1f}%, AUC: {res_lr['AUC']:.4f}, Detected: {res_lr['TP']}/{res_lr['TP']+res_lr['FN']}")
    
    # 2. Neural Network
    m_nn = NeuralNetworkScratch(X_bal_norm.shape[1], 32, 16, lr=0.03, seed=seed).fit(X_bal_norm, y_bal, epochs=80)
    p_nn = m_nn.predict_proba(X_te_norm)
    res_nn = calc_metrics(y_te, p_nn)
    results_nn.append(res_nn)
    print(f"   [2. Deep Neural Net]    Recall: {res_nn['Rec']*100:.1f}%, AUC: {res_nn['AUC']:.4f}, Detected: {res_nn['TP']}/{res_nn['TP']+res_nn['FN']}")
    
    # 3. GBDT
    m_gbdt = GBDTScratch(n_est=25, lr=0.1, max_depth=3, seed=seed).fit(X_bal_norm, y_bal)
    p_gbdt = m_gbdt.predict_proba(X_te_norm)
    res_gbdt = calc_metrics(y_te, p_gbdt)
    results_gbdt.append(res_gbdt)
    print(f"   [3. Gradient Boosted]   Recall: {res_gbdt['Rec']*100:.1f}%, AUC: {res_gbdt['AUC']:.4f}, Detected: {res_gbdt['TP']}/{res_gbdt['TP']+res_gbdt['FN']}")
    
    # 4. Isolation Forest
    m_if = IsolationForestScratch(n_est=60, max_samples=256, seed=seed).fit(X_tr[y_tr == 0])
    p_if = m_if.anomaly_score(X_te)
    th_s = np.linspace(np.min(p_if), np.max(p_if), 101)
    tprs, fprs = [], []
    P, N = np.sum(y_te == 1), np.sum(y_te == 0)
    for t in th_s:
        y_c = (p_if >= t).astype(int)
        tprs.append(np.sum((y_te == 1) & (y_c == 1)) / P)
        fprs.append(np.sum((y_te == 0) & (y_c == 1)) / N)
    auc_if = trapezoid_scratch(np.array(tprs)[np.argsort(fprs)], np.sort(fprs))
    results_iforest.append({'AUC': auc_if})
    print(f"   [4. Isolation Forest]   Unsupervised ROC-AUC: {auc_if:.4f}")
    print(f"   Trial {run_i} completed in {time.time()-t_run0:.2f}s")

def format_summary(name, res_list, is_iforest=False):
    df_res = pd.DataFrame(res_list)
    if is_iforest:
        print(f"{name:32s} | Unsupervised ROC-AUC: {df_res['AUC'].mean():.4f} +/- {df_res['AUC'].std():.4f}")
    else:
        print(f"{name:32s} | Recall: {df_res['Rec'].mean()*100:.2f}% +/- {df_res['Rec'].std()*100:.2f}% | F1: {df_res['F1'].mean()*100:.2f}% +/- {df_res['F1'].std()*100:.2f}% | AUC: {df_res['AUC'].mean():.4f} +/- {df_res['AUC'].std():.4f} | Avg Caught: {df_res['TP'].mean():.1f}/98")

print("\n" + "=" * 85)
print("FINAL REPRODUCIBILITY SUMMARY ACROSS 5 INDEPENDENT RUNS (MEAN +/- STD DEV)")
print("=" * 85)
format_summary("Logistic Regression (Scratch)", results_lr)
format_summary("Deep Neural Network (Scratch)", results_nn)
format_summary("Gradient Boosted Trees (Scratch)", results_gbdt)
format_summary("Isolation Forest (Unsupervised)", results_iforest, is_iforest=True)
print("=" * 85)
print(f"Entire 5-Run Benchmark finished in {time.time() - t_grand0:.2f} seconds!")
