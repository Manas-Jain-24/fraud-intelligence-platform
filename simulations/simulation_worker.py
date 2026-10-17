import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Preload data at module level for worker processes if shared or load inside
def trapezoid_scratch(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))

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

class FastLogReg:
    def __init__(self, lr=0.1, n_iter=450, l2_reg=0.01):
        self.lr, self.n_iter, self.l2_reg = lr, n_iter, l2_reg
    def fit(self, X, y):
        m, d = X.shape
        self.w, self.b = np.zeros(d, dtype=np.float64), 0.0
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
            err = 1.0 / (1.0 + np.exp(-z)) - y
            self.w -= self.lr * ((1.0 / m) * np.dot(X.T, err) + (self.l2_reg / m) * self.w)
            self.b -= self.lr * ((1.0 / m) * np.sum(err))
        return self
    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)))

class FastCSLogReg:
    def __init__(self, lr=0.1, n_iter=450, l2_reg=0.01, kappa=1.5):
        self.lr, self.n_iter, self.l2_reg, self.kappa = lr, n_iter, l2_reg, kappa
    def fit(self, X, y, amounts):
        m, d = X.shape
        self.w, self.b = np.zeros(d, dtype=np.float64), 0.0
        c = np.ones_like(y, dtype=np.float64)
        c[y == 1] += self.kappa * np.log1p(amounts[y == 1])
        c_sum = np.sum(c)
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
            err = c * (1.0 / (1.0 + np.exp(-z)) - y)
            self.w -= self.lr * ((1.0 / c_sum) * np.dot(X.T, err) + (self.l2_reg / m) * self.w)
            self.b -= self.lr * ((1.0 / c_sum) * np.sum(err))
        return self
    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)))

class FastNeuralNet:
    def __init__(self, hidden=(32, 16), lr=0.05, epochs=150, momentum=0.9, seed=42):
        self.hidden, self.lr, self.epochs, self.momentum, self.seed = hidden, lr, epochs, momentum, seed
    def fit(self, X, y):
        np.random.seed(self.seed)
        m, d = X.shape
        h1, h2 = self.hidden
        W1 = np.random.randn(d, h1) * np.sqrt(2.0 / d); b1 = np.zeros((1, h1))
        W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1); b2 = np.zeros((1, h2))
        W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2); b3 = np.zeros((1, 1))
        vW1, vb1, vW2, vb2, vW3, vb3 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)
        y_mat = y.reshape(-1, 1)
        for _ in range(self.epochs):
            Z1 = np.dot(X, W1) + b1; A1 = np.maximum(0.0, Z1)
            Z2 = np.dot(A1, W2) + b2; A2 = np.maximum(0.0, Z2)
            Z3 = np.dot(A2, W3) + b3; A3 = 1.0 / (1.0 + np.exp(-np.clip(Z3, -500.0, 500.0)))
            dZ3 = A3 - y_mat; dW3 = np.dot(A2.T, dZ3) / m; db3 = np.sum(dZ3, axis=0, keepdims=True) / m
            dA2 = np.dot(dZ3, W3.T); dZ2 = dA2 * (Z2 > 0); dW2 = np.dot(A1.T, dZ2) / m; db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            dA1 = np.dot(dZ2, W2.T); dZ1 = dA1 * (Z1 > 0); dW1 = np.dot(X.T, dZ1) / m; db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            vW3 = self.momentum*vW3 + self.lr*dW3; vb3 = self.momentum*vb3 + self.lr*db3
            vW2 = self.momentum*vW2 + self.lr*dW2; vb2 = self.momentum*vb2 + self.lr*db2
            vW1 = self.momentum*vW1 + self.lr*dW1; vb1 = self.momentum*vb1 + self.lr*db1
            W3 -= vW3; b3 -= vb3; W2 -= vW2; b2 -= vb2; W1 -= vW1; b1 -= vb1
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = W1, b1, W2, b2, W3, b3
        return self
    def predict_proba(self, X):
        A1 = np.maximum(0.0, np.dot(X, self.W1) + self.b1)
        A2 = np.maximum(0.0, np.dot(A1, self.W2) + self.b2)
        return (1.0 / (1.0 + np.exp(-np.clip(np.dot(A2, self.W3) + self.b3, -500.0, 500.0)))).flatten()

class FastTreeRegressor:
    def __init__(self, depth=0, max_depth=3):
        self.depth, self.max_depth, self.feature, self.threshold, self.val = depth, max_depth, None, None, None
        self.left, self.right = None, None
    def fit(self, X, r, p):
        if self.depth >= self.max_depth or len(r) <= 10:
            self.val = np.sum(r) / np.sum(p * (1.0 - p) + 1e-10); return self
        m, d = X.shape; best_feat, best_th, min_sse = None, None, np.inf
        current_sse = np.sum((r - np.mean(r)) ** 2)
        for f in np.random.choice(d, size=min(6, d), replace=False):
            for th in np.percentile(X[:, f], [25, 50, 75]):
                left_mask = X[:, f] <= th
                if np.sum(left_mask) < 5 or np.sum(~left_mask) < 5: continue
                sse = np.sum((r[left_mask] - np.mean(r[left_mask]))**2) + np.sum((r[~left_mask] - np.mean(r[~left_mask]))**2)
                if sse < min_sse: min_sse, best_feat, best_th = sse, f, th
        if best_feat is None or (current_sse - min_sse) < 1e-4:
            self.val = np.sum(r) / np.sum(p * (1.0 - p) + 1e-10); return self
        self.feature, self.threshold = best_feat, best_th
        mask = X[:, best_feat] <= best_th
        self.left = FastTreeRegressor(self.depth+1, self.max_depth).fit(X[mask], r[mask], p[mask])
        self.right = FastTreeRegressor(self.depth+1, self.max_depth).fit(X[~mask], r[~mask], p[~mask])
        return self
    def predict(self, X):
        if self.val is not None: return np.full(len(X), self.val)
        mask = X[:, self.feature] <= self.threshold
        out = np.zeros(len(X))
        out[mask] = self.left.predict(X[mask]); out[~mask] = self.right.predict(X[~mask])
        return out

class FastGBDT:
    def __init__(self, n_trees=18, lr=0.15, max_depth=3, seed=42):
        self.n_trees, self.lr, self.max_depth, self.seed, self.trees, self.f0 = n_trees, lr, max_depth, seed, [], 0.0
    def fit(self, X, y):
        np.random.seed(self.seed)
        p_pos = np.clip(np.mean(y), 1e-5, 1.0 - 1e-5); self.f0 = np.log(p_pos / (1.0 - p_pos))
        F = np.full(len(X), self.f0); self.trees = []
        for _ in range(self.n_trees):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500.0, 500.0)))
            tree = FastTreeRegressor(max_depth=self.max_depth).fit(X, y - p, p)
            F += self.lr * tree.predict(X); self.trees.append(tree)
        return self
    def predict_proba(self, X):
        F = np.full(len(X), self.f0)
        for tree in self.trees: F += self.lr * tree.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(F, -500.0, 500.0)))

class FastIsolationTreeNode:
    def __init__(self, depth=0, max_depth=8):
        self.depth, self.max_depth, self.split_feat, self.split_val, self.left, self.right, self.size = depth, max_depth, None, None, None, None, 0
    def fit(self, X):
        n, d = X.shape; self.size = n
        if self.depth >= self.max_depth or n <= 1: return self
        self.split_feat = np.random.randint(0, d); col = X[:, self.split_feat]; mi, ma = np.min(col), np.max(col)
        if mi == ma: return self
        self.split_val = np.random.uniform(mi, ma); left_mask = col < self.split_val
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0: return self
        self.left = FastIsolationTreeNode(self.depth+1, self.max_depth).fit(X[left_mask])
        self.right = FastIsolationTreeNode(self.depth+1, self.max_depth).fit(X[~left_mask])
        return self
    def path_length(self, X):
        if self.split_feat is None or self.left is None or self.right is None or len(X) == 0: return np.full(len(X), self.depth + self._c(self.size))
        mask = X[:, self.split_feat] < self.split_val; res = np.zeros(len(X))
        res[mask] = self.left.path_length(X[mask]); res[~mask] = self.right.path_length(X[~mask])
        return res
    def _c(self, n):
        if n <= 1: return 0.0
        if n == 2: return 1.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)

class FastIsolationForest:
    def __init__(self, n_trees=30, max_samples=256, seed=42):
        self.n_trees, self.max_samples, self.seed, self.trees = n_trees, max_samples, seed, []
    def fit(self, X):
        np.random.seed(self.seed); n = len(X); sub_size = min(self.max_samples, n); max_depth = int(np.ceil(np.log2(max(sub_size, 2))))
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(n, size=sub_size, replace=False)
            self.trees.append(FastIsolationTreeNode(max_depth=max_depth).fit(X[idx]))
        return self
    def predict_proba(self, X):
        paths = np.array([t.path_length(X) for t in self.trees])
        scores = 2.0 ** (-np.mean(paths, axis=0) / 7.0)
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)

def evaluate_metrics(y_true, amounts, prob, threshold=0.5):
    y_pred = (prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    tot_d = np.sum(amounts[y_true == 1])
    caught_d = np.sum(amounts[(y_true == 1) & (y_pred == 1)])
    missed_d = np.sum(amounts[(y_true == 1) & (y_pred == 0)])
    dollar_rec = caught_d / tot_d if tot_d > 0 else 0.0
    net_loss = missed_d + fp * 2.50
    
    # 31 threshold points for fast AUC
    th_s = np.linspace(0, 1, 31)
    tprs, fprs = [], []
    P, N = np.sum(y_true == 1), np.sum(y_true == 0)
    for th in th_s:
        yp = (prob >= th).astype(int)
        tprs.append(np.sum((y_true == 1) & (yp == 1)) / P)
        fprs.append(np.sum((y_true == 0) & (yp == 1)) / N)
    si = np.argsort(fprs)
    auc = trapezoid_scratch(np.array(tprs)[si], np.array(fprs)[si])
    
    return {
        'Precision': prec,
        'Recall': rec,
        'Dollar_Recall': dollar_rec,
        'F1': f1,
        'ROC_AUC': auc,
        'Dollars_Protected': caught_d,
        'Net_Loss': net_loss,
        'FP': int(fp),
        'FN': int(fn)
    }

# Single run worker function
def run_single_simulation(args):
    seed, X_raw, y_raw, amounts_raw = args
    np.random.seed(seed)
    
    # 1. Stratified 80/20 train/test split
    idx_0 = np.where(y_raw == 0)[0]
    idx_1 = np.where(y_raw == 1)[0]
    np.random.shuffle(idx_0); np.random.shuffle(idx_1)
    n_te_0 = int(len(idx_0) * 0.2); n_te_1 = int(len(idx_1) * 0.2)
    test_idx = np.concatenate([idx_0[:n_te_0], idx_1[:n_te_1]])
    train_idx = np.concatenate([idx_0[n_te_0:], idx_1[n_te_1:]])
    
    X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
    y_tr, y_te = y_raw[train_idx], y_raw[test_idx]
    amt_tr, amt_te = amounts_raw[train_idx], amounts_raw[test_idx]
    
    # 2. SMOTE & Balanced sampling
    X_min_tr = X_tr[y_tr == 1]
    amt_min_tr = amt_tr[y_tr == 1]
    n_syn = len(X_min_tr) * 3
    X_syn = smote_scratch(X_min_tr, n_synthetic=n_syn, k=5, seed=seed)
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
    
    # Standardize strictly using unaugmented training partition statistics
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std[std == 0.0] = 1.0
    X_tr_n = (X_bal - mean) / std
    X_te_n = (X_te - mean) / std
    
    # Fit models
    m1 = FastLogReg().fit(X_tr_n, y_bal)
    p1 = m1.predict_proba(X_te_n)
    
    m2 = FastCSLogReg(kappa=1.5).fit(X_tr_n, y_bal, amt_bal)
    p2 = m2.predict_proba(X_te_n)
    
    m3 = FastNeuralNet(seed=seed).fit(X_tr_n, y_bal)
    p3 = m3.predict_proba(X_te_n)
    
    m4 = FastGBDT(seed=seed).fit(X_tr_n, y_bal)
    p4 = m4.predict_proba(X_te_n)
    
    m5 = FastIsolationForest(seed=seed).fit(X_tr[y_tr == 0])
    p5 = m5.predict_proba(X_te)
    
    # Soft Voting
    p6 = 0.35 * p2 + 0.30 * p3 + 0.25 * p4 + 0.10 * p5
    
    # Meta Stacking approximation
    p7 = 0.45 * p2 + 0.40 * p3 + 0.15 * p4
    p7 = 1.0 / (1.0 + np.exp(-(3.0 * p7 - 1.5)))
    
    model_probs = {
        '1. Baseline Logistic Regression': p1,
        '2. Cost-Sensitive LogReg': p2,
        '3. Neural Network (MLP)': p3,
        '4. GBDT Classifier': p4,
        '5. Isolation Forest (Unsupervised)': p5,
        '6. Soft-Voting Ensemble': p6,
        '7. Stacked Meta-Learner': p7
    }
    
    run_output = {'Seed': seed}
    for m_name, p_val in model_probs.items():
        metrics = evaluate_metrics(y_te, amt_te, p_val)
        for k_met, v_met in metrics.items():
            run_output[f"{m_name}__{k_met}"] = v_met
            
    return run_output
