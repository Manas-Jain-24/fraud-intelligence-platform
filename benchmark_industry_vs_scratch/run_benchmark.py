import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn & XGBoost imports
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.tree import DecisionTreeClassifier as SklearnDecTree
from sklearn.ensemble import RandomForestClassifier as SklearnRandForest
from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.ensemble import IsolationForest as SklearnIsoForest
import xgboost as xgb

# Set random seed
SEED = 42
np.random.seed(SEED)

output_dir = "benchmark_industry_vs_scratch"
fig_dir = os.path.join(output_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

print("=" * 85)
print("     HEAD-TO-HEAD BENCHMARK: FROM-SCRATCH (NUMPY) VS. INDUSTRY LIBRARIES")
print("=" * 85)

# 1. Load Data
DATA_PATH = os.path.join("data", "creditcard.csv")
print(f"Loading official dataset: {DATA_PATH} ...")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} transactions in {time.time() - t0:.2f}s.")

# 2. Balanced Undersampling (50:50)
fraud_df = df[df['Class'] == 1]
legit_df = df[df['Class'] == 0]
legit_sample = legit_df.sample(n=len(fraud_df), random_state=SEED)
balanced = pd.concat([fraud_df, legit_sample], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

feature_cols = [c for c in balanced.columns if c not in ['Time', 'Class']]
X_raw = balanced[feature_cols].to_numpy()
y_raw = balanced['Class'].to_numpy()

# 80/20 train/test split
m = len(X_raw)
test_count = int(0.2 * m)
shuffled_idx = np.random.permutation(m)
train_idx, test_idx = shuffled_idx[test_count:], shuffled_idx[:test_count]

X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
y_train, y_test = y_raw[train_idx], y_raw[test_idx]

# Standard Scaler from scratch
mean = np.mean(X_train_raw, axis=0)
std = np.std(X_train_raw, axis=0)
std[std == 0.0] = 1.0

X_train = (X_train_raw - mean) / std
X_test = (X_test_raw - mean) / std

print(f"Dataset split: Train = {len(X_train)} samples, Test = {len(X_test)} samples.\n")

# Evaluation helper
def evaluate(y_true, y_pred, y_probs=None):
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = TP + FP + TN + FN
    acc = (TP + TN) / total if total > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    roc_auc = 0.0
    fprs_sorted, tprs_sorted = [], []
    if y_probs is not None:
        thresholds = np.linspace(0.0, 1.0, 100)
        tprs, fprs = [], []
        for t in thresholds:
            p = (y_probs >= t).astype(int)
            tp_t = np.sum((p == 1) & (y_true == 1))
            fp_t = np.sum((p == 1) & (y_true == 0))
            fn_t = np.sum((p == 0) & (y_true == 1))
            tn_t = np.sum((p == 0) & (y_true == 0))
            tprs.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
            fprs.append(fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0)
        s_idx = np.argsort(fprs)
        fprs_sorted = np.array(fprs)[s_idx]
        tprs_sorted = np.array(tprs)[s_idx]
        roc_auc = float(np.sum((tprs_sorted[:-1] + tprs_sorted[1:]) * 0.5 * np.diff(fprs_sorted)))
        
    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc_auc,
        'ROC_Curve': (fprs_sorted, tprs_sorted)
    }

results = []

# ==============================================================================
# 1. LOGISTIC REGRESSION (SCRATCH VS SKLEARN)
# ==============================================================================
print("--> [1/6] Benchmarking Logistic Regression...")
# Scratch
class ScratchLogReg:
    def __init__(self, lr=0.1, n_iter=1200):
        self.lr, self.n_iter = lr, n_iter
    def fit(self, X, y):
        m, n = X.shape
        self.w, self.b = np.zeros(n), 0.0
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500, 500)
            err = (1.0 / (1.0 + np.exp(-z))) - y
            self.w -= self.lr * (np.dot(X.T, err) / m)
            self.b -= self.lr * np.mean(err)
        return self
    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(np.dot(X, self.w) + self.b, -500, 500)))
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

t0 = time.time()
m_scratch_lr = ScratchLogReg().fit(X_train, y_train)
t_train_scratch_lr = time.time() - t0
t0 = time.time()
y_pred_s_lr = m_scratch_lr.predict(X_test)
t_inf_scratch_lr = (time.time() - t0) * 1000
res_s_lr = evaluate(y_test, y_pred_s_lr, m_scratch_lr.predict_proba(X_test))
res_s_lr.update({'Family': 'Logistic Regression', 'Variant': 'From-Scratch (NumPy)', 'Train_Time_s': t_train_scratch_lr, 'Infer_Time_ms': t_inf_scratch_lr})
results.append(res_s_lr)

# Sklearn
t0 = time.time()
m_sk_lr = SklearnLogReg(max_iter=1200, random_state=SEED).fit(X_train, y_train)
t_train_sk_lr = time.time() - t0
t0 = time.time()
y_pred_sk_lr = m_sk_lr.predict(X_test)
t_inf_sk_lr = (time.time() - t0) * 1000
res_sk_lr = evaluate(y_test, y_pred_sk_lr, m_sk_lr.predict_proba(X_test)[:, 1])
res_sk_lr.update({'Family': 'Logistic Regression', 'Variant': 'Scikit-Learn', 'Train_Time_s': t_train_sk_lr, 'Infer_Time_ms': t_inf_sk_lr})
results.append(res_sk_lr)

# ==============================================================================
# 2. DECISION TREE (SCRATCH VS SKLEARN)
# ==============================================================================
print("--> [2/6] Benchmarking Decision Tree...")
# Scratch
class TreeNode:
    def __init__(self, feat=None, th=None, left=None, right=None, val=None):
        self.feat, self.th, self.left, self.right, self.val = feat, th, left, right, val

class ScratchDecTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
    def _gini(self, y):
        p = np.mean(y) if len(y) > 0 else 0.0
        return 1.0 - (p**2 + (1.0 - p)**2)
    def _build(self, X, y, d=0):
        if d >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 4:
            return TreeNode(val=1 if np.mean(y) >= 0.5 else 0)
        best_g, best_f, best_t = -1, None, None
        cur_g = self._gini(y)
        m, n = X.shape
        for f in range(n):
            for t in np.percentile(X[:, f], np.linspace(15, 85, 5)):
                lm = X[:, f] <= t
                if np.sum(lm) == 0 or np.sum(~lm) == 0: continue
                gain = cur_g - (np.mean(lm) * self._gini(y[lm]) + np.mean(~lm) * self._gini(y[~lm]))
                if gain > best_g:
                    best_g, best_f, best_t = gain, f, t
        if best_f is None:
            return TreeNode(val=1 if np.mean(y) >= 0.5 else 0)
        lm = X[:, best_f] <= best_t
        return TreeNode(feat=best_f, th=best_t, left=self._build(X[lm], y[lm], d+1), right=self._build(X[~lm], y[~lm], d+1))
    def fit(self, X, y):
        self.root = self._build(X, y)
        return self
    def _pred_row(self, node, x):
        if node.val is not None: return node.val
        return self._pred_row(node.left if x[node.feat] <= node.th else node.right, x)
    def predict(self, X):
        return np.array([self._pred_row(self.root, r) for r in X])

t0 = time.time()
m_scratch_dt = ScratchDecTree(max_depth=5).fit(X_train, y_train)
t_train_scratch_dt = time.time() - t0
t0 = time.time()
y_pred_s_dt = m_scratch_dt.predict(X_test)
t_inf_scratch_dt = (time.time() - t0) * 1000
res_s_dt = evaluate(y_test, y_pred_s_dt)
res_s_dt.update({'Family': 'Decision Tree', 'Variant': 'From-Scratch (NumPy)', 'Train_Time_s': t_train_scratch_dt, 'Infer_Time_ms': t_inf_scratch_dt})
results.append(res_s_dt)

# Sklearn
t0 = time.time()
m_sk_dt = SklearnDecTree(max_depth=5, random_state=SEED).fit(X_train, y_train)
t_train_sk_dt = time.time() - t0
t0 = time.time()
y_pred_sk_dt = m_sk_dt.predict(X_test)
t_inf_sk_dt = (time.time() - t0) * 1000
res_sk_dt = evaluate(y_test, y_pred_sk_dt, m_sk_dt.predict_proba(X_test)[:, 1])
res_sk_dt.update({'Family': 'Decision Tree', 'Variant': 'Scikit-Learn', 'Train_Time_s': t_train_sk_dt, 'Infer_Time_ms': t_inf_sk_dt})
results.append(res_sk_dt)

# ==============================================================================
# 3. RANDOM FOREST (SCRATCH VS SKLEARN)
# ==============================================================================
print("--> [3/6] Benchmarking Random Forest...")
# Scratch
class ScratchRandForest:
    def __init__(self, n_trees=25, max_depth=5):
        self.n_trees, self.max_depth = n_trees, max_depth
    def fit(self, X, y):
        m, n = X.shape
        self.trees = []
        for _ in range(self.n_trees):
            b_idx = np.random.choice(m, size=m, replace=True)
            t = ScratchDecTree(max_depth=self.max_depth).fit(X[b_idx], y[b_idx])
            self.trees.append(t)
        return self
    def predict_proba(self, X):
        return np.mean([t.predict(X) for t in self.trees], axis=0)
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

t0 = time.time()
m_scratch_rf = ScratchRandForest(n_trees=25, max_depth=5).fit(X_train, y_train)
t_train_scratch_rf = time.time() - t0
t0 = time.time()
y_pred_s_rf = m_scratch_rf.predict(X_test)
t_inf_scratch_rf = (time.time() - t0) * 1000
res_s_rf = evaluate(y_test, y_pred_s_rf, m_scratch_rf.predict_proba(X_test))
res_s_rf.update({'Family': 'Random Forest', 'Variant': 'From-Scratch (NumPy)', 'Train_Time_s': t_train_scratch_rf, 'Infer_Time_ms': t_inf_scratch_rf})
results.append(res_s_rf)

# Sklearn
t0 = time.time()
m_sk_rf = SklearnRandForest(n_estimators=25, max_depth=5, random_state=SEED).fit(X_train, y_train)
t_train_sk_rf = time.time() - t0
t0 = time.time()
y_pred_sk_rf = m_sk_rf.predict(X_test)
t_inf_sk_rf = (time.time() - t0) * 1000
res_sk_rf = evaluate(y_test, y_pred_sk_rf, m_sk_rf.predict_proba(X_test)[:, 1])
res_sk_rf.update({'Family': 'Random Forest', 'Variant': 'Scikit-Learn', 'Train_Time_s': t_train_sk_rf, 'Infer_Time_ms': t_inf_sk_rf})
results.append(res_sk_rf)

# ==============================================================================
# 4. NEURAL NETWORK / MLP (SCRATCH VS SKLEARN)
# ==============================================================================
print("--> [4/6] Benchmarking Neural Network (MLP)...")
# Scratch 2-layer MLP (d -> 32 -> 16 -> 1)
class ScratchMLP:
    def __init__(self, hidden=(32, 16), lr=0.02, epochs=400):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
    def fit(self, X, y):
        m, n = X.shape
        h1, h2 = self.hidden
        # He Normal Init
        self.W1 = np.random.randn(n, h1) * np.sqrt(2.0 / n)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, 1))
        
        y_col = y.reshape(-1, 1)
        for _ in range(self.epochs):
            # Forward
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = np.maximum(0, Z1) # ReLU
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = np.maximum(0, Z2) # ReLU
            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = 1.0 / (1.0 + np.exp(-np.clip(Z3, -500, 500))) # Sigmoid
            
            # Backward
            dZ3 = A3 - y_col
            dW3 = np.dot(A2.T, dZ3) / m
            db3 = np.mean(dZ3, axis=0, keepdims=True)
            
            dZ2 = np.dot(dZ3, self.W3.T) * (Z2 > 0)
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.mean(dZ2, axis=0, keepdims=True)
            
            dZ1 = np.dot(dZ2, self.W2.T) * (Z1 > 0)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.mean(dZ1, axis=0, keepdims=True)
            
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self
    def predict_proba(self, X):
        A1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        A2 = np.maximum(0, np.dot(A1, self.W2) + self.b2)
        return (1.0 / (1.0 + np.exp(-np.clip(np.dot(A2, self.W3) + self.b3, -500, 500)))).ravel()
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

t0 = time.time()
m_scratch_mlp = ScratchMLP().fit(X_train, y_train)
t_train_scratch_mlp = time.time() - t0
t0 = time.time()
y_pred_s_mlp = m_scratch_mlp.predict(X_test)
t_inf_scratch_mlp = (time.time() - t0) * 1000
res_s_mlp = evaluate(y_test, y_pred_s_mlp, m_scratch_mlp.predict_proba(X_test))
res_s_mlp.update({'Family': 'Neural Network (MLP)', 'Variant': 'From-Scratch (NumPy)', 'Train_Time_s': t_train_scratch_mlp, 'Infer_Time_ms': t_inf_scratch_mlp})
results.append(res_s_mlp)

# Sklearn MLP
t0 = time.time()
m_sk_mlp = SklearnMLP(hidden_layer_sizes=(32, 16), max_iter=400, random_state=SEED).fit(X_train, y_train)
t_train_sk_mlp = time.time() - t0
t0 = time.time()
y_pred_sk_mlp = m_sk_mlp.predict(X_test)
t_inf_sk_mlp = (time.time() - t0) * 1000
res_sk_mlp = evaluate(y_test, y_pred_sk_mlp, m_sk_mlp.predict_proba(X_test)[:, 1])
res_sk_mlp.update({'Family': 'Neural Network (MLP)', 'Variant': 'Scikit-Learn', 'Train_Time_s': t_train_sk_mlp, 'Infer_Time_ms': t_inf_sk_mlp})
results.append(res_sk_mlp)

# ==============================================================================
# 5. GRADIENT BOOSTING (SCRATCH GBDT VS XGBOOST)
# ==============================================================================
print("--> [5/6] Benchmarking Gradient Boosting (Scratch GBDT vs. XGBoost)...")
# Scratch GBDT
class ScratchGBDT:
    def __init__(self, n_estimators=20, lr=0.1, max_depth=3):
        self.n_estimators, self.lr, self.max_depth = n_estimators, lr, max_depth
    def fit(self, X, y):
        m = len(y)
        self.base_pred = np.log(np.mean(y) / (1.0 - np.mean(y) + 1e-15))
        self.trees = []
        F = np.full(m, self.base_pred)
        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            residual = y - p
            tree = ScratchDecTree(max_depth=self.max_depth).fit(X, (residual > 0).astype(int))
            F += self.lr * (tree.predict(X) * 2 - 1)
            self.trees.append(tree)
        return self
    def predict_proba(self, X):
        F = np.full(len(X), self.base_pred)
        for t in self.trees:
            F += self.lr * (t.predict(X) * 2 - 1)
        return 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

t0 = time.time()
m_scratch_gbdt = ScratchGBDT().fit(X_train, y_train)
t_train_scratch_gbdt = time.time() - t0
t0 = time.time()
y_pred_s_gbdt = m_scratch_gbdt.predict(X_test)
t_inf_scratch_gbdt = (time.time() - t0) * 1000
res_s_gbdt = evaluate(y_test, y_pred_s_gbdt, m_scratch_gbdt.predict_proba(X_test))
res_s_gbdt.update({'Family': 'Gradient Boosting', 'Variant': 'From-Scratch (NumPy)', 'Train_Time_s': t_train_scratch_gbdt, 'Infer_Time_ms': t_inf_scratch_gbdt})
results.append(res_s_gbdt)

# XGBoost
t0 = time.time()
m_xgb = xgb.XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=SEED).fit(X_train, y_train)
t_train_xgb = time.time() - t0
t0 = time.time()
y_pred_xgb = m_xgb.predict(X_test)
t_inf_xgb = (time.time() - t0) * 1000
res_xgb = evaluate(y_test, y_pred_xgb, m_xgb.predict_proba(X_test)[:, 1])
res_xgb.update({'Family': 'Gradient Boosting', 'Variant': 'XGBoost (Industry)', 'Train_Time_s': t_train_xgb, 'Infer_Time_ms': t_inf_xgb})
results.append(res_xgb)

# ==============================================================================
# 6. ANOMALY DETECTION (SCRATCH ISOLATION FOREST VS SKLEARN)
# ==============================================================================
print("--> [6/6] Benchmarking Isolation Forest...")
# Scratch Isolation Forest
class ScratchIsoTree:
    def __init__(self, max_depth=8):
        self.max_depth = max_depth
    def _build(self, X, depth=0):
        m, n = X.shape
        if depth >= self.max_depth or m <= 1:
            return {'type': 'leaf', 'size': m}
        feat = np.random.randint(0, n)
        val_min, val_max = np.min(X[:, feat]), np.max(X[:, feat])
        if val_min == val_max:
            return {'type': 'leaf', 'size': m}
        split = np.random.uniform(val_min, val_max)
        lm = X[:, feat] < split
        return {
            'type': 'node', 'feat': feat, 'split': split,
            'left': self._build(X[lm], depth + 1),
            'right': self._build(X[~lm], depth + 1)
        }
    def fit(self, X):
        self.root = self._build(X)
        return self
    def _path_len(self, node, x, current_depth=0):
        if node['type'] == 'leaf':
            m = node['size']
            c_val = 2.0 * (np.log(m - 1) + 0.5772156649) - (2.0 * (m - 1) / m) if m > 2 else (1.0 if m == 2 else 0.0)
            return current_depth + c_val
        if x[node['feat']] < node['split']:
            return self._path_len(node['left'], x, current_depth + 1)
        return self._path_len(node['right'], x, current_depth + 1)

class ScratchIsoForest:
    def __init__(self, n_trees=50, max_depth=8):
        self.n_trees, self.max_depth = n_trees, max_depth
    def fit(self, X):
        self.trees = []
        for _ in range(self.n_trees):
            sub_idx = np.random.choice(len(X), size=min(256, len(X)), replace=False)
            t = ScratchIsoTree(max_depth=self.max_depth).fit(X[sub_idx])
            self.trees.append(t)
        return self
    def anomaly_scores(self, X):
        m = min(256, len(X))
        c_m = 2.0 * (np.log(m - 1) + 0.5772156649) - (2.0 * (m - 1) / m)
        avg_paths = np.mean([[t._path_len(t.root, row) for t in self.trees] for row in X], axis=1)
        return 2.0 ** (- (avg_paths / c_m))
    def predict(self, X, threshold=0.58):
        return (self.anomaly_scores(X) >= threshold).astype(int)

t0 = time.time()
m_scratch_iso = ScratchIsoForest().fit(X_train)
t_train_scratch_iso = time.time() - t0
t0 = time.time()
y_pred_s_iso = m_scratch_iso.predict(X_test)
t_inf_scratch_iso = (time.time() - t0) * 1000
scores_s_iso = m_scratch_iso.anomaly_scores(X_test)
res_s_iso = evaluate(y_test, y_pred_s_iso, scores_s_iso)
res_s_iso.update({'Family': 'Isolation Forest', 'Variant': 'From-Scratch (NumPy)', 'Train_Time_s': t_train_scratch_iso, 'Infer_Time_ms': t_inf_scratch_iso})
results.append(res_s_iso)

# Sklearn Isolation Forest
t0 = time.time()
m_sk_iso = SklearnIsoForest(n_estimators=50, random_state=SEED).fit(X_train)
t_train_sk_iso = time.time() - t0
t0 = time.time()
# Convert Sklearn -1/1 output to 1 (fraud) / 0 (legit)
y_pred_sk_iso = (m_sk_iso.predict(X_test) == -1).astype(int)
t_inf_sk_iso = (time.time() - t0) * 1000
sk_scores = - m_sk_iso.score_samples(X_test) # Higher score = more anomalous
res_sk_iso = evaluate(y_test, y_pred_sk_iso, (sk_scores - sk_scores.min()) / (sk_scores.max() - sk_scores.min()))
res_sk_iso.update({'Family': 'Isolation Forest', 'Variant': 'Scikit-Learn', 'Train_Time_s': t_train_sk_iso, 'Infer_Time_ms': t_inf_sk_iso})
results.append(res_sk_iso)

# ==============================================================================
# CONSOLIDATE RESULTS TABLE
# ==============================================================================
df_results = pd.DataFrame([
    {
        'Algorithm Family': r['Family'],
        'Implementation Variant': r['Variant'],
        'Accuracy': f"{r['Accuracy']*100:.2f}%",
        'Precision': f"{r['Precision']*100:.2f}%",
        'Recall': f"{r['Recall']*100:.2f}%",
        'F1-Score': f"{r['F1']*100:.2f}%",
        'ROC-AUC': f"{r['ROC-AUC']:.4f}",
        'Train Time (s)': f"{r['Train_Time_s']:.3f}",
        'Infer Time (ms)': f"{r['Infer_Time_ms']:.2f}"
    }
    for r in results
])

csv_path = os.path.join(output_dir, "benchmark_results.csv")
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 105)
print("                                 HEAD-TO-HEAD BENCHMARK SUMMARY")
print("=" * 105)
print(df_results.to_string(index=False))
print("=" * 105)
print(f"Results saved to: {csv_path}")

# ==============================================================================
# GENERATE PUBLICATION FIGURES
# ==============================================================================
print("\nGenerating comparison plots...")

# Figure 1: Side-by-side Metric Comparison (Scratch vs Library)
families = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network (MLP)', 'Gradient Boosting', 'Isolation Forest']
scratch_recalls = [r['Recall']*100 for r in results if 'From-Scratch' in r['Variant']]
library_recalls = [r['Recall']*100 for r in results if 'From-Scratch' not in r['Variant']]

scratch_aucs = [r['ROC-AUC'] for r in results if 'From-Scratch' in r['Variant']]
library_aucs = [r['ROC-AUC'] for r in results if 'From-Scratch' not in r['Variant']]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Recall Comparison
x = np.arange(len(families))
w = 0.35
axes[0].bar(x - w/2, scratch_recalls, width=w, label='From-Scratch (NumPy)', color='#ff7f0e', alpha=0.9)
axes[0].bar(x + w/2, library_recalls, width=w, label='Industry Library (Sklearn/XGBoost)', color='#1f77b4', alpha=0.9)
axes[0].set_ylabel('Recall (%) [Catch Rate]', fontsize=11, fontweight='bold')
axes[0].set_title('Event Recall Comparison: Scratch vs. Industry Libraries', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(families, rotation=25, ha='right', fontsize=9.5)
axes[0].set_ylim([0, 110])
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Plot 2: ROC-AUC Comparison
axes[1].bar(x - w/2, scratch_aucs, width=w, label='From-Scratch (NumPy)', color='#ff7f0e', alpha=0.9)
axes[1].bar(x + w/2, library_aucs, width=w, label='Industry Library (Sklearn/XGBoost)', color='#1f77b4', alpha=0.9)
axes[1].set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
axes[1].set_title('ROC-AUC Discriminative Power Comparison', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(families, rotation=25, ha='right', fontsize=9.5)
axes[1].set_ylim([0.8, 1.02])
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig1_path = os.path.join(fig_dir, "fig1_head_to_head_metrics.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: Training Speed Comparison (Log Scale)
scratch_times = [r['Train_Time_s'] for r in results if 'From-Scratch' in r['Variant']]
library_times = [r['Train_Time_s'] for r in results if 'From-Scratch' not in r['Variant']]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w/2, scratch_times, width=w, label='From-Scratch (NumPy)', color='#ff7f0e', alpha=0.9)
ax.bar(x + w/2, library_times, width=w, label='Industry Library (C++/Cython Optimizations)', color='#2ca02c', alpha=0.9)
ax.set_yscale('log')
ax.set_ylabel('Training Time in Seconds (Log Scale)', fontsize=11, fontweight='bold')
ax.set_title('Training Speed & Computational Efficiency Benchmark', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(families, rotation=25, ha='right', fontsize=9.5)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig2_path = os.path.join(fig_dir, "fig2_head_to_head_speed.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Overlaid ROC Curves for GBDT vs XGBoost & Scratch LogReg vs Sklearn
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# LogReg ROC
axes[0].plot(res_s_lr['ROC_Curve'][0], res_s_lr['ROC_Curve'][1], lw=2.5, color='#ff7f0e', label=f"Scratch LogReg (AUC = {res_s_lr['ROC-AUC']:.4f})")
axes[0].plot(res_sk_lr['ROC_Curve'][0], res_sk_lr['ROC_Curve'][1], lw=2.0, linestyle='--', color='#1f77b4', label=f"Sklearn LogReg (AUC = {res_sk_lr['ROC-AUC']:.4f})")
axes[0].plot([0, 1], [0, 1], 'k:', lw=1, label='Random Chance')
axes[0].set_title('ROC Curve: Scratch vs. Sklearn Logistic Regression', fontsize=11, fontweight='bold')
axes[0].set_xlabel('False Positive Rate', fontsize=10)
axes[0].set_ylabel('True Positive Rate', fontsize=10)
axes[0].legend(loc='lower right', fontsize=9.5)
axes[0].grid(True, linestyle='--', alpha=0.5)

# GBDT vs XGBoost ROC
axes[1].plot(res_s_gbdt['ROC_Curve'][0], res_s_gbdt['ROC_Curve'][1], lw=2.5, color='#d62728', label=f"Scratch GBDT (AUC = {res_s_gbdt['ROC-AUC']:.4f})")
axes[1].plot(res_xgb['ROC_Curve'][0], res_xgb['ROC_Curve'][1], lw=2.0, linestyle='--', color='#2ca02c', label=f"XGBoost (AUC = {res_xgb['ROC-AUC']:.4f})")
axes[1].plot([0, 1], [0, 1], 'k:', lw=1, label='Random Chance')
axes[1].set_title('ROC Curve: Scratch GBDT vs. XGBoost', fontsize=11, fontweight='bold')
axes[1].set_xlabel('False Positive Rate', fontsize=10)
axes[1].set_ylabel('True Positive Rate', fontsize=10)
axes[1].legend(loc='lower right', fontsize=9.5)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig3_path = os.path.join(fig_dir, "fig3_roc_curves_comparison.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Saved: {fig3_path}")
print("\nAll figures generated successfully!")
