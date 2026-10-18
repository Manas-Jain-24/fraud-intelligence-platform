"""
paysim_simulation_worker.py
===========================
Worker logic for PaySim Mobile Money Multi-Run Monte Carlo Simulation.
Evaluates From-Scratch pure NumPy models and XGBoost across randomized train/test splits.
Computes financial loss, Dollar Recall, Value-at-Risk (VaR), and Conditional Value-at-Risk (CVaR).
Author: Manas Jain
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


class WorkerLogReg:
    def __init__(self, lr=0.1, n_iter=300):
        self.lr, self.n_iter = lr, n_iter
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        m, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
            err = (1.0 / (1.0 + np.exp(-z))) - y
            self.w -= self.lr * (np.dot(X.T, err) / m)
            self.b -= self.lr * np.mean(err)
        return self

    def predict_proba(self, X):
        z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
        return (1.0 / (1.0 + np.exp(-z))).ravel()


class WorkerCSLogReg:
    def __init__(self, lr=0.08, n_iter=300, fraud_weight=4.0):
        self.lr, self.n_iter, self.fraud_weight = lr, n_iter, fraud_weight
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        m, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0
        weights = np.where(y == 1, self.fraud_weight, 1.0)
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = (p - y) * weights
            self.w -= self.lr * (np.dot(X.T, err) / m)
            self.b -= self.lr * np.mean(err)
        return self

    def predict_proba(self, X):
        z = np.clip(np.dot(X, self.w) + self.b, -500.0, 500.0)
        return (1.0 / (1.0 + np.exp(-z))).ravel()


class WorkerMLP:
    def __init__(self, hidden=(32, 16), lr=0.03, epochs=150, seed=42):
        self.hidden, self.lr, self.epochs, self.seed = hidden, lr, epochs, seed

    def fit(self, X, y):
        np.random.seed(self.seed)
        m, d = X.shape
        h1, h2 = self.hidden
        W1 = np.random.randn(d, h1) * np.sqrt(2.0 / d)
        b1 = np.zeros((1, h1))
        W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        b2 = np.zeros((1, h2))
        W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        b3 = np.zeros((1, 1))

        y_col = y.reshape(-1, 1)
        for _ in range(self.epochs):
            Z1 = np.dot(X, W1) + b1
            A1 = np.maximum(0.0, Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = np.maximum(0.0, Z2)
            Z3 = np.dot(A2, W3) + b3
            A3 = 1.0 / (1.0 + np.exp(-np.clip(Z3, -500.0, 500.0)))

            dZ3 = A3 - y_col
            dW3 = np.dot(A2.T, dZ3) / m
            db3 = np.mean(dZ3, axis=0, keepdims=True)

            dZ2 = np.dot(dZ3, W3.T) * (Z2 > 0)
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.mean(dZ2, axis=0, keepdims=True)

            dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
            dW1 = np.dot(X.T, dZ1) / m
            db1 = np.mean(dZ1, axis=0, keepdims=True)

            W3 -= self.lr * dW3
            b3 -= self.lr * db3
            W2 -= self.lr * dW2
            b2 -= self.lr * db2
            W1 -= self.lr * dW1
            b1 -= self.lr * db1

        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3
        return self

    def predict_proba(self, X):
        A1 = np.maximum(0.0, np.dot(X, self.W1) + self.b1)
        A2 = np.maximum(0.0, np.dot(A1, self.W2) + self.b2)
        p = 1.0 / (1.0 + np.exp(-np.clip(np.dot(A2, self.W3) + self.b3, -500.0, 500.0)))
        return p.ravel()


class WorkerStumpGBDT:
    def __init__(self, n_estimators=20, lr=0.15, seed=42):
        self.n_estimators, self.lr, self.seed = n_estimators, lr, seed
        self.base_pred = 0.0
        self.stumps = []

    def fit(self, X, y):
        np.random.seed(self.seed)
        m, d = X.shape
        p_mean = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_pred = np.log(p_mean / (1.0 - p_mean))
        F = np.full(m, self.base_pred)

        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500.0, 500.0)))
            residual = y - p
            best_gain, best_feat, best_th, best_l, best_r = -1e9, 0, 0.0, 0.0, 0.0
            sample_feats = np.random.choice(d, size=min(5, d), replace=False)
            for f in sample_feats:
                thresholds = np.percentile(X[:, f], [25, 50, 75])
                for th in thresholds:
                    lm = X[:, f] <= th
                    rm = ~lm
                    if np.sum(lm) < 10 or np.sum(rm) < 10:
                        continue
                    val_l = np.mean(residual[lm])
                    val_r = np.mean(residual[rm])
                    gain = np.sum(residual[lm])**2 / (np.sum(lm) + 1.0) + np.sum(residual[rm])**2 / (np.sum(rm) + 1.0)
                    if gain > best_gain:
                        best_gain, best_feat, best_th, best_l, best_r = gain, f, th, val_l, val_r

            self.stumps.append((best_feat, best_th, best_l, best_r))
            pred_update = np.where(X[:, best_feat] <= best_th, best_l, best_r)
            F += self.lr * pred_update
        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.base_pred)
        for feat, th, vl, vr in self.stumps:
            update = np.where(X[:, feat] <= th, vl, vr)
            F += self.lr * update
        return 1.0 / (1.0 + np.exp(-np.clip(F, -500.0, 500.0)))


def evaluate_run_metrics(y_true, amounts, y_prob, threshold=0.5, c_fp=5.0):
    """
    Computes rigorous classification & financial metrics.
    c_fp = $5.00 manual analyst verification cost per false alarm.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Basic classification
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    # Financial Dollar Recall & Losses
    fraud_amounts = amounts[y_true == 1]
    total_fraud_dollars = np.sum(fraud_amounts)

    caught_fraud_dollars = np.sum(amounts[(y_pred == 1) & (y_true == 1)])
    missed_fraud_dollars = np.sum(amounts[(y_pred == 0) & (y_true == 1)])

    dollar_rec = caught_fraud_dollars / total_fraud_dollars if total_fraud_dollars > 0 else 0.0

    # Net Operational Financial Loss = Missed Fraud Dollars + (FP * Verification Cost)
    net_loss = missed_fraud_dollars + (fp * c_fp)

    return {
        "Recall": float(rec),
        "Dollar_Recall": float(dollar_rec),
        "Precision": float(prec),
        "F1": float(f1),
        "ROC_AUC": float(auc),
        "Dollars_Protected": float(caught_fraud_dollars),
        "Dollars_Lost": float(missed_fraud_dollars),
        "Net_Loss": float(net_loss),
        "FP": int(fp),
        "FN": int(fn)
    }


def run_single_paysim_simulation(args):
    """
    Simulates one independent Monte Carlo run on PaySim:
    - Independent stratified 80/20 train/test split
    - Trains Scratch LogReg, Cost-Sensitive LogReg, MLP, GBDT, Soft-Voting Ensemble, and XGBoost
    - Evaluates out-of-sample metrics & financial losses
    """
    seed, X_all, y_all, amounts_all = args
    np.random.seed(seed)

    # Stratified split
    idx_0 = np.where(y_all == 0)[0]
    idx_1 = np.where(y_all == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    n_test_0 = int(len(idx_0) * 0.20)
    n_test_1 = int(len(idx_1) * 0.20)

    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    amt_tr, amt_te = amounts_all[train_idx], amounts_all[test_idx]

    # Balanced training sample for fast gradient learning
    frauds_tr = np.where(y_tr == 1)[0]
    legits_tr = np.where(y_tr == 0)[0]
    sample_legits_tr = np.random.choice(legits_tr, size=len(frauds_tr), replace=False)
    bal_idx = np.concatenate([frauds_tr, sample_legits_tr])
    np.random.shuffle(bal_idx)

    X_bal = X_tr[bal_idx]
    y_bal = y_tr[bal_idx]

    # Standardize strictly on training stats
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std[std == 0.0] = 1.0
    X_bal_norm = (X_bal - mean) / std
    X_te_norm = (X_te - mean) / std

    # Fit models
    # 1. Baseline LogReg
    m_lr = WorkerLogReg(lr=0.1, n_iter=250).fit(X_bal_norm, y_bal)
    p_lr = m_lr.predict_proba(X_te_norm)

    # 2. Cost-Sensitive LogReg
    m_cslr = WorkerCSLogReg(lr=0.08, n_iter=250, fraud_weight=4.0).fit(X_bal_norm, y_bal)
    p_cslr = m_cslr.predict_proba(X_te_norm)

    # 3. Scratch MLP
    m_mlp = WorkerMLP(hidden=(32, 16), lr=0.03, epochs=150, seed=seed).fit(X_bal_norm, y_bal)
    p_mlp = m_mlp.predict_proba(X_te_norm)

    # 4. Scratch Stump GBDT
    m_gbdt = WorkerStumpGBDT(n_estimators=20, lr=0.15, seed=seed).fit(X_bal_norm, y_bal)
    p_gbdt = m_gbdt.predict_proba(X_te_norm)

    # 5. Soft-Voting Ensemble (Champion)
    p_ens = 0.35 * p_mlp + 0.35 * p_gbdt + 0.30 * p_cslr

    # 6. Industry XGBoost
    xgb = XGBClassifier(
        n_estimators=35,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=1
    )
    xgb.fit(X_tr, y_tr)
    p_xgb = xgb.predict_proba(X_te)[:, 1]

    models_dict = {
        "1. Scratch Baseline LogReg": p_lr,
        "2. Scratch Cost-Sensitive LogReg": p_cslr,
        "3. Scratch Neural Net (MLP)": p_mlp,
        "4. Scratch Stump GBDT": p_gbdt,
        "5. Scratch Soft-Voting Ensemble": p_ens,
        "6. Industry XGBoost": p_xgb
    }

    run_output = {"Seed": seed}
    for m_name, p_vals in models_dict.items():
        metrics = evaluate_run_metrics(y_te, amt_te, p_vals)
        for k_met, v_met in metrics.items():
            run_output[f"{m_name}__{k_met}"] = v_met

    return run_output
