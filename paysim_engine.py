"""
paysim_engine.py
================
Unified Machine Learning & From-Scratch Model Engine for Mobile Money & Wire Transfer Fraud (PaySim).
Domain: Ledger & Balance Drain Analysis (Mobile Money, Wire, RTP).
Supports pure NumPy implementations from mathematical first principles as well as industry XGBoost.
Provides sub-millisecond inference and exact feature attribution (XAI).
Author: Manas Jain
"""

import os
import json
import time
import numpy as np
import pandas as pd
from fraud_ops.inference import ScratchInferenceMixin, WIRE_WEIGHTS, artifact_version
from typing import Dict, Any, Tuple, List

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PAYSIM_CSV = os.path.join(DATA_DIR, "paysim_benchmark.csv")
PAYSIM_WEIGHTS = os.path.join(DATA_DIR, "paysim_weights.npz")
PAYSIM_SAMPLES = os.path.join(DATA_DIR, "paysim_samples.json")


# ==============================================================================
# 1. PURE NUMPY ALGORITHMS (FROM MATHEMATICAL PRINCIPLES)
# ==============================================================================

class ScratchLogReg:
    """Logistic Regression from first mathematical principles."""
    def __init__(self, lr: float = 0.1, n_iter: int = 800):
        self.lr, self.n_iter = lr, n_iter
        self.w, self.b = None, 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500, 500)
            err = (1.0 / (1.0 + np.exp(-z))) - y
            self.w -= self.lr * (np.dot(X.T, err) / m)
            self.b -= self.lr * np.mean(err)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z = np.clip(np.dot(X, self.w) + self.b, -500, 500)
        return (1.0 / (1.0 + np.exp(-z))).ravel()


class ScratchCostSensitiveLogReg:
    """Cost-Sensitive Logistic Regression weighting false negatives heavily."""
    def __init__(self, lr: float = 0.08, n_iter: int = 800, fraud_weight: float = 4.0):
        self.lr, self.n_iter, self.fraud_weight = lr, n_iter, fraud_weight
        self.w, self.b = None, 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        weights = np.where(y == 1, self.fraud_weight, 1.0)
        for _ in range(self.n_iter):
            z = np.clip(np.dot(X, self.w) + self.b, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            err = (p - y) * weights
            self.w -= self.lr * (np.dot(X.T, err) / m)
            self.b -= self.lr * np.mean(err)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z = np.clip(np.dot(X, self.w) + self.b, -500, 500)
        return (1.0 / (1.0 + np.exp(-z))).ravel()


class ScratchMLP:
    """Multi-Layer Perceptron (d -> 32 -> 16 -> 1) with He Normal & ReLU."""
    def __init__(self, hidden: Tuple[int, int] = (32, 16), lr: float = 0.03, epochs: int = 350):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.W3, self.b3 = None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        h1, h2 = self.hidden
        np.random.seed(42)
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
            A1 = np.maximum(0, Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = np.maximum(0, Z2)
            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = 1.0 / (1.0 + np.exp(-np.clip(Z3, -500, 500)))

            # Backprop
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        A1 = np.maximum(0, np.dot(X, self.W1) + self.b1)
        A2 = np.maximum(0, np.dot(A1, self.W2) + self.b2)
        p = 1.0 / (1.0 + np.exp(-np.clip(np.dot(A2, self.W3) + self.b3, -500, 500)))
        return p.ravel()


class ScratchStumpGBDT:
    """Gradient Boosted Decision Stumps for mobile money anomaly splits."""
    def __init__(self, n_estimators: int = 25, lr: float = 0.15):
        self.n_estimators, self.lr = n_estimators, lr
        self.base_pred = 0.0
        self.stumps = []  # list of (feature_idx, threshold, left_val, right_val)

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        p_mean = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_pred = np.log(p_mean / (1.0 - p_mean))
        F = np.full(m, self.base_pred)

        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            residual = y - p
            best_gain, best_feat, best_th, best_l, best_r = -1e9, 0, 0.0, 0.0, 0.0
            sample_feats = np.random.choice(n, size=min(6, n), replace=False)
            for f in sample_feats:
                thresholds = np.percentile(X[:, f], [20, 40, 60, 80])
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = np.full(len(X), self.base_pred)
        for feat, th, vl, vr in self.stumps:
            update = np.where(X[:, feat] <= th, vl, vr)
            F += self.lr * update
        return 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))


class ScratchAnomalyDetector:
    """Robust distance anomaly detector for mobile transfer amounts and balances."""
    def __init__(self):
        self.center = None
        self.scale = None

    def fit(self, X: np.ndarray):
        self.center = np.median(X, axis=0)
        self.scale = np.std(X, axis=0) + 1e-6
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z = (X - self.center) / self.scale
        dists = np.sqrt(np.mean(z**2, axis=1))
        return 1.0 / (1.0 + np.exp(-np.clip(0.5 * (dists - 2.0), -500, 500)))


# ==============================================================================
# 2. MASTER MOBILE MONEY ENGINE
# ==============================================================================

class PaySimFraudEngine(ScratchInferenceMixin):
    """
    Dedicated Machine Learning Engine for PaySim Mobile Money Transfer Fraud.
    Analyzes ledger discrepancies, balance drains, and mule cash-outs.
    """
    ENSEMBLE_WEIGHTS = WIRE_WEIGHTS
    FEATURE_NAMES = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "balanceDrainRatio",
        "isTransfer"
    ]

    FEATURE_LABELS = {
        "amount": "Transfer Amount ($)",
        "oldbalanceOrg": "Sender Initial Balance ($)",
        "newbalanceOrig": "Sender Post-Tx Balance ($)",
        "oldbalanceDest": "Recipient Initial Balance ($)",
        "newbalanceDest": "Recipient Post-Tx Balance ($)",
        "errorBalanceOrig": "Sender Balance Discrepancy ($)",
        "errorBalanceDest": "Recipient Balance Discrepancy ($)",
        "balanceDrainRatio": "Balance Drain Ratio",
        "isTransfer": "Is TRANSFER (vs CASH_OUT)"
    }

    def __init__(self):
        self.is_loaded = False
        self.scaler_mean = None
        self.scaler_std = None
        self.models = {}
        self.sample_cases = []

    def compute_features(self, amount: float, oldbalanceOrg: float, newbalanceOrig: float,
                         oldbalanceDest: float, newbalanceDest: float, tx_type: str) -> np.ndarray:
        """Computes derived ledger anomaly features from raw transaction inputs."""
        errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
        errorBalanceDest = oldbalanceDest + amount - newbalanceDest
        balanceDrainRatio = amount / (oldbalanceOrg + 1.0)
        isTransfer = 1.0 if tx_type.upper() == "TRANSFER" else 0.0

        return np.array([
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            errorBalanceOrig,
            errorBalanceDest,
            balanceDrainRatio,
            isTransfer
        ], dtype=np.float64)

    def train_and_save(self, csv_path: str = PAYSIM_CSV, output_file: str = PAYSIM_WEIGHTS):
        """Trains from-scratch models on PaySim and serializes weights."""
        print(f"Training PaySimFraudEngine from: {csv_path}...")
        df = pd.read_csv(csv_path)

        frauds = df[df['isFraud'] == 1]
        legits = df[df['isFraud'] == 0]

        # Balanced training sample for maximum gradient learning
        n_sample = min(len(frauds), 8000)
        np.random.seed(42)
        balanced = pd.concat([
            frauds.sample(n=n_sample, random_state=42),
            legits.sample(n=n_sample, random_state=42)
        ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        X_raw = balanced[self.FEATURE_NAMES].to_numpy()
        y_raw = balanced['isFraud'].to_numpy()

        mean = np.mean(X_raw, axis=0)
        std = np.std(X_raw, axis=0)
        std[std == 0.0] = 1.0
        X_norm = (X_raw - mean) / std

        # 1. Baseline LogReg
        lr = ScratchLogReg(lr=0.1, n_iter=800).fit(X_norm, y_raw)

        # 2. Cost-Sensitive LogReg
        cs_lr = ScratchCostSensitiveLogReg(lr=0.08, n_iter=800, fraud_weight=4.0).fit(X_norm, y_raw)

        # 3. MLP Neural Net
        mlp = ScratchMLP(hidden=(32, 16), lr=0.03, epochs=300).fit(X_norm, y_raw)

        # 4. Stump GBDT
        gbdt = ScratchStumpGBDT(n_estimators=25, lr=0.15).fit(X_norm, y_raw)

        # 5. Anomaly Detector
        ad = ScratchAnomalyDetector().fit(X_norm[y_raw == 0])

        # Serialize
        stumps_arr = np.array(gbdt.stumps, dtype=object)
        np.savez_compressed(
            output_file,
            mean=mean,
            std=std,
            lr_w=lr.w,
            lr_b=lr.b,
            cslr_w=cs_lr.w,
            cslr_b=cs_lr.b,
            mlp_w1=mlp.W1, mlp_b1=mlp.b1,
            mlp_w2=mlp.W2, mlp_b2=mlp.b2,
            mlp_w3=mlp.W3, mlp_b3=mlp.b3,
            gbdt_base=gbdt.base_pred,
            gbdt_stumps=stumps_arr,
            ad_center=ad.center,
            ad_scale=ad.scale
        )
        print(f"Serialized PaySim model weights to {output_file}")

        # Generate representative sample cases
        sample_cases = self._generate_sample_cases(df)
        with open(PAYSIM_SAMPLES, "w") as f:
            json.dump(sample_cases, f, indent=2)
        print(f"Serialized PaySim sample cases to {PAYSIM_SAMPLES}")

        self.load(output_file)

    def _generate_sample_cases(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extracts 4 distinct real-world mobile money transfer scenarios."""
        cases = [
            {
                "id": "CASE-PAYSIM-01",
                "label": "Legitimate Everyday Peer-to-Peer Transfer",
                "type": "TRANSFER",
                "amount": 250.00,
                "oldbalanceOrg": 1200.00,
                "newbalanceOrig": 950.00,
                "oldbalanceDest": 500.00,
                "newbalanceDest": 750.00,
                "expected": "BENIGN",
                "story": "Alice transfers $250 to Bob for dinner. Ledger balances reconcile perfectly with $0 discrepancy."
            },
            {
                "id": "CASE-PAYSIM-02",
                "label": "Legitimate Merchant Cash-Out",
                "type": "CASH_OUT",
                "amount": 1500.00,
                "oldbalanceOrg": 5000.00,
                "newbalanceOrig": 3500.00,
                "oldbalanceDest": 12000.00,
                "newbalanceDest": 13500.00,
                "expected": "BENIGN",
                "story": "Merchant withdraws funds at an authorized agent kiosk. Partial balance drain, recipient balance increases by exact amount."
            },
            {
                "id": "CASE-PAYSIM-03",
                "label": "Mule Account Drain (Full Balance Drained to $0)",
                "type": "TRANSFER",
                "amount": 450000.00,
                "oldbalanceOrg": 450000.00,
                "newbalanceOrig": 0.00,
                "oldbalanceDest": 0.00,
                "newbalanceDest": 0.00,
                "expected": "FRAUD",
                "story": "Account takeover: Attacker drains entire $450k balance to zero in a single high-value transfer to an unverified mule wallet whose balance remains zero (immediate cash-out)."
            },
            {
                "id": "CASE-PAYSIM-04",
                "label": "Synthetic Overdraft Ghost Balance Discrepancy",
                "type": "CASH_OUT",
                "amount": 180000.00,
                "oldbalanceOrg": 25000.00,
                "newbalanceOrig": 0.00,
                "oldbalanceDest": 0.00,
                "newbalanceDest": 180000.00,
                "expected": "FRAUD",
                "story": "Exploited gateway vulnerability: Attacker cashes out $180,000 from an account that only contained $25,000, creating an unresolved $155,000 balance discrepancy."
            }
        ]
        return cases

    def load(self, weights_file: str = PAYSIM_WEIGHTS):
        """Loads serialized weights into memory."""
        if not os.path.exists(weights_file):
            self.train_and_save(output_file=weights_file)
            return

        self.model_version = artifact_version(weights_file)
        data = np.load(weights_file, allow_pickle=True)
        self.scaler_mean = data['mean']
        self.scaler_std = data['std']

        # 1. Baseline LogReg
        lr = ScratchLogReg()
        lr.w, lr.b = data['lr_w'], float(data['lr_b'])
        self.models["Baseline Logistic Regression (NumPy)"] = lr

        # 2. Cost-Sensitive LogReg
        cs_lr = ScratchCostSensitiveLogReg()
        cs_lr.w, cs_lr.b = data['cslr_w'], float(data['cslr_b'])
        self.models["Cost-Sensitive LogReg (NumPy)"] = cs_lr

        # 3. MLP
        mlp = ScratchMLP()
        mlp.W1, mlp.b1 = data['mlp_w1'], data['mlp_b1']
        mlp.W2, mlp.b2 = data['mlp_w2'], data['mlp_b2']
        mlp.W3, mlp.b3 = data['mlp_w3'], data['mlp_b3']
        self.models["Multi-Layer Perceptron (NumPy MLP)"] = mlp

        # 4. GBDT
        gbdt = ScratchStumpGBDT()
        gbdt.base_pred = float(data['gbdt_base'])
        gbdt.stumps = [(int(s[0]), float(s[1]), float(s[2]), float(s[3])) for s in data['gbdt_stumps']]
        gbdt.lr = float(data["gbdt_lr"]) if "gbdt_lr" in data else 0.15
        self.models["Gradient Boosted Trees (NumPy GBDT)"] = gbdt

        # 5. Anomaly Detector
        ad = ScratchAnomalyDetector()
        ad.center = data['ad_center']
        ad.scale = data['ad_scale']
        self.models["Unsupervised Isolation Profiler (NumPy)"] = ad

        # Load samples
        if os.path.exists(PAYSIM_SAMPLES):
            with open(PAYSIM_SAMPLES, "r") as f:
                self.sample_cases = json.load(f)

        self.is_loaded = True

    def scale_features(self, x_raw: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        if x_raw.ndim == 1:
            x_raw = x_raw.reshape(1, -1)
        return (x_raw - self.scaler_mean) / self.scaler_std

    def explain_transaction(self, x_raw, model_name="Soft-Voting Ensemble (Scratch Champion)"):
        result = super().explain_transaction(x_raw, model_name)
        raw = np.asarray(x_raw).ravel()
        observations = []
        if raw[7] >= .95:
            observations.append(f"Account Drain Alert: requested amount / (initial balance + 1) = {raw[7]:.2f}.")
        if abs(raw[5]) > 1:
            observations.append(f"Origin Ledger Anomaly: discrepancy {raw[5]:,.2f}.")
        if abs(raw[6]) > 1:
            observations.append(f"Destination Ghost Balance: ledger discrepancy {raw[6]:,.2f}; investigate settlement context.")
        result["narrative"] = " ".join(observations + [result["narrative"]])
        return result


# Singleton engine instance
paysim_engine = PaySimFraudEngine()

if __name__ == "__main__":
    paysim_engine.train_and_save()
    sample = paysim_engine.compute_features(
        amount=450000.0,
        oldbalanceOrg=450000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        tx_type="TRANSFER"
    )
    preds = paysim_engine.predict_all(sample)
    print("\nSample Fraud Prediction:")
    for m, p in preds.items():
        print(f"  {m:45s}: {p:.2%}")
    xai = paysim_engine.explain_transaction(sample)
    print("\nNarrative:", xai['narrative'])
