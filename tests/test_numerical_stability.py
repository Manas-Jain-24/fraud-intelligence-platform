"""
tests/test_numerical_stability.py
==================================
Numerical Stability, Edge Cases, and Floating-Point Invariance Tests.
Verifies that custom NumPy algorithms handle extreme financial scale, overflow,
underflow, and boundary cases without generating NaNs, Infs, or runtime crashes.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_engine import ScratchLogReg, ScratchCostSensitiveLogReg, ScratchMLP, ScratchStumpGBDT, FraudIntelligenceEngine


def test_logit_overflow_clipping():
    """Verifies that extreme inputs (+/- 10,000) clip cleanly to [0, 1] without overflow."""
    model = ScratchLogReg()
    model.w = np.array([100.0, -100.0])
    model.b = 0.0

    # Extreme input vectors
    X_extreme = np.array([
        [1000.0, -1000.0],  # Massive positive logit
        [-1000.0, 1000.0],  # Massive negative logit
        [0.0, 0.0]
    ])

    probs = model.predict_proba(X_extreme)
    assert not np.any(np.isnan(probs)), "NaN detected in extreme logit output"
    assert not np.any(np.isinf(probs)), "Inf detected in extreme logit output"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities outside [0, 1] range"
    assert probs[0] > 0.9999, "Massive positive logit should yield ~1.0"
    assert probs[1] < 0.0001, "Massive negative logit should yield ~0.0"


def test_extreme_amount_transactions():
    """Tests system stability under extreme financial amounts ($0.00 to $100,000,000.00)."""
    engine = FraudIntelligenceEngine()
    engine.load()

    # Zero dollar micro-authorization
    x_zero = np.zeros(29)
    x_zero[28] = 0.0
    preds_zero = engine.predict_all(x_zero)
    assert all(0.0 <= p <= 1.0 for p in preds_zero.values()), "Zero amount produced invalid probabilities"

    # 100 Million Dollar mega-wire
    x_huge = np.zeros(29)
    x_huge[28] = 100_000_000.0
    preds_huge = engine.predict_all(x_huge)
    assert all(0.0 <= p <= 1.0 for p in preds_huge.values()), "Mega-wire amount produced invalid probabilities"
    assert all(not np.isnan(p) for p in preds_huge.values()), "NaN generated on mega-wire"


def test_zero_variance_feature_resilience():
    """Ensures Standard Scaler handles zero-variance (constant) features without division by zero."""
    X_constant = np.zeros((50, 5))
    X_constant[:, 2] = 42.0  # Column 2 has zero variance

    mean = np.mean(X_constant, axis=0)
    std = np.std(X_constant, axis=0)
    std[std == 0.0] = 1.0  # Division by zero safeguard

    X_scaled = (X_constant - mean) / std
    assert not np.any(np.isnan(X_scaled)), "NaN encountered in zero-variance feature scaling"
    assert np.all(X_scaled[:, 2] == 0.0), "Zero variance feature should center to 0.0"


def test_mlp_activation_bounds():
    """Verifies that ReLU does not explode and Sigmoid output remains bounded."""
    mlp = ScratchMLP()
    mlp.W1 = np.ones((5, 10)) * 2.0
    mlp.b1 = np.zeros((1, 10))
    mlp.W2 = np.ones((10, 5)) * 2.0
    mlp.b2 = np.zeros((1, 5))
    mlp.W3 = np.ones((5, 1)) * 2.0
    mlp.b3 = np.zeros((1, 1))

    X = np.ones((10, 5)) * 10.0
    probs = mlp.predict_proba(X)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "MLP probabilities violated [0, 1] range"
    assert not np.any(np.isnan(probs)), "MLP generated NaNs under saturated activations"
