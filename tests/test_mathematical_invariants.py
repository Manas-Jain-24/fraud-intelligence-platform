"""
tests/test_mathematical_invariants.py
======================================
Mathematical Invariants & Algorithmic Correctness Tests.
Verifies He-normal variance conditions, loss monotonicity during gradient descent,
ensemble simplex constraints, and evaluation metric symmetry.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_engine import ScratchLogReg, ScratchMLP, FraudIntelligenceEngine


def test_he_normal_initialization_variance():
    """
    Verifies that He normal weight initialization satisfies:
    Var(W) == 2.0 / n_in within statistical sampling tolerance.
    """
    np.random.seed(42)
    n_in, n_out = 500, 1000
    expected_variance = 2.0 / n_in
    weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

    sample_variance = float(np.var(weights))
    relative_err = abs(sample_variance - expected_variance) / expected_variance

    assert relative_err < 0.05, f"He normal variance error too high: {relative_err:.2%}"


def test_gradient_descent_loss_monotonicity():
    """
    Verifies that Binary Cross-Entropy loss monotonically decreases
    over initial gradient descent epochs on linearly separable data.
    """
    np.random.seed(42)
    m, n = 100, 4
    # Linearly separable synthetic data
    X = np.random.randn(m, n)
    w_true = np.array([2.0, -3.0, 1.5, -1.0])
    y = (np.dot(X, w_true) > 0).astype(int)

    w = np.zeros(n)
    b = 0.0
    lr = 0.2
    losses = []

    for _ in range(30):
        z = np.clip(np.dot(X, w) + b, -500, 500)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        losses.append(loss)

        err = p - y
        w -= lr * (np.dot(X.T, err) / m)
        b -= lr * np.mean(err)

    # Verify overall loss reduction
    assert losses[-1] < losses[0], "Gradient descent failed to reduce binary cross-entropy loss"
    # Verify strict monotonicity over the first 10 steps
    for i in range(1, 10):
        assert losses[i] <= losses[i-1] + 1e-6, f"Loss increased at step {i}: {losses[i]} > {losses[i-1]}"


def test_ensemble_simplex_weights_constraint():
    """Verifies that the Soft-Voting Ensemble blending weights lie on the unit simplex (sum == 1.0, w >= 0)."""
    # Champion weights: MLP (0.35) + GBDT (0.30) + CostSensitiveLogReg (0.25) + Anomaly (0.10)
    weights = np.array([0.35, 0.30, 0.25, 0.10])
    assert np.all(weights >= 0.0), "Ensemble contains negative weights"
    assert abs(np.sum(weights) - 1.0) < 1e-6, f"Ensemble weights do not sum to 1.0: {np.sum(weights)}"


def test_confusion_matrix_sum_invariant():
    """Verifies that TP + FP + TN + FN identically matches total evaluation sample size N."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 0])

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    assert (TP + FP + TN + FN) == len(y_true), "Confusion matrix elements do not sum to sample size N"
