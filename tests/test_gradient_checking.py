"""
tests/test_gradient_checking.py
================================
Mathematical Finite-Difference Numerical Gradient Verification.
Compares exact analytical backpropagation gradients against two-sided finite difference approximations.
Ensures zero calculus or implementation errors in custom NumPy algorithms.
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_engine import ScratchLogReg, ScratchCostSensitiveLogReg, ScratchMLP


def relative_error(grad_analytic: np.ndarray, grad_numeric: np.ndarray) -> float:
    """Computes normalized L2 relative error between analytical and numerical gradients."""
    diff = np.linalg.norm(grad_analytic - grad_numeric)
    denom = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric) + 1e-12
    return float(diff / denom)


def test_logistic_regression_gradient():
    """Finite difference gradient check on ScratchLogReg binary cross-entropy."""
    np.random.seed(42)
    m, n = 30, 5
    X = np.random.randn(m, n)
    y = np.random.randint(0, 2, size=m)

    w = np.random.randn(n) * 0.1
    b = 0.05
    eps = 1e-5

    # Analytical gradient
    z = np.dot(X, w) + b
    p = 1.0 / (1.0 + np.exp(-z))
    err = p - y
    grad_w_analytic = np.dot(X.T, err) / m
    grad_b_analytic = np.mean(err)

    # Numerical gradient for w
    grad_w_numeric = np.zeros(n)
    for j in range(n):
        w_plus = w.copy()
        w_plus[j] += eps
        p_plus = 1.0 / (1.0 + np.exp(-(np.dot(X, w_plus) + b)))
        loss_plus = -np.mean(y * np.log(p_plus + 1e-15) + (1 - y) * np.log(1 - p_plus + 1e-15))

        w_minus = w.copy()
        w_minus[j] -= eps
        p_minus = 1.0 / (1.0 + np.exp(-(np.dot(X, w_minus) + b)))
        loss_minus = -np.mean(y * np.log(p_minus + 1e-15) + (1 - y) * np.log(1 - p_minus + 1e-15))

        grad_w_numeric[j] = (loss_plus - loss_minus) / (2 * eps)

    # Numerical gradient for b
    p_b_plus = 1.0 / (1.0 + np.exp(-(np.dot(X, w) + (b + eps))))
    loss_b_plus = -np.mean(y * np.log(p_b_plus + 1e-15) + (1 - y) * np.log(1 - p_b_plus + 1e-15))
    p_b_minus = 1.0 / (1.0 + np.exp(-(np.dot(X, w) + (b - eps))))
    loss_b_minus = -np.mean(y * np.log(p_b_minus + 1e-15) + (1 - y) * np.log(1 - p_b_minus + 1e-15))
    grad_b_numeric = (loss_b_plus - loss_b_minus) / (2 * eps)

    err_w = relative_error(grad_w_analytic, grad_w_numeric)
    err_b = abs(grad_b_analytic - grad_b_numeric) / (abs(grad_b_analytic) + abs(grad_b_numeric) + 1e-12)

    assert err_w < 1e-5, f"LogReg weight gradient relative error too high: {err_w:.2e}"
    assert err_b < 1e-5, f"LogReg bias gradient relative error too high: {err_b:.2e}"


def test_cost_sensitive_gradient():
    """Finite difference check on Cost-Sensitive weighted cross-entropy."""
    np.random.seed(42)
    m, n = 25, 4
    X = np.random.randn(m, n)
    y = np.random.randint(0, 2, size=m)
    fraud_weight = 4.0
    weights = np.where(y == 1, fraud_weight, 1.0)

    w = np.random.randn(n) * 0.1
    b = -0.1
    eps = 1e-5

    # Analytical
    z = np.dot(X, w) + b
    p = 1.0 / (1.0 + np.exp(-z))
    grad_w_analytic = np.dot(X.T, (p - y) * weights) / m

    # Numerical
    grad_w_numeric = np.zeros(n)
    for j in range(n):
        w_plus = w.copy()
        w_plus[j] += eps
        p_plus = 1.0 / (1.0 + np.exp(-(np.dot(X, w_plus) + b)))
        l_plus = -np.mean(weights * (y * np.log(p_plus + 1e-15) + (1 - y) * np.log(1 - p_plus + 1e-15)))

        w_minus = w.copy()
        w_minus[j] -= eps
        p_minus = 1.0 / (1.0 + np.exp(-(np.dot(X, w_minus) + b)))
        l_minus = -np.mean(weights * (y * np.log(p_minus + 1e-15) + (1 - y) * np.log(1 - p_minus + 1e-15)))

        grad_w_numeric[j] = (l_plus - l_minus) / (2 * eps)

    err = relative_error(grad_w_analytic, grad_w_numeric)
    assert err < 1e-5, f"Cost-sensitive gradient relative error too high: {err:.2e}"


def test_neural_network_mlp_backpropagation_gradient():
    """Finite difference gradient check on 2-layer Neural Network (MLP) backprop."""
    np.random.seed(42)
    m, n = 20, 6
    h1, h2 = 8, 4
    X = np.random.randn(m, n)
    y = np.random.randint(0, 2, size=m).reshape(-1, 1)

    W1 = np.random.randn(n, h1) * np.sqrt(2.0 / n)
    b1 = np.zeros((1, h1))
    W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
    b2 = np.zeros((1, h2))
    W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
    b3 = np.zeros((1, 1))

    # Analytical Forward + Backward
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = 1.0 / (1.0 + np.exp(-Z3))

    dZ3 = A3 - y
    dW3_analytic = np.dot(A2.T, dZ3) / m

    # Numerical gradient check for W3
    eps = 1e-5
    dW3_numeric = np.zeros_like(W3)
    for i in range(h2):
        for j in range(1):
            W3_p = W3.copy()
            W3_p[i, j] += eps
            A3_p = 1.0 / (1.0 + np.exp(-(np.dot(A2, W3_p) + b3)))
            l_p = -np.mean(y * np.log(A3_p + 1e-15) + (1 - y) * np.log(1 - A3_p + 1e-15))

            W3_m = W3.copy()
            W3_m[i, j] -= eps
            A3_m = 1.0 / (1.0 + np.exp(-(np.dot(A2, W3_m) + b3)))
            l_m = -np.mean(y * np.log(A3_m + 1e-15) + (1 - y) * np.log(1 - A3_m + 1e-15))

            dW3_numeric[i, j] = (l_p - l_m) / (2 * eps)

    err = relative_error(dW3_analytic, dW3_numeric)
    assert err < 1e-5, f"Neural network W3 backprop gradient relative error too high: {err:.2e}"
