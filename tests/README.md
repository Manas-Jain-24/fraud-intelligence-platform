# 🧪 Mathematical Gradient Verification & Numerical Stability Test Suite

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)](https://pytest.org/)
[![Zero Black-Box](https://img.shields.io/badge/Algorithms-100%25_From_Scratch-orange.svg)](https://numpy.org/)

---

## 📌 Overview

This test suite provides mathematical proof and numerical validation for the custom algorithms implemented in [models_engine.py](file:///c:/Users/HP/Desktop/PDS/models_engine.py). It ensures:
1. **Zero Calculus / Gradient Errors:** Exact analytical gradients match two-sided finite difference approximations within $10^{-5}$.
2. **Numerical Invariance & Stability:** Algorithms resist floating-point overflow ($z \in [-10000, 10000]$), zero variance, and extreme dollar amounts without generating `NaN` or `inf`.
3. **Architectural Invariants:** Neural network He normal initialization variance ($\text{Var}(W) = \frac{2}{n_{in}}$), binary cross-entropy loss monotonicity, and unit simplex ensemble weights.

---

## 🏃 Quickstart: Running the Tests

Execute via the standalone colored runner:
```bash
python tests/run_tests.py
```
Or execute directly using standard `pytest`:
```bash
pytest -v tests/
```

---

## 📋 Test Matrix Breakdown

| Test File | Verification Target | Mathematical Metric / Condition |
| :--- | :--- | :--- |
| [`test_gradient_checking.py`](test_gradient_checking.py) | **Logistic Regression Gradient** | $\frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|_2}{\|\nabla_{\text{analytical}}\|_2 + \|\nabla_{\text{numerical}}\|_2} < 10^{-5}$ |
| [`test_gradient_checking.py`](test_gradient_checking.py) | **Cost-Sensitive BCE Gradient** | Weighted gradient finite difference check |
| [`test_gradient_checking.py`](test_gradient_checking.py) | **Neural Network MLP Backprop** | 2-layer He normal initialized backprop relative error $< 10^{-5}$ |
| [`test_numerical_stability.py`](test_numerical_stability.py) | **Logit Overflow Clipping** | Saturated $\pm 10,000$ inputs bound cleanly to $[0, 1]$ |
| [`test_numerical_stability.py`](test_numerical_stability.py) | **Extreme Transaction Scale** | Micro-auths ($0.00) to Mega-wires ($100,000,000) evaluated without NaNs |
| [`test_numerical_stability.py`](test_numerical_stability.py) | **Zero-Variance Feature Guard** | Standard Scaler safely normalizes constant vectors ($\sigma = 0 \to 1$) |
| [`test_numerical_stability.py`](test_numerical_stability.py) | **MLP Activation Bounds** | ReLU / Sigmoid activations remain strictly finite |
| [`test_mathematical_invariants.py`](test_mathematical_invariants.py) | **He Normal Variance** | $\text{Var}(W) \approx \frac{2}{n_{in}}$ within 5% tolerance |
| [`test_mathematical_invariants.py`](test_mathematical_invariants.py) | **Loss Monotonicity** | Strict monotonic decrease $\mathcal{L}_{t} \le \mathcal{L}_{t-1}$ over gradient descent |
| [`test_mathematical_invariants.py`](test_mathematical_invariants.py) | **Simplex Ensemble Weights** | $\sum_{i=1}^k w_i = 1.0$ and $w_i \ge 0$ |
| [`test_mathematical_invariants.py`](test_mathematical_invariants.py) | **Confusion Matrix Conservation** | $TP + FP + TN + FN \equiv N_{\text{samples}}$ |
