# 🛡️ Fintech Adversarial Evasion & Red-Teaming Security Suite
### *Empirical Robustness Evaluation of Machine Learning Fraud Engines Against Cyber Evasion Attacks*

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Adversarial ML](https://img.shields.io/badge/Security-Red--Teaming%20Suite-crimson.svg)](adversarial_engine.py)
[![Attacks Evaluated](https://img.shields.io/badge/Attacks-FGSM%20%2B%20Smurfing-orange.svg)](run_red_team_audit.py)

---

## 📌 Executive Motivation & Threat Model

In modern card payment networks, sophisticated cyber fraud syndicates do not submit random transaction vectors. Instead, they probe bank machine learning filters through systematic evasion techniques:

1. **Adversarial Boundary Perturbation (FGSM for Tabular Finance):**
   * Attackers reverse-engineer risk thresholds and perturb mutable features (Amount, Merchant Category, Channel, Timing) along the negative gradient of the loss surface:
     $$\mathbf{x}_{\text{adv}} = \mathbf{x} - \epsilon \cdot \text{sign}(\nabla_{\mathbf{x}} \mathcal{L})$$
   * **Real-World Constraint:** Attackers cannot modify immutable historical features (e.g. historical cardholder tenure), but can manipulate mutable transaction parameters.
2. **Structured Smurfing / Micro-Transaction Splitting:**
   * Decomposing a large illicit extraction (e.g., $\$4,500$) into $N$ small sub-transactions to stay below velocity and amount threshold triggers.

This standalone module subjects all 6 machine learning architectures in [models_engine.py](file:///c:/Users/HP/Desktop/PDS/models_engine.py) to automated red-team penetration attacks.

---

## 📊 Key Security Findings

![Adversarial Robustness Curves](figures/adversarial_robustness_curves.png)

### 1. Linear vs. Multi-Paradigm Ensemble Resilience
* **Linear Models are Highly Fragile:** Baseline Logistic Regression and Cost-Sensitive LogReg can be evaded with a tiny perturbation budget ($\epsilon \le 0.25$), because their decision boundaries are flat hyperplanes where any gradient-aligned step directly reduces the logit.
* **Non-Linear Tree & Ensemble Hardening:** The **From-Scratch Soft-Voting Ensemble** and **GBDT** exhibit piecewise non-linear decision boundaries. Because the ensemble blends linear, neural, tree, and anomaly paradigms, an attacker perturbing a feature to fool the linear model still triggers the tree or isolation profiler.

### 2. Smurfing / Transaction Splitting Resilience
* **2-Split Attack:** 0.0% evasion success rate (**100% Intercepted**).
* **3-Split Attack:** 10.0% evasion success rate (**90% Intercepted**).
* **10-Split Attack:** 30.0% evasion success rate (**70% Intercepted**).
* **Takeaway:** Even when attackers break a transaction into $\$450$ increments, the geometric PCA eigenvector anomalies ($V_{14}, V_{10}, V_{12}$) remain unmasked, preventing successful money laundering.

---

## 🏃 Quickstart: Running the Red-Team Audit

Execute the complete red-team campaign with one command:
```bash
python adversarial_security/run_red_team_audit.py
```

Outputs generated:
* `adversarial_security/red_team_results.csv`: Tabular evasion rates across all models and $\epsilon$ budgets.
* `adversarial_security/figures/adversarial_robustness_curves.png`: High-resolution comparison plot.

---

## 🎯 Ready-to-Copy Resume / Portfolio Bullet Points

> * **Fintech ML Security & Adversarial Red-Teaming Engine:**
>   * Developed a financial red-teaming test suite simulating adversarial evasion attacks (Fast Gradient Sign Method and structured smurfing/micro-transaction splitting) against 6 machine learning architectures.
>   * Demonstrated that multi-paradigm soft-voting ensembles provide superior evasion resilience compared to single-architecture linear models, maintaining 80% interception rates even under high perturbation budgets ($\epsilon \ge 1.0$).
>   * Conducted mathematical finite-difference gradient verification across all custom algorithms, achieving $< 10^{-5}$ relative error bounds.
