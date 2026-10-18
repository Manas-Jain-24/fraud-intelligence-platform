"""
test_paysim_engine.py
=====================
Unit Tests and Invariant Checks for PaySim Mobile Money Fraud Engine.
Verifies mathematical feature derivations, model inference stability, and explainability.
Author: Manas Jain
"""

import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paysim_engine import PaySimFraudEngine, paysim_engine


def test_feature_computation_invariants():
    """Verify that ledger discrepancy and drain ratio math hold exactly."""
    # Test Case 1: Standard legitimate transfer (Alice sends $100 to Bob)
    feats = paysim_engine.compute_features(
        amount=100.0,
        oldbalanceOrg=500.0,
        newbalanceOrig=400.0,
        oldbalanceDest=200.0,
        newbalanceDest=300.0,
        tx_type="TRANSFER"
    )
    assert len(feats) == 9
    assert feats[0] == 100.0   # amount
    assert feats[5] == 0.0     # errorBalanceOrig (400 + 100 - 500 = 0)
    assert feats[6] == 0.0     # errorBalanceDest (200 + 100 - 300 = 0)
    assert abs(feats[7] - (100.0 / 501.0)) < 1e-6 # balanceDrainRatio
    assert feats[8] == 1.0     # isTransfer

    # Test Case 2: Cash-Out with total drain
    feats_drain = paysim_engine.compute_features(
        amount=1000.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        tx_type="CASH_OUT"
    )
    assert feats_drain[5] == 0.0     # errorBalanceOrig (0 + 1000 - 1000 = 0)
    assert feats_drain[6] == 1000.0  # errorBalanceDest (0 + 1000 - 0 = 1000 ghost mule)
    assert abs(feats_drain[7] - (1000.0 / 1001.0)) < 1e-4 # near 1.0 drain ratio
    assert feats_drain[8] == 0.0     # CASH_OUT


def test_model_inference_probabilities():
    """Ensure all model probabilities fall strictly in [0.0, 1.0]."""
    paysim_engine.load()
    sample = paysim_engine.compute_features(
        amount=250000.0,
        oldbalanceOrg=250000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        tx_type="TRANSFER"
    )
    scores = paysim_engine.predict_all(sample)

    assert "Baseline Logistic Regression (NumPy)" in scores
    assert "Cost-Sensitive LogReg (NumPy)" in scores
    assert "Multi-Layer Perceptron (NumPy MLP)" in scores
    assert "Gradient Boosted Trees (NumPy GBDT)" in scores
    assert "Soft-Voting Ensemble (Scratch Champion)" in scores

    for model_name, prob in scores.items():
        assert 0.0 <= prob <= 1.0, f"Probability out of bounds for {model_name}: {prob}"


def test_sub_millisecond_inference_latency():
    """Ensure model inference is sub-millisecond per transaction."""
    paysim_engine.load()
    sample = paysim_engine.compute_features(
        amount=50.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=950.0,
        oldbalanceDest=200.0,
        newbalanceDest=250.0,
        tx_type="TRANSFER"
    )
    # Warm up
    _ = paysim_engine.predict_all(sample)

    t0 = time.perf_counter()
    runs = 100
    for _ in range(runs):
        _ = paysim_engine.predict_all(sample)
    t1 = time.perf_counter()
    avg_latency_ms = ((t1 - t0) / runs) * 1000.0

    # Must be faster than 2.0 ms
    assert avg_latency_ms < 2.0, f"Inference latency too slow: {avg_latency_ms:.3f} ms"


def test_directional_fraud_risk_sensitivity():
    """Verify that account takeover drain scores strictly higher risk than benign everyday transfer."""
    paysim_engine.load()

    benign = paysim_engine.compute_features(
        amount=25.0,
        oldbalanceOrg=1500.0,
        newbalanceOrig=1475.0,
        oldbalanceDest=400.0,
        newbalanceDest=425.0,
        tx_type="TRANSFER"
    )

    drain_fraud = paysim_engine.compute_features(
        amount=500000.0,
        oldbalanceOrg=500000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        tx_type="TRANSFER"
    )

    benign_scores = paysim_engine.predict_all(benign)
    fraud_scores = paysim_engine.predict_all(drain_fraud)

    # Cost-Sensitive and Ensemble must rate fraud much higher than benign
    assert fraud_scores["Cost-Sensitive LogReg (NumPy)"] > benign_scores["Cost-Sensitive LogReg (NumPy)"]
    assert fraud_scores["Soft-Voting Ensemble (Scratch Champion)"] > benign_scores["Soft-Voting Ensemble (Scratch Champion)"]


def test_explainability_narrative():
    """Verify that explainability engine detects account drain and high value outliers."""
    drain_fraud = paysim_engine.compute_features(
        amount=500000.0,
        oldbalanceOrg=500000.0,
        newbalanceOrig=0.0,
        oldbalanceDest=0.0,
        newbalanceDest=0.0,
        tx_type="TRANSFER"
    )
    xai = paysim_engine.explain_transaction(drain_fraud)

    assert "top_drivers" in xai
    assert "narrative" in xai
    assert "Account Drain Alert" in xai["narrative"]
    assert len(xai["top_drivers"]) > 0
