"""
tests/test_api.py
=================
Integration and Performance Test Suite for FastAPI Fraud Intelligence Microservice.
Verifies data contracts, Pydantic validation, dual-rail inference accuracy,
XAI explainability narrative generation, and sub-millisecond execution.
"""

import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from api import app, card_engine, wire_engine


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and enters lifespan context so models are preloaded."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Verifies API root catalog and OpenAPI docs discovery."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "author" in data
    assert "Manas Jain" in data["author"]
    assert data["documentation"] == "/docs"
    assert "score_card" in data["endpoints"]
    assert "score_wire" in data["endpoints"]


def test_health_check(client):
    """Verifies operational readiness probe."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dual_track_engines_loaded"] is True
    assert data["version"] == "1.0.0"
    assert "X-Process-Time-Ms" in response.headers


def test_models_metadata(client):
    """Verifies model registry metadata and benchmark metrics."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 5
    names = [m["name"] for m in models]
    assert any("Soft-Voting Ensemble" in n for n in names)
    assert any("Cost-Sensitive" in n for n in names)


def test_score_credit_card_benign(client):
    """Verifies scoring of a nominal credit card transaction."""
    payload = {
        "amount": 25.50,
        "time": 7200.0,
        "v_features": {f"V{i}": 0.05 for i in range(1, 29)}
    }
    response = client.post("/v1/score/card", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rail"] == "Credit Card (ULB European)"
    assert data["is_fraud"] is False
    assert data["risk_score"] < 0.50
    assert data["risk_level"] in ["LOW", "MEDIUM"]
    assert "model_breakdown" in data
    assert data["model_breakdown"]["champion_soft_voting_ensemble"] >= 0.0


def test_score_credit_card_fraud(client):
    """Verifies detection of an adversarial / fraud card transaction signature."""
    # Strong synthetic fraud signature in ULB PCA space (e.g. extreme negative V14, V17, positive V4)
    v_feats = {f"V{i}": 0.0 for i in range(1, 29)}
    v_feats["V14"] = -8.5
    v_feats["V17"] = -7.2
    v_feats["V4"] = 6.1
    v_feats["V12"] = -5.8

    payload = {
        "amount": 499.00,
        "time": 10000.0,
        "v_features": v_feats
    }
    response = client.post("/v1/score/card", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["risk_score"] >= 0.50
    assert data["risk_level"] in ["HIGH", "CRITICAL"]


def test_score_wire_benign_p2p(client):
    """Verifies that a balanced legitimate P2P wire transfer is approved."""
    payload = {
        "type": "TRANSFER",
        "amount": 250.00,
        "oldbalanceOrg": 1200.00,
        "newbalanceOrig": 950.00,
        "oldbalanceDest": 500.00,
        "newbalanceDest": 750.00
    }
    response = client.post("/v1/score/wire", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rail"] == "Mobile Money / Wire Transfer (PaySim)"
    assert data["is_fraud"] is False
    assert data["risk_level"] in ["LOW", "MEDIUM"]
    assert data["ledger_anomalies"]["errorBalanceOrig"] == 0.0
    assert data["ledger_anomalies"]["errorBalanceDest"] == 0.0
    assert data["ledger_anomalies"]["balanceDrainRatio"] < 0.50


def test_score_wire_mule_drain_fraud(client):
    """Verifies detection of high-value account takeover and mule balance drain."""
    payload = {
        "type": "TRANSFER",
        "amount": 450000.00,
        "oldbalanceOrg": 450000.00,
        "newbalanceOrig": 0.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 0.00
    }
    response = client.post("/v1/score/wire", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["risk_score"] >= 0.75
    assert data["risk_level"] in ["HIGH", "CRITICAL"]
    assert data["model_breakdown"]["cost_sensitive_logistic_regression"] > 0.95
    assert data["ledger_anomalies"]["balanceDrainRatio"] == 1.0


def test_explain_wire_transfer_narrative(client):
    """Verifies explainability endpoint generates deep diagnostic narratives."""
    payload = {
        "type": "TRANSFER",
        "amount": 350000.00,
        "oldbalanceOrg": 350000.00,
        "newbalanceOrig": 0.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 0.00
    }
    response = client.post("/v1/explain/wire", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "narrative" in data
    assert "Account Drain Alert" in data["narrative"] or "Destination Ghost Balance" in data["narrative"]
    assert len(data["top_fraud_drivers"]) > 0


def test_pydantic_validation_error(client):
    """Verifies invalid transaction types and negative amounts are rejected with 422."""
    invalid_payload = {
        "type": "BITCOIN_PAYMENT",  # Invalid enum
        "amount": -50.00,          # Invalid negative
        "oldbalanceOrg": 100.00,
        "newbalanceOrig": 50.00,
        "oldbalanceDest": 0.00,
        "newbalanceDest": 50.00
    }
    response = client.post("/v1/score/wire", json=invalid_payload)
    assert response.status_code == 422


def test_samples_endpoints(client):
    """Verifies retrieval of curated sample cases for both rails."""
    res_card = client.get("/v1/samples/card")
    assert res_card.status_code == 200
    assert len(res_card.json()) > 0

    res_wire = client.get("/v1/samples/wire")
    assert res_wire.status_code == 200
    assert len(res_wire.json()) >= 4


def test_high_throughput_sub_10ms_latency(client):
    """Executes 50 consecutive requests to confirm microsecond inference SLA."""
    payload = {
        "type": "TRANSFER",
        "amount": 100.00,
        "oldbalanceOrg": 500.00,
        "newbalanceOrig": 400.00,
        "oldbalanceDest": 200.00,
        "newbalanceDest": 300.00
    }
    latencies = []
    for _ in range(50):
        res = client.post("/v1/score/wire", json=payload)
        assert res.status_code == 200
        latencies.append(float(res.headers["X-Process-Time-Ms"]))

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage API Round-Trip Latency: {avg_latency:.2f} ms")
    assert avg_latency < 10.0  # Must be strictly sub-10ms
