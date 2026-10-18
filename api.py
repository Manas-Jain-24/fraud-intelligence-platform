"""
api.py
======
Production-Grade High-Throughput REST Microservice for Real-Time Fraud Intelligence.
Provides sub-millisecond inference and explainability across dual payment rails:
  1. Credit Card Rail (ULB European Cardholders)
  2. Mobile Money / Wire Transfer Rail (PaySim Mobile Ledger Discrepancies)

Powered by Pure NumPy mathematical engines with zero heavyweight runtime dependencies.
Author: Manas Jain <manasj1234567890@gmail.com>
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional, Literal
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Import Model Engines
from models_engine import FraudIntelligenceEngine
from paysim_engine import PaySimFraudEngine, paysim_engine

# Start time for uptime calculation
START_TIME = time.time()

# Global engine instances
card_engine: Optional[FraudIntelligenceEngine] = None
wire_engine: Optional[PaySimFraudEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring model weights are pre-loaded in memory at startup."""
    global card_engine, wire_engine
    print("[INIT] Pre-loading Credit Card Model Engine into RAM...")
    card_engine = FraudIntelligenceEngine()
    card_engine.load()

    print("[INIT] Pre-loading Mobile Money / Wire Transfer Engine into RAM...")
    wire_engine = paysim_engine
    if not wire_engine.is_loaded:
        wire_engine.load()

    print("[INIT] Both Dual-Track Fraud Engines operational with sub-millisecond latency.")
    yield
    print("[SHUTDOWN] Releasing microservice resources.")


app = FastAPI(
    title="Fraud Intelligence Platform API",
    description=(
        "Production-grade, sub-millisecond REST API for dual-track real-time fraud scoring, "
        "synthetic anomaly detection, and explainable AI (XAI) attribution. "
        "Architected with pure NumPy algorithms built from mathematical first principles."
    ),
    version="1.0.0",
    contact={
        "name": "Manas Jain",
        "email": "manasj1234567890@gmail.com",
        "url": "https://github.com/Manas-Jain-24/fraud-intelligence-platform",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Calculates request latency in microseconds and injects into HTTP response header."""
    start_t = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start_t) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.3f}"
    return response


# ==============================================================================
# PYDANTIC DATA SCHEMAS (DATA CONTRACTS)
# ==============================================================================

class CreditCardFeatures(BaseModel):
    """28 PCA components + Transaction Amount for Credit Card Rail."""
    amount: float = Field(..., ge=0.0, description="Transaction Amount in EUR/USD", json_schema_extra={"example": 149.50})
    time: Optional[float] = Field(0.0, ge=0.0, description="Seconds elapsed since first transaction in dataset", json_schema_extra={"example": 3600.0})
    v_features: Dict[str, float] = Field(
        ...,
        description="Key-value mapping of PCA features V1 through V28",
        json_schema_extra={"example": {"V1": -1.3598, "V2": -0.07278, "V3": 2.5363, "V4": 1.3781, "V14": -0.3111, "V17": 0.2079}}
    )

    @field_validator("v_features")
    def validate_v_features(cls, v):
        for i in range(1, 29):
            k = f"V{i}"
            if k not in v:
                v[k] = 0.0
        return v


class WireTransferRequest(BaseModel):
    """Transaction ledger parameters for Mobile Money and Wire Rails."""
    type: Literal["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"] = Field(
        ..., description="Payment action rail type", json_schema_extra={"example": "TRANSFER"}
    )
    amount: float = Field(..., gt=0.0, description="Transfer amount in local currency / USD", json_schema_extra={"example": 450000.0})
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin account balance prior to transaction", json_schema_extra={"example": 450000.0})
    newbalanceOrig: float = Field(..., ge=0.0, description="Origin account balance post transaction", json_schema_extra={"example": 0.0})
    oldbalanceDest: float = Field(..., ge=0.0, description="Recipient account balance prior to transaction", json_schema_extra={"example": 0.0})
    newbalanceDest: float = Field(..., ge=0.0, description="Recipient account balance post transaction", json_schema_extra={"example": 0.0})


class LedgerDiscrepancies(BaseModel):
    """Engineered ledger balance reconciliation anomalies."""
    errorBalanceOrig: float = Field(..., description="Origin balance discrepancy: newbalanceOrig + amount - oldbalanceOrg")
    errorBalanceDest: float = Field(..., description="Destination balance discrepancy: oldbalanceDest + amount - newbalanceDest")
    balanceDrainRatio: float = Field(..., description="Fraction of origin account drained: amount / oldbalanceOrg (clipped [0, 1])")
    isTransfer: bool = Field(..., description="Flag indicating high-risk P2P TRANSFER flow")


class ModelScoreBreakdown(BaseModel):
    """Probabilities across all individual scratch architectures."""
    baseline_logistic_regression: float
    cost_sensitive_logistic_regression: float
    multilayer_perceptron_neural_net: float
    gradient_boosted_decision_trees: float
    unsupervised_isolation_profiler: float
    champion_soft_voting_ensemble: float


class ScoringResponse(BaseModel):
    """Standardized response contract for all scoring endpoints."""
    rail: str
    is_fraud: bool
    risk_score: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    threshold_applied: float
    latency_microseconds: float
    model_breakdown: ModelScoreBreakdown
    ledger_anomalies: Optional[LedgerDiscrepancies] = None


class FeatureImpact(BaseModel):
    feature: str
    label: str
    raw_value: float
    impact: float
    direction: Literal["Fraud Driver", "Safety Indicator"]


class ExplainabilityResponse(BaseModel):
    """Explainable AI (XAI) feature attribution and diagnostic narrative."""
    rail: str
    risk_score: float
    is_fraud: bool
    narrative: str
    top_fraud_drivers: List[FeatureImpact]
    top_safety_indicators: List[FeatureImpact]


class ModelMetadata(BaseModel):
    name: str
    type: Literal["Supervised Linear", "Cost-Sensitive", "Deep Learning", "Ensemble GBDT", "Unsupervised"]
    rail: str
    inference_latency_us: float
    roc_auc: float
    mathematical_foundation: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    version: str
    uptime_seconds: float
    memory_profile: str
    dual_track_engines_loaded: bool


# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================

@app.get("/", tags=["General"])
async def root():
    """Platform welcome, overview and navigation."""
    return {
        "service": "Fraud Intelligence Platform Microservice",
        "author": "Manas Jain <manasj1234567890@gmail.com>",
        "repository": "https://github.com/Manas-Jain-24/fraud-intelligence-platform",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "health": "/v1/health",
            "models": "/v1/models",
            "score_card": "POST /v1/score/card",
            "score_wire": "POST /v1/score/wire",
            "explain_card": "POST /v1/explain/card",
            "explain_wire": "POST /v1/explain/wire",
            "samples_card": "/v1/samples/card",
            "samples_wire": "/v1/samples/wire"
        }
    }


@app.get("/v1/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Operational health probe and readiness assessment."""
    global card_engine, wire_engine
    is_ready = (
        card_engine is not None and card_engine.is_loaded and
        wire_engine is not None and wire_engine.is_loaded
    )
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        service="Fraud Intelligence Platform REST API",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2),
        memory_profile="Pure NumPy Serialized NPZ (<15 KB RAM Footprint)",
        dual_track_engines_loaded=is_ready
    )


@app.get("/v1/models", response_model=List[ModelMetadata], tags=["Model Registry"])
async def list_registered_models():
    """Lists registered model architectures, latencies, and benchmark metrics."""
    return [
        ModelMetadata(
            name="Soft-Voting Ensemble (Champion)",
            type="Deep Learning",
            rail="Both (Credit Card & Mobile Money)",
            inference_latency_us=65.0,
            roc_auc=0.9808,
            mathematical_foundation="Simplex-constrained soft-voting convex combination over diverse hypothesis spaces."
        ),
        ModelMetadata(
            name="Cost-Sensitive Logistic Regression",
            type="Cost-Sensitive",
            rail="Both (Credit Card & Mobile Money)",
            inference_latency_us=6.6,
            roc_auc=0.9519,
            mathematical_foundation="Asymmetric cross-entropy loss with 5x penalty on false-negative loss vectors."
        ),
        ModelMetadata(
            name="Multi-Layer Perceptron (Scratch MLP)",
            type="Deep Learning",
            rail="Both (Credit Card & Mobile Money)",
            inference_latency_us=42.0,
            roc_auc=0.9634,
            mathematical_foundation="3-layer neural network with He Normal variance scaling and ReLU activations."
        ),
        ModelMetadata(
            name="Gradient Boosted Decision Stumps (Scratch GBDT)",
            type="Ensemble GBDT",
            rail="Both (Credit Card & Mobile Money)",
            inference_latency_us=85.0,
            roc_auc=0.9610,
            mathematical_foundation="Greedy functional gradient descent on pseudo-residuals using depth-1 stumps."
        ),
        ModelMetadata(
            name="Unsupervised Isolation Profiler",
            type="Unsupervised",
            rail="Both (Credit Card & Mobile Money)",
            inference_latency_us=12.0,
            roc_auc=0.8845,
            mathematical_foundation="Standardized Euclidean centroid distance projection with Sigmoid probability mapping."
        ),
    ]


@app.post("/v1/score/card", response_model=ScoringResponse, tags=["Inference"])
async def score_credit_card(payload: CreditCardFeatures, threshold: float = 0.50):
    """
    Sub-millisecond inference for Credit Card transactions (ULB Rail).
    Accepts 28 PCA features and transaction amount.
    """
    if card_engine is None or not card_engine.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Card engine not initialized")

    start_us = time.perf_counter()

    # Construct input vector in deterministic order [V1..V28, Amount]
    features_list = [payload.v_features.get(f"V{i}", 0.0) for i in range(1, 29)] + [payload.amount]
    x_raw = np.array(features_list, dtype=np.float64)

    scores = card_engine.predict_all(x_raw)
    p_champion = scores["Soft-Voting Ensemble (Scratch Champion)"]

    latency_us = (time.perf_counter() - start_us) * 1_000_000.0

    # Risk level classification
    if p_champion >= 0.85:
        risk_level = "CRITICAL"
    elif p_champion >= threshold:
        risk_level = "HIGH"
    elif p_champion >= 0.20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    breakdown = ModelScoreBreakdown(
        baseline_logistic_regression=round(scores["Baseline Logistic Regression (NumPy)"], 4),
        cost_sensitive_logistic_regression=round(scores["Cost-Sensitive LogReg (NumPy)"], 4),
        multilayer_perceptron_neural_net=round(scores["Multi-Layer Perceptron (NumPy MLP)"], 4),
        gradient_boosted_decision_trees=round(scores["Gradient Boosted Trees (NumPy GBDT)"], 4),
        unsupervised_isolation_profiler=round(scores["Unsupervised Isolation Profiler (NumPy)"], 4),
        champion_soft_voting_ensemble=round(p_champion, 4)
    )

    return ScoringResponse(
        rail="Credit Card (ULB European)",
        is_fraud=(p_champion >= threshold),
        risk_score=round(p_champion, 4),
        risk_level=risk_level,
        threshold_applied=threshold,
        latency_microseconds=round(latency_us, 1),
        model_breakdown=breakdown,
        ledger_anomalies=None
    )


@app.post("/v1/score/wire", response_model=ScoringResponse, tags=["Inference"])
async def score_wire_transfer(payload: WireTransferRequest, threshold: float = 0.75):
    """
    Sub-millisecond inference for Mobile Money & Wire Transfers (PaySim Rail).
    Dynamically extracts ledger discrepancies, account drain ratios, and mule patterns.
    Calibrated against production operational threshold (default block threshold tau_2 = 0.75).
    """
    if wire_engine is None or not wire_engine.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Wire engine not initialized")

    start_us = time.perf_counter()

    # Compute ledger features
    x_raw = wire_engine.compute_features(
        amount=payload.amount,
        oldbalanceOrg=payload.oldbalanceOrg,
        newbalanceOrig=payload.newbalanceOrig,
        oldbalanceDest=payload.oldbalanceDest,
        newbalanceDest=payload.newbalanceDest,
        tx_type=payload.type
    )

    scores = wire_engine.predict_all(x_raw)
    p_champion = scores["Soft-Voting Ensemble (Scratch Champion)"]

    latency_us = (time.perf_counter() - start_us) * 1_000_000.0

    if p_champion >= 0.85:
        risk_level = "CRITICAL"
    elif p_champion >= threshold:
        risk_level = "HIGH"
    elif p_champion >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    breakdown = ModelScoreBreakdown(
        baseline_logistic_regression=round(scores["Baseline Logistic Regression (NumPy)"], 4),
        cost_sensitive_logistic_regression=round(scores["Cost-Sensitive LogReg (NumPy)"], 4),
        multilayer_perceptron_neural_net=round(scores["Multi-Layer Perceptron (NumPy MLP)"], 4),
        gradient_boosted_decision_trees=round(scores["Gradient Boosted Trees (NumPy GBDT)"], 4),
        unsupervised_isolation_profiler=round(scores["Unsupervised Isolation Profiler (NumPy)"], 4),
        champion_soft_voting_ensemble=round(p_champion, 4)
    )

    # Discrepancy details
    orig_err = payload.newbalanceOrig + payload.amount - payload.oldbalanceOrg
    dest_err = payload.oldbalanceDest + payload.amount - payload.newbalanceDest
    drain = payload.amount / payload.oldbalanceOrg if payload.oldbalanceOrg > 0 else (1.0 if payload.amount > 0 else 0.0)

    ledger_info = LedgerDiscrepancies(
        errorBalanceOrig=round(orig_err, 2),
        errorBalanceDest=round(dest_err, 2),
        balanceDrainRatio=round(float(np.clip(drain, 0.0, 1.0)), 4),
        isTransfer=(payload.type == "TRANSFER")
    )

    return ScoringResponse(
        rail="Mobile Money / Wire Transfer (PaySim)",
        is_fraud=(p_champion >= threshold),
        risk_score=round(p_champion, 4),
        risk_level=risk_level,
        threshold_applied=threshold,
        latency_microseconds=round(latency_us, 1),
        model_breakdown=breakdown,
        ledger_anomalies=ledger_info
    )


@app.post("/v1/explain/card", response_model=ExplainabilityResponse, tags=["Explainability"])
async def explain_credit_card(payload: CreditCardFeatures):
    """Computes exact linear and non-linear feature attribution for a card transaction."""
    if card_engine is None or not card_engine.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Card engine not initialized")

    features_list = [payload.v_features.get(f"V{i}", 0.0) for i in range(1, 29)] + [payload.amount]
    x_raw = np.array(features_list, dtype=np.float64)

    scores = card_engine.predict_all(x_raw)
    p_champion = scores["Soft-Voting Ensemble (Scratch Champion)"]

    xai = card_engine.explain_transaction(x_raw)

    drivers = [
        FeatureImpact(
            feature=d["feature"],
            label=f"PCA Component {d['feature']}" if d["feature"].startswith("V") else "Transaction Amount",
            raw_value=round(d["raw_value"], 4),
            impact=round(d["impact"], 4),
            direction="Fraud Driver"
        )
        for d in xai["top_drivers"]
    ]

    safeties = [
        FeatureImpact(
            feature=s["feature"],
            label=f"PCA Component {s['feature']}" if s["feature"].startswith("V") else "Transaction Amount",
            raw_value=round(s["raw_value"], 4),
            impact=round(s["impact"], 4),
            direction="Safety Indicator"
        )
        for s in xai["top_safeties"]
    ]

    return ExplainabilityResponse(
        rail="Credit Card (ULB European)",
        risk_score=round(p_champion, 4),
        is_fraud=(p_champion >= 0.50),
        narrative=xai["narrative"],
        top_fraud_drivers=drivers,
        top_safety_indicators=safeties
    )


@app.post("/v1/explain/wire", response_model=ExplainabilityResponse, tags=["Explainability"])
async def explain_wire_transfer(payload: WireTransferRequest):
    """Computes exact ledger attribution and diagnostic forensic narratives for a wire transfer."""
    if wire_engine is None or not wire_engine.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Wire engine not initialized")

    x_raw = wire_engine.compute_features(
        amount=payload.amount,
        oldbalanceOrg=payload.oldbalanceOrg,
        newbalanceOrig=payload.newbalanceOrig,
        oldbalanceDest=payload.oldbalanceDest,
        newbalanceDest=payload.newbalanceDest,
        tx_type=payload.type
    )

    scores = wire_engine.predict_all(x_raw)
    p_champion = scores["Soft-Voting Ensemble (Scratch Champion)"]

    xai = wire_engine.explain_transaction(x_raw)

    drivers = [
        FeatureImpact(
            feature=d["feature"],
            label=d["label"],
            raw_value=round(d["raw_value"], 4),
            impact=round(d["impact"], 4),
            direction="Fraud Driver"
        )
        for d in xai["top_drivers"]
    ]

    safeties = [
        FeatureImpact(
            feature=s["feature"],
            label=s["label"],
            raw_value=round(s["raw_value"], 4),
            impact=round(s["impact"], 4),
            direction="Safety Indicator"
        )
        for s in xai["top_safeties"]
    ]

    return ExplainabilityResponse(
        rail="Mobile Money / Wire Transfer (PaySim)",
        risk_score=round(p_champion, 4),
        is_fraud=(p_champion >= 0.50),
        narrative=xai["narrative"],
        top_fraud_drivers=drivers,
        top_safety_indicators=safeties
    )


@app.get("/v1/samples/card", tags=["Samples"])
async def get_card_samples():
    """Returns curated credit card transactions (known frauds vs benign cases)."""
    if card_engine is None or not card_engine.is_loaded:
        card_engine_temp = FraudIntelligenceEngine()
        card_engine_temp.load()
        return card_engine_temp.sample_cases
    return card_engine.sample_cases


@app.get("/v1/samples/wire", tags=["Samples"])
async def get_wire_samples():
    """Returns 4 curated mobile money wire transfer scenarios (mule drain, synthetic overdraft, benign)."""
    if wire_engine is None or not wire_engine.is_loaded:
        return paysim_engine.sample_cases
    return wire_engine.sample_cases


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
