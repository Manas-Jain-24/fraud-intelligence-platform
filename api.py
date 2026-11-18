"""
api.py
======
Local scoring and investigation API for the fraud research project.
Provides evaluated inference and explanations across payment profiles:
  1. Credit Card Rail (ULB European Cardholders)
  2. Mobile Money / Wire Transfer Rail (PaySim Mobile Ledger Discrepancies)

Supports saved NumPy engines and locally evaluated scikit-learn candidates.
Author: Manas Jain <manasj1234567890@gmail.com>
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional, Literal
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Import Model Engines
from models_engine import FraudIntelligenceEngine
from paysim_engine import PaySimFraudEngine, paysim_engine
from fraud_ops.artifacts import get_bundle
from fraud_ops.inference import BASELINE, COST, MLP, GBDT, ANOMALY, ENSEMBLE
from fraud_ops.service import assess, ingest
from fraud_ops.store import InvestigationStore, ConflictError

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
    balanceDrainRatio: float = Field(..., description="Engine feature: amount / (oldbalanceOrg + 1), without clipping")
    isTransfer: bool = Field(..., description="Flag indicating high-risk P2P TRANSFER flow")


class ModelScoreBreakdown(BaseModel):
    """Uncalibrated scratch diagnostics, separate from the active model's score."""
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
    action: str
    flag_threshold: float
    policy_version: str
    model_name: str
    model_version: str
    score_kind: str
    assessment_stage: str
    policy: Dict[str, Any]
    evaluation_status: Optional[str] = None
    diagnostic_model_version: str
    model_breakdown_score_kind: str = "uncalibrated_score"
    workflow_policy_version: Optional[str] = None


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
    model_name: str
    model_version: str
    explained_model: str
    explained_score: float
    baseline_score: float
    threshold_applied: float
    flag_threshold: float
    action: str
    risk_level: str
    score_kind: str
    assessment_stage: str
    policy_version: str
    attribution_method: str
    impact_units: str
    feature_order: List[str]
    reference: List[float]
    all_impacts: List[FeatureImpact]
    workflow_policy_version: Optional[str] = None


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
            "score_wire_pre": "POST /v1/score/wire/pre",
            "explain_card": "POST /v1/explain/card",
            "explain_wire": "POST /v1/explain/wire",
            "samples_card": "/v1/samples/card",
            "samples_wire": "/v1/samples/wire",
            "ingest": "POST /v1/transactions",
            "cases": "GET /v1/cases",
            "case_action": "PATCH /v1/cases/{case_id}",
            "feedback": "GET /v1/feedback"
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


@app.get("/v1/models", tags=["Model Registry"])
async def list_registered_models():
    """Artifact-specific measurements; legacy scratch scores remain available for research."""
    entries = []
    for profile in ("card", "wire", "wire_pre"):
        try:
            bundle = get_bundle(profile)
        except FileNotFoundError:
            continue
        entries.append({"name": bundle.model_name, "rail": profile, "active": True,
                        "model_version": bundle.version, "score_kind": bundle.score_kind,
                        "assessment_stage": bundle.stage, "policy": bundle.policy.to_dict(),
                        "evaluation": bundle.metadata})
    for name in (BASELINE, COST, MLP, GBDT, ANOMALY, ENSEMBLE):
        entries.append({"name": name, "rail": "card and wire", "active": False,
                        "purpose": "Uncalibrated research diagnostics; active model is identified above"})
    return entries


def _score(profile, payload, threshold=None):
    if (card_engine if profile == "card" else wire_engine) is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    values = payload.model_dump()
    try:
        result = assess(profile, values, threshold)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    from fraud_ops.artifacts import feature_vector
    diagnostic_engine = card_engine if profile == "card" else wire_engine
    result["diagnostic_model_version"] = diagnostic_engine.model_version
    scores = diagnostic_engine.predict_all(feature_vector(profile, values))
    result["model_breakdown"] = ModelScoreBreakdown(
        baseline_logistic_regression=scores[BASELINE], cost_sensitive_logistic_regression=scores[COST],
        multilayer_perceptron_neural_net=scores[MLP], gradient_boosted_decision_trees=scores[GBDT],
        unsupervised_isolation_profiler=scores[ANOMALY], champion_soft_voting_ensemble=scores[ENSEMBLE])
    if profile == "wire":
        raw = feature_vector(profile, values)
        result["ledger_anomalies"] = LedgerDiscrepancies(errorBalanceOrig=raw[5], errorBalanceDest=raw[6],
            balanceDrainRatio=raw[7], isTransfer=bool(raw[8]))
    return result


@app.post("/v1/score/card", response_model=ScoringResponse, tags=["Inference"])
async def score_credit_card(payload: CreditCardFeatures, threshold: Optional[float] = None):
    return _score("card", payload, threshold)


@app.post("/v1/score/wire", response_model=ScoringResponse, tags=["Inference"])
async def score_wire_transfer(payload: WireTransferRequest, threshold: Optional[float] = None):
    """Post-transaction assessment: uses observed settlement balances; action is REVIEW or PASS."""
    return _score("wire", payload, threshold)


def _explain(profile, payload, threshold=None):
    try:
        result = assess(profile, payload.model_dump(), threshold, explain=True)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    explanation = result.pop("explanation")
    return {**result, **explanation, "top_fraud_drivers": explanation["top_drivers"],
            "top_safety_indicators": explanation["top_safeties"]}


@app.post("/v1/explain/card", response_model=ExplainabilityResponse, tags=["Explainability"])
async def explain_credit_card(payload: CreditCardFeatures, threshold: Optional[float] = None):
    return _explain("card", payload, threshold)


@app.post("/v1/explain/wire", response_model=ExplainabilityResponse, tags=["Explainability"])
async def explain_wire_transfer(payload: WireTransferRequest, threshold: Optional[float] = None):
    return _explain("wire", payload, threshold)


class WirePreRequest(BaseModel):
    type: Literal["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)


@app.post("/v1/score/wire/pre", tags=["Inference"])
async def score_wire_pre(payload: WirePreRequest, threshold: Optional[float] = None):
    """Uses only pre-transaction fields; never substitutes zero for an unknown post balance."""
    try:
        return assess("wire_pre", payload.model_dump(), threshold, explain=True)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class TransactionContext(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    event_time: datetime
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    synthetic: bool = False


class TransactionSubmission(BaseModel):
    profile: Literal["card", "wire", "wire_pre"]
    payload: Dict[str, Any]
    context: TransactionContext
    force_case: bool = False


@app.post("/v1/transactions", tags=["Investigations"])
async def ingest_transaction(submission: TransactionSubmission):
    from pydantic import ValidationError
    schema = {"card": CreditCardFeatures, "wire": WireTransferRequest, "wire_pre": WirePreRequest}[submission.profile]
    try:
        payload = schema.model_validate(submission.payload).model_dump()
        context = submission.context.model_dump(mode="json")
        return ingest(submission.profile, payload, context, force_case=submission.force_case)
    except ConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (ValueError, ValidationError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/v1/transactions", tags=["Investigations"])
async def list_transactions(sender_id: Optional[str] = None, recipient_id: Optional[str] = None, limit: int = 100):
    return InvestigationStore().list_transactions(limit, sender_id, recipient_id)


@app.get("/v1/transactions/{transaction_id}", tags=["Investigations"])
async def get_transaction(transaction_id: str):
    try:
        return InvestigationStore().get_transaction(transaction_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/v1/cases", tags=["Investigations"])
async def list_cases(status: Optional[str] = None, assignee: Optional[str] = None, synthetic: Optional[bool] = None):
    return InvestigationStore().list_cases(status, assignee, synthetic)


@app.get("/v1/cases/{case_id}", tags=["Investigations"])
async def get_case(case_id: int):
    try:
        return InvestigationStore().get_case(case_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


class CaseUpdate(BaseModel):
    action: Literal["ASSIGN", "RESOLVE", "REOPEN", "NOTE"]
    actor: str
    expected_version: int
    assignee: Optional[str] = None
    decision: Optional[Literal["CONFIRMED_FRAUD", "LEGITIMATE", "INCONCLUSIVE"]] = None
    reason: Optional[str] = None


@app.patch("/v1/cases/{case_id}", tags=["Investigations"])
async def update_case(case_id: int, update: CaseUpdate):
    try:
        return InvestigationStore().update_case(case_id, **update.model_dump())
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/feedback", tags=["Investigations"])
async def investigator_feedback(include_synthetic: bool = False):
    return InvestigationStore().feedback_rows(include_synthetic)


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
