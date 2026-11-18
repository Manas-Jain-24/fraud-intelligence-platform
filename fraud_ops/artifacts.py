"""Local, reproducible model bundles used by evaluation and serving alike."""
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any
import numpy as np

from fraud_ops.inference import ENSEMBLE, explain_score
from fraud_ops.policy import DEFAULT_POLICIES, DecisionPolicy

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "artifacts"
PRE_FEATURES = ["amount", "oldbalanceOrg", "oldbalanceDest", "balanceDrainRatio", "isTransfer"]


def logit_features(scores):
    scores = np.clip(np.asarray(scores).ravel(), 1e-7, 1 - 1e-7)
    return np.log(scores / (1 - scores)).reshape(-1, 1)


@dataclass
class ModelBundle:
    model: Any
    model_name: str
    profile: str
    feature_names: list
    reference: np.ndarray
    policy: DecisionPolicy
    metadata: dict
    calibrator: Any = None
    scratch_model_name: str = ENSEMBLE

    def raw_scores(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if hasattr(self.model, "predict_batch"):
            return self.model.predict_batch(x)[self.scratch_model_name]
        return self.model.predict_proba(x)[:, 1]

    def predict(self, x):
        raw = self.raw_scores(x)
        if self.calibrator is not None:
            raw = self.calibrator.predict_proba(logit_features(raw))[:, 1]
        # Preserve a valid all-pass policy at threshold=1, including saturated logits.
        return np.clip(raw, 0., np.nextafter(1., 0.))

    def explain(self, x):
        result = explain_score(self.predict, x, self.reference, self.feature_names, self.model_name)
        if self.profile == "wire":
            raw = np.asarray(x).ravel()
            observations = []
            if raw[7] >= .95:
                observations.append(f"Account Drain Alert: requested amount / (initial balance + 1) = {raw[7]:.2f}.")
            if abs(raw[6]) > 1:
                observations.append(f"Destination Ghost Balance: discrepancy {raw[6]:,.2f}; investigate settlement context.")
            result["narrative"] = " ".join(observations + [result["narrative"]])
        return result

    @property
    def version(self):
        return self.metadata["model_version"]

    @property
    def stage(self):
        return "post_transaction" if self.profile == "wire" else "pre_transaction"

    @property
    def score_kind(self):
        return "calibrated_probability" if self.calibrator is not None else "uncalibrated_score"


def feature_vector(profile, payload):
    if profile == "card":
        return np.array([payload["v_features"][f"V{i}"] for i in range(1, 29)] + [payload["amount"]], dtype=float)
    values = {
        "amount": payload["amount"], "oldbalanceOrg": payload["oldbalanceOrg"],
        "oldbalanceDest": payload["oldbalanceDest"],
        "balanceDrainRatio": payload["amount"] / (payload["oldbalanceOrg"] + 1.0),
        "isTransfer": float(payload["type"] == "TRANSFER"),
    }
    if profile == "wire_pre":
        return np.array([values[name] for name in PRE_FEATURES])
    from paysim_engine import paysim_engine
    return paysim_engine.compute_features(**{key: payload[key] for key in
        ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]}, tx_type=payload["type"])


def artifact_root():
    return Path(os.environ.get("FRAUD_ARTIFACT_DIR", ARTIFACT_DIR))


@lru_cache(maxsize=16)
def _load_bundle(path, mtime_ns):
    import joblib
    return joblib.load(path)


def get_bundle(profile):
    """Read the locally selected chronological artifact, or an explicit legacy fallback."""
    pointer = artifact_root() / profile / "active.json"
    if pointer.exists():
        target = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["bundle"]
        from fraud_ops.workflow import apply_workflow_policy
        return apply_workflow_policy(_load_bundle(str(target), target.stat().st_mtime_ns))
    if profile == "wire_pre":
        raise FileNotFoundError("No pre-transaction model. Run python scripts/evaluate_operations.py --profiles wire_pre")
    return legacy_bundle(profile)


@lru_cache(maxsize=2)
def legacy_bundle(profile):
    if profile == "card":
        from models_engine import FraudIntelligenceEngine
        engine = FraudIntelligenceEngine()
    else:
        from paysim_engine import PaySimFraudEngine
        engine = PaySimFraudEngine()
    engine.load()
    return ModelBundle(engine, ENSEMBLE, profile, list(engine.FEATURE_NAMES), engine.scaler_mean,
                       DEFAULT_POLICIES[profile], {"model_version": engine.model_version,
                       "evaluation_status": "legacy_weights_not_validated_by_new_protocol"})
