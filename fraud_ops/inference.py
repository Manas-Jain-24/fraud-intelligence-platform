"""Shared batch inference and score explanations for the saved NumPy engines."""
from hashlib import sha256
from pathlib import Path
import numpy as np

BASELINE = "Baseline Logistic Regression (NumPy)"
COST = "Cost-Sensitive LogReg (NumPy)"
MLP = "Multi-Layer Perceptron (NumPy MLP)"
GBDT = "Gradient Boosted Trees (NumPy GBDT)"
ANOMALY = "Unsupervised Isolation Profiler (NumPy)"
ENSEMBLE = "Soft-Voting Ensemble (Scratch Champion)"
CARD_WEIGHTS = {MLP: .35, GBDT: .30, COST: .25, ANOMALY: .10}
WIRE_WEIGHTS = {MLP: .35, GBDT: .35, COST: .20, ANOMALY: .10}


def artifact_version(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()[:16]


def explain_score(predict, raw, reference, feature_names, model_name, labels=None):
    """Ordered reference replacement. Contributions telescope to score - baseline.

    This is a path-dependent decomposition of the actual scored model, not SHAP,
    a causal explanation, or an assertion about the meaning of anonymized PCA.
    """
    raw = np.asarray(raw, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    path = np.tile(reference, (len(raw) + 1, 1))
    for i in range(len(raw)):
        path[i + 1:, i] = raw[i]
    scores = np.asarray(predict(path)).ravel()
    impacts = [
        {"feature": name, "label": (labels or {}).get(name, name), "raw_value": float(value),
         "impact": float(impact), "direction": "Fraud Driver" if impact > 0 else "Safety Indicator"}
        for name, value, impact in zip(feature_names, raw, np.diff(scores))
    ]
    impacts.sort(key=lambda row: abs(row["impact"]), reverse=True)
    drivers = [row for row in impacts if row["impact"] > 0][:5]
    safeties = [row for row in impacts if row["impact"] < 0][:5]
    narrative = f"{model_name}: score {scores[-1]:.4f}, reference score {scores[0]:.4f}. "
    if drivers:
        narrative += f"Largest positive contribution along this feature order: {drivers[0]['label']}. "
    narrative += "Contributions depend on feature order and reference; they describe the model, not the cause of fraud."
    return {
        "top_drivers": drivers, "top_safeties": safeties, "all_impacts": impacts,
        "narrative": narrative, "bias": float(scores[0]), "baseline_score": float(scores[0]),
        "explained_score": float(scores[-1]), "explained_model": model_name,
        "attribution_method": "ordered_reference_replacement", "impact_units": "risk_score_difference",
        "feature_order": list(feature_names), "reference": reference.tolist(),
    }


class ScratchInferenceMixin:
    def predict_batch(self, x_raw):
        if not self.is_loaded:
            self.load()
        x = np.asarray(x_raw, dtype=float)
        x = np.atleast_2d(x)
        if x.shape[1] != len(self.FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Input must contain one finite value per engine feature")
        x = (x - self.scaler_mean) / self.scaler_std
        result = {name: np.asarray(model.anomaly_score(x) if name == ANOMALY else model.predict_proba(x)).ravel()
                  for name, model in self.models.items()}
        result[ENSEMBLE] = np.clip(sum(weight * result[name] for name, weight in self.ENSEMBLE_WEIGHTS.items()), 0, 1)
        return result

    def predict_all(self, x_raw):
        if np.atleast_2d(x_raw).shape[0] != 1:
            raise ValueError("predict_all accepts one transaction; use predict_batch for batches")
        return {name: float(scores[0]) for name, scores in self.predict_batch(x_raw).items()}

    def explain_transaction(self, x_raw, model_name=ENSEMBLE):
        if not self.is_loaded:
            self.load()
        return explain_score(lambda batch: self.predict_batch(batch)[model_name], x_raw, self.scaler_mean,
                             self.FEATURE_NAMES, model_name, getattr(self, "FEATURE_LABELS", None))
