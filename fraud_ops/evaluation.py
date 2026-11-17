"""Four-part evaluation: fit -> calibration -> policy/model selection -> untouched test."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_ops.artifacts import ModelBundle, PRE_FEATURES, logit_features
from fraud_ops.inference import BASELINE, COST, MLP, GBDT, ANOMALY, ENSEMBLE
from fraud_ops.policy import DecisionPolicy


def split_indices(y, times, protocol, seed=42):
    """55/15/15/15; chronological boundaries never separate equal timestamps."""
    y, times = np.asarray(y), np.asarray(times)
    if protocol == "chronological":
        order = np.argsort(times, kind="stable")
        sorted_times = times[order]
        cuts = [int(np.searchsorted(sorted_times, sorted_times[min(int(len(y) * f), len(y)-1)], side="left"))
                for f in (.55, .70, .85)]
        parts = np.split(order, cuts)
    elif protocol == "stratified":
        train, rest = train_test_split(np.arange(len(y)), test_size=.45, stratify=y, random_state=seed)
        cal, rest = train_test_split(rest, test_size=2/3, stratify=y[rest], random_state=seed + 1)
        val, test = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=seed + 2)
        parts = [train, cal, val, test]
    else:
        raise ValueError("Unknown split protocol")
    for name, ids in zip(["train", "calibration", "validation", "test"], parts):
        if not len(ids) or len(np.unique(y[ids])) != 2:
            raise ValueError(f"{protocol} {name} must contain both classes; provide more data")
    return dict(zip(["train", "calibration", "validation", "test"], parts))


def operational_metrics(y, scores, amounts, threshold, review_cost=2.5, false_decline_cost=5., loss_fraction=1.):
    y = np.asarray(y, dtype=bool)
    scores, amounts = np.asarray(scores), np.asarray(amounts)
    predicted = scores >= threshold
    tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
    fn, tn = int((~predicted & y).sum()), int((~predicted & ~y).sum())
    missed_loss = float(amounts[~predicted & y].sum() * loss_fraction)
    review = (tp + fp) * review_cost
    friction = fp * false_decline_cost
    return {
        "n": len(y), "frauds": int(y.sum()), "prevalence": float(y.mean()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "alerts_per_10000": float(predicted.mean() * 10000), "false_alerts_per_10000": fp / len(y) * 10000,
        "alert_rate": float(predicted.mean()), "threshold": float(threshold),
        "missed_fraud_amount": missed_loss, "review_cost": review, "false_decline_cost": friction,
        "expected_loss": missed_loss + review + friction,
        "expected_loss_per_10000": (missed_loss + review + friction) / len(y) * 10000,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)) if y.any() else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, np.clip(scores, 1e-7, 1-1e-7), labels=[0, 1])),
    }


def choose_policy(y, scores, amounts, max_alert_rate=.02, **costs):
    # Sort once: each distinct threshold's confusion matrix/loss is available cumulatively.
    order = np.argsort(-scores, kind="stable")
    s, y_sorted, a = np.asarray(scores)[order], np.asarray(y)[order], np.asarray(amounts)[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
    counts = ends + 1
    tp = np.cumsum(y_sorted)[ends]
    fp = counts - tp
    missed = float(np.sum(a * y_sorted)) - np.cumsum(a * y_sorted)[ends]
    losses = missed * costs.get("loss_fraction", 1.) + counts * costs.get("review_cost", 2.5) + fp * costs.get("false_decline_cost", 5.)
    allowed = counts / len(s) <= max_alert_rate
    thresholds = np.r_[1., s[ends][allowed]]
    losses = np.r_[float(np.sum(a * y_sorted)) * costs.get("loss_fraction", 1.), losses[allowed]]
    threshold = float(thresholds[int(np.argmin(losses))])
    # One threshold drives routing and evaluation, avoiding a second unmeasured alert volume.
    return DecisionPolicy(threshold, threshold, "validation-cost-policy-v1")


def reliability(y, scores):
    edges = np.linspace(0, 1, 11)
    bins = np.minimum(np.searchsorted(edges, scores, side="right") - 1, 9)
    rows = []
    for i in range(10):
        mask = bins == i
        if mask.any():
            rows.append({"bin": i, "count": int(mask.sum()), "mean_score": float(np.mean(scores[mask])),
                         "fraud_fraction": float(np.mean(np.asarray(y)[mask]))})
    return rows


def fit_scratch(profile, X, y, max_per_class=4000):
    # The exact engine classes and ensemble weights used by serving. Resampling is TRAIN ONLY.
    if profile == "card":
        import models_engine as module
        engine = module.FraudIntelligenceEngine()
    else:
        import paysim_engine as module
        engine = module.PaySimFraudEngine()
    rng = np.random.default_rng(42)
    n = min(int((y == 1).sum()), int((y == 0).sum()), max_per_class)
    idx = np.r_[rng.choice(np.flatnonzero(y == 1), n, replace=False), rng.choice(np.flatnonzero(y == 0), n, replace=False)]
    rng.shuffle(idx)
    engine.scaler_mean = X.mean(axis=0)
    engine.scaler_std = X.std(axis=0)
    engine.scaler_std[engine.scaler_std == 0] = 1
    normalized = (X[idx] - engine.scaler_mean) / engine.scaler_std
    sampled_y = y[idx]
    np.random.seed(42)
    engine.models = {
        BASELINE: module.ScratchLogReg(n_iter=800).fit(normalized, sampled_y),
        COST: module.ScratchCostSensitiveLogReg(n_iter=800).fit(normalized, sampled_y),
        MLP: module.ScratchMLP(epochs=300).fit(normalized, sampled_y),
        GBDT: module.ScratchStumpGBDT(n_estimators=30).fit(normalized, sampled_y),
        ANOMALY: module.ScratchAnomalyDetector().fit(normalized[sampled_y == 0]),
    }
    engine.is_loaded = True
    engine.model_version = "fitted-in-evaluation"
    return engine


def evaluate_profile(frame, profile, protocol, output_dir, provenance, max_alert_rate=.02, costs=None):
    import joblib
    from threadpoolctl import threadpool_limits
    costs = costs or {"review_cost": 2.5, "false_decline_cost": 5., "loss_fraction": 1.}
    if profile == "card":
        from models_engine import FraudIntelligenceEngine
        features, target, clock = list(FraudIntelligenceEngine.FEATURE_NAMES), "Class", "Time"
    else:
        from paysim_engine import PaySimFraudEngine
        features = PRE_FEATURES if profile == "wire_pre" else list(PaySimFraudEngine.FEATURE_NAMES)
        target, clock = "isFraud", "step"
    y, times = frame[target].to_numpy(dtype=int), frame[clock].to_numpy()
    X = frame[features].to_numpy(dtype=float)
    amounts = frame["Amount" if profile == "card" else "amount"].to_numpy(dtype=float)
    ids = split_indices(y, times, protocol)
    tr, ca, va, te = (ids[k] for k in ["train", "calibration", "validation", "test"])
    out = Path(output_dir) / profile / protocol
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    with threadpool_limits(limits=1):
        if profile != "wire_pre":
            engine = fit_scratch(profile, X[tr], y[tr])
            for model_name in [BASELINE, COST, MLP, GBDT, ENSEMBLE]:
                candidates.append((model_name, engine, model_name))
        for name, estimator in [
            ("Sklearn Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, class_weight="balanced", random_state=42))),
            ("Sklearn Histogram Gradient Boosting", HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=1., random_state=42)),
        ]:
            estimator.fit(X[tr], y[tr])
            candidates.append((name, estimator, ENSEMBLE))
        bundles, rows = [], []
        for name, estimator, scratch_name in candidates:
            bundle = ModelBundle(estimator, name, profile, features, X[tr].mean(axis=0), DecisionPolicy(.5, .5), {}, scratch_model_name=scratch_name)
            raw_cal = bundle.raw_scores(X[ca])
            calibrator = LogisticRegression(C=1., max_iter=500).fit(logit_features(raw_cal), y[ca])
            for calibrated in (False, True):
                variant = replace(bundle, calibrator=calibrator if calibrated else None)
                val_scores = variant.predict(X[va])
                policy = choose_policy(y[va], val_scores, amounts[va], max_alert_rate, **costs)
                variant = replace(variant, policy=policy)
                metrics = operational_metrics(y[va], val_scores, amounts[va], policy.block_threshold, **costs)
                # Calibration is an explicit selection dimension, not a promise that raw scores are probabilities.
                rows.append({"model": name, "calibrated": calibrated, **metrics})
                bundles.append(variant)
        best = min(range(len(rows)), key=lambda i: (rows[i]["expected_loss"], rows[i]["brier_score"], rows[i]["log_loss"]))
        selected = bundles[best]
        metadata = {
            "profile": profile, "protocol": protocol, "seed": 42, "features": features,
            "stage": selected.stage, "source": provenance, "cost_assumptions": costs,
            "max_validation_alert_rate": max_alert_rate,
            "selection": "minimum validation expected loss within alert budget; Brier/log loss break ties",
            "partition_summary": {k: {"n": len(v), "frauds": int(y[v].sum()), "prevalence": float(y[v].mean()),
                                         "time_min": float(times[v].min()), "time_max": float(times[v].max())} for k, v in ids.items()},
            "model_name": selected.model_name, "score_kind": selected.score_kind,
            "calibrated": selected.calibrator is not None, "policy": selected.policy.to_dict(),
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "model_fingerprint": joblib.hash((selected.model, selected.calibrator)),
            "limitations": ["Cost assumptions are illustrative, not observed financial savings.",
                             "A single chronological holdout does not establish live performance.",
                             "Behavioural rules are evaluated separately from model-only metrics."],
        }
        # Include exact source row membership; repeated random splits do not create new independent observations.
        pd.DataFrame({"source_row": np.concatenate(list(ids.values())),
                      "partition": np.repeat(list(ids), [len(v) for v in ids.values()])}).to_csv(out / "split_membership.csv", index=False)
        version_input = json.dumps(metadata, sort_keys=True) + repr([(r["model"], r["calibrated"], r["expected_loss"]) for r in rows])
        metadata["model_version"] = "eval-v2-" + sha256(version_input.encode()).hexdigest()[:16]
        selected.metadata = metadata
        joblib.dump(selected, out / "bundle.joblib", compress=3)
        loaded = joblib.load(out / "bundle.joblib")
        test_scores = loaded.predict(X[te])
        np.testing.assert_allclose(test_scores, selected.predict(X[te]), rtol=0, atol=0)
        metadata["test"] = operational_metrics(y[te], test_scores, amounts[te], loaded.policy.block_threshold, **costs)
        metadata["test_reliability"] = reliability(y[te], test_scores)
        # Compare before/after calibration on the same selected base model, without reselecting on test.
        metadata["test_raw_reliability"] = reliability(y[te], loaded.raw_scores(X[te]))
        metadata["test_raw_brier"] = float(brier_score_loss(y[te], loaded.raw_scores(X[te])))
        metadata["artifact_reload_verified"] = True
        # Save the report inside the bundle too, so API metadata refers to these exact measured weights.
        loaded.metadata = metadata
        joblib.dump(loaded, out / "bundle.joblib", compress=3)
    pd.DataFrame(rows).to_csv(out / "validation_comparison.csv", index=False)
    pd.DataFrame({"source_row": te, "time": times[te], "actual_fraud": y[te], "amount": amounts[te],
                  "score": test_scores, "alert": test_scores >= selected.policy.block_threshold}).to_csv(out / "test_predictions.csv", index=False)
    (out / "report.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata
