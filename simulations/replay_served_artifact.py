"""Replay untouched holdout rows through the exact selected artifact.

Optional time-block bootstrap describes holdout uncertainty for a FIXED model.
It does not claim new independent training runs or assess training uncertainty.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraud_ops.evaluation import operational_metrics


def replay(bundle_path, source, bootstrap_runs=200, seed=42):
    import joblib
    from threadpoolctl import threadpool_limits
    bundle_path, source = Path(bundle_path), Path(source)
    bundle = joblib.load(bundle_path)
    source_hash = sha256(source.read_bytes()).hexdigest()
    if source_hash != bundle.metadata["source"]["sha256"]:
        raise ValueError("Source data does not match the evaluated artifact's dataset hash")
    frame = pd.read_csv(source)
    membership = pd.read_csv(bundle_path.parent / "split_membership.csv")
    test_ids = membership.loc[membership.partition == "test", "source_row"].to_numpy(dtype=int)
    test = frame.iloc[test_ids]
    profile = bundle.profile
    target, amount, clock = ("Class", "Amount", "Time") if profile == "card" else ("isFraud", "amount", "step")
    with threadpool_limits(limits=1):
        scores = bundle.predict(test[bundle.feature_names].to_numpy(dtype=float))
    costs = bundle.metadata["cost_assumptions"]
    y, amounts = test[target].to_numpy(), test[amount].to_numpy()
    measured = operational_metrics(y, scores, amounts, bundle.policy.block_threshold, **costs)
    for key in ["tp", "fp", "fn", "tn", "expected_loss", "brier_score"]:
        if not np.isclose(measured[key], bundle.metadata["test"][key], rtol=1e-10, atol=1e-10):
            raise AssertionError(f"Artifact replay differs for {key}")
    # Card Time is in seconds; PaySim step is already in hours.
    blocks = np.floor(test[clock].to_numpy() / (3600 if profile == "card" else 1)).astype(int)
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng, rows = np.random.default_rng(seed), []
    for run in range(bootstrap_runs):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        rows.append({"run": run, **operational_metrics(y[idx], scores[idx], amounts[idx], bundle.policy.block_threshold, **costs)})
    result = {"model_version": bundle.version, "source_sha256": source_hash, "replay_matches_report": True,
              "test": measured, "bootstrap_runs": bootstrap_runs, "bootstrap_seed": seed,
              "bootstrap_unit": "observed one-hour time blocks, with replacement", "distinct_time_blocks": len(groups),
              "limitation": "Conditional uncertainty for this fitted model and observed holdout; no new independent observations or training fits."}
    if rows:
        metrics = pd.DataFrame(rows)
        metrics.to_csv(bundle_path.parent / "bootstrap_metrics.csv", index=False)
        result["bootstrap_95_percentile_intervals"] = {key: [float(metrics[key].quantile(.025)), float(metrics[key].quantile(.975))]
            for key in ["precision", "recall", "false_alerts_per_10000", "expected_loss_per_10000"]}
    (bundle_path.parent / "replay_report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["card", "wire", "wire_pre"], default="card")
    parser.add_argument("--protocol", choices=["chronological", "stratified"], default="chronological")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--bootstrap-runs", type=int, default=200)
    args = parser.parse_args()
    source = args.input or ROOT / "data" / ("creditcard.csv" if args.profile == "card" else "paysim_natural.csv")
    print(json.dumps(replay(ROOT / "data/artifacts" / args.profile / args.protocol / "bundle.joblib", source, args.bootstrap_runs), indent=2))
