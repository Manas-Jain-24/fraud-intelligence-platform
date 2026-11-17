"""Evaluate and select local operational models; never tune on the test partition."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from fraud_ops.evaluation import evaluate_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=["card", "wire", "wire_pre"], default=["card", "wire", "wire_pre"])
    parser.add_argument("--protocols", nargs="+", choices=["chronological", "stratified"], default=["chronological", "stratified"])
    parser.add_argument("--card-input", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--wire-input", type=Path, default=ROOT / "data/paysim_natural.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/artifacts")
    parser.add_argument("--max-alert-rate", type=float, default=.02)
    parser.add_argument("--review-cost", type=float, default=2.5)
    parser.add_argument("--false-decline-cost", type=float, default=5.)
    parser.add_argument("--loss-fraction", type=float, default=1.)
    parser.add_argument("--no-activate", action="store_true", help="Write reports without changing local serving selection")
    args = parser.parse_args()
    if not 0 < args.max_alert_rate <= 1 or min(args.review_cost, args.false_decline_cost, args.loss_fraction) < 0:
        parser.error("Alert rate must be within (0, 1]; costs must be nonnegative")
    summary = []
    cached = {}
    for profile in args.profiles:
        source = args.card_input if profile == "card" else args.wire_input
        if not source.exists():
            parser.error(f"Missing {source}. For natural PaySim, run python scripts/prepare_paysim_natural.py first.")
        if source not in cached:
            frame = pd.read_csv(source)
            provenance = {"file": source.name, "sha256": sha256(source.read_bytes()).hexdigest(),
                          "distribution": "original card dataset" if profile == "card" else "see sampling manifest"}
            manifest = source.with_suffix(".manifest.json")
            if manifest.exists():
                provenance["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
            elif profile != "card":
                provenance["distribution"] = "unverified prevalence; enriched curated benchmark is not natural"
            cached[source] = frame, provenance
        frame, provenance = cached[source]
        for protocol in args.protocols:
            print(f"Evaluating {profile} / {protocol}: {len(frame):,} rows", flush=True)
            report = evaluate_profile(frame, profile, protocol, args.output, provenance, args.max_alert_rate,
                {"review_cost": args.review_cost, "false_decline_cost": args.false_decline_cost, "loss_fraction": args.loss_fraction})
            row = {"profile": profile, "protocol": protocol, "model": report["model_name"],
                   "calibrated": report["calibrated"], "version": report["model_version"], **report["test"]}
            summary.append(row)
            print(json.dumps(row), flush=True)
            if protocol == "chronological" and not args.no_activate:
                pointer = args.output / profile / "active.json"
                pointer.write_text(json.dumps({"bundle": "chronological/bundle.joblib", "model_version": report["model_version"]}, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(args.output / "evaluation_summary.csv", index=False)


if __name__ == "__main__":
    main()
