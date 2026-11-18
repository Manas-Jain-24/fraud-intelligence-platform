"""Compare the complete model-plus-rules queue and tune against daily review capacity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraud_ops.workflow_replay import ReviewCapacity, run_full_replay
from scripts.prepare_paysim_natural import DEFAULT_SOURCE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data/artifacts/wire_pre/chronological/bundle.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "data/artifacts/workflow")
    parser.add_argument("--analysts", type=int, default=2)
    parser.add_argument("--hours-per-day", type=float, default=8.)
    parser.add_argument("--minutes-per-case", type=float, default=8.)
    parser.add_argument("--activate", action="store_true", help="Use the validation-selected policy locally for this exact model")
    parser.add_argument("--activate-rules-only", action="store_true", help="Use an existing report to select rules at the original model cutoff; no replay or cutoff change")
    args = parser.parse_args()
    if args.activate_rules_only:
        import joblib
        from fraud_ops.workflow import activate_rules_from_report
        if args.activate:
            parser.error("Choose --activate or --activate-rules-only, not both")
        report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
        config = activate_rules_from_report(report, joblib.load(args.bundle))
        print(json.dumps(config, indent=2))
        sys.exit(0)
    report = run_full_replay(args.input, args.bundle, args.output,
        ReviewCapacity(args.analysts, args.hours_per_day, args.minutes_per_case), activate=args.activate)
    print(json.dumps({"version": report["workflow_policy_version"], "selection": report["selection"],
                      "comparison": report["comparison"]}, indent=2), flush=True)
