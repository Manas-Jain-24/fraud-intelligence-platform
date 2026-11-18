"""Capacity-aware policy overrides bound to an exact evaluated model version."""
from dataclasses import replace
import json
import os
from pathlib import Path
from fraud_ops.history import RulePolicy
from fraud_ops.policy import DecisionPolicy


def policy_path(profile):
    root = Path(os.environ.get("FRAUD_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "data/artifacts"))
    return root / "workflow" / f"{profile}_policy.json"


def selected_workflow_policy(profile, model_version):
    path = policy_path(profile)
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["model_version"] != model_version:
        return None
    return config


def active_rule_policy(profile, model_version):
    config = selected_workflow_policy(profile, model_version)
    return RulePolicy(**config["rule_policy"]) if config else RulePolicy()


def apply_workflow_policy(bundle):
    config = selected_workflow_policy(bundle.profile, bundle.version)
    if not config:
        return bundle
    return replace(bundle, policy=DecisionPolicy(**config["decision_policy"]),
                   metadata={**bundle.metadata, "workflow_policy_version": config["workflow_policy_version"],
                             "workflow_rule_policy": config["rule_policy"],
                             "base_model_evaluation_policy": bundle.policy.to_dict(),
                             "workflow_test_metrics": config.get("workflow_test_metrics"),
                             "evaluation_scope": "Original model metrics use the base cutoff; workflow metrics apply to causal ingestion and case routing."})


def activate_rules_from_report(report, bundle):
    """Choose between default/disabled rules on VALIDATION at the existing model cutoff.

    Staffing scenarios never silently replace the model cutoff. This narrower
    activation removes rules only when their validation cost does not improve.
    """
    from hashlib import sha256
    if report["model_version"] != bundle.version:
        raise ValueError("Workflow report does not match the selected model weights")
    validation = {row["strategy"]: row for row in report["comparison"] if row["partition"] == "validation"}
    baseline, combined = validation["model_only_current"], validation["model_plus_default_rules"]
    if baseline["model_threshold"] != bundle.policy.flag_threshold:
        raise ValueError("Workflow report does not match the model's original cutoff")
    strategy = "model_only_current" if baseline["expected_loss"] <= combined["expected_loss"] else "model_plus_default_rules"
    rules = RulePolicy(enabled_rules=()) if strategy == "model_only_current" else RulePolicy()
    fingerprint = {"model_version": bundle.version, "rule_policy": rules.to_dict(),
                   "source_sha256": report["source_sha256"], "threshold": bundle.policy.flag_threshold,
                   "scope": "rules_only_at_existing_model_cutoff"}
    version = "workflow-rules-" + sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    config = {**fingerprint, "workflow_policy_version": version,
              "decision_policy": {**bundle.policy.to_dict(), "version": version},
              "validation_selected_strategy": strategy,
              "selection": "minimum validation expected loss between default and disabled rules at the existing cutoff; no test selection",
              "workflow_test_metrics": next(row for row in report["comparison"] if row["partition"] == "test" and row["strategy"] == strategy)}
    path = policy_path(bundle.profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config
