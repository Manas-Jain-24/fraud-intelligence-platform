"""Replay full PaySim histories, select on validation, then measure a held-out queue."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

from fraud_ops.history import RulePolicy, rule_masks
from fraud_ops.evaluation import operational_metrics


@dataclass(frozen=True)
class ReviewCapacity:
    analysts: int = 2
    hours_per_day: float = 8.
    minutes_per_case: float = 8.

    def __post_init__(self):
        if self.analysts < 1 or not 0 < self.hours_per_day <= 24 or self.minutes_per_case <= 0:
            raise ValueError("Choose positive analyst count, working hours and case duration")

    @property
    def cases_per_day(self):
        return int(self.analysts * self.hours_per_day * 60 / self.minutes_per_case)


def encode_accounts(values):
    """PaySim uses C/M plus a nonnegative 31-bit identifier; preserve the prefix."""
    if not values.str.match(r"^[CM][0-9]+$").all():
        raise ValueError("Unexpected PaySim account identifier")
    numbers = values.str.slice(1).to_numpy(dtype=np.uint64)
    if (numbers > 2**31-1).any():
        raise ValueError("Account ID is outside PaySim's supported range")
    return (numbers * 2 + values.str.startswith("M").to_numpy(dtype=np.uint64)).astype(np.uint32)


def read_full_stream(source, chunksize=250000):
    columns = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "oldbalanceDest", "isFraud"]
    chunks = []
    for raw in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        frame = pd.DataFrame({
            "step": raw.step.to_numpy(dtype=np.int16), "amount": raw.amount.to_numpy(dtype=float),
            "sender": encode_accounts(raw.nameOrig), "recipient": encode_accounts(raw.nameDest),
            "oldbalanceOrg": raw.oldbalanceOrg.to_numpy(dtype=float),
            "oldbalanceDest": raw.oldbalanceDest.to_numpy(dtype=float),
            "is_transfer": (raw.type == "TRANSFER").to_numpy(), "is_cash_out": (raw.type == "CASH_OUT").to_numpy(),
            "actual_fraud": raw.isFraud.to_numpy(dtype=np.int8),
        })
        chunks.append(frame)
    frame = pd.concat(chunks, ignore_index=True)
    if (frame.step.diff().dropna() < 0).any():
        raise ValueError("Replay requires nondecreasing source steps; do not invent within-step ordering")
    return frame


def _group_history(keys, amounts, query, distinct=None):
    """Exact one-step lookups, keeping full-source aggregation separate from labels."""
    order = np.argsort(keys, kind="stable") if distinct is None else np.lexsort((distinct, keys))
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    unique = sorted_keys[starts]
    if distinct is None:
        counts = np.diff(np.r_[starts, len(keys)])
    else:
        sorted_distinct = distinct[order]
        changes = np.r_[True, (sorted_keys[1:] != sorted_keys[:-1]) | (sorted_distinct[1:] != sorted_distinct[:-1])]
        counts = np.add.reduceat(changes.astype(np.int32), starts)
    totals = np.add.reduceat(amounts[order], starts)
    locations = np.searchsorted(unique, query)
    clipped = np.minimum(locations, len(unique)-1)
    found = (locations < len(unique)) & (unique[clipped] == query)
    return np.where(found, counts[clipped], 0), np.where(found, totals[clipped], 0.)


def full_history_features(frame, target_ids):
    """Exact features at recorded hour boundaries, using all original source rows.

    At hourly precision, strictly earlier timestamps are at least an hour apart.
    Thus sub-hour windows are unobservable; no intra-hour order is manufactured.
    """
    steps = frame.step.to_numpy(dtype=np.uint64)
    senders, recipients = frame.sender.to_numpy(dtype=np.uint64), frame.recipient.to_numpy(dtype=np.uint64)
    amounts = frame.amount.to_numpy(dtype=float)
    base = int(steps.max()) + 2
    query_sender = senders[target_ids] * base + steps[target_ids] - 1
    sender_count, sender_amount = _group_history(senders * base + steps, amounts, query_sender)
    query_recipient = recipients[target_ids] * base + steps[target_ids] - 1
    recipient_count, recipient_amount = _group_history(recipients * base + steps, amounts, query_recipient, distinct=senders)
    pairs = (senders << np.uint64(32)) | recipients
    unique_pairs, first = np.unique(pairs, return_index=True)
    first_step = steps[first[np.searchsorted(unique_pairs, pairs[target_ids])]]
    return pd.DataFrame({
        "sender_prior_count_5m": np.zeros(len(target_ids), dtype=np.int32),
        "sender_prior_count_1h": sender_count, "sender_prior_amount_1h": sender_amount,
        "recipient_distinct_senders_1h": recipient_count, "recipient_prior_amount_1h": recipient_amount,
        "sender_received_amount_15m": np.zeros(len(target_ids)),
        "new_beneficiary_in_observed_history": first_step >= steps[target_ids],
    })


def routed_mask(table, threshold, rule_policy):
    flags = rule_masks(table, table.is_cash_out, table.amount, rule_policy)
    rule_alerts = np.zeros(len(table), dtype=bool)
    for flag in flags.values():
        rule_alerts |= np.asarray(flag)
    rule_alerts &= table.score.to_numpy() >= rule_policy.minimum_model_score
    model_alerts = table.score.to_numpy() >= threshold
    return model_alerts | rule_alerts, model_alerts, rule_alerts


def day_capacity(steps, capacity, start_step=None, end_step=None):
    start, end = int(min(steps)) if start_step is None else int(start_step), int(max(steps)) if end_step is None else int(end_step)
    days = np.arange((start-1)//24, (end-1)//24+1)
    hours = np.array([max(0, min(end, (day+1)*24) - max(start, day*24+1) + 1) for day in days])
    return days, np.floor(capacity.cases_per_day * hours / 24).astype(int), hours


def daily_workload(steps, alerts, capacity, start_step=None, end_step=None):
    days, limits, hours = day_capacity(steps, capacity, start_step, end_step)
    counts = np.bincount((np.asarray(steps)-1)//24-days[0], weights=np.asarray(alerts, dtype=int), minlength=len(days)).astype(int)
    backlog, rows = 0, []
    for day, limit, hours_in_day, count in zip(days, limits, hours, counts):
        available = backlog + int(count)
        processed = min(available, int(limit))
        backlog = available - processed
        rows.append({"day_offset": int(day), "observed_hours": int(hours_in_day), "new_cases": int(count),
                     "capacity": int(limit), "processed": processed, "backlog_end": backlog,
                     "review_minutes_requested": float(count * capacity.minutes_per_case)})
    return pd.DataFrame(rows)


def queue_metrics(table, threshold, rule_policy, capacity, costs, bounds):
    alerts, model_alerts, rule_alerts = routed_mask(table, threshold, rule_policy)
    metrics = operational_metrics(table.actual_fraud, alerts.astype(float), table.amount, .5, **costs)
    for key in ["roc_auc", "average_precision", "brier_score", "log_loss", "threshold"]:
        metrics.pop(key)
    workload = daily_workload(table.step.to_numpy(), alerts, capacity, *bounds)
    metrics.update({
        "model_threshold": float(threshold), "cases": int(alerts.sum()),
        "model_alerts": int(model_alerts.sum()), "rule_alerts": int(rule_alerts.sum()),
        "rule_only_cases": int((rule_alerts & ~model_alerts).sum()),
        "incremental_frauds_from_rules": int((rule_alerts & ~model_alerts & table.actual_fraud.astype(bool)).sum()),
        "overlapping_alerts_merged": int((rule_alerts & model_alerts).sum()),
        "duplicate_cases_created": 0, "peak_new_cases_per_day": int(workload.new_cases.max()),
        "days_over_capacity": int((workload.new_cases > workload.capacity).sum()),
        "peak_backlog": int(workload.backlog_end.max()), "final_backlog": int(workload.backlog_end.iloc[-1]),
    })
    return metrics, workload


class ThresholdSearch:
    """Enumerate every distinct validation score cutoff without quadratic masks."""
    def __init__(self, table, capacity, costs, bounds):
        self.table, self.costs = table, costs
        self.unique, self.groups = np.unique(table.score.to_numpy(), return_inverse=True)
        self.thresholds = np.r_[1., self.unique[::-1]]
        self.days, self.limits, _ = day_capacity(table.step.to_numpy(), capacity, *bounds)
        self.day_ids = (table.step.to_numpy()-1)//24-self.days[0]
        self.y, self.amount = table.actual_fraud.to_numpy(dtype=bool), table.amount.to_numpy()

    def best(self, policy):
        _, _, rules = routed_mask(self.table, 1., policy)
        remaining = ~rules
        n = len(self.unique)
        def prefix(weights):
            return np.r_[0., np.cumsum(np.bincount(self.groups, weights=weights, minlength=n)[::-1])]
        alerts = rules.sum() + prefix(remaining.astype(int))
        true_alerts = (rules & self.y).sum() + prefix(remaining & self.y)
        false_alerts = alerts - true_alerts
        captured = np.sum(self.amount[rules & self.y]) + prefix(self.amount * remaining * self.y)
        missed = np.maximum(0., np.sum(self.amount[self.y]) - captured)
        loss = missed * self.costs["loss_fraction"] + alerts * self.costs["review_cost"] + false_alerts * self.costs["false_decline_cost"]
        combined = self.groups * len(self.days) + self.day_ids
        per_score_day = np.bincount(combined, weights=remaining, minlength=n*len(self.days)).reshape(n, len(self.days))
        cumulative_day = np.vstack([np.zeros(len(self.days)), np.cumsum(per_score_day[::-1], axis=0)])
        cumulative_day += np.bincount(self.day_ids, weights=rules, minlength=len(self.days))
        feasible = (cumulative_day <= self.limits).all(axis=1)
        if not feasible.any():
            return None
        indexes = np.flatnonzero(feasible)
        best = min(indexes, key=lambda i: (loss[i], false_alerts[i], alerts[i]))
        return {"threshold": float(self.thresholds[best]), "validation_expected_loss": float(loss[best]),
                "validation_cases": int(alerts[best]), "validation_false_alerts": int(false_alerts[best]),
                "rule_policy": policy.to_dict()}


def select_capacity_policy(validation, capacity, costs, bounds):
    search = ThresholdSearch(validation, capacity, costs, bounds)
    policies = [RulePolicy(enabled_rules=())]
    # Sub-hour rules are intentionally excluded from a dataset with hourly timestamps.
    for fan_in in [3, 10, 25, 50, 100]:
        for gate in [0., .001, .01, .05, .1, .25, .5]:
            policies.append(RulePolicy(recipient_prior_senders=fan_in, minimum_model_score=gate,
                enabled_rules=("RECIPIENT_FAN_IN", "NEW_BENEFICIARY_BURST")))
    candidates = [search.best(policy) for policy in policies]
    feasible = [candidate for candidate in candidates if candidate is not None]
    selected = min(feasible, key=lambda c: (c["validation_expected_loss"], c["validation_false_alerts"],
                                          len(c["rule_policy"]["enabled_rules"]), c["validation_cases"]))
    return selected, candidates[0], feasible


def run_full_replay(source, bundle_path, output_dir, capacity, *, activate=False):
    import joblib
    from threadpoolctl import threadpool_limits
    source, bundle_path, out = Path(source), Path(bundle_path), Path(output_dir)
    bundle = joblib.load(bundle_path)
    if bundle.profile != "wire_pre" or bundle.metadata["protocol"] != "chronological":
        raise ValueError("Use the chronological pre-transfer artifact")
    digest = sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    expected = bundle.metadata["source"]["manifest"]["source_sha256"]
    if digest.hexdigest() != expected:
        raise ValueError("Original source checksum differs from the model's recorded provenance")
    print("Reading the complete source stream...", flush=True)
    full = read_full_stream(source)
    partitions = bundle.metadata["partition_summary"]
    val_bounds = (int(partitions["validation"]["time_min"]), int(partitions["validation"]["time_max"]))
    test_bounds = (int(partitions["test"]["time_min"]), int(partitions["test"]["time_max"]))
    target_ids = np.flatnonzero(full.step.between(val_bounds[0], test_bounds[1]).to_numpy())
    print(f"Reconstructing complete history for {len(target_ids):,} validation/test events...", flush=True)
    features = full_history_features(full, target_ids)
    target = full.iloc[target_ids].reset_index(drop=True)
    source_rows = len(full)
    del full
    model_inputs = np.column_stack([target.amount, target.oldbalanceOrg, target.oldbalanceDest,
                                    target.amount/(target.oldbalanceOrg+1.), target.is_transfer])
    print("Scoring the fixed pre-transfer model...", flush=True)
    with threadpool_limits(limits=1):
        scores = np.concatenate([bundle.predict(model_inputs[start:start+100000]) for start in range(0, len(target), 100000)])
    table = pd.concat([target[["step", "amount", "is_cash_out", "actual_fraud"]], features], axis=1)
    table["score"], table["source_row"] = scores, target_ids
    del model_inputs, features, target
    validation = table.loc[table.step.between(*val_bounds)].reset_index(drop=True)
    test = table.loc[table.step.between(*test_bounds)].reset_index(drop=True)
    del table
    costs = bundle.metadata["cost_assumptions"]
    print(f"Selecting policy on {len(validation):,} validation events; test labels are not used for selection...", flush=True)
    selected, model_only, candidates = select_capacity_policy(validation, capacity, costs, val_bounds)
    policy_version = "workflow-" + sha256(json.dumps({"selection": selected, "model": bundle.version,
        "capacity": capacity.__dict__, "source": digest.hexdigest()}, sort_keys=True).encode()).hexdigest()[:16]
    strategies = {
        "model_only_current": (bundle.policy.flag_threshold, RulePolicy(enabled_rules=())),
        "model_plus_default_rules": (bundle.policy.flag_threshold, RulePolicy()),
        "capacity_tuned_model_only": (model_only["threshold"], RulePolicy(**model_only["rule_policy"])),
        "capacity_tuned_combined": (selected["threshold"], RulePolicy(**selected["rule_policy"])),
    }
    out.mkdir(parents=True, exist_ok=True)
    summary, daily = [], []
    for partition, frame, bounds in [("validation", validation, val_bounds), ("test", test, test_bounds)]:
        for strategy, (threshold, policy) in strategies.items():
            metrics, workload = queue_metrics(frame, threshold, policy, capacity, costs, bounds)
            summary.append({"partition": partition, "strategy": strategy, **metrics})
            daily.append(workload.assign(partition=partition, strategy=strategy))
    pd.DataFrame(summary).to_csv(out / "queue_comparison.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(out / "daily_workload.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in candidate.items() if k != "rule_policy"},
                   "rule_policy": json.dumps(candidate["rule_policy"], sort_keys=True)} for candidate in candidates]).to_csv(out / "validation_policy_search.csv", index=False)
    test_alerts, test_ml, test_rules = routed_mask(test, selected["threshold"], RulePolicy(**selected["rule_policy"]))
    test.assign(alert=test_alerts, model_alert=test_ml, rule_alert=test_rules).to_csv(out / "test_workflow_predictions.csv", index=False)
    print("Comparing staffing scenarios using validation-selected model cutoffs...", flush=True)
    sensitivity = []
    for analysts in [1, 2, 4, 8, 16, 24]:
        scenario = ReviewCapacity(analysts, capacity.hours_per_day, capacity.minutes_per_case)
        candidate = ThresholdSearch(validation, scenario, costs, val_bounds).best(RulePolicy(enabled_rules=()))
        for partition, frame, bounds in [("validation", validation, val_bounds), ("test", test, test_bounds)]:
            metrics, _ = queue_metrics(frame, candidate["threshold"], RulePolicy(enabled_rules=()), scenario, costs, bounds)
            sensitivity.append({"partition": partition, "analysts": analysts, "cases_per_day": scenario.cases_per_day, **metrics})
    pd.DataFrame(sensitivity).to_csv(out / "capacity_sensitivity.csv", index=False)
    report = {
        "model_version": bundle.version, "workflow_policy_version": policy_version, "source_file": source.name,
        "source_sha256": digest.hexdigest(), "source_rows": source_rows,
        "history_scope": "complete original PaySim stream; strictly earlier recorded steps only",
        "time_resolution_seconds": 3600, "unobservable_rules": ["TRANSFER_VELOCITY", "RAPID_CASH_OUT"],
        "validation_step_bounds": val_bounds, "test_step_bounds": test_bounds,
        "capacity": {**capacity.__dict__, "cases_per_day": capacity.cases_per_day,
                     "partial_day_rule": "capacity prorated by hours in the evaluation window"},
        "cost_assumptions": costs, "selection": selected, "feasible_candidate_count": len(candidates),
        "selection_rule": "minimum validation expected loss subject to each day's capacity; no test tuning",
        "comparison": summary,
        "capacity_sensitivity": sensitivity,
        "case_deduplication": "One case per unique source row after union of model and rule alerts; repeated-ingestion idempotence is tested separately.",
        "limitations": ["PaySim is simulated data; outcomes do not establish live fraud performance.",
                        "Hourly timestamps cannot establish sub-hour ordering or short-window fraud patterns.",
                        "Daily capacity and case duration are explicit scenarios, not measured investigator productivity.",
                        "Full-population holdouts differ from the earlier uniform-sample model evaluation.",
                        "Capacity on validation is not a guarantee on later periods; test overload remains visible."],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if activate:
        from fraud_ops.workflow import policy_path
        config = {"model_version": bundle.version, "workflow_policy_version": policy_version,
                  "decision_policy": {"flag_threshold": selected["threshold"], "block_threshold": selected["threshold"], "version": policy_version},
                  "rule_policy": selected["rule_policy"], "capacity": report["capacity"],
                  "selection": report["selection_rule"],
                  "workflow_test_metrics": next(row for row in summary if row["partition"] == "test" and row["strategy"] == "capacity_tuned_combined")}
        path = policy_path("wire_pre")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return report
