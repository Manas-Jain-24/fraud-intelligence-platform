"""Exact causal history and capacity search against independent small references."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import pytest

from fraud_ops.history import RulePolicy, behavioural_rules, history_features
from fraud_ops.policy import DecisionPolicy
from fraud_ops.workflow import activate_rules_from_report, active_rule_policy, apply_workflow_policy
from fraud_ops.workflow_replay import ReviewCapacity, ThresholdSearch, daily_workload, encode_accounts, full_history_features, queue_metrics, routed_mask, run_full_replay
from fraud_ops.artifacts import ModelBundle, PRE_FEATURES, legacy_bundle
from fraud_ops.store import InvestigationStore


def history_frame():
    return pd.DataFrame({"step": [1, 1, 1, 2, 2, 3], "sender": np.array([1, 2, 3, 1, 1, 1], dtype=np.uint32),
                         "recipient": np.array([8, 8, 8, 8, 9, 8], dtype=np.uint32), "amount": [10., 20., 30., 40., 50., 1000.]})


def test_complete_hourly_history_matches_scalar_history_and_excludes_same_hour():
    frame = history_frame()
    calculated = full_history_features(frame, np.arange(len(frame)))
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    transactions = [{"transaction_id": f"row-{i}", "event_time": (epoch+timedelta(hours=int(row.step))).isoformat(),
                     "sender_id": int(row.sender), "recipient_id": int(row.recipient), "rail": "wire",
                     "synthetic": False, "amount": float(row.amount)} for i, row in frame.iterrows()]
    for i, current in enumerate(transactions):
        reference = history_features(transactions, current)
        for column in calculated:
            assert calculated.iloc[i][column] == reference[column]
    assert calculated.iloc[1].recipient_distinct_senders_1h == 0
    assert calculated.iloc[3].recipient_distinct_senders_1h == 3
    assert calculated.iloc[5].new_beneficiary_in_observed_history == False
    assert calculated.iloc[5].sender_prior_count_1h == 2


def queue_frame():
    return pd.DataFrame({"step": [1, 1, 24, 25, 26, 48], "amount": [10., 1000., 20., 2000., 10., 10.],
        "actual_fraud": [0, 1, 0, 1, 0, 0], "score": [.1, .8, .5, .9, .6, .5], "is_cash_out": False,
        "sender_prior_count_5m": 0, "sender_prior_count_1h": 0, "sender_received_amount_15m": 0.,
        "recipient_distinct_senders_1h": [0, 0, 3, 4, 0, 0], "new_beneficiary_in_observed_history": False})


def test_vector_rule_routing_matches_scalar_routing_with_a_score_gate():
    frame = queue_frame()
    policy = RulePolicy(minimum_model_score=.55)
    _, _, fired = routed_mask(frame, .8, policy)
    for index, row in frame.iterrows():
        scalar = behavioural_rules(row.to_dict(), {"amount": row.amount, "transaction_type": "TRANSFER"}, policy, row.score)
        assert bool(fired[index]) == bool(scalar)
    assert fired[2] == False
    assert fired[3] == True


def test_capacity_search_matches_exhaustive_search_including_tied_scores():
    frame = queue_frame()
    capacity = ReviewCapacity(analysts=1, hours_per_day=1, minutes_per_case=30)  # 2/day
    costs = {"review_cost": 2.5, "false_decline_cost": 5., "loss_fraction": 1.}
    policy = RulePolicy()
    selected = ThresholdSearch(frame, capacity, costs, (1, 48)).best(policy)
    options = []
    for threshold in np.r_[1., np.unique(frame.score)[::-1]]:
        metrics, daily = queue_metrics(frame, threshold, policy, capacity, costs, (1, 48))
        if (daily.new_cases <= daily.capacity).all():
            options.append((metrics["expected_loss"], metrics["fp"], metrics["cases"], threshold))
    expected = min(options)
    assert selected["validation_expected_loss"] == pytest.approx(expected[0])
    assert selected["threshold"] == expected[3]


def test_partial_days_and_backlog_are_not_hidden_by_capacity_limits():
    capacity = ReviewCapacity(2, 8, 8)
    frame = daily_workload(np.array([13, 13, 25, 25, 25]), np.ones(5, dtype=bool), capacity, 13, 25)
    assert frame.capacity.tolist() == [60, 5]
    tiny = daily_workload(np.array([1, 1, 1, 25]), [1, 1, 1, 1], ReviewCapacity(1, 1, 30), 1, 48)
    assert tiny.new_cases.tolist() == [3, 1]
    assert tiny.backlog_end.tolist() == [1, 0]
    assert tiny.processed.tolist() == [2, 2]


def test_model_rule_overlap_creates_one_case_and_preserves_snapshot(tmp_path):
    store = InvestigationStore(tmp_path / "workflow.db")
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    policy = DecisionPolicy(.4, .75)
    assessment = {"risk_score": .1, "assessment_stage": "pre_transaction", "model_version": "test-v1"}
    for i in range(4):
        tx = {"transaction_id": f"t-{i}", "event_time": (epoch+timedelta(seconds=30*i)).isoformat(),
              "rail": "wire", "profile": "wire_pre", "sender_id": "A", "recipient_id": "B", "amount": 10.,
              "transaction_type": "TRANSFER", "synthetic": True, "payload": {}}
        current = {**assessment, "risk_score": .9 if i == 3 else .1}
        record = store.ingest(tx, current, policy)
    assert record["assessment"]["decision_reasons"] == ["MODEL_THRESHOLD", "TRANSFER_VELOCITY"]
    assert len(store.list_cases()) == 1
    assert record["rule_version"] == RulePolicy().version
    assert record["rule_policy"] == RulePolicy().to_dict()


def test_workflow_policy_override_requires_matching_model_version(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path))
    bundle = legacy_bundle("card")
    path = tmp_path / "workflow/card_policy.json"
    path.parent.mkdir()
    config = {"model_version": "different", "workflow_policy_version": "wf-test",
              "decision_policy": {"flag_threshold": .8, "block_threshold": .8, "version": "wf-test"},
              "rule_policy": RulePolicy(minimum_model_score=.2).to_dict()}
    path.write_text(json.dumps(config))
    assert apply_workflow_policy(bundle).policy == bundle.policy
    config["model_version"] = bundle.version
    path.write_text(json.dumps(config))
    assert apply_workflow_policy(bundle).policy.block_threshold == .8
    assert active_rule_policy("card", bundle.version).minimum_model_score == .2
    assert active_rule_policy("card", "different").minimum_model_score == 0.


def test_paysim_identity_encoding_preserves_customer_and_merchant_prefixes():
    encoded = encode_accounts(pd.Series(["C42", "M42", "C2147483647"]))
    assert encoded.tolist() == [84, 85, 4294967294]
    with pytest.raises(ValueError):
        encode_accounts(pd.Series(["C2147483648"]))


def test_rule_only_activation_uses_validation_and_preserves_model_cutoff(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path))
    bundle = legacy_bundle("card")
    baseline = {"model_threshold": bundle.policy.flag_threshold, "strategy": "model_only_current"}
    combined = {"model_threshold": bundle.policy.flag_threshold, "strategy": "model_plus_default_rules"}
    report = {"model_version": bundle.version, "source_sha256": "fixture", "comparison": [
        {**baseline, "partition": "validation", "expected_loss": 10.},
        {**combined, "partition": "validation", "expected_loss": 20.},
        # Opposite held-out ranking must not influence selection.
        {**baseline, "partition": "test", "expected_loss": 999.},
        {**combined, "partition": "test", "expected_loss": 0.}]}
    config = activate_rules_from_report(report, bundle)
    assert config["validation_selected_strategy"] == "model_only_current"
    assert active_rule_policy("card", bundle.version).enabled_rules == ()
    overridden = apply_workflow_policy(bundle)
    assert overridden.policy.flag_threshold == bundle.policy.flag_threshold
    assert overridden.policy.block_threshold == bundle.policy.block_threshold
    assert overridden.metadata["workflow_policy_version"] == config["workflow_policy_version"]


def test_end_to_end_workflow_replay_keeps_policies_separate_and_writes_sensitivity(tmp_path, monkeypatch):
    import joblib
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    n = 1200
    labels = (np.arange(n) % 20 == 0).astype(int)
    raw = pd.DataFrame({"step": np.arange(n)//10+1, "type": "TRANSFER", "amount": 10.+labels*1500,
        "nameOrig": [f"C{i}" for i in range(n)], "nameDest": [f"C{i%17+2000}" for i in range(n)],
        "oldbalanceOrg": 5000., "oldbalanceDest": 0., "isFraud": labels})
    source = tmp_path / "original.csv"
    raw.to_csv(source, index=False)
    inputs = np.column_stack([raw.amount, raw.oldbalanceOrg, raw.oldbalanceDest, raw.amount/(raw.oldbalanceOrg+1), np.ones(n)])
    model = LogisticRegression().fit(inputs[:600], labels[:600])
    metadata = {"model_version": "fixture-weights", "protocol": "chronological",
        "source": {"manifest": {"source_sha256": sha256(source.read_bytes()).hexdigest()}},
        "partition_summary": {"validation": {"time_min": 85, "time_max": 108}, "test": {"time_min": 109, "time_max": 120}},
        "cost_assumptions": {"review_cost": 2.5, "false_decline_cost": 5., "loss_fraction": 1.}}
    bundle = ModelBundle(model, "fixture", "wire_pre", PRE_FEATURES, inputs[:600].mean(axis=0), DecisionPolicy(.5, .5), metadata)
    artifact = tmp_path / "bundle.joblib"
    joblib.dump(bundle, artifact)
    out = tmp_path / "artifacts/workflow"
    report = run_full_replay(source, artifact, out, ReviewCapacity(), activate=False)
    assert report["source_rows"] == n
    assert not (out / "wire_pre_policy.json").exists()
    assert len(pd.read_csv(out / "capacity_sensitivity.csv")) == 12
    assert len(pd.read_csv(out / "test_workflow_predictions.csv")) == 120
    assert report["selection"]["validation_cases"] <= 120
    # Activation is explicit and tied to this exact model; no retraining or held-out selection.
    activated = run_full_replay(source, artifact, out, ReviewCapacity(), activate=True)
    assert activated["selection"] == report["selection"]
    assert active_rule_policy("wire_pre", bundle.version).to_dict() == report["selection"]["rule_policy"]
    assert apply_workflow_policy(bundle).policy.block_threshold == report["selection"]["threshold"]
