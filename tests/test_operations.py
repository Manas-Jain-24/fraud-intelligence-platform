"""Regression coverage for causal history, investigation state, and evaluated serving."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from fraud_ops.artifacts import ModelBundle, PRE_FEATURES, get_bundle, legacy_bundle
from fraud_ops.evaluation import choose_policy, evaluate_profile, operational_metrics, split_indices
from fraud_ops.history import history_features
from fraud_ops.inference import ENSEMBLE, COST
from fraud_ops.policy import DecisionPolicy
from fraud_ops.service import assess, ingest
from fraud_ops.store import ConflictError, InvestigationStore

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def tx(identifier, minute, sender="A", recipient="B", amount=100, synthetic=True):
    return {"transaction_id": identifier, "event_time": (START+timedelta(minutes=minute)).isoformat(),
            "rail": "wire", "profile": "wire_pre", "sender_id": sender, "recipient_id": recipient,
            "transaction_type": "TRANSFER", "amount": amount, "synthetic": synthetic,
            "payload": {"amount": amount}}


def assessment(score=.1):
    return {"risk_score": score, "assessment_stage": "pre_transaction", "model_version": "test-v1",
            "model_name": "test", "score_kind": "uncalibrated_score"}


def test_history_excludes_future_equal_time_current_and_other_sources():
    current = tx("current", 10)
    prior = [tx("past", 9), tx("future", 11), tx("equal", 10), tx("current", 9), tx("external", 8, synthetic=False)]
    features = history_features(prior, current)
    assert features["sender_prior_count_5m"] == 1
    assert features["sender_prior_amount_5m"] == 100
    assert features["new_beneficiary_in_observed_history"] is False
    assert features["seconds_since_sender_previous"] == 60


def test_behavioural_routing_does_not_change_ml_score_and_is_idempotent(tmp_path):
    store = InvestigationStore(tmp_path / "cases.db")
    policy = DecisionPolicy(.4, .75)
    for i in range(3):
        store.ingest(tx(str(i), i), assessment(), policy)
    last = tx("last", 3)
    saved = store.ingest(last, assessment(), policy)
    assert saved["assessment"]["risk_score"] == .1
    assert saved["assessment"]["action"] == "REVIEW"
    assert "TRANSFER_VELOCITY" in saved["assessment"]["decision_reasons"]
    # Duplicate requests retain the original model/policy/history snapshot, even after an update.
    duplicate = store.ingest(last, assessment(.9), DecisionPolicy(.1, .2))
    assert duplicate["duplicate"]
    assert duplicate["assessment"] == saved["assessment"]
    assert len(store.list_cases()) == 1
    with pytest.raises(ConflictError):
        store.ingest({**last, "amount": 200}, assessment(), policy)


def test_recipient_fan_in_and_rapid_cashout(tmp_path):
    store = InvestigationStore(tmp_path / "cases.db")
    policy = DecisionPolicy(.4, .75)
    for i in range(3):
        store.ingest(tx(f"in-{i}", i, sender=f"S{i}", recipient="MULE"), assessment(), policy)
    fourth = store.ingest(tx("in-4", 3, sender="S4", recipient="MULE"), assessment(), policy)
    assert {r["code"] for r in fourth["behavioural_rules"]} == {"RECIPIENT_FAN_IN"}
    cashout = tx("out", 4, sender="MULE", recipient="AGENT")
    cashout["transaction_type"] = "CASH_OUT"
    saved = store.ingest(cashout, assessment(), policy)
    assert {r["code"] for r in saved["behavioural_rules"]} == {"RAPID_CASH_OUT"}
    assert saved["behavioural_features"]["sender_received_amount_15m"] == 400


def test_case_lifecycle_survives_restart_and_rejects_stale_updates(tmp_path):
    path = tmp_path / "cases.db"
    store = InvestigationStore(path)
    store.ingest(tx("flagged", 0, synthetic=False), assessment(.9), DecisionPolicy(.4, .75))
    case_id = store.list_cases()[0]["case_id"]
    assigned = store.update_case(case_id, action="ASSIGN", actor="Asha", assignee="Asha", expected_version=1)
    assert assigned["status"] == "IN_REVIEW"
    with pytest.raises(ConflictError):
        store.update_case(case_id, action="NOTE", actor="Ben", reason="stale", expected_version=1)
    with pytest.raises(ValueError):
        store.update_case(case_id, action="RESOLVE", actor="Asha", decision="LEGITIMATE", reason=" ", expected_version=2)
    resolved = store.update_case(case_id, action="RESOLVE", actor="Asha", decision="LEGITIMATE", reason="Verified with customer", expected_version=2)
    restarted = InvestigationStore(path)
    assert restarted.get_case(case_id)["reason"] == "Verified with customer"
    assert restarted.feedback_rows()[0]["confirmed_label"] == 0
    reopened = restarted.update_case(case_id, action="REOPEN", actor="Asha", reason="New evidence", expected_version=resolved["version"])
    assert reopened["decision"] is None
    assert len(reopened["events"]) == 4
    assert restarted.feedback_rows() == []


def test_post_transaction_high_risk_is_review_not_retroactive_decline():
    policy = DecisionPolicy(.2, .7)
    assert policy.decide(.9, stage="post_transaction")["action"] == "REVIEW"
    assert policy.decide(.9, stage="pre_transaction")["action"] == "BLOCK"
    assert policy.decide(.699)["is_fraud"] is False


@pytest.mark.parametrize("profile", ["card", "wire"])
def test_explanation_decomposes_the_actual_requested_model(profile):
    bundle = legacy_bundle(profile)
    raw = np.array(bundle.reference, copy=True)
    raw[0] += 2
    raw[-1] += 3
    for name in [ENSEMBLE, COST]:
        selected = replace(bundle, model_name=name, scratch_model_name=name)
        explanation = selected.explain(raw)
        assert explanation["explained_model"] == name
        assert explanation["explained_score"] == pytest.approx(selected.predict(raw)[0])
        assert sum(row["impact"] for row in explanation["all_impacts"]) + explanation["baseline_score"] == pytest.approx(explanation["explained_score"])
    batch = bundle.model.predict_batch(np.vstack([raw, bundle.reference]))[ENSEMBLE]
    assert batch[0] == bundle.model.predict_all(raw)[ENSEMBLE]


def test_chronological_split_has_no_timestamp_overlap_and_no_resampling():
    times = np.repeat(np.arange(100), 10)
    y = np.arange(1000) % 20 == 0
    parts = split_indices(y, times, "chronological")
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(1000))
    for first, second in zip(list(parts.values()), list(parts.values())[1:]):
        assert times[first].max() < times[second].min()
    assert sum(y[parts["test"]]) == 7 or sum(y[parts["test"]]) == 8
    stratified = split_indices(y, times, "stratified")
    assert y[stratified["test"]].mean() < .06


def test_policy_selection_respects_budget_and_amount_costs():
    y = np.array([0, 0, 1, 1])
    scores = np.array([.1, .2, .7, .9])
    amounts = np.array([10, 10, 1000, 100])
    policy = choose_policy(y, scores, amounts, max_alert_rate=.5)
    metrics = operational_metrics(y, scores, amounts, policy.block_threshold)
    assert metrics["recall"] == 1
    assert metrics["alert_rate"] <= .5
    assert metrics["false_alerts_per_10000"] == 0
    assert metrics["expected_loss"] == 5


def test_evaluation_saves_reloadable_pre_transaction_model_and_untouched_test(tmp_path):
    rng = np.random.default_rng(7)
    n = 2000
    amounts = rng.uniform(1, 1000, n)
    labels = ((np.arange(n) % 13 == 0) & (amounts > 100)).astype(int)
    frame = pd.DataFrame({"step": np.arange(n)//10, "amount": amounts, "oldbalanceOrg": rng.uniform(0, 1000, n),
                          "oldbalanceDest": rng.uniform(0, 1000, n), "isTransfer": np.arange(n) % 2, "isFraud": labels})
    frame["balanceDrainRatio"] = frame.amount / (frame.oldbalanceOrg + 1)
    report = evaluate_profile(frame, "wire_pre", "chronological", tmp_path, {"file": "fixture"})
    import joblib
    bundle = joblib.load(tmp_path / "wire_pre/chronological/bundle.joblib")
    assert bundle.feature_names == PRE_FEATURES
    assert all("newbalance" not in feature for feature in bundle.feature_names)
    assert report["artifact_reload_verified"]
    membership = pd.read_csv(tmp_path / "wire_pre/chronological/split_membership.csv")
    predictions = pd.read_csv(tmp_path / "wire_pre/chronological/test_predictions.csv")
    assert set(predictions.source_row) == set(membership.loc[membership.partition == "test", "source_row"])
    assert report["test"]["n"] == len(predictions)


def test_api_score_and_explain_share_model_threshold_and_decision(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path / "empty"))
    payload = {"type": "TRANSFER", "amount": 450000., "oldbalanceOrg": 450000., "newbalanceOrig": 0., "oldbalanceDest": 0., "newbalanceDest": 0.}
    with TestClient(app) as client:
        for threshold in [None, .95, .4]:
            query = "" if threshold is None else f"?threshold={threshold}"
            score = client.post("/v1/score/wire"+query, json=payload)
            explanation = client.post("/v1/explain/wire"+query, json=payload)
            assert score.status_code == explanation.status_code == 200
            for key in ["risk_score", "model_name", "model_version", "is_fraud", "action", "threshold_applied"]:
                assert score.json()[key] == explanation.json()[key]
            detail = explanation.json()
            assert detail["explained_model"] == score.json()["model_name"]
            assert detail["baseline_score"] + sum(row["impact"] for row in detail["all_impacts"]) == pytest.approx(detail["risk_score"])


def test_api_ingestion_case_actions_and_conflict(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path / "empty"))
    request = {"profile": "card", "payload": {"amount": 20., "v_features": {f"V{i}": 0. for i in range(1, 29)}},
               "context": {"transaction_id": "API-1", "event_time": START.isoformat(), "sender_id": "A", "recipient_id": "B", "synthetic": True},
               "force_case": True}
    with TestClient(app) as client:
        response = client.post("/v1/transactions", json=request)
        assert response.status_code == 200
        assert client.post("/v1/transactions", json=request).json()["duplicate"]
        request["payload"]["amount"] = 21.
        assert client.post("/v1/transactions", json=request).status_code == 409
        case = client.get("/v1/cases").json()[0]
        update = {"action": "RESOLVE", "actor": "Asha", "expected_version": case["version"], "decision": "CONFIRMED_FRAUD", "reason": "Synthetic investigation"}
        assert client.patch(f"/v1/cases/{case['case_id']}", json=update).status_code == 200
        assert client.patch(f"/v1/cases/{case['case_id']}", json=update).status_code == 409
        assert client.get("/v1/feedback").json() == []
        assert client.get("/v1/feedback?include_synthetic=true").json()[0]["confirmed_label"] == 1
        assert client.get("/v1/cases/9999").status_code == 404
        assert client.get("/v1/transactions/API-1").json()["assessment"]["model_version"]
        assert client.post("/v1/score/wire/pre", json={"type": "TRANSFER", "amount": 100, "oldbalanceOrg": 500, "oldbalanceDest": 0}).status_code == 503


def test_natural_paysim_selection_is_independent_of_labels(tmp_path):
    from scripts.prepare_paysim_natural import prepare
    n = 1000
    frame = pd.DataFrame({"step": np.arange(n), "type": "TRANSFER", "amount": 100.,
                          "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": 500., "newbalanceOrig": 400.,
                          "nameDest": "D", "oldbalanceDest": 0., "newbalanceDest": 100., "isFraud": np.arange(n)%10 == 0})
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    frame.to_csv(first, index=False)
    frame["isFraud"] = ~frame["isFraud"]
    frame.to_csv(second, index=False)
    first_out, second_out = tmp_path / "sample1.csv", tmp_path / "sample2.csv"
    prepare(first, first_out, rows=100)
    prepare(second, second_out, rows=100)
    selected1, selected2 = pd.read_csv(first_out), pd.read_csv(second_out)
    assert selected1.nameOrig.tolist() == selected2.nameOrig.tolist()
    assert len(selected1) == 100
    assert selected1.isFraud.sum() < 100


@pytest.mark.parametrize("profile", ["card", "wire", "wire_pre"])
def test_locally_selected_artifacts_are_served_with_recorded_predictions(profile):
    """Also exercise the real measured artifacts when present; CI can use legacy fallbacks."""
    from fraud_ops.artifacts import ARTIFACT_DIR
    directory = ARTIFACT_DIR / profile / "chronological"
    source = ARTIFACT_DIR.parent / ("creditcard.csv" if profile == "card" else "paysim_natural.csv")
    if not (directory / "bundle.joblib").exists() or not source.exists():
        pytest.skip("Local measured artifacts/data have not been generated")
    report = json.loads((directory / "report.json").read_text())
    bundle = get_bundle(profile)
    assert bundle.version == report["model_version"]
    predictions = pd.read_csv(directory / "test_predictions.csv")
    ids = predictions.source_row.iloc[:20].to_numpy(dtype=int)
    frame = pd.read_csv(source).iloc[ids]
    scores = bundle.predict(frame[bundle.feature_names].to_numpy(dtype=float))
    np.testing.assert_allclose(scores, predictions.score.iloc[:20], rtol=1e-12, atol=1e-12)
    row = frame.iloc[0]
    if profile == "card":
        payload = {"amount": float(row.Amount), "v_features": {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}}
    else:
        payload = {"type": row.type, **{key: float(row[key]) for key in ["amount", "oldbalanceOrg", "oldbalanceDest"]}}
        if profile == "wire":
            payload.update(newbalanceOrig=float(row.newbalanceOrig), newbalanceDest=float(row.newbalanceDest))
    score = assess(profile, payload, explain=True)
    assert score["model_version"] == bundle.version
    assert score["risk_score"] == pytest.approx(scores[0])
    assert score["explanation"]["explained_score"] == pytest.approx(score["risk_score"])
    with TestClient(app) as client:
        endpoint = "wire/pre" if profile == "wire_pre" else profile
        response = client.post(f"/v1/score/{endpoint}", json=payload)
        assert response.status_code == 200
        assert response.json()["risk_score"] == pytest.approx(scores[0])
        if profile == "wire_pre":
            # Post balances cannot influence the pre-transfer artifact, even when supplied by a caller.
            extra = {**payload, "newbalanceOrig": 9999999, "newbalanceDest": 1}
            assert client.post(f"/v1/score/{endpoint}", json=extra).json()["risk_score"] == response.json()["risk_score"]
