"""Model selection, scoring, explanations, and ingestion shared by UI and API."""
import time
from fraud_ops.artifacts import feature_vector, get_bundle
from fraud_ops.store import InvestigationStore


def assess(profile, payload, threshold=None, *, explain=False, bundle=None):
    bundle = bundle or get_bundle(profile)
    policy = bundle.policy.with_threshold(threshold)
    raw = feature_vector(profile, payload)
    start = time.perf_counter()
    score = float(bundle.predict(raw)[0])
    response = {
        "rail": {"card": "Credit Card (ULB European)", "wire": "Mobile Money / Wire Transfer (PaySim)",
                 "wire_pre": "Mobile Money / Wire Transfer (Pre-transaction)"}[profile],
        "risk_score": score, **policy.decide(score, stage=bundle.stage),
        "model_name": bundle.model_name, "model_version": bundle.version,
        "score_kind": bundle.score_kind, "assessment_stage": bundle.stage,
        "policy": policy.to_dict(), "evaluation_status": bundle.metadata.get("protocol", bundle.metadata.get("evaluation_status")),
        "workflow_policy_version": bundle.metadata.get("workflow_policy_version"),
        "latency_microseconds": (time.perf_counter()-start) * 1e6,
    }
    if explain:
        response["explanation"] = bundle.explain(raw)
    return response


def ingest(profile, payload, context, *, store=None, force_case=False):
    bundle = get_bundle(profile)
    assessment = assess(profile, payload, explain=True, bundle=bundle)
    transaction = {
        "transaction_id": context["transaction_id"], "event_time": context["event_time"],
        "sender_id": context.get("sender_id"), "recipient_id": context.get("recipient_id"),
        "synthetic": bool(context.get("synthetic", False)),
        "rail": "card" if profile == "card" else "wire", "profile": profile,
        "transaction_type": "CARD" if profile == "card" else payload["type"],
        "amount": float(payload["amount"]), "payload": payload,
    }
    return (store or InvestigationStore()).ingest(transaction, assessment, bundle.policy, force_case=force_case)
