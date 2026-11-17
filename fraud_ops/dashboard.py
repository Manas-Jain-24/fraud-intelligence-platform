"""Streamlit investigation workflow and artifact-backed evaluation views."""
from datetime import datetime, timedelta, timezone
import json
import uuid
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_ops.artifacts import artifact_root, get_bundle
from fraud_ops.service import ingest
from fraud_ops.store import InvestigationStore


def seed_demo(store):
    """A repeatable story with new IDs each run; no asserted real fraud labels."""
    batch = "DEMO-" + uuid.uuid4().hex[:8]
    start = datetime.now(timezone.utc) - timedelta(minutes=12)
    events = [("ALICE", "MULE", "TRANSFER", 120.), ("BOB", "MULE", "TRANSFER", 180.),
              ("CAROL", "MULE", "TRANSFER", 250.), ("DAN", "MULE", "TRANSFER", 400.),
              ("MULE", "AGENT", "CASH_OUT", 950.),
              ("BURST", "SHOP-1", "TRANSFER", 110.), ("BURST", "SHOP-2", "TRANSFER", 120.),
              ("BURST", "SHOP-3", "TRANSFER", 130.), ("BURST", "NEW-PAYEE", "TRANSFER", 1200.)]
    try:
        get_bundle("wire_pre")
        profile = "wire_pre"
    except FileNotFoundError:
        profile = "wire"
    for i, (sender, recipient, kind, amount) in enumerate(events):
        payload = {"type": kind, "amount": amount, "oldbalanceOrg": 5000., "oldbalanceDest": 1000.}
        if profile == "wire":
            payload.update(newbalanceOrig=5000.-amount, newbalanceDest=1000.+amount)
        ingest(profile, payload, {"transaction_id": f"{batch}-{i+1:02d}",
            "event_time": (start + timedelta(seconds=i * 45)).isoformat(),
            "sender_id": f"{batch}-{sender}", "recipient_id": f"{batch}-{recipient}", "synthetic": True}, store=store)
    return batch


def render_investigations():
    st.subheader("Investigation queue")
    st.caption("Scores, policy, history snapshots and analyst decisions persist across sessions. Actions here record recommendations; they do not move money.")
    store = InvestigationStore()
    if st.button("Create synthetic investigation scenario", key="ops_seed_demo"):
        batch = seed_demo(store)
        st.success(f"Created {batch}: recipient fan-in, rapid cash-out and a sender burst.")

    with st.expander("Ingest a transaction", expanded=False):
        st.caption("Pre-transfer inputs exclude post-transaction balances. Entity IDs enable history rules; an unseen beneficiary means unseen in this local history.")
        profile = st.selectbox("Assessment profile", ["wire_pre", "wire", "card"], key="ops_profile")
        if profile == "card":
            template = {"amount": 45.2, "v_features": {f"V{i}": 0. for i in range(1, 29)}}
        else:
            template = {"type": "TRANSFER", "amount": 250., "oldbalanceOrg": 1200., "oldbalanceDest": 500.}
            if profile == "wire":
                template.update(newbalanceOrig=950., newbalanceDest=750.)
        st.session_state.setdefault("ops_tx_id", "TX-" + uuid.uuid4().hex[:12])
        with st.form(f"ops_ingest_{profile}"):
            transaction_id = st.text_input("Transaction ID", value=st.session_state.ops_tx_id)
            event_time = st.text_input("Event time (ISO 8601, with timezone)", value=datetime.now(timezone.utc).replace(microsecond=0).isoformat())
            left, right = st.columns(2)
            sender = left.text_input("Sender / cardholder ID", value="LOCAL-SENDER")
            recipient = right.text_input("Recipient / merchant ID", value="LOCAL-RECIPIENT")
            payload_json = st.text_area("Transaction inputs (JSON)", value=json.dumps(template, indent=2), height=210)
            synthetic = st.checkbox("Synthetic example", value=True)
            force_case = st.checkbox("Open a case even if the policy passes this transaction", value=False)
            submit = st.form_submit_button("Score and save transaction")
        if submit:
            try:
                from api import CreditCardFeatures, WireTransferRequest, WirePreRequest
                schema = {"card": CreditCardFeatures, "wire": WireTransferRequest, "wire_pre": WirePreRequest}[profile]
                payload = schema.model_validate(json.loads(payload_json)).model_dump()
                result = ingest(profile, payload, {"transaction_id": transaction_id, "event_time": event_time,
                    "sender_id": sender.strip() or None, "recipient_id": recipient.strip() or None, "synthetic": synthetic}, store=store, force_case=force_case)
                st.success(f"{'Existing record' if result['duplicate'] else 'Saved'}: {transaction_id} — {result['assessment']['action']}")
                st.json({"assessment": result["assessment"], "behavioural_rules": result["behavioural_rules"]}, expanded=False)
            except (ValueError, FileNotFoundError) as error:
                st.error(str(error))

    filters = st.columns(3)
    status = filters[0].selectbox("Case status", ["All", "OPEN", "IN_REVIEW", "RESOLVED"], key="ops_status")
    source = filters[1].selectbox("Transaction source", ["All", "External", "Synthetic"], key="ops_source")
    assignee_filter = filters[2].text_input("Filter by assignee", key="ops_assignee_filter")
    rows = store.list_cases(None if status == "All" else status, assignee_filter.strip() or None,
                            None if source == "All" else source == "Synthetic")
    if rows:
        st.dataframe(pd.DataFrame(rows)[["case_id", "status", "assignee", "amount", "rail", "risk_score", "action", "synthetic", "decision"]], hide_index=True, use_container_width=True)
        case_id = st.selectbox("Open case", [row["case_id"] for row in rows], key="ops_selected_case")
        case = store.get_case(case_id)
        tx, assessment = case["transaction"], case["transaction"]["assessment"]
        st.markdown(f"**Case #{case_id} · {case['status']} · {tx['transaction_id']}**")
        st.caption(f"{assessment['model_name']} · {assessment['model_version']} · {assessment['score_kind']} · {assessment['assessment_stage']}")
        info, history = st.columns([1, 1])
        with info:
            st.metric("Recorded model score", f"{assessment['risk_score']:.6f}")
            st.write("Recorded recommendation:", assessment["action"])
            st.json({"policy": assessment["policy"], "history_snapshot": tx["behavioural_features"],
                     "rules": tx["behavioural_rules"]}, expanded=False)
            st.write(assessment["explanation"]["narrative"])
            impacts = pd.DataFrame(assessment["explanation"]["all_impacts"])
            st.dataframe(impacts[["feature", "raw_value", "impact"]], hide_index=True, use_container_width=True)
        with history:
            st.markdown("**Observed sender history**")
            sender_rows = store.list_transactions(sender_id=tx.get("sender_id")) if tx.get("sender_id") else []
            timeline = [{"transaction_id": item["transaction_id"], "event_time": item["event_time"],
                         "recipient": item.get("recipient_id"), "amount": item["amount"], "synthetic": item["synthetic"]} for item in sender_rows]
            st.dataframe(pd.DataFrame(timeline), hide_index=True, use_container_width=True)
            st.caption("The table includes later observed events; the recorded feature snapshot used strictly earlier events only.")
            st.markdown("**Transaction inputs**")
            st.json(tx["payload"], expanded=False)

        actor = st.text_input("Your analyst name", key="ops_actor")
        with st.form(f"ops_update_{case_id}_{case['version']}"):
            action = st.selectbox("Case action", ["ASSIGN", "NOTE", "RESOLVE", "REOPEN"])
            assignee = st.text_input("Assignee", value=case["assignee"] or "")
            outcome = st.selectbox("Outcome (for resolution)", ["CONFIRMED_FRAUD", "LEGITIMATE", "INCONCLUSIVE"])
            reason = st.text_area("Reason / investigation note")
            save = st.form_submit_button("Save case action")
        if save:
            try:
                store.update_case(case_id, action=action, actor=actor, expected_version=case["version"],
                                  assignee=assignee, decision=outcome if action == "RESOLVE" else None, reason=reason)
                st.success("Case action saved.")
                st.rerun()
            except (ValueError, KeyError) as error:
                st.error(str(error))
        st.markdown("**Investigation timeline**")
        st.dataframe(pd.DataFrame([{k: event[k] for k in ["event_time", "actor", "action", "details"]}
                                   for event in case["events"]]), hide_index=True, use_container_width=True)
    else:
        st.info("No cases match these filters. Ingest a transaction or create a synthetic scenario.")

    with st.expander("Confirmed feedback export"):
        include_synthetic = st.checkbox("Include synthetic investigations in export", value=False)
        feedback = pd.DataFrame(store.feedback_rows(include_synthetic))
        st.caption("Only resolved CONFIRMED_FRAUD and LEGITIMATE cases receive labels. Feedback is not automatically used for training.")
        st.download_button("Download confirmed feedback CSV", feedback.to_csv(index=False).encode(),
                           file_name="confirmed_feedback.csv", mime="text/csv")


def render_evaluation():
    st.subheader("Measured model performance")
    left, right = st.columns(2)
    profile = left.selectbox("Model profile", ["card", "wire", "wire_pre"], key="evaluation_profile")
    protocol = right.selectbox("Evaluation protocol", ["chronological", "stratified"], key="evaluation_protocol")
    report_path = artifact_root() / profile / protocol / "report.json"
    if not report_path.exists():
        st.info("Run python scripts/evaluate_operations.py to create natural-prevalence evaluation reports.")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.write(f"**{report['model_name']}** · {report['score_kind']} · {report['stage']}")
    st.caption(f"Artifact {report['model_version']}. Selection used validation cost; the final test partition was kept untouched.")
    metrics = report["test"]
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Precision", "Recall", "False alerts / 10,000", "Alerts / 10,000"],
                                [f"{metrics['precision']:.1%}", f"{metrics['recall']:.1%}", f"{metrics['false_alerts_per_10000']:.2f}", f"{metrics['alerts_per_10000']:.2f}"]):
        col.metric(label, value)
    st.dataframe(pd.DataFrame(report["partition_summary"]).T, use_container_width=True)
    st.caption("Chronological partitions keep equal timestamps together. Their prevalence can differ over time; stratified results preserve the sampled overall prevalence.")
    st.json({"policy": report["policy"], "cost_assumptions": report["cost_assumptions"],
             "test_expected_loss": metrics["expected_loss"], "test_missed_fraud_amount": metrics["missed_fraud_amount"],
             "source": report["source"]}, expanded=False)
    st.markdown("**Reliability on the untouched test partition**")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash"})
    for label, key in [("Selected output", "test_reliability"), ("Base model before calibration", "test_raw_reliability")]:
        points = report[key]
        fig.add_scatter(x=[p["mean_score"] for p in points], y=[p["fraud_fraction"] for p in points],
                       mode="lines+markers", name=label, text=[f"n={p['count']}" for p in points])
    fig.update_layout(xaxis_title="Mean score in bin", yaxis_title="Observed fraud fraction", height=330)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"Selected Brier score: {metrics['brier_score']:.6f}; base-model Brier score: {report['test_raw_brier']:.6f}.")
    st.caption("Sparse fraud counts make reliability estimates uncertain. An uncalibrated selected output is displayed as a score, not a probability.")
    with st.expander("Model and calibration competition (validation only)"):
        st.dataframe(pd.read_csv(report_path.parent / "validation_comparison.csv").sort_values("expected_loss"), hide_index=True, use_container_width=True)
    for limitation in report["limitations"]:
        st.caption(limitation)
