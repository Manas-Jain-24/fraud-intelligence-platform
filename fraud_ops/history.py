"""Causal, observable-history features and transparent review rules.

No fraud labels are used. Current and future transactions are excluded. Rules
route investigations independently of the ML score; they are not calibrated ML.
"""
from datetime import datetime, timezone


def parse_time(value):
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Event timestamps must include a UTC offset")
    return timestamp.astimezone(timezone.utc)


def history_features(prior_rows, transaction):
    now = parse_time(transaction["event_time"])
    prior = [(row, parse_time(row["event_time"])) for row in prior_rows
             if row["transaction_id"] != transaction["transaction_id"]]
    prior = [(row, when) for row, when in prior if when < now and row["rail"] == transaction["rail"]
             and bool(row.get("synthetic")) == bool(transaction.get("synthetic"))]
    sender, recipient = transaction.get("sender_id"), transaction.get("recipient_id")
    sent = [(row, when) for row, when in prior if sender and row.get("sender_id") == sender]
    last5 = [row for row, when in sent if (now-when).total_seconds() <= 300]
    last_hour = [row for row, when in sent if (now-when).total_seconds() <= 3600]
    incoming = [row for row, when in prior if recipient and row.get("recipient_id") == recipient and (now-when).total_seconds() <= 3600]
    # Sender acting now as a cash-out wallet may have recently received funds.
    received = [row for row, when in prior if sender and row.get("recipient_id") == sender and (now-when).total_seconds() <= 900]
    last = max((when for _, when in sent), default=None)
    observed_before = any(row.get("recipient_id") == recipient for row, _ in sent) if recipient else False
    return {
        "sender_history_available": bool(sender), "recipient_history_available": bool(recipient),
        "sender_prior_count_5m": len(last5), "sender_prior_amount_5m": sum(row["amount"] for row in last5),
        "sender_prior_count_1h": len(last_hour), "sender_prior_amount_1h": sum(row["amount"] for row in last_hour),
        "recipient_distinct_senders_1h": len({row["sender_id"] for row in incoming if row.get("sender_id")}),
        "recipient_prior_amount_1h": sum(row["amount"] for row in incoming),
        "sender_received_amount_15m": sum(row["amount"] for row in received),
        "new_beneficiary_in_observed_history": bool(sender and recipient and not observed_before),
        "seconds_since_sender_previous": (now-last).total_seconds() if last else None,
        "history_scope": "strictly earlier events observed locally; synthetic and external histories are separate",
    }


def behavioural_rules(features, transaction):
    rules = []
    if features["sender_prior_count_5m"] >= 3:
        rules.append({"code": "TRANSFER_VELOCITY", "reason": "At least three prior sender transactions in five minutes."})
    if features["recipient_distinct_senders_1h"] >= 3:
        rules.append({"code": "RECIPIENT_FAN_IN", "reason": "Recipient received from at least three distinct senders in one hour."})
    if transaction.get("transaction_type") == "CASH_OUT" and features["sender_received_amount_15m"] > 0:
        rules.append({"code": "RAPID_CASH_OUT", "reason": "Wallet received funds within fifteen minutes before this cash-out."})
    if features["new_beneficiary_in_observed_history"] and features["sender_prior_count_1h"] >= 2 and transaction["amount"] >= 1000:
        rules.append({"code": "NEW_BENEFICIARY_BURST", "reason": "High-value new beneficiary after at least two sender transactions in one hour."})
    return rules
