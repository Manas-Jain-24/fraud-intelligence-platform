"""Causal, observable-history features and transparent review rules.

No fraud labels are used. Current and future transactions are excluded. Rules
route investigations independently of the ML score; they are not calibrated ML.
"""
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from hashlib import sha256
import json


RULE_CODES = ("TRANSFER_VELOCITY", "RECIPIENT_FAN_IN", "RAPID_CASH_OUT", "NEW_BENEFICIARY_BURST")


@dataclass(frozen=True)
class RulePolicy:
    velocity_prior_count: int = 3
    recipient_prior_senders: int = 3
    beneficiary_prior_count: int = 2
    beneficiary_min_amount: float = 1000.
    minimum_model_score: float = 0.
    enabled_rules: tuple = RULE_CODES

    def __post_init__(self):
        object.__setattr__(self, "enabled_rules", tuple(self.enabled_rules))
        if min(self.velocity_prior_count, self.recipient_prior_senders, self.beneficiary_prior_count) < 1:
            raise ValueError("Rule counts must be positive")
        if self.beneficiary_min_amount < 0 or not 0 <= self.minimum_model_score <= 1:
            raise ValueError("Invalid rule amount or model score gate")
        if not set(self.enabled_rules).issubset(RULE_CODES):
            raise ValueError("Unknown behavioural rule")

    def to_dict(self):
        return asdict(self)

    @property
    def version(self):
        return "history-rules-" + sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def rule_masks(features, is_cash_out, amount, policy=None):
    """The same expressions accept scalar snapshots or pandas/numpy replay columns."""
    policy = policy or RulePolicy()
    masks = {
        "TRANSFER_VELOCITY": features["sender_prior_count_5m"] >= policy.velocity_prior_count,
        "RECIPIENT_FAN_IN": features["recipient_distinct_senders_1h"] >= policy.recipient_prior_senders,
        "RAPID_CASH_OUT": is_cash_out & (features["sender_received_amount_15m"] > 0),
        "NEW_BENEFICIARY_BURST": (features["new_beneficiary_in_observed_history"] &
            (features["sender_prior_count_1h"] >= policy.beneficiary_prior_count) & (amount >= policy.beneficiary_min_amount)),
    }
    return {code: mask for code, mask in masks.items() if code in policy.enabled_rules}


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


def behavioural_rules(features, transaction, policy=None, model_score=None):
    policy = policy or RulePolicy()
    if policy.minimum_model_score > 0 and (model_score is None or model_score < policy.minimum_model_score):
        return []
    masks = rule_masks(features, transaction.get("transaction_type") == "CASH_OUT", transaction["amount"], policy)
    reasons = {
        "TRANSFER_VELOCITY": f"At least {policy.velocity_prior_count} prior sender transactions in five minutes.",
        "RECIPIENT_FAN_IN": f"Recipient received from at least {policy.recipient_prior_senders} distinct senders in one hour.",
        "RAPID_CASH_OUT": "Wallet received funds within fifteen minutes before this cash-out.",
        "NEW_BENEFICIARY_BURST": f"New beneficiary, amount at least {policy.beneficiary_min_amount:g}, after {policy.beneficiary_prior_count} prior sender transactions in one hour.",
    }
    rules = []
    for code, fired in masks.items():
        if fired:
            rules.append({"code": code, "reason": reasons[code]})
    return rules
