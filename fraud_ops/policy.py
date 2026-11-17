"""One versioned decision policy for APIs, dashboard, and transaction replay."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class DecisionPolicy:
    flag_threshold: float
    block_threshold: float
    version: str = "policy-v2"

    def __post_init__(self):
        if not (0 <= self.flag_threshold <= self.block_threshold <= 1):
            raise ValueError("Thresholds must satisfy 0 <= flag <= block <= 1")

    def with_threshold(self, threshold=None):
        if threshold is None:
            return self
        return replace(self, flag_threshold=min(self.flag_threshold, threshold), block_threshold=threshold)

    def decide(self, score: float, *, stage="pre_transaction", behavioural_alert=False):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Risk score must be finite and within [0, 1]")
        is_fraud = score >= self.block_threshold
        flagged = score >= self.flag_threshold or behavioural_alert
        # A settled transaction can be investigated, but cannot be declined retroactively.
        action = ("REVIEW" if stage == "post_transaction" else "BLOCK") if is_fraud else ("REVIEW" if flagged else "PASS")
        level = ("CRITICAL" if score >= max(0.85, self.block_threshold) else "HIGH") if is_fraud else ("MEDIUM" if flagged else "LOW")
        return {
            "is_fraud": is_fraud, "risk_level": level, "action": action,
            "threshold_applied": self.block_threshold, "flag_threshold": self.flag_threshold,
            "policy_version": self.version,
        }

    def to_dict(self):
        return asdict(self)


DEFAULT_POLICIES = {
    "card": DecisionPolicy(0.20, 0.50),
    "wire": DecisionPolicy(0.40, 0.75),
    "wire_pre": DecisionPolicy(0.40, 0.75),
}
