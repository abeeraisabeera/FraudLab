"""Decision policy: maps calibrated risk probability to ALLOW / REVIEW / BLOCK.

Kept separate from the model so thresholds can change without retraining.
"""

from __future__ import annotations

from fraud.schema import Decision, PolicyConfig

DEFAULT_POLICY = PolicyConfig()


def decide(probability: float, policy: PolicyConfig | None = None) -> Decision:
    cfg = policy or DEFAULT_POLICY
    p = float(probability)
    if p < cfg.allow_threshold:
        return "ALLOW"
    if p < cfg.review_threshold:
        return "REVIEW"
    return "BLOCK"


def decide_many(probabilities: list[float] | tuple[float, ...], policy: PolicyConfig | None = None) -> list[Decision]:
    return [decide(p, policy) for p in probabilities]


def policy_payload(policy: PolicyConfig | None = None) -> dict:
    cfg = policy or DEFAULT_POLICY
    return {
        "version": cfg.version,
        "allow_threshold": cfg.allow_threshold,
        "review_threshold": cfg.review_threshold,
        "block_threshold": cfg.block_threshold,
        "mapping": {
            "LOW_RISK": "ALLOW",
            "MEDIUM_RISK": "REVIEW",
            "HIGH_RISK": "BLOCK",
        },
    }
