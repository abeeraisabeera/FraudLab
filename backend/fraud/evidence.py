"""Canonical fraud evidence package — single source of truth for UI and audit."""

from __future__ import annotations

from typing import Any

from fraud.decision import calibrated_decision
from fraud.policy import policy_payload
from fraud.schema import CostConfig, PolicyConfig


def build_evidence(
    *,
    transaction: dict[str, Any],
    features: list[dict[str, Any]],
    explanation: dict[str, Any],
    raw_score: float,
    calibrated_probability: float,
    calibration_method: str,
    policy: PolicyConfig,
    costs: CostConfig,
    model_version: str,
    evaluation_mode: str,
    metrics_snapshot: dict[str, Any] | None = None,
    split_role: str | None = None,
) -> dict[str, Any]:
    """Build evidence using the canonical decision path only.

    ``decision`` is always ``calibrated_decision(calibrated_probability, policy)``.
    Contributions come from the model explanation package (not fabricated).
    """
    decision = calibrated_decision(calibrated_probability, policy)
    contribs = list(explanation.get("contributions") or [])
    positive = list(explanation.get("positive_contributions") or [])
    negative = list(explanation.get("negative_contributions") or [])

    feature_by_name = {f["feature"]: f for f in features}

    def enrich(items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        out = []
        for item in items[:limit]:
            name = str(item["feature"])
            meta = feature_by_name.get(name, {})
            members = item.get("members") or [name]
            if meta.get("value") is None and item.get("term_type") == "interaction":
                # Compose a readable definition for pairwise terms.
                member_defs = [
                    feature_by_name.get(m, {}).get("definition") or m for m in members
                ]
                definition = "Interaction: " + " × ".join(str(d) for d in member_defs)
            else:
                definition = meta.get("definition")
            out.append(
                {
                    "feature": name,
                    "value": meta.get("value"),
                    "contribution": float(item["contribution"]),
                    "definition": definition,
                    "term_type": item.get("term_type", "main"),
                    "members": members,
                }
            )
        return out

    return {
        "transaction_id": transaction.get("transaction_id"),
        "transaction": transaction,
        "risk_probability": float(calibrated_probability),
        "decision": decision,
        "top_contributors": enrich(positive),
        "counter_signals": enrich(negative),
        "positive_contributions": enrich(positive, limit=20),
        "negative_contributions": enrich(negative, limit=20),
        "features": features,
        "feature_contributions": contribs,
        "explanation": {
            "intercept": explanation.get("intercept"),
            "contribution_sum": explanation.get("contribution_sum"),
            "model_logit": explanation.get("model_logit"),
            "additivity_error": explanation.get("additivity_error"),
        },
        "model": {
            "model_version": model_version,
            "raw_score": float(raw_score),
            "calibrated_probability": float(calibrated_probability),
            "calibration_method": calibration_method,
            **(
                {
                    "pr_auc": metrics_snapshot.get("pr_auc"),
                    "brier_score": metrics_snapshot.get("brier_score"),
                }
                if metrics_snapshot
                else {}
            ),
        },
        "calibration": {
            "raw_score": float(raw_score),
            "calibrated_probability": float(calibrated_probability),
            "calibration_method": calibration_method,
        },
        "policy": policy_payload(policy),
        "policy_version": policy.version,
        "model_version": model_version,
        "thresholds": {
            "allow_threshold": policy.allow_threshold,
            "review_threshold": policy.review_threshold,
            "block_threshold": policy.block_threshold,
            "version": policy.version,
        },
        "cost_assumptions": costs.model_dump(),
        "evaluation_mode": evaluation_mode,
        "split_role": split_role,
    }
