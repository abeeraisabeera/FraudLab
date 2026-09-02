"""Canonical decision path used by evaluation, scoring tables, and evidence.

raw_score → calibrated_probability → policy.decide → decision

Nothing else may invent a decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraud.policy import decide
from fraud.schema import Decision, PolicyConfig


def calibrated_decision(
    calibrated_probability: float,
    policy: PolicyConfig,
) -> Decision:
    """Single canonical mapping from calibrated probability to action."""
    return decide(float(calibrated_probability), policy)


def score_row(
    *,
    raw_score: float,
    calibrated_probability: float,
    policy: PolicyConfig,
) -> dict[str, Any]:
    """Canonical scored-row payload shared by holdout tables and evidence."""
    p = float(calibrated_probability)
    return {
        "raw_score": float(raw_score),
        "risk_probability": p,
        "calibrated_probability": p,
        "decision": calibrated_decision(p, policy),
    }


def score_arrays(
    raw_scores: np.ndarray,
    calibrated_probabilities: np.ndarray,
    policy: PolicyConfig,
) -> list[dict[str, Any]]:
    raw = np.asarray(raw_scores, dtype=float)
    cal = np.asarray(calibrated_probabilities, dtype=float)
    return [
        score_row(raw_score=float(r), calibrated_probability=float(p), policy=policy)
        for r, p in zip(raw, cal)
    ]
