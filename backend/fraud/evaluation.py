"""Cost-sensitive fraud evaluation.

Accuracy is intentionally not a primary metric under extreme class imbalance.
Primary metrics: PR-AUC, precision, recall, F1, FPR, FNR, Brier, confusion matrix,
and expected operational cost under ALLOW/REVIEW/BLOCK policy.

Holdout evaluation must only receive scores for the untouched evaluation fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from fraud.calibration import brier_score, calibration_curve_points
from fraud.decision import calibrated_decision
from fraud.schema import CostConfig, Decision, PolicyConfig


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def policy_confusion(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy: PolicyConfig,
) -> dict[str, Any]:
    """Recompute decisions and TP/FP/TN/FN/review under the three-way policy.

    Binary confusion treats BLOCK as the positive (fraud) prediction.
    REVIEW is counted separately and is not folded into TP/FP.
    FPR/FNR are therefore defined among non-REVIEW decisions only.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    decisions = [calibrated_decision(float(v), policy) for v in p]
    tp = fp = tn = fn = reviews = reviewed_fraud = reviewed_legit = 0
    for label, decision in zip(y, decisions):
        if decision == "REVIEW":
            reviews += 1
            if label == 1:
                reviewed_fraud += 1
            else:
                reviewed_legit += 1
            continue
        if decision == "BLOCK" and label == 1:
            tp += 1
        elif decision == "BLOCK" and label == 0:
            fp += 1
        elif decision == "ALLOW" and label == 0:
            tn += 1
        elif decision == "ALLOW" and label == 1:
            fn += 1
    return {
        "decisions": decisions,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "reviews": reviews,
        "reviewed_fraud": reviewed_fraud,
        "reviewed_legit": reviewed_legit,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "decision_counts": {
            "ALLOW": int(sum(d == "ALLOW" for d in decisions)),
            "REVIEW": int(sum(d == "REVIEW" for d in decisions)),
            "BLOCK": int(sum(d == "BLOCK" for d in decisions)),
        },
    }


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Treat probability >= threshold as positive (fraud) for binary metrics."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return {
            "threshold": threshold,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }
    y_hat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return 0.0
    if y.sum() == len(y):
        return 1.0
    return float(average_precision_score(y, p))


def precision_recall_curve_points(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    max_points: int = 40,
) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return []
    precision, recall, thresholds = precision_recall_curve(y, p)
    pts = []
    for i in range(len(thresholds)):
        pts.append(
            {
                "threshold": float(thresholds[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
            }
        )
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
    return [pts[i] for i in idx]


def expected_cost_from_counts(
    *,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    reviews: int,
    costs: CostConfig,
    reviewed_fraud: int = 0,
) -> dict[str, float]:
    """Cost model from explicit counts (recomputed per threshold).

    Defaults (v1):
    - ALLOW on fraud (FN) → false_negative_cost
    - BLOCK on legit (FP) → false_positive_cost
    - REVIEW → review_cost only

    Optional: ``charge_false_negative_on_review`` also charges FN for REVIEW∧fraud.
    """
    fn_charged = fn + (reviewed_fraud if costs.charge_false_negative_on_review else 0)
    total = (
        fp * costs.false_positive_cost
        + fn_charged * costs.false_negative_cost
        + reviews * costs.review_cost
    )
    n = max(tp + fp + tn + fn + reviews, 1)
    return {
        "expected_cost": float(total),
        "expected_cost_per_transaction": float(total / n),
        "false_positives_charged": float(fp),
        "false_negatives_charged": float(fn_charged),
        "reviewed_fraud": float(reviewed_fraud),
        "reviews": float(reviews),
        "review_rate": float(reviews / n),
    }


def expected_cost_for_decisions(
    y_true: np.ndarray,
    decisions: list[Decision],
    costs: CostConfig,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    tp = fp = tn = fn = reviews = reviewed_fraud = 0
    for label, decision in zip(y, decisions):
        if decision == "REVIEW":
            reviews += 1
            if label == 1:
                reviewed_fraud += 1
        elif decision == "BLOCK" and label == 1:
            tp += 1
        elif decision == "BLOCK" and label == 0:
            fp += 1
        elif decision == "ALLOW" and label == 0:
            tn += 1
        elif decision == "ALLOW" and label == 1:
            fn += 1
    return expected_cost_from_counts(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        reviews=reviews,
        costs=costs,
        reviewed_fraud=reviewed_fraud,
    )


def policy_operating_point(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy: PolicyConfig,
    costs: CostConfig,
) -> dict[str, Any]:
    conf = policy_confusion(y_true, probabilities, policy)
    cost = expected_cost_from_counts(
        tp=conf["tp"],
        fp=conf["fp"],
        tn=conf["tn"],
        fn=conf["fn"],
        reviews=conf["reviews"],
        costs=costs,
        reviewed_fraud=conf["reviewed_fraud"],
    )
    # Precision/recall among non-reviewed decisions using BLOCK as positive.
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    n_neg = conf["tn"] + conf["fp"]
    n_pos = conf["tp"] + conf["fn"]
    return {
        "threshold": policy.review_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": _safe_div(conf["fp"], n_neg),
        "false_negative_rate": _safe_div(conf["fn"], n_pos),
        "confusion_matrix": conf["confusion_matrix"],
        "tp": conf["tp"],
        "fp": conf["fp"],
        "tn": conf["tn"],
        "fn": conf["fn"],
        "reviews": conf["reviews"],
        "reviewed_fraud": conf["reviewed_fraud"],
        **cost,
        "review_rate": cost["review_rate"],
        "block_rate": _safe_div(conf["decision_counts"]["BLOCK"], max(len(y_true), 1)),
        "allow_rate": _safe_div(conf["decision_counts"]["ALLOW"], max(len(y_true), 1)),
        "decision_counts": conf["decision_counts"],
        "metric_definitions": {
            "fpr_fnr": "Among non-REVIEW decisions only (BLOCK=positive).",
            "cost": (
                "FP*fp_cost + FN*fn_cost + REVIEW*review_cost; "
                "optional FN charge on REVIEW∧fraud via charge_false_negative_on_review."
            ),
        },
    }


def threshold_cost_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    costs: CostConfig,
    allow_grid: list[float] | None = None,
    review_grid: list[float] | None = None,
) -> dict[str, Any]:
    """For every threshold pair, recompute decisions, confusion, reviews, cost."""
    if allow_grid is None:
        allow_grid = [round(x, 2) for x in np.linspace(0.05, 0.45, 9)]
    if review_grid is None:
        review_grid = [round(x, 2) for x in np.linspace(0.35, 0.9, 12)]

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for allow_t in allow_grid:
        for review_t in review_grid:
            if allow_t > review_t:
                continue
            policy = PolicyConfig(
                version="sweep",
                allow_threshold=float(allow_t),
                review_threshold=float(review_t),
                block_threshold=float(review_t),
            )
            point = policy_operating_point(y_true, probabilities, policy, costs)
            row = {
                "allow_threshold": float(allow_t),
                "review_threshold": float(review_t),
                "expected_cost": point["expected_cost"],
                "expected_cost_per_transaction": point["expected_cost_per_transaction"],
                "precision": point["precision"],
                "recall": point["recall"],
                "f1": point["f1"],
                "review_rate": point["review_rate"],
                "false_positive_rate": point["false_positive_rate"],
                "false_negative_rate": point["false_negative_rate"],
                "tp": point["tp"],
                "fp": point["fp"],
                "tn": point["tn"],
                "fn": point["fn"],
                "reviews": point["reviews"],
                "decision_counts": point["decision_counts"],
            }
            results.append(row)
            if best is None or row["expected_cost"] < best["expected_cost"]:
                best = row
    return {"sweep": results, "best_thresholds": best}


def evaluate_scores(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy: PolicyConfig,
    costs: CostConfig,
    *,
    evaluation_mode: str = "holdout",
    raw_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    operating = policy_operating_point(y, p, policy, costs)
    brier_cal = brier_score(y, p)
    payload: dict[str, Any] = {
        "evaluation_mode": evaluation_mode,
        "pr_auc": pr_auc(y, p),
        "brier_score": brier_cal,
        "brier_calibrated": brier_cal,
        "precision": operating["precision"],
        "recall": operating["recall"],
        "f1": operating["f1"],
        "false_positive_rate": operating["false_positive_rate"],
        "false_negative_rate": operating["false_negative_rate"],
        "expected_cost": operating["expected_cost"],
        "expected_cost_per_transaction": operating["expected_cost_per_transaction"],
        "review_rate": operating["review_rate"],
        "block_rate": operating["block_rate"],
        "allow_rate": operating["allow_rate"],
        "confusion_matrix": operating["confusion_matrix"],
        "tp": operating["tp"],
        "fp": operating["fp"],
        "tn": operating["tn"],
        "fn": operating["fn"],
        "reviews": operating["reviews"],
        "reviewed_fraud": operating["reviewed_fraud"],
        "decision_counts": operating["decision_counts"],
        "precision_recall_curve": precision_recall_curve_points(y, p),
        "calibration_curve": calibration_curve_points(y, p),
        "threshold_cost_analysis": threshold_cost_sweep(y, p, costs),
        "policy": {
            "version": policy.version,
            "allow_threshold": policy.allow_threshold,
            "review_threshold": policy.review_threshold,
            "block_threshold": policy.block_threshold,
        },
        "costs": costs.model_dump(),
        "metric_definitions": operating["metric_definitions"],
        "n_samples": int(len(y)),
        "n_positives": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
    }
    if raw_scores is not None:
        brier_raw = brier_score(y, np.asarray(raw_scores, dtype=float))
        payload["brier_raw"] = brier_raw
        payload["calibration_improves_brier"] = bool(brier_cal <= brier_raw + 1e-12)
    return payload
