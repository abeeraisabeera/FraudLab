"""Fraud Lab pipeline orchestration.

Holdout protocol (evaluate + analyze):
  sort by timestamp
       → temporal split (earliest train / latest holdout)
       → features on chronological frame (train never sees future holdout)
       → carve calibration fold from train only
       → fit EBM on train\\calibration
       → fit calibrator on calibration fold
       → score holdout for metrics (holdout untouched until final scoring)
       → optionally score all rows with the same frozen stack for browsing

``analyze`` uses the identical seeded protocol (including ``test_size``) so
evidence decisions match the scored-transaction table from ``evaluate``.

``evaluation_mode``:
- ``holdout`` — metrics from untouched evaluation fold (trustworthy)
- exploratory in-sample metrics are never returned as ``holdout``
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud.calibration import (
    calibration_safety_notes,
    choose_calibration_method,
    fit_calibrator,
)
from fraud.decision import score_row
from fraud.evaluation import evaluate_scores
from fraud.evidence import build_evidence
from fraud.features import (
    MODEL_FEATURE_NAMES,
    compute_features,
    dataset_baselines,
    feature_matrix,
    provenance_for_row,
)
from fraud.generator import generate_fraud_dataset, realized_fraud_rate
from fraud.model import ADDITIVITY_MAX_ERROR, MODEL_VERSION, FraudEBMModel
from fraud.policy import policy_payload
from fraud.schema import (
    CostConfig,
    FraudAnalyzeRequest,
    FraudEvaluateRequest,
    FraudGenerateRequest,
    FraudTransaction,
    PolicyConfig,
)

_STACK_CACHE: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
_STACK_CACHE_LOCK = threading.Lock()
_STACK_CACHE_MAX = 4


def _stack_cache_key(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    calibration_method: str | None,
) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(X_train, dtype=np.float64).tobytes())
    h.update(np.asarray(y_train, dtype=np.int64).tobytes())
    h.update(str(seed).encode())
    h.update(str(calibration_method).encode())
    return h.hexdigest()


def _cache_get(key: str) -> tuple[Any, ...] | None:
    with _STACK_CACHE_LOCK:
        value = _STACK_CACHE.get(key)
        if value is not None:
            _STACK_CACHE.move_to_end(key)
        return value


def _cache_put(key: str, value: tuple[Any, ...]) -> None:
    with _STACK_CACHE_LOCK:
        _STACK_CACHE[key] = value
        _STACK_CACHE.move_to_end(key)
        while len(_STACK_CACHE) > _STACK_CACHE_MAX:
            _STACK_CACHE.popitem(last=False)


def _transactions_to_frame(transactions: list[FraudTransaction] | list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(transactions, pd.DataFrame):
        return transactions.copy()
    rows = []
    for item in transactions:
        if isinstance(item, FraudTransaction):
            rows.append(item.model_dump(mode="json"))
        else:
            rows.append(dict(item))
    return pd.DataFrame(rows)


def _require_labels(df: pd.DataFrame) -> np.ndarray:
    if "is_fraud" not in df.columns or df["is_fraud"].isna().all():
        raise ValueError("Labeled transactions (is_fraud) are required for training/evaluation.")
    return df["is_fraud"].astype(int).to_numpy()


def generate(request: FraudGenerateRequest) -> dict[str, Any]:
    df = generate_fraud_dataset(request)
    return {
        "transactions": df.to_dict(orient="records"),
        "meta": {
            "number_of_users": request.number_of_users,
            "number_of_transactions": int(len(df)),
            "requested_fraud_rate": request.fraud_rate,
            "realized_fraud_rate": realized_fraud_rate(df),
            "time_range_days": request.time_range_days,
            "seed": request.seed,
            "hard_fraud_rate": request.hard_fraud_rate,
            "suspicious_legit_rate": request.suspicious_legit_rate,
        },
    }


def _stratify_or_none(y: np.ndarray):
    return y if y.sum() > 1 and (y == 0).sum() > 1 else None


def _fit_stack_from_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    calibration_method: str | None,
) -> tuple[FraudEBMModel, Any, str, dict[str, int], list[str]]:
    """Fit EBM + calibrator using only the training partition.

    Holdout must never enter this function.
    """
    if len(y_train) < 20 or len(np.unique(y_train)) < 2:
        raise ValueError("Training partition is too small or single-class.")

    cache_key = _stack_cache_key(X_train, y_train, seed, calibration_method)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    reused_train_for_cal = False
    # Carve calibration fold from train only.
    if len(y_train) >= 80:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=seed,
            stratify=_stratify_or_none(y_train),
        )
    else:
        X_fit, y_fit = X_train, y_train
        X_cal, y_cal = X_train, y_train
        reused_train_for_cal = True

    model = FraudEBMModel().fit(X_fit, y_fit, seed=seed)
    raw_cal = model.predict_raw(X_cal)
    method = choose_calibration_method(len(y_cal), forced=calibration_method)  # type: ignore[arg-type]
    calibrator = fit_calibrator(raw_cal, y_cal, method=method)
    sizes = {
        "n_model_fit": int(len(y_fit)),
        "n_calibration": int(len(y_cal)),
        "n_train_total": int(len(y_train)),
    }
    warnings = calibration_safety_notes(
        n_calibration=sizes["n_calibration"],
        n_model_fit=sizes["n_model_fit"],
        reused_train_for_cal=reused_train_for_cal,
        requested_method=calibration_method,  # type: ignore[arg-type]
        chosen_method=method,  # type: ignore[arg-type]
    )
    payload = (model, calibrator, method, sizes, warnings)
    _cache_put(cache_key, payload)
    return payload


def _temporal_holdout_split(df: pd.DataFrame, y: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split by time so train never sees future holdout history.

    Rows are assumed sortable by timestamp. The earliest ``1 - test_size`` fraction
    is train; the latest ``test_size`` fraction is holdout. Features computed on the
    full chronological frame therefore cannot leak holdout events into earlier
    training rows (the failure mode of a random split after global featureization).
    """
    n = len(df)
    if n < 4:
        raise ValueError("Need at least 4 transactions for a temporal holdout split.")
    order = np.argsort(
        pd.to_datetime(df["timestamp"], utc=True).to_numpy(),
        kind="mergesort",
    )
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 2)
    n_train = n - n_test
    train_idx = order[:n_train]
    test_idx = order[n_train:]
    y_train = y[train_idx]
    y_test = y[test_idx]
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Temporal train fold is single-class. Increase sample size or fraud_rate, "
            "or shorten the holdout fraction."
        )
    if len(np.unique(y_test)) < 1:
        raise ValueError("Temporal holdout fold is empty.")
    return train_idx.astype(int), test_idx.astype(int)


def _run_holdout_protocol(
    df: pd.DataFrame,
    *,
    policy: PolicyConfig,
    costs: CostConfig,
    seed: int,
    test_size: float,
    calibration_method: str | None,
) -> dict[str, Any]:
    y = _require_labels(df)
    if len(np.unique(y)) < 2:
        raise ValueError("Both fraud and non-fraud examples are required.")

    # Chronological order before featureization so prior-history windows are well-defined.
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    y = _require_labels(df)

    train_idx, test_idx = _temporal_holdout_split(df, y, test_size)

    # Features on full chronological frame: train rows only see earlier (train) history.
    feature_df = compute_features(df)
    X, _ = feature_matrix(feature_df)
    baselines = dataset_baselines(feature_df.iloc[train_idx].reset_index(drop=True))

    model, calibrator, method, fit_sizes, cal_warnings = _fit_stack_from_train(
        X[train_idx],
        y[train_idx],
        seed=seed,
        calibration_method=calibration_method,
    )

    # Holdout untouched until final scoring.
    raw_test = model.predict_raw(X[test_idx])
    cal_test = calibrator.transform(raw_test)
    metrics = evaluate_scores(
        y[test_idx],
        cal_test,
        policy,
        costs,
        evaluation_mode="holdout",
        raw_scores=raw_test,
    )

    # Browse scores: same frozen stack applied to all rows (not used for metrics).
    raw_all = model.predict_raw(X)
    cal_all = calibrator.transform(raw_all)
    split_role = np.array(["train"] * len(df), dtype=object)
    split_role[test_idx] = "holdout"

    scored = []
    for i, row in df.iterrows():
        payload = score_row(
            raw_score=float(raw_all[i]),
            calibrated_probability=float(cal_all[i]),
            policy=policy,
        )
        scored.append(
            {
                "transaction_id": row["transaction_id"],
                "is_fraud": int(row["is_fraud"]) if "is_fraud" in row and pd.notna(row["is_fraud"]) else None,
                "split_role": str(split_role[i]),
                **payload,
            }
        )

    explanations = model.feature_contributions(X)
    max_additivity = max((float(e["additivity_error"]) for e in explanations), default=0.0)
    additivity_warnings: list[str] = []
    if max_additivity > ADDITIVITY_MAX_ERROR:
        additivity_warnings.append(
            f"Max EBM additivity error {max_additivity:.4f} exceeds soft cap {ADDITIVITY_MAX_ERROR}."
        )
    all_warnings = list(cal_warnings) + additivity_warnings
    if all_warnings:
        metrics["calibration_warnings"] = all_warnings

    return {
        "df": df,
        "feature_df": feature_df,
        "baselines": baselines,
        "X": X,
        "y": y,
        "model": model,
        "calibrator": calibrator,
        "calibration_method": method,
        "calibration_warnings": all_warnings,
        "raw_all": raw_all,
        "cal_all": cal_all,
        "explanations": explanations,
        "metrics": metrics,
        "scored_transactions": scored,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "fit_sizes": fit_sizes,
        "max_additivity_error": max_additivity,
        "split": {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_model_fit": fit_sizes["n_model_fit"],
            "n_calibration": fit_sizes["n_calibration"],
            "test_size": test_size,
            "seed": seed,
            "protocol": "temporal_split__train_cal_from_train__metrics_on_holdout",
            "split_method": "temporal",
            "calibration_warnings": all_warnings,
            "max_additivity_error": max_additivity,
        },
    }


def evaluate(request: FraudEvaluateRequest) -> dict[str, Any]:
    df = _transactions_to_frame(request.transactions)
    state = _run_holdout_protocol(
        df,
        policy=request.policy,
        costs=request.costs,
        seed=request.seed,
        test_size=request.test_size,
        calibration_method=request.calibration_method,
    )
    return {
        "evaluation": state["metrics"],
        "evaluation_mode": "holdout",
        "model_version": MODEL_VERSION,
        "feature_version": "v2",
        "calibration_method": state["calibration_method"],
        "calibration_warnings": state.get("calibration_warnings") or [],
        "policy": policy_payload(request.policy),
        "feature_baselines": state["baselines"],
        "feature_names": MODEL_FEATURE_NAMES,
        "scored_transactions": state["scored_transactions"],
        "split": state["split"],
        "notes": {
            "metrics": "Computed only on the temporal holdout fold after model and calibrator were frozen.",
            "scored_transactions": (
                "Scored with the frozen train/cal stack for browsing. "
                "Rows with split_role=holdout are the latest time slice; "
                "train rows precede holdout and are not used in reported metrics."
            ),
            "split": "Temporal split: earliest rows train, latest rows holdout (no random leakage).",
            "fpr_fnr": "Defined among non-REVIEW decisions (BLOCK=positive).",
            "cost": (
                "Default cost charges REVIEW with review_cost only. "
                "Set costs.charge_false_negative_on_review to also charge FN on REVIEW∧fraud."
            ),
        },
    }


def analyze(request: FraudAnalyzeRequest) -> dict[str, Any]:
    """Inspect one transaction using the same holdout protocol as evaluate.

    Same seed + same test_size + same transactions ⇒ identical raw/calibrated/decision
    values as the scored_transactions table from evaluate.
    """
    df = _transactions_to_frame(request.transactions)
    state = _run_holdout_protocol(
        df,
        policy=request.policy,
        costs=request.costs,
        seed=request.seed,
        test_size=request.test_size,
        calibration_method=request.calibration_method,
    )

    target_id = request.transaction_id
    if target_id is None:
        # Prefer highest-risk holdout row; fall back to global max.
        holdout_mask = np.array([s["split_role"] == "holdout" for s in state["scored_transactions"]])
        cal = state["cal_all"]
        if holdout_mask.any():
            holdout_pos = np.where(holdout_mask)[0]
            target_pos = int(holdout_pos[np.argmax(cal[holdout_mask])])
        else:
            target_pos = int(np.argmax(cal))
        target_id = str(state["df"].iloc[target_pos]["transaction_id"])

    matches = state["df"].index[state["df"]["transaction_id"] == target_id].tolist()
    if not matches:
        raise KeyError(f"transaction_id not found: {target_id}")
    pos = int(matches[0])

    tx = state["df"].iloc[pos].to_dict()
    if hasattr(tx.get("timestamp"), "isoformat"):
        tx["timestamp"] = tx["timestamp"].isoformat()

    scored = score_row(
        raw_score=float(state["raw_all"][pos]),
        calibrated_probability=float(state["cal_all"][pos]),
        policy=request.policy,
    )
    split_role = state["scored_transactions"][pos]["split_role"]

    evidence = build_evidence(
        transaction=tx,
        features=provenance_for_row(state["feature_df"], target_id, state["baselines"]),
        explanation=state["explanations"][pos],
        raw_score=scored["raw_score"],
        calibrated_probability=scored["risk_probability"],
        calibration_method=state["calibration_method"],
        policy=request.policy,
        costs=request.costs,
        model_version=MODEL_VERSION,
        evaluation_mode="holdout",
        metrics_snapshot=state["metrics"],
        split_role=split_role,
    )

    # Sanity: evidence decision must match scored table for this id.
    table_decision = state["scored_transactions"][pos]["decision"]
    if evidence["decision"] != table_decision:
        raise RuntimeError(
            f"Decision inconsistency for {target_id}: evidence={evidence['decision']} table={table_decision}"
        )

    return {
        "evidence": evidence,
        "evaluation": state["metrics"],
        "evaluation_mode": "holdout",
        "model_version": MODEL_VERSION,
        "calibration_method": state["calibration_method"],
        "policy": policy_payload(request.policy),
        "scored_transaction": {
            "transaction_id": target_id,
            "split_role": split_role,
            **scored,
        },
        "split": state["split"],
        "notes": {
            "metrics": "Holdout metrics from the untouched evaluation fold.",
            "evidence": (
                "Scores/decisions use the same frozen train/cal stack as /fraud/evaluate "
                "for the same seed and transactions."
            ),
        },
    }


class FraudPipeline:
    """Thin façade for tests and future streaming feature stores."""

    def generate(self, request: FraudGenerateRequest) -> dict[str, Any]:
        return generate(request)

    def evaluate(self, request: FraudEvaluateRequest) -> dict[str, Any]:
        return evaluate(request)

    def analyze(self, request: FraudAnalyzeRequest) -> dict[str, Any]:
        return analyze(request)
