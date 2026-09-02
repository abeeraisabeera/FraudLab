"""Phase 0 P1: temporal holdout + analyze/evaluate contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.features import compute_features
from fraud.generator import generate_fraud_dataset
from fraud.pipeline import analyze, evaluate, _temporal_holdout_split
from fraud.schema import FraudAnalyzeRequest, FraudEvaluateRequest, FraudGenerateRequest, PolicyConfig


def test_temporal_split_train_precedes_holdout():
    df = generate_fraud_dataset(
        FraudGenerateRequest(number_of_users=80, number_of_transactions=800, fraud_rate=0.06, seed=11)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    y = df["is_fraud"].astype(int).to_numpy()
    train_idx, test_idx = _temporal_holdout_split(df, y, test_size=0.3)
    train_max = df.iloc[train_idx]["timestamp"].max()
    test_min = df.iloc[test_idx]["timestamp"].min()
    assert train_max <= test_min


def test_no_holdout_history_in_train_features():
    """Regression: train rows must not use later holdout txs as prior history.

    With a temporal split + chronological featureization, every prior event for a
    train row is also in the train window. We assert that for a train row, no
    holdout transaction shares the same user with an earlier-or-equal timestamp
    that could have been counted — equivalently, all same-user priors are train.
    """
    df = generate_fraud_dataset(
        FraudGenerateRequest(number_of_users=100, number_of_transactions=1000, fraud_rate=0.05, seed=21)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    y = df["is_fraud"].astype(int).to_numpy()
    train_idx, test_idx = _temporal_holdout_split(df, y, 0.3)
    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())

    for i in train_idx:
        user = df.iloc[i]["user_id"]
        ts = df.iloc[i]["timestamp"]
        priors = df.index[(df["user_id"] == user) & (df["timestamp"] < ts)].tolist()
        for p in priors:
            assert p in train_set, "Train feature history must not include holdout rows"
            assert p not in test_set


def test_analyze_respects_test_size_match_evaluate():
    df = generate_fraud_dataset(
        FraudGenerateRequest(number_of_users=90, number_of_transactions=900, fraud_rate=0.07, seed=8)
    )
    txs = df.to_dict(orient="records")
    policy = PolicyConfig(allow_threshold=0.2, review_threshold=0.55, block_threshold=0.55)
    ev = evaluate(FraudEvaluateRequest(transactions=txs, seed=8, test_size=0.25, policy=policy))
    assert ev["split"]["test_size"] == 0.25
    assert ev["split"]["split_method"] == "temporal"

    target = next(r for r in ev["scored_transactions"] if r["split_role"] == "holdout")["transaction_id"]
    an = analyze(
        FraudAnalyzeRequest(
            transactions=txs,
            transaction_id=target,
            seed=8,
            test_size=0.25,
            policy=policy,
        )
    )
    assert an["split"]["test_size"] == 0.25
    scored = next(r for r in ev["scored_transactions"] if r["transaction_id"] == target)
    assert an["evidence"]["decision"] == scored["decision"]
    assert abs(an["evidence"]["risk_probability"] - scored["risk_probability"]) < 1e-9
