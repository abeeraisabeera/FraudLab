"""Statistical validity tests for Fraud Lab.

These tests protect trustworthiness of metrics and explanations — not high scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud.calibration import brier_score, calibration_curve_points, fit_calibrator
from fraud.decision import calibrated_decision, score_row
from fraud.evaluation import evaluate_scores, threshold_cost_sweep
from fraud.features import FEATURE_DEFINITIONS, MODEL_FEATURE_NAMES, compute_features, feature_matrix
from fraud.generator import generate_fraud_dataset
from fraud.model import FraudEBMModel
from fraud.pipeline import analyze, evaluate
from fraud.policy import decide
from fraud.schema import (
    CostConfig,
    FraudAnalyzeRequest,
    FraudEvaluateRequest,
    FraudGenerateRequest,
    PolicyConfig,
)


def _gen(**kwargs):
    defaults = dict(number_of_users=120, number_of_transactions=1200, fraud_rate=0.05, seed=42)
    defaults.update(kwargs)
    return generate_fraud_dataset(FraudGenerateRequest(**defaults))


def test_no_near_perfect_feature_label_separation():
    df = _gen(seed=11, number_of_transactions=2500, number_of_users=300, fraud_rate=0.05)
    y = df["is_fraud"].astype(int).to_numpy()
    feats = compute_features(df)
    X, cols = feature_matrix(feats)
    for i, name in enumerate(cols):
        x = X[:, i]
        if np.std(x) < 1e-12:
            continue
        auc = roc_auc_score(y, x)
        sep = max(auc, 1 - auc)
        assert sep < 0.92, f"{name} nearly separates labels (AUC={sep:.3f})"

    # No deterministic encoding: failed_attempts>=2 must occur in both classes.
    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]
    assert (fraud["failed_attempts"] >= 2).mean() < 0.95
    assert (legit["failed_attempts"] >= 2).mean() > 0.0
    # Shared devices appear in both classes.
    shared_fraud = fraud["device_id"].astype(str).str.contains("SHARED").mean()
    shared_legit = legit["device_id"].astype(str).str.contains("SHARED").mean()
    assert shared_fraud > 0.0 and shared_legit > 0.0


def test_legitimate_and_fraud_distributions_overlap():
    df = _gen(seed=22, number_of_transactions=2000)
    fraud = df[df["is_fraud"] == 1]["amount"]
    legit = df[df["is_fraud"] == 0]["amount"]
    # Interquartile ranges must overlap.
    f_q1, f_q3 = fraud.quantile(0.25), fraud.quantile(0.75)
    l_q1, l_q3 = legit.quantile(0.25), legit.quantile(0.75)
    assert f_q1 <= l_q3 and l_q1 <= f_q3


def test_holdout_never_used_for_train_or_calibration():
    df = _gen(seed=7, number_of_transactions=1000)
    txs = df.where(pd.notna(df), None).to_dict(orient="records")
    out = evaluate(FraudEvaluateRequest(transactions=txs, seed=7, test_size=0.3))
    assert out["evaluation_mode"] == "holdout"
    assert out["evaluation"]["evaluation_mode"] == "holdout"
    assert out["split"]["n_calibration"] + out["split"]["n_model_fit"] >= out["split"]["n_train"]
    # Holdout size matches scored holdout rows.
    n_holdout_scored = sum(1 for r in out["scored_transactions"] if r["split_role"] == "holdout")
    assert n_holdout_scored == out["split"]["n_test"]
    assert out["evaluation"]["n_samples"] == out["split"]["n_test"]


def test_evaluate_analyze_decision_consistency():
    df = _gen(seed=5, number_of_transactions=900)
    txs = df.to_dict(orient="records")
    policy = PolicyConfig(allow_threshold=0.2, review_threshold=0.55, block_threshold=0.55)
    ev = evaluate(FraudEvaluateRequest(transactions=txs, seed=5, policy=policy))
    by_id = {r["transaction_id"]: r for r in ev["scored_transactions"]}
    # Check several rows including train and holdout.
    for row in ev["scored_transactions"][:15]:
        an = analyze(
            FraudAnalyzeRequest(
                transactions=txs,
                transaction_id=row["transaction_id"],
                seed=5,
                policy=policy,
            )
        )
        evidence = an["evidence"]
        assert evidence["decision"] == row["decision"]
        assert abs(evidence["risk_probability"] - row["risk_probability"]) < 1e-9
        assert abs(evidence["calibration"]["raw_score"] - row["raw_score"]) < 1e-9
        # Canonical path
        assert evidence["decision"] == calibrated_decision(evidence["risk_probability"], policy)
        assert by_id[row["transaction_id"]]["decision"] == evidence["decision"]


def test_calibration_bins_include_counts_and_improve_brier_fixture():
    # Known fixture: raw scores are biased; calibration should reduce Brier.
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=800)
    raw = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.15, size=800), 0.01, 0.99)
    cal = fit_calibrator(raw, y, method="isotonic")
    probs = cal.transform(raw)
    bins = calibration_curve_points(y, probs, n_bins=8)
    assert bins
    assert all("n" in b and b["n"] > 0 for b in bins)
    assert all("mean_predicted_probability" in b and "fraction_positive" in b for b in bins)

    # Analytical Brier fixture
    y_fix = np.array([0, 0, 1, 1])
    p_fix = np.array([0.1, 0.2, 0.8, 0.9])
    expected = float(np.mean((p_fix - y_fix) ** 2))
    assert abs(brier_score(y_fix, p_fix) - expected) < 1e-12

    # Calibrated probs should be closer to empirical rates than constant 0.5 on this fixture
    assert brier_score(y, probs) < brier_score(y, np.full_like(raw, 0.5))


def test_threshold_cost_recomputes_confusion_and_changes():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.25, 0.4, 0.45, 0.6, 0.7, 0.9, 0.55, 0.3])
    costs = CostConfig(false_positive_cost=5, false_negative_cost=100, review_cost=2)
    sweep = threshold_cost_sweep(
        y,
        p,
        costs,
        allow_grid=[0.1, 0.3],
        review_grid=[0.4, 0.7],
    )
    assert sweep["sweep"]
    for row in sweep["sweep"]:
        assert set(row).issuperset({"tp", "fp", "tn", "fn", "reviews", "expected_cost"})
        assert row["tp"] + row["fp"] + row["tn"] + row["fn"] + row["reviews"] == len(y)

    costs_low = [r["expected_cost"] for r in sweep["sweep"]]
    assert len(set(costs_low)) > 1  # curve changes across thresholds

    # Threshold change must change decisions where expected.
    assert decide(0.35, PolicyConfig(allow_threshold=0.2, review_threshold=0.55)) == "REVIEW"
    assert decide(0.35, PolicyConfig(allow_threshold=0.4, review_threshold=0.7)) == "ALLOW"
    assert decide(0.8, PolicyConfig(allow_threshold=0.2, review_threshold=0.55)) == "BLOCK"


def test_shuffled_labels_destroy_performance():
    df = _gen(seed=9, number_of_transactions=1400, fraud_rate=0.06)
    txs = df.to_dict(orient="records")
    real = evaluate(FraudEvaluateRequest(transactions=txs, seed=9))
    shuffled = df.copy()
    rng = np.random.default_rng(9)
    shuffled["is_fraud"] = rng.permutation(shuffled["is_fraud"].to_numpy())
    junk = evaluate(FraudEvaluateRequest(transactions=shuffled.to_dict(orient="records"), seed=9))
    # With shuffled labels, PR-AUC should collapse toward prevalence.
    prevalence = float(df["is_fraud"].mean())
    assert junk["evaluation"]["pr_auc"] < real["evaluation"]["pr_auc"]
    assert junk["evaluation"]["pr_auc"] < prevalence + 0.15


def test_random_features_near_chance_discrimination():
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, len(MODEL_FEATURE_NAMES)))
    y = rng.binomial(1, 0.1, size=n)
    # Ensure both classes
    y[0] = 0
    y[1] = 1
    model = FraudEBMModel().fit(X[:560], y[:560], seed=0)
    p = model.predict_proba(X[560:])
    y_te = y[560:]
    from fraud.evaluation import pr_auc

    score = pr_auc(y_te, p)
    # Near-chance for imbalanced random data (allow loose bound).
    assert score < 0.35


def test_fraud_prevalence_does_not_change_feature_definitions():
    defs_a = dict(FEATURE_DEFINITIONS)
    _gen(fraud_rate=0.02, seed=1)
    _gen(fraud_rate=0.15, seed=1)
    assert FEATURE_DEFINITIONS == defs_a
    assert list(FEATURE_DEFINITIONS.keys()) == MODEL_FEATURE_NAMES


def test_seed_reproduces_dataset_and_holdout_metrics():
    req = FraudGenerateRequest(number_of_users=80, number_of_transactions=800, fraud_rate=0.05, seed=123)
    a = generate_fraud_dataset(req)
    b = generate_fraud_dataset(req)
    pd.testing.assert_frame_equal(a, b)
    ev1 = evaluate(FraudEvaluateRequest(transactions=a.to_dict(orient="records"), seed=123))
    ev2 = evaluate(FraudEvaluateRequest(transactions=b.to_dict(orient="records"), seed=123))
    assert abs(ev1["evaluation"]["pr_auc"] - ev2["evaluation"]["pr_auc"]) < 1e-9
    assert abs(ev1["evaluation"]["brier_score"] - ev2["evaluation"]["brier_score"]) < 1e-9
    assert ev1["scored_transactions"][0]["decision"] == ev2["scored_transactions"][0]["decision"]


def test_canonical_score_row_matches_policy():
    policy = PolicyConfig(allow_threshold=0.25, review_threshold=0.6, block_threshold=0.6)
    row = score_row(raw_score=0.71, calibrated_probability=0.58, policy=policy)
    assert row["decision"] == "REVIEW"
    assert row["risk_probability"] == 0.58
    assert row["decision"] == decide(0.58, policy)


def test_ebm_explanations_additive_and_signed():
    df = _gen(seed=3, number_of_transactions=700, fraud_rate=0.08)
    X, _ = feature_matrix(compute_features(df))
    y = df["is_fraud"].astype(int).to_numpy()
    model = FraudEBMModel().fit(X, y, seed=3)
    expl = model.feature_contributions(X[:5])
    for item in expl:
        assert "positive_contributions" in item and "negative_contributions" in item
        for c in item["positive_contributions"]:
            assert float(c["contribution"]) > 0
        for c in item["negative_contributions"]:
            assert float(c["contribution"]) < 0
        assert item["additivity_error"] < 0.20
