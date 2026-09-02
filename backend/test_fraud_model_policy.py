from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.calibration import brier_score, choose_calibration_method, fit_calibrator
from fraud.evaluation import binary_metrics_at_threshold, evaluate_scores, expected_cost_for_decisions
from fraud.features import compute_features, feature_matrix
from fraud.generator import generate_fraud_dataset
from fraud.model import FraudEBMModel
from fraud.policy import decide
from fraud.schema import CostConfig, FraudGenerateRequest, PolicyConfig


def test_policy_boundaries():
    policy = PolicyConfig(allow_threshold=0.2, review_threshold=0.55, block_threshold=0.55)
    assert decide(0.19, policy) == "ALLOW"
    assert decide(0.2, policy) == "REVIEW"
    assert decide(0.549, policy) == "REVIEW"
    assert decide(0.55, policy) == "BLOCK"
    assert decide(0.99, policy) == "BLOCK"


def test_calibration_method_choice_and_brier():
    assert choose_calibration_method(50) == "sigmoid"
    assert choose_calibration_method(250) == "isotonic"
    assert choose_calibration_method(250, forced="sigmoid") == "sigmoid"

    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=300)
    y = (raw > 0.7).astype(int)
    cal = fit_calibrator(raw, y, method="sigmoid")
    probs = cal.transform(raw)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    score = brier_score(y, probs)
    assert 0.0 <= score <= 1.0


def test_ebm_train_predict_contributions_reproducible():
    df = generate_fraud_dataset(
        FraudGenerateRequest(
            number_of_users=80,
            number_of_transactions=800,
            fraud_rate=0.08,
            seed=7,
        )
    )
    feats = compute_features(df)
    X, _ = feature_matrix(feats)
    y = df["is_fraud"].astype(int).to_numpy()

    m1 = FraudEBMModel().fit(X, y, seed=11)
    m2 = FraudEBMModel().fit(X, y, seed=11)
    p1 = m1.predict_proba(X[:20])
    p2 = m2.predict_proba(X[:20])
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5)

    explanations = m1.feature_contributions(X[:3])
    assert len(explanations) == 3
    assert "contributions" in explanations[0]
    assert "positive_contributions" in explanations[0]
    assert "negative_contributions" in explanations[0]
    if explanations[0]["contributions"]:
        assert "feature" in explanations[0]["contributions"][0]
        assert "contribution" in explanations[0]["contributions"][0]
    # Contributions must correspond to the model logit (no fabrication).
    assert explanations[0]["additivity_error"] < 0.20


def test_evaluation_imbalanced_and_zero_positive():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.15, 0.8, 0.7, 0.3, 0.05, 0.9, 0.4, 0.1])
    metrics = evaluate_scores(y, p, PolicyConfig(), CostConfig())
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "expected_cost" in metrics
    assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"] == 3

    zero_pos = binary_metrics_at_threshold(np.zeros(10, dtype=int), p, 0.5)
    assert zero_pos["recall"] == 0.0
    assert zero_pos["confusion_matrix"]["tp"] == 0

    costs = expected_cost_for_decisions(
        np.array([0, 1, 0]),
        ["BLOCK", "ALLOW", "REVIEW"],
        CostConfig(false_positive_cost=5, false_negative_cost=100, review_cost=2),
    )
    assert costs["expected_cost"] == 5 + 100 + 2


def test_generator_reproducible_and_behavioral():
    req = FraudGenerateRequest(number_of_users=40, number_of_transactions=400, fraud_rate=0.05, seed=99)
    a = generate_fraud_dataset(req)
    b = generate_fraud_dataset(req)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["is_fraud"].unique()) == {0, 1}
    assert abs(a["is_fraud"].mean() - 0.05) < 0.03
