import { expect, test } from "@playwright/test";

function makeFraudTransactions(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    transaction_id: `e2e-tx-${i}`,
    user_id: `u-${i % 10}`,
    timestamp: `2025-06-${String((i % 28) + 1).padStart(2, "0")}T12:00:00Z`,
    amount: 20 + (i % 40),
    currency: "USD",
    merchant_id: `m-${i % 5}`,
    merchant_category: "retail",
    payment_method: "card_online",
    device_id: `d-${i % 8}`,
    ip_address: "203.0.113.10",
    country: "US",
    billing_country: "US",
    account_age_days: 90,
    is_fraud: i % 15 === 0 ? 1 : 0,
  }));
}

test.beforeEach(async ({ page }) => {
  const fraudTxs = makeFraudTransactions(80);
  await page.route("**/fraud/generate", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        transactions: fraudTxs,
        meta: {
          number_of_users: 40,
          number_of_transactions: fraudTxs.length,
          requested_fraud_rate: 0.05,
          realized_fraud_rate: 0.05,
          time_range_days: 14,
          seed: 42,
        },
      }),
    });
  });
  await page.route("**/fraud/evaluate", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        evaluation: {
          evaluation_mode: "holdout",
          pr_auc: 0.4512,
          brier_score: 0.12,
          precision: 0.4,
          recall: 0.35,
          f1: 0.37,
          false_positive_rate: 0.04,
          false_negative_rate: 0.65,
          expected_cost: 90,
          expected_cost_per_transaction: 1.5,
          review_rate: 0.12,
          block_rate: 0.06,
          allow_rate: 0.82,
          confusion_matrix: { tn: 40, fp: 2, fn: 4, tp: 2 },
          decision_counts: { ALLOW: 50, REVIEW: 8, BLOCK: 4 },
          precision_recall_curve: [{ threshold: 0.5, precision: 0.4, recall: 0.35 }],
          calibration_curve: [{ mean_predicted_probability: 0.2, fraction_positive: 0.18, n: 10 }],
          threshold_cost_analysis: { sweep: [], best_thresholds: null },
          policy: { version: "policy-v1", allow_threshold: 0.2, review_threshold: 0.55, block_threshold: 0.55 },
          costs: { false_positive_cost: 5, false_negative_cost: 100, review_cost: 2 },
          n_samples: 24,
          n_positives: 2,
          prevalence: 0.05,
        },
        evaluation_mode: "holdout",
        model_version: "ebm-v1",
        feature_version: "v2",
        calibration_method: "isotonic",
        policy: { version: "policy-v1", allow_threshold: 0.2, review_threshold: 0.55, block_threshold: 0.55 },
        feature_baselines: {},
        feature_names: [],
        scored_transactions: fraudTxs.map((t, i) => ({
          transaction_id: t.transaction_id,
          is_fraud: t.is_fraud,
          raw_score: 0.1,
          risk_probability: i === 0 ? 0.88 : 0.12,
          decision: i === 0 ? "BLOCK" : "ALLOW",
          split_role: i % 3 === 0 ? "holdout" : "train",
        })),
        split: { n_train: 56, n_test: 24, test_size: 0.3, seed: 42, split_method: "temporal" },
      }),
    });
  });
});

test("fraud lab happy path: generate → evaluate holdout metrics", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/Synthetic banking/i)).toBeVisible();
  await page.getByRole("button", { name: /Generate Fraud Dataset/i }).click();
  await page.getByRole("button", { name: /Train · Calibrate · Evaluate/i }).click();
  await expect(page.getByTestId("fraud-holdout-metrics")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByTestId("fraud-holdout-metrics")).toContainText("0.4512");
});
