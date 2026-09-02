import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { FraudLab } from "@/components/fraud-lab";
import type {
  FraudAnalyzeResponse,
  FraudEvaluateResponse,
  FraudGenerateResponse,
  FraudTransaction,
} from "@/lib/fraud-types";

vi.mock("@/components/fraud-charts", () => ({
  CalibrationChart: () => <div data-testid="chart-cal" />,
  ConfusionMatrixBlock: () => <div data-testid="chart-cm" />,
  DecisionRateBars: () => <div data-testid="chart-rates" />,
  PrecisionRecallChart: () => <div data-testid="chart-pr" />,
  ThresholdCostChart: () => <div data-testid="chart-cost" />,
}));

const api = vi.hoisted(() => ({
  generateFraudDataset: vi.fn(),
  evaluateFraudDataset: vi.fn(),
  analyzeFraudTransaction: vi.fn(),
}));

vi.mock("@/lib/api", () => api);

function makeTx(id: string, fraud = 0): FraudTransaction {
  return {
    transaction_id: id,
    user_id: "u1",
    timestamp: "2025-01-01T00:00:00Z",
    amount: 40,
    currency: "USD",
    merchant_id: "m1",
    merchant_category: "retail",
    payment_method: "card_online",
    device_id: "d1",
    ip_address: "1.1.1.1",
    country: "US",
    billing_country: "US",
    account_age_days: 100,
    is_fraud: fraud,
  };
}

function makeGenerate(n = 60): FraudGenerateResponse {
  const transactions = Array.from({ length: n }, (_, i) => makeTx(`tx-${i}`, i % 12 === 0 ? 1 : 0));
  return {
    transactions,
    meta: {
      number_of_users: 30,
      number_of_transactions: n,
      requested_fraud_rate: 0.04,
      realized_fraud_rate: 0.05,
      time_range_days: 30,
      seed: 42,
    },
  };
}

function makeEvaluate(prAuc = 0.4123): FraudEvaluateResponse {
  const txs = makeGenerate(60).transactions;
  return {
    evaluation: {
      evaluation_mode: "holdout",
      pr_auc: prAuc,
      brier_score: 0.11,
      precision: 0.4,
      recall: 0.3,
      f1: 0.34,
      false_positive_rate: 0.05,
      false_negative_rate: 0.7,
      expected_cost: 120,
      expected_cost_per_transaction: 2,
      review_rate: 0.1,
      block_rate: 0.05,
      allow_rate: 0.85,
      confusion_matrix: { tn: 50, fp: 3, fn: 5, tp: 2 },
      decision_counts: { ALLOW: 51, REVIEW: 6, BLOCK: 3 },
      precision_recall_curve: [{ threshold: 0.5, precision: 0.4, recall: 0.3 }],
      calibration_curve: [{ mean_predicted_probability: 0.2, fraction_positive: 0.18, n: 10 }],
      threshold_cost_analysis: {
        sweep: [],
        best_thresholds: {
          allow_threshold: 0.2,
          review_threshold: 0.55,
          expected_cost: 100,
          precision: 0.4,
          recall: 0.3,
          f1: 0.34,
          review_rate: 0.1,
        },
      },
      policy: { version: "policy-v1", allow_threshold: 0.2, review_threshold: 0.55, block_threshold: 0.55 },
      costs: { false_positive_cost: 5, false_negative_cost: 100, review_cost: 2 },
      n_samples: 18,
      n_positives: 2,
      prevalence: 0.05,
    },
    evaluation_mode: "holdout",
    model_version: "ebm-v1",
    feature_version: "features-v1",
    calibration_method: "isotonic",
    policy: {
      version: "policy-v1",
      allow_threshold: 0.2,
      review_threshold: 0.55,
      block_threshold: 0.55,
    },
    feature_baselines: {},
    feature_names: [],
    scored_transactions: txs.map((t, i) => ({
      transaction_id: t.transaction_id,
      is_fraud: t.is_fraud ?? 0,
      raw_score: 0.1,
      risk_probability: i === 0 ? 0.9 : 0.1,
      decision: i === 0 ? "BLOCK" : "ALLOW",
      split_role: i % 3 === 0 ? "holdout" : "train",
    })),
    split: {
      n_train: 42,
      n_test: 18,
      test_size: 0.3,
      seed: 42,
      split_method: "temporal",
    },
  };
}

function makeAnalyze(): FraudAnalyzeResponse {
  const evalResp = makeEvaluate();
  const tx = makeTx("tx-0", 1);
  return {
    evidence: {
      transaction_id: "tx-0",
      transaction: tx,
      risk_probability: 0.91,
      decision: "BLOCK",
      top_contributors: [],
      counter_signals: [],
      features: [],
      feature_contributions: [{ feature: "amount", value: 40, contribution: 0.2 }],
      model: {
        model_version: "ebm-v1",
        raw_score: 0.2,
        calibrated_probability: 0.91,
        calibration_method: "isotonic",
      },
      calibration: {
        raw_score: 0.2,
        calibrated_probability: 0.91,
        calibration_method: "isotonic",
      },
      policy: evalResp.policy,
      policy_version: "policy-v1",
      model_version: "ebm-v1",
      thresholds: {
        allow_threshold: 0.2,
        review_threshold: 0.55,
        block_threshold: 0.55,
        version: "policy-v1",
      },
      cost_assumptions: evalResp.evaluation.costs,
      positive_contributions: [],
      negative_contributions: [],
    },
    evaluation: {
      ...evalResp.evaluation,
      pr_auc: 0.9999, // stale analyze metric must NOT drive UI metrics
    },
    evaluation_mode: "holdout",
    model_version: "ebm-v1",
    calibration_method: "isotonic",
    policy: evalResp.policy,
  };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("FraudLab state machine", () => {
  it("disables evaluate until a dataset is generated", () => {
    render(<FraudLab />);
    expect(screen.getByRole("button", { name: /Train · Calibrate · Evaluate/i })).toBeDisabled();
  });

  it("generate → evaluate shows holdout metrics from evaluate only", async () => {
    const user = userEvent.setup();
    api.generateFraudDataset.mockResolvedValue(makeGenerate(60));
    api.evaluateFraudDataset.mockResolvedValue(makeEvaluate(0.4123));

    render(<FraudLab />);
    await user.click(screen.getByRole("button", { name: /Generate Fraud Dataset/i }));
    await waitFor(() => expect(api.generateFraudDataset).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: /Train · Calibrate · Evaluate/i }));
    await waitFor(() => expect(api.evaluateFraudDataset).toHaveBeenCalled());

    const metrics = await screen.findByTestId("fraud-holdout-metrics");
    expect(metrics).toHaveTextContent("0.4123");
    expect(screen.queryByTestId("fraud-evidence")).not.toBeInTheDocument();
  });

  it("re-evaluate clears prior evidence and ignores analyze pr_auc", async () => {
    const user = userEvent.setup();
    api.generateFraudDataset.mockResolvedValue(makeGenerate(60));
    api.evaluateFraudDataset
      .mockResolvedValueOnce(makeEvaluate(0.4123))
      .mockResolvedValueOnce(makeEvaluate(0.3333));
    api.analyzeFraudTransaction.mockResolvedValue(makeAnalyze());

    render(<FraudLab />);
    await user.click(screen.getByRole("button", { name: /Generate Fraud Dataset/i }));
    await waitFor(() => expect(api.generateFraudDataset).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: /Train · Calibrate · Evaluate/i }));
    await waitFor(() => expect(api.evaluateFraudDataset).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: /Inspect Evidence/i }));
    expect(await screen.findByTestId("fraud-evidence")).toBeInTheDocument();
    expect(screen.getByTestId("fraud-holdout-metrics")).toHaveTextContent("0.4123");
    expect(screen.getByTestId("fraud-holdout-metrics")).not.toHaveTextContent("0.9999");

    await user.click(screen.getByRole("button", { name: /Train · Calibrate · Evaluate/i }));
    await waitFor(() => expect(api.evaluateFraudDataset).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId("fraud-evidence")).not.toBeInTheDocument();
    expect(screen.getByTestId("fraud-holdout-metrics")).toHaveTextContent("0.3333");
  });

  it("shows policy dirty banner when thresholds change after evaluate", async () => {
    const user = userEvent.setup();
    api.generateFraudDataset.mockResolvedValue(makeGenerate(60));
    api.evaluateFraudDataset.mockResolvedValue(makeEvaluate(0.4));

    render(<FraudLab />);
    await user.click(screen.getByRole("button", { name: /Generate Fraud Dataset/i }));
    await waitFor(() => expect(api.generateFraudDataset).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: /Train · Calibrate · Evaluate/i }));
    await waitFor(() => expect(screen.getByTestId("fraud-holdout-metrics")).toBeInTheDocument());

    const allow = screen.getByLabelText(/Allow threshold/i);
    await user.clear(allow);
    await user.type(allow, "0.35");

    expect(await screen.findByTestId("fraud-policy-dirty")).toBeInTheDocument();
  });
});
