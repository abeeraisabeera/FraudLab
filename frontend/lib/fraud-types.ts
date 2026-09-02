export type FraudDecision = "ALLOW" | "REVIEW" | "BLOCK";

export type FraudTransaction = {
  transaction_id: string;
  user_id: string;
  timestamp: string;
  amount: number;
  currency: string;
  merchant_id: string;
  merchant_category: string;
  payment_method: string;
  device_id: string;
  ip_address: string;
  country: string;
  billing_country: string;
  account_age_days: number;
  failed_attempts?: number;
  previous_chargebacks?: number;
  shipping_country?: string | null;
  is_fraud?: number | null;
};

export type FraudGeneratePayload = {
  number_of_users: number;
  number_of_transactions: number;
  fraud_rate: number;
  time_range_days: number;
  seed: number;
  hard_fraud_rate?: number;
  suspicious_legit_rate?: number;
};

export type FraudPolicy = {
  version: string;
  allow_threshold: number;
  review_threshold: number;
  block_threshold: number;
  mapping?: Record<string, string>;
};

export type FraudCosts = {
  false_positive_cost: number;
  false_negative_cost: number;
  review_cost: number;
  charge_false_negative_on_review?: boolean;
};

export type FeatureProvenance = {
  feature: string;
  value: number | null;
  baseline: number | null;
  definition: string;
  source: string;
  version: string;
  note?: string;
};

export type FeatureContribution = {
  feature: string;
  value?: number | null;
  contribution: number;
  definition?: string;
  term_type?: "main" | "interaction" | string;
  members?: string[];
};

export type FraudEvidence = {
  transaction_id: string;
  transaction: FraudTransaction;
  risk_probability: number;
  decision: FraudDecision;
  top_contributors: FeatureContribution[];
  counter_signals: FeatureContribution[];
  features: FeatureProvenance[];
  feature_contributions: FeatureContribution[];
  model: {
    model_version: string;
    raw_score: number;
    calibrated_probability: number;
    calibration_method: string;
    pr_auc?: number;
    brier_score?: number;
  };
  calibration: {
    raw_score: number;
    calibrated_probability: number;
    calibration_method: string;
  };
  policy: FraudPolicy;
  policy_version: string;
  model_version: string;
  thresholds: {
    allow_threshold: number;
    review_threshold: number;
    block_threshold: number;
    version: string;
  };
  cost_assumptions: FraudCosts;
  evaluation_mode?: "holdout" | string;
  split_role?: "train" | "holdout" | string;
  positive_contributions?: FeatureContribution[];
  negative_contributions?: FeatureContribution[];
  explanation?: {
    intercept?: number;
    contribution_sum?: number;
    model_logit?: number;
    additivity_error?: number;
  };
};

export type FraudEvaluation = {
  evaluation_mode?: "holdout" | string;
  pr_auc: number;
  brier_score: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  false_negative_rate: number;
  expected_cost: number;
  expected_cost_per_transaction: number;
  review_rate: number;
  block_rate: number;
  allow_rate: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  decision_counts: Record<FraudDecision, number>;
  precision_recall_curve: Array<{ threshold: number; precision: number; recall: number }>;
  calibration_curve: Array<{
    mean_predicted_probability: number;
    fraction_positive: number;
    n?: number;
    bin_lower?: number;
    bin_upper?: number;
  }>;
  threshold_cost_analysis: {
    sweep: Array<{
      allow_threshold: number;
      review_threshold: number;
      expected_cost: number;
      expected_cost_per_transaction: number;
      precision: number;
      recall: number;
      f1: number;
      review_rate: number;
      false_positive_rate: number;
      false_negative_rate: number;
      tp?: number;
      fp?: number;
      tn?: number;
      fn?: number;
      reviews?: number;
    }>;
    best_thresholds: {
      allow_threshold: number;
      review_threshold: number;
      expected_cost: number;
      precision: number;
      recall: number;
      f1: number;
      review_rate: number;
    } | null;
  };
  policy: {
    version: string;
    allow_threshold: number;
    review_threshold: number;
    block_threshold: number;
  };
  costs: FraudCosts;
  n_samples: number;
  n_positives: number;
  prevalence: number;
};

export type FraudGenerateResponse = {
  transactions: FraudTransaction[];
  meta: {
    number_of_users: number;
    number_of_transactions: number;
    requested_fraud_rate: number;
    realized_fraud_rate: number;
    time_range_days: number;
    seed: number;
  };
};

export type FraudEvaluateResponse = {
  evaluation: FraudEvaluation;
  evaluation_mode?: "holdout" | string;
  model_version: string;
  feature_version: string;
  calibration_method: string;
  policy: FraudPolicy;
  feature_baselines: Record<string, number>;
  feature_names: string[];
  scored_transactions: Array<{
    transaction_id: string;
    is_fraud: number | null;
    raw_score: number;
    risk_probability: number;
    decision: FraudDecision;
    split_role?: "train" | "holdout" | string;
  }>;
  split: {
    n_train: number;
    n_test: number;
    test_size: number;
    seed: number;
    n_model_fit?: number;
    n_calibration?: number;
    protocol?: string;
    split_method?: string;
  };
  notes?: Record<string, string>;
};

export type FraudAnalyzeResponse = {
  evidence: FraudEvidence;
  evaluation: FraudEvaluation;
  evaluation_mode?: "holdout" | string;
  model_version: string;
  calibration_method: string;
  policy: FraudPolicy;
  scored_transaction?: {
    transaction_id: string;
    split_role?: string;
    raw_score: number;
    risk_probability: number;
    decision: FraudDecision;
  };
  split?: FraudEvaluateResponse["split"];
  notes?: Record<string, string>;
};
