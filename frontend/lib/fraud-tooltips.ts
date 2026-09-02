export const FRAUD_TOOLTIPS = {
  search: "Filter scored transactions by id, decision, or train/holdout split.",
  fraudRateSummary: "Share of generated transactions marked as fraud (synthetic ground truth).",
  expectedCostSummary: "Estimated business cost on the holdout set from missed fraud, false blocks, and manual reviews.",

  number_of_users: "How many synthetic customer accounts to create.",
  number_of_transactions: "Total payment rows to generate (100–20,000). Need at least 50 before evaluate.",
  fraud_rate: "Target fraction of transactions that are fraud. Actual rate may differ slightly.",
  hard_fraud_rate: "Among fraud cases, how many look obviously fraudulent vs subtle.",
  suspicious_legit_rate: "Share of legitimate txs that look suspicious (tests false positives).",
  time_range_days: "Spread transaction timestamps across this many days.",
  seed: "Same seed → identical dataset every time. Change seed for a new random sample.",

  generateButton: "Build a fresh synthetic banking dataset with the settings above.",
  allowThreshold: "Risk below this → ALLOW automatically. Lower = stricter auto-approve.",
  reviewThreshold: "Risk between allow and this → REVIEW (human check). At or above → BLOCK.",
  fpCost: "Cost when a legitimate transaction is wrongly blocked (customer friction + ops).",
  fnCost: "Cost when fraud is allowed through (chargebacks, loss). Usually much higher than FP.",
  reviewCost: "Cost per transaction sent to manual review (analyst time).",
  evaluateButton: "Train EBM on train split, calibrate probabilities, score all txs, report holdout metrics.",
  policyNote: "Model predicts risk; policy decides action. They are versioned separately.",

  tabDashboard: "Overview: KPI cards, holdout metrics, charts, and quick evidence.",
  tabTransactions: "Full scored table — sort and open any transaction.",
  tabPolicy: "Notes on generator and policy (controls live in sidebar).",
  tabEvidence: "Deep dive: feature contributions and model info for one transaction.",

  sideGenerate: "Jump to generator settings in the sidebar.",
  sidePolicy: "Jump to threshold and cost controls.",
  sideEvidence: "Open evidence tab for inspected transaction details.",
  sideModel: "Dashboard model summary tiles.",
  sideCharts: "Dashboard charts section.",
  sideMore: "Open full transactions table.",

  runCtaGenerate: "Shortcut: generate dataset if none exists yet.",
  runCtaEvaluate: "Shortcut: re-run train · calibrate · evaluate on current data.",

  holdoutBalance: "Metrics computed only on the holdout fold (~30%) the model never trained on.",
  prAuc: "Can the model rank fraud above normal txs? 1.0 = perfect; ~prevalence ≈ random guessing.",
  precision: "Of auto-BLOCK decisions, how many were actually fraud. REVIEW does not count as caught.",
  recall: "Of fraud in ALLOW/BLOCK decisions, how many were BLOCKed. REVIEW fraud is excluded here.",
  f1: "Balance of precision and recall. Low when either is low.",
  brier: "Calibration error — are predicted probabilities honest? Lower is better.",
  fpr: "Legitimate txs wrongly blocked, among ALLOW/BLOCK only (not REVIEW).",
  fnr: "Fraud wrongly allowed, among ALLOW/BLOCK only. 100% means no fraud was auto-blocked.",
  expectedCost: "FP×cost + FN×cost + reviews×review cost on holdout.",
  reviewRate: "Fraction of holdout txs sent to manual review.",
  prevalence: "Fraction of holdout txs that are actually fraud (ground truth).",
  holdoutSplitNote: "Temporal split: train for learning, holdout for honest scoring. Seed controls reproducibility.",

  metricCardPrAuc: "Primary ranking quality on holdout. Better than accuracy when fraud is rare.",
  metricCardPrecision: "Auto-block precision at current policy. See holdout panel for full breakdown.",
  metricCardCost: "Total expected operational cost on holdout at current thresholds and costs.",

  recentTransactions: "Highest-risk scored transactions. Click Open to inspect evidence.",
  confusionMatrix: "TN/FP/FN/TP using BLOCK as “predicted fraud”. REVIEW rows are tracked separately.",
  tn: "True negative — legit transaction allowed.",
  fp: "False positive — legit transaction blocked.",
  fn: "False negative — fraud allowed (missed).",
  tp: "True positive — fraud blocked.",
  prChart: "Trade-off between precision and recall at different score cutoffs.",
  calibrationChart: "Predicted probability vs actual fraud rate per bin. Closer to diagonal = better calibrated.",
  thresholdCostChart: "Expected cost across allow/review threshold pairs to find cheaper policies.",
  decisionMix: "Share of ALLOW, REVIEW, and BLOCK decisions on holdout.",

  riskProbability: "Calibrated fraud probability (0–100%) after isotonic/Platt scaling.",
  decision: "ALLOW = auto approve · REVIEW = manual check · BLOCK = auto reject.",
  transaction: "Raw transaction fields used to build features.",
  featureEvidence: "Which features pushed risk up or down (EBM contributions).",
  modelPanel: "Model version, calibration method, and holdout metrics snapshot.",

  transactionId: "Paste or pick a transaction id, then inspect evidence.",
  inspectEvidence: "Explain one transaction: risk score, decision, and feature breakdown.",
  scoredTable: "All transactions ranked by calibrated risk after evaluate.",

  colId: "Unique synthetic transaction identifier.",
  colSplit: "train = used for learning · holdout = untouched test fold.",
  colLabel: "Ground truth: 1 = fraud, 0 = legitimate (synthetic only).",
  colRisk: "Calibrated fraud probability from the model.",
  colDecision: "Policy outcome: ALLOW, REVIEW, or BLOCK.",
  colInspect: "Open full evidence for this row.",

  ebm: "Explainable Boosting Machine — glass-box model with per-feature effects.",
  calibration: "Method that maps raw scores to calibrated probabilities.",
  policyDirty: "You changed thresholds or costs since last evaluate. Re-run evaluate before trusting metrics or evidence alignment.",

  featureCol: "Engineered signal name (may include feature interactions).",
  valueCol: "Feature value for this transaction.",
  effectCol: "Contribution to raw risk score. Positive = pushes toward fraud.",
} as const;

export type FraudTooltipKey = keyof typeof FRAUD_TOOLTIPS;

export function tip(key: FraudTooltipKey): string {
  return FRAUD_TOOLTIPS[key];
}
