"use client";

import { useMemo, useState } from "react";

import {
  CalibrationChart,
  ConfusionMatrixBlock,
  DecisionRateBars,
  PrecisionRecallChart,
  ThresholdCostChart,
} from "@/components/fraud-charts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { analyzeFraudTransaction, evaluateFraudDataset, generateFraudDataset } from "@/lib/api";
import type {
  FraudAnalyzeResponse,
  FraudEvaluateResponse,
  FraudGeneratePayload,
  FraudTransaction,
} from "@/lib/fraud-types";

function formatPct(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function decisionTone(decision: string) {
  if (decision === "BLOCK") return "bg-[#3b2f1f] text-paper";
  if (decision === "REVIEW") return "bg-[#9c6f2c] text-paper";
  return "bg-olive text-paper";
}

const defaultGenerator: FraudGeneratePayload = {
  number_of_users: 300,
  number_of_transactions: 2500,
  fraud_rate: 0.04,
  time_range_days: 30,
  seed: 42,
  hard_fraud_rate: 0.25,
  suspicious_legit_rate: 0.18,
};

export function FraudLab() {
  const [generator, setGenerator] = useState<FraudGeneratePayload>(defaultGenerator);
  const [allowThreshold, setAllowThreshold] = useState(0.2);
  const [reviewThreshold, setReviewThreshold] = useState(0.55);
  const [fpCost, setFpCost] = useState(5);
  const [fnCost, setFnCost] = useState(100);
  const [reviewCost, setReviewCost] = useState(2);

  const [transactions, setTransactions] = useState<FraudTransaction[]>([]);
  const [meta, setMeta] = useState<FraudEvaluateResponse["split"] | null>(null);
  const [generateMeta, setGenerateMeta] = useState<{
    realized_fraud_rate: number;
    number_of_transactions: number;
    number_of_users: number;
  } | null>(null);
  const [evaluation, setEvaluation] = useState<FraudEvaluateResponse | null>(null);
  const [analysis, setAnalysis] = useState<FraudAnalyzeResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const policy = useMemo(
    () => ({
      version: "policy-v1",
      allow_threshold: allowThreshold,
      review_threshold: reviewThreshold,
      block_threshold: reviewThreshold,
    }),
    [allowThreshold, reviewThreshold],
  );

  const costs = useMemo(
    () => ({
      false_positive_cost: fpCost,
      false_negative_cost: fnCost,
      review_cost: reviewCost,
    }),
    [fpCost, fnCost, reviewCost],
  );

  const scored = evaluation?.scored_transactions ?? [];
  const evidence = analysis?.evidence;
  // Metrics always come from the last evaluate call — never from a stale analyze payload.
  const evalBlock = evaluation?.evaluation ?? null;

  const evaluatedPolicy = evaluation?.policy;
  const policyDirty =
    evaluation != null &&
    (Math.abs((evaluatedPolicy?.allow_threshold ?? allowThreshold) - allowThreshold) > 1e-9 ||
      Math.abs((evaluatedPolicy?.review_threshold ?? reviewThreshold) - reviewThreshold) > 1e-9 ||
      Math.abs((evaluation.evaluation.costs?.false_positive_cost ?? fpCost) - fpCost) > 1e-9 ||
      Math.abs((evaluation.evaluation.costs?.false_negative_cost ?? fnCost) - fnCost) > 1e-9 ||
      Math.abs((evaluation.evaluation.costs?.review_cost ?? reviewCost) - reviewCost) > 1e-9);

  const runGenerate = async () => {
    try {
      setLoading("generate");
      setError(null);
      const response = await generateFraudDataset(generator);
      setTransactions(response.transactions);
      setGenerateMeta({
        realized_fraud_rate: response.meta.realized_fraud_rate,
        number_of_transactions: response.meta.number_of_transactions,
        number_of_users: response.meta.number_of_users,
      });
      setEvaluation(null);
      setAnalysis(null);
      setSelectedId("");
      setMeta(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to generate fraud dataset.");
    } finally {
      setLoading(null);
    }
  };

  const runEvaluate = async () => {
    if (transactions.length < 50) {
      setError("Generate at least 50 transactions before evaluation.");
      return;
    }
    try {
      setLoading("evaluate");
      setError(null);
      // Clear prior inspect evidence so metrics/decision cannot diverge from the new evaluate.
      setAnalysis(null);
      const response = await evaluateFraudDataset({
        transactions,
        policy,
        costs,
        seed: generator.seed,
        test_size: 0.3,
      });
      setEvaluation(response);
      setMeta(response.split);
      const highest = [...response.scored_transactions].sort(
        (a, b) => b.risk_probability - a.risk_probability,
      )[0];
      if (highest) {
        setSelectedId(highest.transaction_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to evaluate fraud model.");
    } finally {
      setLoading(null);
    }
  };

  const runAnalyze = async (transactionId?: string) => {
    if (!transactions.length) {
      setError("Generate a dataset first.");
      return;
    }
    try {
      setLoading("analyze");
      setError(null);
      const response = await analyzeFraudTransaction({
        transactions,
        transaction_id: transactionId || selectedId || undefined,
        policy,
        costs,
        seed: generator.seed,
        test_size: evaluation?.split.test_size ?? 0.3,
      });
      setAnalysis(response);
      setSelectedId(response.evidence.transaction_id);
      // Keep evaluation metrics as the workstation source of truth; refresh scored table
      // alignment only happens via evaluate. Analyze updates evidence cards only.
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze transaction.");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="space-y-4">
      <Card className="bg-[#f0ebdf]">
        <CardHeader>
          <CardTitle>Fraud Lab</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm leading-relaxed">
          <p>
            Deterministic features → Explainable Boosting Machine → probability calibration →
            versioned decision policy. Evidence before automation.
          </p>
          <p className="text-xs uppercase tracking-[0.18em] text-stone-600">
            Synthetic banking / fintech workstation · not a production gateway
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[340px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Synthetic Generator</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {(
                [
                  ["number_of_users", "Users"],
                  ["number_of_transactions", "Transactions"],
                  ["fraud_rate", "Fraud rate"],
                  ["hard_fraud_rate", "Hard fraud rate"],
                  ["suspicious_legit_rate", "Suspicious legit rate"],
                  ["time_range_days", "Time range (days)"],
                  ["seed", "Seed"],
                ] as const
              ).map(([key, label]) => (
                <div key={key}>
                  <Label htmlFor={key}>{label}</Label>
                  <Input
                    id={key}
                    type="number"
                    step={
                      key === "fraud_rate" || key === "hard_fraud_rate" || key === "suspicious_legit_rate"
                        ? 0.01
                        : 1
                    }
                    value={generator[key] ?? 0}
                    onChange={(event) =>
                      setGenerator((prev) => ({
                        ...prev,
                        [key]: Number(event.target.value),
                      }))
                    }
                  />
                </div>
              ))}
              <Button onClick={runGenerate} disabled={loading !== null} className="w-full">
                {loading === "generate" ? "Generating…" : "Generate Fraud Dataset"}
              </Button>
              {generateMeta ? (
                <div className="border border-ink p-3 text-xs">
                  <div>Users: {generateMeta.number_of_users.toLocaleString()}</div>
                  <div>Transactions: {generateMeta.number_of_transactions.toLocaleString()}</div>
                  <div>Realized fraud rate: {formatPct(generateMeta.realized_fraud_rate)}</div>
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Decision Policy</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <Label htmlFor="allow">Allow threshold</Label>
                <Input
                  id="allow"
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={allowThreshold}
                  onChange={(event) => setAllowThreshold(Number(event.target.value))}
                />
              </div>
              <div>
                <Label htmlFor="review">Review / block threshold</Label>
                <Input
                  id="review"
                  type="number"
                  step={0.01}
                  min={0}
                  max={1}
                  value={reviewThreshold}
                  onChange={(event) => setReviewThreshold(Number(event.target.value))}
                />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <Label htmlFor="fp">FP cost</Label>
                  <Input id="fp" type="number" value={fpCost} onChange={(e) => setFpCost(Number(e.target.value))} />
                </div>
                <div>
                  <Label htmlFor="fn">FN cost</Label>
                  <Input id="fn" type="number" value={fnCost} onChange={(e) => setFnCost(Number(e.target.value))} />
                </div>
                <div>
                  <Label htmlFor="rv">Review</Label>
                  <Input
                    id="rv"
                    type="number"
                    value={reviewCost}
                    onChange={(e) => setReviewCost(Number(e.target.value))}
                  />
                </div>
              </div>
              <p className="text-[11px] leading-relaxed text-stone-600">
                LOW → ALLOW · MEDIUM → REVIEW · HIGH → BLOCK. Thresholds are versioned separately
                from the EBM.
              </p>
              <Button onClick={runEvaluate} disabled={loading !== null || !transactions.length} className="w-full">
                {loading === "evaluate" ? "Evaluating…" : "Train · Calibrate · Evaluate"}
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          {error ? (
            <div
              className="border border-ink bg-[#f3e7d3] px-4 py-3 text-sm"
              role="alert"
              aria-live="assertive"
            >
              {error}
            </div>
          ) : null}

          {policyDirty ? (
            <div
              data-testid="fraud-policy-dirty"
              className="border border-ink bg-[#f3e7d3] px-4 py-3 text-xs leading-relaxed"
              role="status"
              aria-live="polite"
            >
              Policy or cost inputs changed since the last evaluate. Scored table and holdout metrics still
              reflect the previous thresholds — run <span className="font-bold">Train · Calibrate · Evaluate</span>{" "}
              again before inspecting evidence.
            </div>
          ) : null}

          {evidence ? (
            <div data-testid="fraud-evidence" className="grid gap-4 lg:grid-cols-3">
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle>Risk Probability</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold">{formatPct(evidence.risk_probability)}</div>
                  <div className="mt-2 text-xs text-stone-600">
                    Raw score {evidence.calibration.raw_score.toFixed(4)} ·{" "}
                    {evidence.calibration.calibration_method}
                  </div>
                </CardContent>
              </Card>
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle>Decision</CardTitle>
                </CardHeader>
                <CardContent>
                  <div
                    className={`inline-block border border-ink px-4 py-2 text-lg font-bold uppercase tracking-[0.18em] ${decisionTone(evidence.decision)}`}
                  >
                    {evidence.decision}
                  </div>
                  <div className="mt-3 text-xs text-stone-600">
                    Policy {evidence.policy_version} · Model {evidence.model_version}
                  </div>
                </CardContent>
              </Card>
              <Card className="lg:col-span-1">
                <CardHeader>
                  <CardTitle>Transaction</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-xs">
                  <div>{evidence.transaction_id}</div>
                  <div>
                    {evidence.transaction.amount} {evidence.transaction.currency} ·{" "}
                    {evidence.transaction.merchant_category}
                  </div>
                  <div>
                    {evidence.transaction.country} / bill {evidence.transaction.billing_country}
                  </div>
                  <div className="truncate">{evidence.transaction.device_id}</div>
                </CardContent>
              </Card>
            </div>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Feature Evidence</CardTitle>
              </CardHeader>
              <CardContent>
                {evidence ? (
                  <div className="max-h-[360px] overflow-auto">
                    <table className="min-w-full border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-ink text-left uppercase tracking-[0.14em]">
                          <th className="py-2 pr-2">Feature</th>
                          <th className="py-2 pr-2">Value</th>
                          <th className="py-2">Effect</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evidence.feature_contributions.slice(0, 17).map((row) => {
                          const prov = evidence.features.find((f) => f.feature === row.feature);
                          return (
                            <tr key={row.feature} className="border-b border-stone-300">
                              <td className="py-2 pr-2 font-medium">{row.feature}</td>
                              <td className="py-2 pr-2">
                                {prov?.value == null ? "—" : Number(prov.value).toFixed(3)}
                              </td>
                              <td className="py-2">{Number(row.contribution).toFixed(4)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-xs text-stone-600">
                    Analyze a scored transaction to inspect feature provenance and EBM contributions.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="grid grid-cols-2 gap-2">
                  <div className="border border-ink p-3">
                    <div className="text-[11px] uppercase tracking-[0.18em] text-stone-600">EBM</div>
                    <div className="mt-1 font-bold">
                      {evaluation?.model_version ?? analysis?.model_version ?? "ebm-v1"}
                    </div>
                  </div>
                  <div className="border border-ink p-3">
                    <div className="text-[11px] uppercase tracking-[0.18em] text-stone-600">
                      Calibration
                    </div>
                    <div className="mt-1 font-bold">
                      {evaluation?.calibration_method ?? analysis?.calibration_method ?? "—"}
                    </div>
                  </div>
                  <div className="border border-ink p-3">
                    <div className="text-[11px] uppercase tracking-[0.18em] text-stone-600">PR-AUC</div>
                    <div className="mt-1 font-bold">
                      {evalBlock ? evalBlock.pr_auc.toFixed(4) : "—"}
                    </div>
                  </div>
                  <div className="border border-ink p-3">
                    <div className="text-[11px] uppercase tracking-[0.18em] text-stone-600">Brier</div>
                    <div className="mt-1 font-bold">
                      {evalBlock ? evalBlock.brier_score.toFixed(4) : "—"}
                    </div>
                  </div>
                </div>
                <div className="border border-ink p-3 text-xs leading-relaxed text-stone-700">
                  Accuracy is not the primary metric. Fraud is rare; PR-AUC, calibration, and expected
                  cost govern the workstation.
                </div>
              </CardContent>
            </Card>
          </div>

          {evalBlock ? (
            <>
              <Card data-testid="fraud-holdout-metrics">
                <CardHeader>
                  <CardTitle>Evaluation Workstation · Holdout</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="mb-3 text-xs leading-relaxed text-stone-600">
                    Metrics are computed only on the untouched holdout fold. Model fit and
                    calibration use the training partition exclusively. Scored train rows are for
                    browsing — not for reported metrics.
                  </p>
                  <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-5">
                    {(
                      [
                        ["PR-AUC", evalBlock.pr_auc.toFixed(4)],
                        ["Precision", formatPct(evalBlock.precision)],
                        ["Recall", formatPct(evalBlock.recall)],
                        ["F1", evalBlock.f1.toFixed(4)],
                        ["Brier", evalBlock.brier_score.toFixed(4)],
                        ["FPR", formatPct(evalBlock.false_positive_rate)],
                        ["FNR", formatPct(evalBlock.false_negative_rate)],
                        ["Expected cost", evalBlock.expected_cost.toFixed(2)],
                        ["Review rate", formatPct(evalBlock.review_rate)],
                        ["Prevalence", formatPct(evalBlock.prevalence)],
                      ] as const
                    ).map(([label, value]) => (
                      <div key={label} className="border border-ink p-3">
                        <div className="text-[11px] uppercase tracking-[0.18em] text-stone-600">
                          {label}
                        </div>
                        <div className="mt-2 text-lg font-bold">{value}</div>
                      </div>
                    ))}
                  </div>
                  {meta ? (
                    <p className="mt-3 text-xs text-stone-600">
                      Temporal holdout · train {meta.n_train.toLocaleString()}
                      {meta.n_calibration != null
                        ? ` (fit ${meta.n_model_fit?.toLocaleString()} / cal ${meta.n_calibration.toLocaleString()})`
                        : ""}
                      {" · "}holdout {meta.n_test.toLocaleString()} · seed {meta.seed}
                      {meta.split_method ? ` · split=${meta.split_method}` : ""}
                    </p>
                  ) : null}
                  {evalBlock.threshold_cost_analysis.best_thresholds ? (
                    <p className="mt-2 text-xs text-stone-700">
                      Lowest expected cost in sweep at allow=
                      {evalBlock.threshold_cost_analysis.best_thresholds.allow_threshold}, review=
                      {evalBlock.threshold_cost_analysis.best_thresholds.review_threshold} (cost{" "}
                      {evalBlock.threshold_cost_analysis.best_thresholds.expected_cost.toFixed(2)}).
                    </p>
                  ) : null}
                </CardContent>
              </Card>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Confusion Matrix</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ConfusionMatrixBlock matrix={evalBlock.confusion_matrix} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Decision Mix</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <DecisionRateBars
                      allowRate={evalBlock.allow_rate}
                      reviewRate={evalBlock.review_rate}
                      blockRate={evalBlock.block_rate}
                    />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Precision–Recall</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <PrecisionRecallChart data={evalBlock.precision_recall_curve} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Calibration</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CalibrationChart data={evalBlock.calibration_curve} />
                  </CardContent>
                </Card>
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle>Threshold / Cost Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ThresholdCostChart sweep={evalBlock.threshold_cost_analysis.sweep} />
                  </CardContent>
                </Card>
              </div>
            </>
          ) : null}

          <Card>
            <CardHeader>
              <CardTitle>Scored Transactions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap items-end gap-3">
                <div className="min-w-[220px] flex-1">
                  <Label htmlFor="tx">Transaction id</Label>
                  <Input
                    id="tx"
                    value={selectedId}
                    onChange={(event) => setSelectedId(event.target.value)}
                    placeholder="TX-000123"
                  />
                </div>
                <Button
                  onClick={() => runAnalyze(selectedId || undefined)}
                  disabled={loading !== null || !transactions.length}
                >
                  {loading === "analyze" ? "Analyzing…" : "Inspect Evidence"}
                </Button>
              </div>
              {scored.length ? (
                <div className="max-h-[320px] overflow-auto">
                  <table className="min-w-full border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-ink text-left uppercase tracking-[0.14em]">
                        <th className="py-2 pr-2">Id</th>
                        <th className="py-2 pr-2">Split</th>
                        <th className="py-2 pr-2">Label</th>
                        <th className="py-2 pr-2">Risk</th>
                        <th className="py-2 pr-2">Decision</th>
                        <th className="py-2">Inspect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...scored]
                        .sort((a, b) => b.risk_probability - a.risk_probability)
                        .slice(0, 40)
                        .map((row) => (
                          <tr key={row.transaction_id} className="border-b border-stone-300">
                            <td className="py-2 pr-2">{row.transaction_id}</td>
                            <td className="py-2 pr-2">{row.split_role ?? "—"}</td>
                            <td className="py-2 pr-2">{row.is_fraud ?? "—"}</td>
                            <td className="py-2 pr-2">{formatPct(row.risk_probability)}</td>
                            <td className="py-2 pr-2">{row.decision}</td>
                            <td className="py-2">
                              <button
                                className="underline decoration-amber underline-offset-2"
                                onClick={() => {
                                  setSelectedId(row.transaction_id);
                                  void runAnalyze(row.transaction_id);
                                }}
                              >
                                Open
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-xs text-stone-600">
                  Run evaluation to rank transactions by calibrated risk.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
