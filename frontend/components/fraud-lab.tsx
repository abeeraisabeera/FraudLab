"use client";

import { useMemo, useState } from "react";
import {
  ArrowDownLeft,
  ArrowRight,
  ArrowUpRight,
  CalendarDays,
  FileText,
  FolderOpen,
  LayoutDashboard,
  Menu,
  Plus,
  Receipt,
  Search,
  Shield,
  SlidersHorizontal,
  UserRound,
  Users,
} from "lucide-react";

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
import { cn } from "@/lib/utils";

type MainTab = "dashboard" | "transactions" | "policy" | "evidence";
type SideNav = "generate" | "policy" | "evidence" | "model" | "charts" | "more";

function formatPct(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function decisionTone(decision: string) {
  if (decision === "BLOCK") return "bg-ink text-white";
  if (decision === "REVIEW") return "bg-blush text-white";
  return "bg-mint text-ink";
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

const TABS: { id: MainTab; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "transactions", label: "Transactions" },
  { id: "policy", label: "Policy" },
  { id: "evidence", label: "Evidence" },
];

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
  const [tab, setTab] = useState<MainTab>("dashboard");
  const [sideNav, setSideNav] = useState<SideNav>("generate");
  const [query, setQuery] = useState("");

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
  const evalBlock = evaluation?.evaluation ?? null;

  const evaluatedPolicy = evaluation?.policy;
  const policyDirty =
    evaluation != null &&
    (Math.abs((evaluatedPolicy?.allow_threshold ?? allowThreshold) - allowThreshold) > 1e-9 ||
      Math.abs((evaluatedPolicy?.review_threshold ?? reviewThreshold) - reviewThreshold) > 1e-9 ||
      Math.abs((evaluation.evaluation.costs?.false_positive_cost ?? fpCost) - fpCost) > 1e-9 ||
      Math.abs((evaluation.evaluation.costs?.false_negative_cost ?? fnCost) - fnCost) > 1e-9 ||
      Math.abs((evaluation.evaluation.costs?.review_cost ?? reviewCost) - reviewCost) > 1e-9);

  const filteredScored = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = [...scored].sort((a, b) => b.risk_probability - a.risk_probability);
    if (!q) return rows;
    return rows.filter(
      (row) =>
        row.transaction_id.toLowerCase().includes(q) ||
        row.decision.toLowerCase().includes(q) ||
        (row.split_role ?? "").toLowerCase().includes(q),
    );
  }, [scored, query]);

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
      setTab("dashboard");
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
      setTab("dashboard");
      setSideNav("evidence");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze transaction.");
    } finally {
      setLoading(null);
    }
  };

  const sideItems: {
    id: SideNav;
    label: string;
    icon: typeof Users;
    onClick: () => void;
  }[] = [
    {
      id: "generate",
      label: "Generate",
      icon: Users,
      onClick: () => {
        setSideNav("generate");
        setTab("policy");
      },
    },
    {
      id: "policy",
      label: "Policy",
      icon: SlidersHorizontal,
      onClick: () => {
        setSideNav("policy");
        setTab("policy");
      },
    },
    {
      id: "evidence",
      label: "Evidence",
      icon: FolderOpen,
      onClick: () => {
        setSideNav("evidence");
        setTab("evidence");
      },
    },
    {
      id: "model",
      label: "Model",
      icon: FileText,
      onClick: () => {
        setSideNav("model");
        setTab("dashboard");
      },
    },
    {
      id: "charts",
      label: "Charts",
      icon: CalendarDays,
      onClick: () => {
        setSideNav("charts");
        setTab("dashboard");
      },
    },
    {
      id: "more",
      label: "More",
      icon: Menu,
      onClick: () => {
        setSideNav("more");
        setTab("transactions");
      },
    },
  ];

  const metricCards = [
    {
      gradient: "bg-grad-coral",
      label: "PR-AUC",
      value: evalBlock ? evalBlock.pr_auc.toFixed(4) : "—",
      hint: evalBlock ? "Holdout ranking quality" : "Run evaluate to unlock",
      icon: "A",
    },
    {
      gradient: "bg-grad-violet",
      label: "Precision",
      value: evalBlock ? formatPct(evalBlock.precision) : "—",
      hint: evalBlock ? `Recall ${formatPct(evalBlock.recall)}` : "Awaiting model fold",
      icon: "P",
    },
    {
      gradient: "bg-grad-sky",
      label: "Expected cost",
      value: evalBlock ? evalBlock.expected_cost.toFixed(2) : "—",
      hint: evalBlock ? `Review ${formatPct(evalBlock.review_rate)}` : "Policy sweep pending",
      icon: "C",
    },
  ];

  const policyControls = (
    <div className="space-y-4">
      <div className="space-y-3">
        <h2 className="font-display text-sm font-semibold">Synthetic Generator</h2>
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
        <Button onClick={runGenerate} disabled={loading !== null} className="w-full" variant="accent">
          {loading === "generate" ? "Generating…" : "Generate Fraud Dataset"}
        </Button>
        {generateMeta ? (
          <div className="rounded-2xl bg-white/70 p-3 text-xs leading-relaxed text-muted">
            <div>Users: {generateMeta.number_of_users.toLocaleString()}</div>
            <div>Transactions: {generateMeta.number_of_transactions.toLocaleString()}</div>
            <div>Realized fraud rate: {formatPct(generateMeta.realized_fraud_rate)}</div>
          </div>
        ) : null}
      </div>

      <div className="space-y-3 border-t border-white/50 pt-4">
        <h2 className="font-display text-sm font-semibold">Decision Policy</h2>
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
        <p className="text-[11px] leading-relaxed text-muted">
          LOW → ALLOW · MEDIUM → REVIEW · HIGH → BLOCK. Thresholds are versioned separately from the
          EBM.
        </p>
        <Button
          onClick={runEvaluate}
          disabled={loading !== null || !transactions.length}
          className="w-full"
        >
          {loading === "evaluate" ? "Evaluating…" : "Train · Calibrate · Evaluate"}
        </Button>
      </div>
    </div>
  );

  return (
    <div className="mx-auto grid min-h-screen max-w-[1500px] grid-cols-1 gap-4 p-3 md:p-5 lg:grid-cols-[320px_minmax(0,1fr)] lg:gap-5 lg:p-6">
      <aside className="flex max-h-[calc(100vh-1.5rem)] flex-col gap-5 overflow-y-auto rounded-[2rem] border border-white/50 bg-white/35 p-5 shadow-soft backdrop-blur-xl lg:sticky lg:top-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-grad-coral shadow-card">
              <Shield className="h-4 w-4 text-white" aria-hidden />
            </div>
            <div>
              <div className="font-display text-xl font-bold tracking-tight">Fraud Lab
    
              </div>
              <div className="text-[11px] font-medium text-muted" lang="ur" dir="rtl">
              Evidence before automation
              </div>
            </div>
          </div>
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/80 shadow-card">
            <UserRound className="h-5 w-5 text-muted" aria-hidden />
          </div>
        </div>

        <div className="relative">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search"
            className="rounded-full border-white/90 bg-white/80 pl-10"
            aria-label="Search transactions"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-2xl bg-white/70 p-3 shadow-card">
            <div className="flex items-center gap-1 text-[11px] font-medium text-muted">
              <ArrowUpRight className="h-3.5 w-3.5 text-mint" />
              Fraud rate
            </div>
            <div className="mt-1 font-display text-lg font-bold text-mint">
              {generateMeta ? `+${formatPct(generateMeta.realized_fraud_rate)}` : "—"}
            </div>
          </div>
          <div className="rounded-2xl bg-white/70 p-3 shadow-card">
            <div className="flex items-center gap-1 text-[11px] font-medium text-muted">
              <ArrowDownLeft className="h-3.5 w-3.5 text-muted" />
              Exp. cost
            </div>
            <div className="mt-1 font-display text-lg font-bold text-ink">
              {evalBlock ? `-${evalBlock.expected_cost.toFixed(0)}` : "—"}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2.5">
          {sideItems.map((item) => {
            const Icon = item.icon;
            const active = sideNav === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={item.onClick}
                className={cn(
                  "flex aspect-square flex-col items-center justify-center gap-1.5 rounded-2xl transition",
                  active
                    ? "bg-white text-ink shadow-lift"
                    : "bg-white/25 text-muted hover:bg-white/50",
                )}
              >
                <Icon className="h-5 w-5" />
                <span className="text-[10px] font-semibold">{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="rounded-[1.5rem] bg-white/55 p-4 shadow-card">{policyControls}</div>

        <p className="text-xs leading-relaxed text-muted">
          Synthetic banking / fintech workstation · not a production gateway. Evidence before
          automation.
        </p>
      </aside>

      <section className="glass-panel flex min-h-[calc(100vh-2.5rem)] flex-col rounded-[2rem] p-4 md:p-6">
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <nav className="flex flex-wrap items-center gap-1 rounded-full bg-soft/80 p-1">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setTab(item.id);
                  if (item.id === "policy") setSideNav("policy");
                  if (item.id === "evidence") setSideNav("evidence");
                  if (item.id === "dashboard") setSideNav("charts");
                  if (item.id === "transactions") setSideNav("more");
                }}
                className={cn(
                  "rounded-full px-4 py-2 text-sm font-semibold transition",
                  tab === item.id ? "bg-blush text-white shadow-card" : "text-muted hover:text-ink",
                )}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <Button
            variant="secondary"
            onClick={() => {
              if (!transactions.length) void runGenerate();
              else void runEvaluate();
            }}
            disabled={loading !== null}
            className="rounded-full"
          >
            {!transactions.length ? (
              <>
                <Plus className="h-4 w-4" />
                Generate dataset
              </>
            ) : (
              <>
                <Receipt className="h-4 w-4" />
                Run evaluate
              </>
            )}
          </Button>
        </div>

        {error ? (
          <div
            className="mb-4 rounded-2xl border border-rose/30 bg-rose/10 px-4 py-3 text-sm text-ink"
            role="alert"
            aria-live="assertive"
          >
            {error}
          </div>
        ) : null}

        {policyDirty ? (
          <div
            data-testid="fraud-policy-dirty"
            className="mb-4 rounded-2xl border border-lilac/40 bg-lilac/15 px-4 py-3 text-xs leading-relaxed text-ink"
            role="status"
            aria-live="polite"
          >
            Policy or cost inputs changed since the last evaluate. Scored table and holdout metrics
            still reflect the previous thresholds — run{" "}
            <span className="font-bold">Train · Calibrate · Evaluate</span> again before inspecting
            evidence.
          </div>
        ) : null}

        {tab === "dashboard" || tab === "policy" || tab === "evidence" ? (
          <div
            className={cn(
              "flex flex-1 flex-col gap-6",
              (tab === "policy" || tab === "evidence") && "hidden",
            )}
          >
            <div className="flex items-end justify-between gap-3">
              <h1 className="font-display text-3xl font-bold tracking-tight md:text-4xl">Dashboard</h1>
              <div className="hidden items-center gap-2 text-xs font-medium text-muted sm:flex">
                <LayoutDashboard className="h-4 w-4" />
                EBM · calibration · policy
              </div>
            </div>

            {evidence ? (
              <div data-testid="fraud-evidence" className="grid gap-4 lg:grid-cols-3">
                <Card className="overflow-hidden bg-grad-coral text-white lg:col-span-1">
                  <CardHeader>
                    <CardTitle className="text-white/90">Risk Probability</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="font-display text-4xl font-bold">
                      {formatPct(evidence.risk_probability)}
                    </div>
                    <div className="mt-2 text-xs text-white/80">
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
                      className={cn(
                        "inline-block rounded-full px-4 py-2 text-sm font-bold uppercase tracking-wider",
                        decisionTone(evidence.decision),
                      )}
                    >
                      {evidence.decision}
                    </div>
                    <div className="mt-3 text-xs text-muted">
                      Policy {evidence.policy_version} · Model {evidence.model_version}
                    </div>
                  </CardContent>
                </Card>
                <Card className="lg:col-span-1">
                  <CardHeader>
                    <CardTitle>Transaction</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1 text-sm">
                    <div className="font-semibold">{evidence.transaction_id}</div>
                    <div className="text-muted">
                      {evidence.transaction.amount} {evidence.transaction.currency} ·{" "}
                      {evidence.transaction.merchant_category}
                    </div>
                    <div className="text-muted">
                      {evidence.transaction.country} / bill {evidence.transaction.billing_country}
                    </div>
                    <div className="truncate text-muted">{evidence.transaction.device_id}</div>
                  </CardContent>
                </Card>
              </div>
            ) : null}

            <div className="flex items-stretch gap-3 overflow-x-auto pb-1">
              {metricCards.map((card) => (
                <div
                  key={card.label}
                  className={cn(
                    "relative min-h-[200px] min-w-[200px] flex-1 overflow-hidden rounded-[1.75rem] p-5 text-white shadow-lift",
                    card.gradient,
                  )}
                >
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/25 font-display text-lg font-bold backdrop-blur-sm">
                    {card.icon}
                  </div>
                  <div className="mt-10 text-sm font-medium text-white/90">{card.hint}</div>
                  <div className="mt-1 font-display text-3xl font-bold tracking-tight">{card.value}</div>
                  <div className="mt-1 text-xs font-semibold uppercase tracking-wider text-white/75">
                    {card.label}
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => setTab("transactions")}
                className="flex h-[200px] w-12 shrink-0 items-center justify-center self-center rounded-full bg-white/80 text-ink shadow-card transition hover:bg-white"
                aria-label="View transactions"
              >
                <ArrowRight className="h-5 w-5" />
              </button>
            </div>

            <div className="grid flex-1 gap-4 lg:grid-cols-2">
              <Card className="bg-white/95">
                <CardHeader>
                  <CardTitle>Holdout balance</CardTitle>
                </CardHeader>
                <CardContent>
                  {evalBlock ? (
                    <div data-testid="fraud-holdout-metrics" className="space-y-4">
                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
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
                          <div key={label} className="rounded-2xl bg-soft/80 p-3">
                            <div className="text-[11px] font-medium text-muted">{label}</div>
                            <div className="mt-1 font-display text-lg font-bold">{value}</div>
                          </div>
                        ))}
                      </div>
                      <DecisionRateBars
                        allowRate={evalBlock.allow_rate}
                        reviewRate={evalBlock.review_rate}
                        blockRate={evalBlock.block_rate}
                      />
                      {meta ? (
                        <p className="text-xs text-muted">
                          Temporal holdout · train {meta.n_train.toLocaleString()}
                          {meta.n_calibration != null
                            ? ` (fit ${meta.n_model_fit?.toLocaleString()} / cal ${meta.n_calibration.toLocaleString()})`
                            : ""}
                          {" · "}holdout {meta.n_test.toLocaleString()} · seed {meta.seed}
                        </p>
                      ) : null}
                    </div>
                  ) : (
                    <p className="text-sm text-muted">
                      Generate a dataset and run evaluate to see holdout metrics and decision mix.
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card className="bg-white/95">
                <CardHeader>
                  <CardTitle>Recent transactions</CardTitle>
                </CardHeader>
                <CardContent>
                  {filteredScored.length ? (
                    <ul className="space-y-3">
                      {filteredScored.slice(0, 6).map((row) => {
                        const inbound = row.decision === "ALLOW";
                        return (
                          <li
                            key={row.transaction_id}
                            className="flex items-center gap-3 rounded-2xl bg-soft/60 px-3 py-2.5"
                          >
                            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-card">
                              <UserRound className="h-4 w-4 text-muted" />
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="truncate text-sm font-semibold">{row.transaction_id}</div>
                              <div className="text-xs text-muted">
                                {row.split_role ?? "—"} · {row.decision}
                              </div>
                            </div>
                            {inbound ? (
                              <ArrowUpRight className="h-4 w-4 text-mint" />
                            ) : (
                              <ArrowDownLeft className="h-4 w-4 text-muted" />
                            )}
                            <div
                              className={cn(
                                "min-w-[72px] text-right text-sm font-bold",
                                inbound ? "text-mint" : "text-ink",
                              )}
                            >
                              {formatPct(row.risk_probability)}
                            </div>
                            <button
                              type="button"
                              className="text-xs font-semibold text-blush underline-offset-2 hover:underline"
                              onClick={() => {
                                setSelectedId(row.transaction_id);
                                void runAnalyze(row.transaction_id);
                              }}
                            >
                              Open
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="text-sm text-muted">
                      Run evaluation to rank transactions by calibrated risk.
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>

            {evalBlock ? (
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
                <Card>
                  <CardHeader>
                    <CardTitle>Threshold / Cost</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ThresholdCostChart sweep={evalBlock.threshold_cost_analysis.sweep} />
                    {evalBlock.threshold_cost_analysis.best_thresholds ? (
                      <p className="mt-2 text-xs text-muted">
                        Lowest expected cost at allow=
                        {evalBlock.threshold_cost_analysis.best_thresholds.allow_threshold}, review=
                        {evalBlock.threshold_cost_analysis.best_thresholds.review_threshold} (cost{" "}
                        {evalBlock.threshold_cost_analysis.best_thresholds.expected_cost.toFixed(2)}
                        ).
                      </p>
                    ) : null}
                  </CardContent>
                </Card>
              </div>
            ) : null}
          </div>
        ) : null}

        {tab === "policy" ? (
          <div className="space-y-4">
            <h1 className="font-display text-3xl font-bold tracking-tight">Policy</h1>
            <p className="text-sm text-muted">
              Generator and decision thresholds live in the sidebar — adjust them there, then run
              evaluate. Dashboard metrics refresh after each fold.
            </p>
            <Card className="bg-white/95">
              <CardContent className="space-y-2 py-6 text-sm text-muted">
                <p>
                  Deterministic features → Explainable Boosting Machine → probability calibration →
                  versioned decision policy.
                </p>
                <p className="text-xs">Synthetic banking / fintech workstation · not a production gateway</p>
              </CardContent>
            </Card>
          </div>
        ) : null}

        {tab === "transactions" ? (
          <div className="space-y-4">
            <h1 className="font-display text-3xl font-bold tracking-tight">Transactions</h1>
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
                    variant="accent"
                  >
                    {loading === "analyze" ? "Analyzing…" : "Inspect Evidence"}
                  </Button>
                </div>
                {filteredScored.length ? (
                  <div className="max-h-[480px] overflow-auto rounded-2xl">
                    <table className="min-w-full border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-soft text-left text-xs font-semibold uppercase tracking-wider text-muted">
                          <th className="py-3 pr-3">Id</th>
                          <th className="py-3 pr-3">Split</th>
                          <th className="py-3 pr-3">Label</th>
                          <th className="py-3 pr-3">Risk</th>
                          <th className="py-3 pr-3">Decision</th>
                          <th className="py-3">Inspect</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredScored.slice(0, 40).map((row) => (
                          <tr key={row.transaction_id} className="border-b border-soft/80">
                            <td className="py-3 pr-3 font-medium">{row.transaction_id}</td>
                            <td className="py-3 pr-3 text-muted">{row.split_role ?? "—"}</td>
                            <td className="py-3 pr-3">{row.is_fraud ?? "—"}</td>
                            <td className="py-3 pr-3">{formatPct(row.risk_probability)}</td>
                            <td className="py-3 pr-3">
                              <span
                                className={cn(
                                  "inline-flex rounded-full px-2.5 py-0.5 text-xs font-semibold",
                                  decisionTone(row.decision),
                                )}
                              >
                                {row.decision}
                              </span>
                            </td>
                            <td className="py-3">
                              <button
                                className="font-semibold text-blush underline-offset-2 hover:underline"
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
                  <p className="text-sm text-muted">
                    Run evaluation to rank transactions by calibrated risk.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        ) : null}

        {tab === "evidence" ? (
          <div className="space-y-4">
            <h1 className="font-display text-3xl font-bold tracking-tight">Evidence</h1>
            {!evidence ? (
              <Card>
                <CardContent className="py-8 text-sm text-muted">
                  Analyze a scored transaction to inspect feature provenance and EBM contributions.
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4 lg:grid-cols-3">
                <Card className="overflow-hidden bg-grad-coral text-white">
                  <CardHeader>
                    <CardTitle className="text-white/90">Risk Probability</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="font-display text-4xl font-bold">
                      {formatPct(evidence.risk_probability)}
                    </div>
                    <div className="mt-2 text-xs text-white/80">
                      Raw score {evidence.calibration.raw_score.toFixed(4)} ·{" "}
                      {evidence.calibration.calibration_method}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Decision</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div
                      className={cn(
                        "inline-block rounded-full px-4 py-2 text-sm font-bold uppercase tracking-wider",
                        decisionTone(evidence.decision),
                      )}
                    >
                      {evidence.decision}
                    </div>
                    <div className="mt-3 text-xs text-muted">
                      Policy {evidence.policy_version} · Model {evidence.model_version}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Transaction</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1 text-sm">
                    <div className="font-semibold">{evidence.transaction_id}</div>
                    <div className="text-muted">
                      {evidence.transaction.amount} {evidence.transaction.currency} ·{" "}
                      {evidence.transaction.merchant_category}
                    </div>
                    <div className="text-muted">
                      {evidence.transaction.country} / bill {evidence.transaction.billing_country}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Feature Evidence</CardTitle>
                </CardHeader>
                <CardContent>
                  {evidence ? (
                    <div className="max-h-[360px] overflow-auto">
                      <table className="min-w-full border-collapse text-sm">
                        <thead>
                          <tr className="border-b border-soft text-left text-xs font-semibold uppercase tracking-wider text-muted">
                            <th className="py-2 pr-2">Feature</th>
                            <th className="py-2 pr-2">Value</th>
                            <th className="py-2">Effect</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evidence.feature_contributions.slice(0, 17).map((row) => {
                            const prov = evidence.features.find((f) => f.feature === row.feature);
                            return (
                              <tr key={row.feature} className="border-b border-soft/80">
                                <td className="py-2 pr-2 font-medium">{row.feature}</td>
                                <td className="py-2 pr-2 text-muted">
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
                    <p className="text-sm text-muted">
                      Analyze a scored transaction to inspect feature provenance and EBM
                      contributions.
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
                    <div className="rounded-2xl bg-soft/80 p-3">
                      <div className="text-[11px] font-medium text-muted">EBM</div>
                      <div className="mt-1 font-display font-bold">
                        {evaluation?.model_version ?? analysis?.model_version ?? "ebm-v1"}
                      </div>
                    </div>
                    <div className="rounded-2xl bg-soft/80 p-3">
                      <div className="text-[11px] font-medium text-muted">Calibration</div>
                      <div className="mt-1 font-display font-bold">
                        {evaluation?.calibration_method ?? analysis?.calibration_method ?? "—"}
                      </div>
                    </div>
                    <div className="rounded-2xl bg-soft/80 p-3">
                      <div className="text-[11px] font-medium text-muted">PR-AUC</div>
                      <div className="mt-1 font-display font-bold">
                        {evalBlock ? evalBlock.pr_auc.toFixed(4) : "—"}
                      </div>
                    </div>
                    <div className="rounded-2xl bg-soft/80 p-3">
                      <div className="text-[11px] font-medium text-muted">Brier</div>
                      <div className="mt-1 font-display font-bold">
                        {evalBlock ? evalBlock.brier_score.toFixed(4) : "—"}
                      </div>
                    </div>
                  </div>
                  <div className="rounded-2xl bg-soft/60 p-3 text-xs leading-relaxed text-muted">
                    Accuracy is not the primary metric. Fraud is rare; PR-AUC, calibration, and
                    expected cost govern the workstation.
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        ) : null}

        {/* Inspect Evidence always available for tests when on dashboard after evaluate */}
        {tab === "dashboard" && transactions.length > 0 ? (
          <div className="mt-4 flex flex-wrap items-end gap-3 border-t border-soft/80 pt-4">
            <div className="min-w-[220px] flex-1">
              <Label htmlFor="tx-dash">Transaction id</Label>
              <Input
                id="tx-dash"
                value={selectedId}
                onChange={(event) => setSelectedId(event.target.value)}
                placeholder="TX-000123"
              />
            </div>
            <Button
              onClick={() => runAnalyze(selectedId || undefined)}
              disabled={loading !== null || !transactions.length}
              variant="accent"
            >
              {loading === "analyze" ? "Analyzing…" : "Inspect Evidence"}
            </Button>
          </div>
        ) : null}
      </section>
    </div>
  );
}
