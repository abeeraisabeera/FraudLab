"use client";

import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FraudEvaluation } from "@/lib/fraud-types";

const AXIS = "#6b6b80";
const GRID = "#e8dff0";
const INK = "#1a1a2e";
const BLUSH = "#ff8fab";
const VIOLET = "#8b5cf6";
const MINT = "#34d399";
const MUTED = "#94a3b8";

function ChartFrame({
  label,
  children,
  table,
}: {
  label: string;
  children: ReactNode;
  table?: ReactNode;
}) {
  return (
    <div role="img" aria-label={label}>
      {children}
      {table ? <div className="sr-only">{table}</div> : null}
    </div>
  );
}

export function PrecisionRecallChart({
  data,
}: {
  data: FraudEvaluation["precision_recall_curve"];
}) {
  if (!data.length) {
    return <p className="text-xs text-muted">Insufficient positives for a PR curve.</p>;
  }
  return (
    <ChartFrame
      label="Precision-recall curve"
      table={
        <table>
          <caption>Precision-recall points</caption>
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Precision</th>
              <th>Recall</th>
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 12).map((row) => (
              <tr key={`${row.threshold}-${row.recall}`}>
                <td>{row.threshold.toFixed(3)}</td>
                <td>{row.precision.toFixed(3)}</td>
                <td>{row.recall.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="recall" stroke={AXIS} domain={[0, 1]} type="number" />
          <YAxis dataKey="precision" stroke={AXIS} domain={[0, 1]} type="number" />
          <Tooltip />
          <Line type="monotone" dataKey="precision" stroke={VIOLET} strokeWidth={2.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function CalibrationChart({ data }: { data: FraudEvaluation["calibration_curve"] }) {
  if (!data.length) {
    return <p className="text-xs text-muted">Calibration curve unavailable for this fold.</p>;
  }
  const diagonal = [
    { mean_predicted_probability: 0, fraction_positive: 0 },
    { mean_predicted_probability: 1, fraction_positive: 1 },
  ];
  return (
    <ChartFrame
      label="Calibration reliability diagram"
      table={
        <table>
          <caption>Calibration bins</caption>
          <thead>
            <tr>
              <th>Mean predicted</th>
              <th>Fraction positive</th>
              <th>n</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx}>
                <td>{row.mean_predicted_probability.toFixed(3)}</td>
                <td>{row.fraction_positive.toFixed(3)}</td>
                <td>{row.n ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <ResponsiveContainer width="100%" height={240}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis
            dataKey="mean_predicted_probability"
            type="number"
            domain={[0, 1]}
            stroke={AXIS}
            allowDuplicatedCategory={false}
          />
          <YAxis dataKey="fraction_positive" type="number" domain={[0, 1]} stroke={AXIS} />
          <Tooltip />
          <Line
            data={diagonal}
            type="linear"
            dataKey="fraction_positive"
            stroke={MUTED}
            strokeDasharray="4 4"
            dot={false}
            name="perfect"
          />
          <Line
            data={data}
            type="monotone"
            dataKey="fraction_positive"
            stroke={BLUSH}
            strokeWidth={2.5}
            name="model"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function ConfusionMatrixBlock({
  matrix,
}: {
  matrix: FraudEvaluation["confusion_matrix"];
}) {
  const cells = [
    { label: "TN", value: matrix.tn, tone: "bg-soft" },
    { label: "FP", value: matrix.fp, tone: "bg-blush/15" },
    { label: "FN", value: matrix.fn, tone: "bg-blush/15" },
    { label: "TP", value: matrix.tp, tone: "bg-mint/20" },
  ];
  return (
    <div
      className="grid grid-cols-2 gap-2"
      role="table"
      aria-label={`Confusion matrix TN ${matrix.tn}, FP ${matrix.fp}, FN ${matrix.fn}, TP ${matrix.tp}`}
    >
      {cells.map((cell) => (
        <div key={cell.label} className={`rounded-2xl p-3 ${cell.tone}`} role="cell">
          <div className="text-[11px] font-medium text-muted">{cell.label}</div>
          <div className="mt-1 font-display text-xl font-bold">{cell.value}</div>
        </div>
      ))}
    </div>
  );
}

export function ThresholdCostChart({
  sweep,
}: {
  sweep: FraudEvaluation["threshold_cost_analysis"]["sweep"];
}) {
  if (!sweep.length) {
    return <p className="text-xs text-muted">No threshold sweep points.</p>;
  }
  const byReview = new Map<number, number>();
  for (const row of sweep) {
    const prev = byReview.get(row.review_threshold);
    if (prev === undefined || row.expected_cost < prev) {
      byReview.set(row.review_threshold, row.expected_cost);
    }
  }
  const data = Array.from(byReview.entries())
    .map(([review_threshold, expected_cost]) => ({ review_threshold, expected_cost }))
    .sort((a, b) => a.review_threshold - b.review_threshold);

  return (
    <ChartFrame
      label="Expected cost by review threshold"
      table={
        <table>
          <caption>Threshold cost sweep</caption>
          <thead>
            <tr>
              <th>Review threshold</th>
              <th>Expected cost</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.review_threshold}>
                <td>{row.review_threshold}</td>
                <td>{row.expected_cost.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="review_threshold" stroke={AXIS} />
          <YAxis stroke={AXIS} />
          <Tooltip />
          <Bar dataKey="expected_cost" fill={BLUSH} radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function DecisionRateBars({
  allowRate,
  reviewRate,
  blockRate,
}: {
  allowRate: number;
  reviewRate: number;
  blockRate: number;
}) {
  const data = [
    { name: "ALLOW", value: allowRate },
    { name: "REVIEW", value: reviewRate },
    { name: "BLOCK", value: blockRate },
  ];
  const colors = [MINT, BLUSH, INK];
  return (
    <ChartFrame
      label="Decision rate bars for ALLOW, REVIEW, and BLOCK"
      table={
        <table>
          <caption>Decision rates</caption>
          <thead>
            <tr>
              <th>Decision</th>
              <th>Rate</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
                <td>{(row.value * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
          <XAxis dataKey="name" stroke={AXIS} />
          <YAxis stroke={AXIS} domain={[0, 1]} />
          <Tooltip />
          <Bar dataKey="value" radius={[8, 8, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={entry.name} fill={colors[index % colors.length]} />
            ))}
          </Bar>
          <ReferenceLine y={0} stroke={AXIS} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
