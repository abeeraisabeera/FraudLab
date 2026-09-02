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
    return <p className="text-xs text-stone-600">Insufficient positives for a PR curve.</p>;
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
          <CartesianGrid strokeDasharray="2 2" stroke="#8f8a7b" />
          <XAxis dataKey="recall" stroke="#111111" domain={[0, 1]} type="number" />
          <YAxis dataKey="precision" stroke="#111111" domain={[0, 1]} type="number" />
          <Tooltip />
          <Line type="monotone" dataKey="precision" stroke="#1d3328" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function CalibrationChart({ data }: { data: FraudEvaluation["calibration_curve"] }) {
  if (!data.length) {
    return <p className="text-xs text-stone-600">Calibration curve unavailable for this fold.</p>;
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
          <CartesianGrid strokeDasharray="2 2" stroke="#8f8a7b" />
          <XAxis
            dataKey="mean_predicted_probability"
            type="number"
            domain={[0, 1]}
            stroke="#111111"
            allowDuplicatedCategory={false}
          />
          <YAxis dataKey="fraction_positive" type="number" domain={[0, 1]} stroke="#111111" />
          <Tooltip />
          <Line
            data={diagonal}
            type="linear"
            dataKey="fraction_positive"
            stroke="#8f8a7b"
            strokeDasharray="4 4"
            dot={false}
            name="perfect"
          />
          <Line
            data={data}
            type="monotone"
            dataKey="fraction_positive"
            stroke="#9c6f2c"
            strokeWidth={2}
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
    { label: "TN", value: matrix.tn, tone: "bg-stone-100" },
    { label: "FP", value: matrix.fp, tone: "bg-[#f3e7d3]" },
    { label: "FN", value: matrix.fn, tone: "bg-[#f3e7d3]" },
    { label: "TP", value: matrix.tp, tone: "bg-stone-100" },
  ];
  return (
    <div
      className="grid grid-cols-2 gap-2"
      role="table"
      aria-label={`Confusion matrix TN ${matrix.tn}, FP ${matrix.fp}, FN ${matrix.fn}, TP ${matrix.tp}`}
    >
      {cells.map((cell) => (
        <div key={cell.label} className={`border border-ink p-3 ${cell.tone}`} role="cell">
          <div className="text-[11px] uppercase tracking-[0.18em] text-stone-600">{cell.label}</div>
          <div className="mt-1 text-xl font-bold">{cell.value}</div>
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
    return <p className="text-xs text-stone-600">No threshold sweep points.</p>;
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
          <CartesianGrid strokeDasharray="2 2" stroke="#8f8a7b" />
          <XAxis dataKey="review_threshold" stroke="#111111" />
          <YAxis stroke="#111111" />
          <Tooltip />
          <Bar dataKey="expected_cost" fill="#1d3328" />
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
  const colors = ["#1d3328", "#9c6f2c", "#4b5563"];
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
          <CartesianGrid strokeDasharray="2 2" stroke="#8f8a7b" />
          <XAxis dataKey="name" stroke="#111111" />
          <YAxis stroke="#111111" domain={[0, 1]} />
          <Tooltip />
          <Bar dataKey="value">
            {data.map((entry, index) => (
              <Cell key={entry.name} fill={colors[index % colors.length]} />
            ))}
          </Bar>
          <ReferenceLine y={0} stroke="#111111" />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
