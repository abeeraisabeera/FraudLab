import type {
  FraudAnalyzeResponse,
  FraudEvaluateResponse,
  FraudGeneratePayload,
  FraudGenerateResponse,
  FraudTransaction,
  FraudCosts,
  FraudPolicy,
} from "@/lib/fraud-types";

function resolveApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const fallback = "http://127.0.0.1:8000";
  const base = configured && configured.length > 0 ? configured : fallback;
  const isProd = process.env.NODE_ENV === "production";
  const isLocalhost = /localhost|127\.0\.0\.1/i.test(base);
  if (isProd && isLocalhost) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE_URL must be set to your deployed API origin in production (refusing localhost).",
    );
  }
  return base.replace(/\/$/, "");
}

const API_BASE = resolveApiBase();

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = `API request failed: ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
      }
    } catch {
      // ignore parse errors
    }
    throw new Error(detail);
  }
  return response.json();
}

export function assertFraudGeneratePayload(payload: FraudGeneratePayload): void {
  if (payload.number_of_transactions < 100 || payload.number_of_transactions > 20000) {
    throw new Error("number_of_transactions must be between 100 and 20000.");
  }
  if (payload.number_of_users < 20 || payload.number_of_users > 5000) {
    throw new Error("number_of_users must be between 20 and 5000.");
  }
  if (payload.fraud_rate < 0.001 || payload.fraud_rate > 0.4) {
    throw new Error("fraud_rate must be between 0.001 and 0.4.");
  }
}

export async function generateFraudDataset(payload: FraudGeneratePayload): Promise<FraudGenerateResponse> {
  assertFraudGeneratePayload(payload);
  return request("/fraud/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function evaluateFraudDataset(payload: {
  transactions: FraudTransaction[];
  policy?: Partial<FraudPolicy>;
  costs?: FraudCosts;
  seed?: number;
  test_size?: number;
}): Promise<FraudEvaluateResponse> {
  if (payload.transactions.length < 50) {
    throw new Error("Need at least 50 transactions to evaluate.");
  }
  if (payload.transactions.length > 20000) {
    throw new Error("Too many transactions to evaluate (max 20000).");
  }
  return request("/fraud/evaluate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function analyzeFraudTransaction(payload: {
  transactions: FraudTransaction[];
  transaction_id?: string;
  policy?: Partial<FraudPolicy>;
  costs?: FraudCosts;
  seed?: number;
  test_size?: number;
}): Promise<FraudAnalyzeResponse> {
  if (!payload.transactions.length) {
    throw new Error("transactions must be non-empty.");
  }
  if (payload.transactions.length > 20000) {
    throw new Error("Too many transactions to analyze (max 20000).");
  }
  return request("/fraud/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getApiBaseForTests(): string {
  return API_BASE;
}
