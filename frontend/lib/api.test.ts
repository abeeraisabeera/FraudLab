import { describe, expect, it } from "vitest";

import { assertFraudGeneratePayload } from "@/lib/api";
import type { FraudGeneratePayload } from "@/lib/fraud-types";

const ok: FraudGeneratePayload = {
  number_of_users: 100,
  number_of_transactions: 1000,
  fraud_rate: 0.05,
  time_range_days: 14,
  seed: 1,
};

describe("fraud api contract guards", () => {
  it("accepts a valid payload", () => {
    expect(() => assertFraudGeneratePayload(ok)).not.toThrow();
  });

  it("rejects out-of-range transaction counts", () => {
    expect(() => assertFraudGeneratePayload({ ...ok, number_of_transactions: 10 })).toThrow();
  });
});
