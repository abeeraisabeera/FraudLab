import { describe, expect, it } from "vitest";

import { INDUSTRIES } from "@/lib/industries";
import { cn } from "@/lib/utils";
import { createExperimentId, defaultExperimentName } from "@/lib/history";
import type { AnalysisResponse } from "@/lib/types";

const mockAnalysis = {
  statistics: {
    relative_lift: 0.0482,
  },
} as AnalysisResponse;

describe("cn", () => {
  it("merges class names and resolves conflicts", () => {
    expect(cn("px-2", "px-4", false && "hidden")).toBe("px-4");
  });
});

describe("history helpers", () => {
  it("creates unique experiment ids", () => {
    const a = createExperimentId();
    const b = createExperimentId();
    expect(a).toMatch(/^exp_/);
    expect(b).toMatch(/^exp_/);
    expect(a).not.toBe(b);
  });

  it("builds a readable default name from lift", () => {
    const name = defaultExperimentName("generated", mockAnalysis);
    expect(name).toContain("Synthetic");
    expect(name).toContain("4.82%");
  });
});

describe("industry configs", () => {
  it("exposes food delivery and banking verticals", () => {
    expect(INDUSTRIES.food_delivery.defaults.industry).toBe("food_delivery");
    expect(INDUSTRIES.banking.defaults.industry).toBe("banking");
    expect(INDUSTRIES.banking.defaults.baseline_conversion).toBeLessThan(
      INDUSTRIES.food_delivery.defaults.baseline_conversion,
    );
  });

  it("provides test and question scenario buttons per industry", () => {
    for (const industry of ["food_delivery", "banking"] as const) {
      const scenarios = INDUSTRIES[industry].scenarios;
      expect(scenarios.some((item) => item.kind === "test")).toBe(true);
      expect(scenarios.some((item) => item.kind === "question")).toBe(true);
    }
  });
});
