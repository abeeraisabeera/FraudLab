"use client";

import { FraudLab } from "@/components/fraud-lab";
import { LayoutShell } from "@/components/layout-shell";

export default function HomePage() {
  return (
    <LayoutShell>
      <FraudLab />
    </LayoutShell>
  );
}
