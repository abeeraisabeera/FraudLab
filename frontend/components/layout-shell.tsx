"use client";

import type { ReactNode } from "react";

export function LayoutShell({ children }: { children: ReactNode }) {
  return <div className="min-h-screen">{children}</div>;
}
