"use client";

import type { ReactNode } from "react";
import { useState } from "react";
import { Moon, ShieldAlert } from "lucide-react";

import { cn } from "@/lib/utils";

export function LayoutShell({ children }: { children: ReactNode }) {
  const [crtMode, setCrtMode] = useState(false);

  return (
    <div className={cn("min-h-screen", crtMode && "crt-mode")}>
      <div className="mx-auto grid min-h-screen max-w-[1600px] grid-cols-1 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className="border-r border-ink bg-paper p-4">
          <div className="border border-ink p-4 shadow-print">
            <div className="flex items-center gap-2 text-xl font-bold uppercase tracking-[0.22em]">
              <ShieldAlert className="h-5 w-5" aria-hidden="true" />
              Fraud Lab
            </div>
            <div className="mt-1 text-lg tracking-[0.12em] text-amber" lang="ur" dir="rtl">
              کساؤٹی
            </div>
            <p className="mt-3 text-xs leading-relaxed text-stone-700">
              Evidence before automation. Deterministic features → EBM → calibration → policy.
            </p>
          </div>

          <button
            type="button"
            onClick={() => setCrtMode((value) => !value)}
            className="mt-6 flex w-full items-center justify-center gap-2 border border-ink bg-paper px-3 py-3 text-xs uppercase tracking-[0.18em] hover:bg-stone-200"
          >
            <Moon className="h-4 w-4" />
            {crtMode ? "Disable CRT Mode" : "Enable CRT Mode"}
          </button>
        </aside>

        <main className="p-4 lg:p-6">
          <header className="mb-6 border border-ink bg-paper px-4 py-3 shadow-print">
            <div className="text-[11px] uppercase tracking-[0.25em] text-stone-600">
              Kasauti · Fraud Evidence Workstation
            </div>
            <h1 className="mt-1 text-2xl font-bold uppercase tracking-[0.24em]">Fraud Lab</h1>
          </header>
          {children}
        </main>
      </div>
    </div>
  );
}
