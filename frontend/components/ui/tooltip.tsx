"use client";

import type { ReactNode } from "react";
import { CircleHelp } from "lucide-react";

import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

type Side = "top" | "bottom" | "left" | "right";

const sideClasses: Record<Side, string> = {
  top: "bottom-full left-1/2 mb-2 -translate-x-1/2",
  bottom: "top-full left-1/2 mt-2 -translate-x-1/2",
  left: "right-full top-1/2 mr-2 -translate-y-1/2",
  right: "left-full top-1/2 ml-2 -translate-y-1/2",
};

export function Tooltip({
  content,
  children,
  side = "top",
  className,
  wide = false,
}: {
  content: string;
  children: ReactNode;
  side?: Side;
  className?: string;
  wide?: boolean;
}) {
  if (!content) return <>{children}</>;

  return (
    <span className={cn("group/tip relative inline-flex max-w-full", className)}>
      {children}
      <span
        role="tooltip"
        className={cn(
          "pointer-events-none absolute z-50 rounded-xl border border-white/80 bg-ink px-3 py-2 text-left text-[11px] font-normal normal-case leading-relaxed tracking-normal text-white opacity-0 shadow-lift transition-opacity duration-150 group-hover/tip:opacity-100 group-focus-within/tip:opacity-100",
          wide ? "w-64" : "w-56 max-w-[min(16rem,calc(100vw-2rem))]",
          sideClasses[side],
        )}
      >
        {content}
      </span>
    </span>
  );
}

export function TipLabel({
  htmlFor,
  tip,
  children,
}: {
  htmlFor?: string;
  tip: string;
  children: ReactNode;
}) {
  return (
    <Tooltip content={tip} side="right">
      <Label htmlFor={htmlFor} className="inline-flex cursor-help items-center gap-1">
        {children}
        <CircleHelp className="h-3 w-3 shrink-0 text-muted/80" aria-hidden />
      </Label>
    </Tooltip>
  );
}

export function TipTitle({
  tip,
  children,
  className,
}: {
  tip: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Tooltip content={tip} side="top" wide>
      <span
        className={cn(
          "inline-flex cursor-help items-center gap-1.5 font-display text-base font-semibold tracking-tight",
          className,
        )}
      >
        {children}
        <CircleHelp className="h-3.5 w-3.5 shrink-0 text-muted/70" aria-hidden />
      </span>
    </Tooltip>
  );
}

export function TipText({
  tip,
  children,
  className,
}: {
  tip: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Tooltip content={tip} side="top">
      <span className={cn("cursor-help", className)}>{children}</span>
    </Tooltip>
  );
}

export function MetricTile({
  label,
  value,
  tip,
}: {
  label: string;
  value: ReactNode;
  tip: string;
}) {
  return (
    <Tooltip content={tip} side="top" wide className="block w-full">
      <div className="rounded-2xl bg-soft/80 p-3">
        <div className="flex items-center gap-1 text-[11px] font-medium text-muted">
          {label}
          <CircleHelp className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
        </div>
        <div className="mt-1 font-display text-lg font-bold">{value}</div>
      </div>
    </Tooltip>
  );
}
