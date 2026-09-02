"use client";

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { CircleHelp } from "lucide-react";

import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

type Side = "top" | "bottom" | "left" | "right";

function computePosition(rect: DOMRect, side: Side, gap = 10): CSSProperties {
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;

  switch (side) {
    case "bottom":
      return {
        top: rect.bottom + gap,
        left: centerX,
        transform: "translateX(-50%)",
      };
    case "left":
      return {
        top: centerY,
        left: rect.left - gap,
        transform: "translate(-100%, -50%)",
      };
    case "right":
      return {
        top: centerY,
        left: rect.right + gap,
        transform: "translateY(-50%)",
      };
    case "top":
    default:
      return {
        top: rect.top - gap,
        left: centerX,
        transform: "translate(-50%, -100%)",
      };
  }
}

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
  const tipId = useId();
  const triggerRef = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [style, setStyle] = useState<CSSProperties>({ top: -9999, left: -9999 });

  const updatePosition = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    setStyle(computePosition(el.getBoundingClientRect(), side));
  }, [side]);

  const show = useCallback(() => {
    updatePosition();
    setOpen(true);
  }, [updatePosition]);

  const hide = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    const onScrollOrResize = () => updatePosition();
    window.addEventListener("scroll", onScrollOrResize, true);
    window.addEventListener("resize", onScrollOrResize);
    return () => {
      window.removeEventListener("scroll", onScrollOrResize, true);
      window.removeEventListener("resize", onScrollOrResize);
    };
  }, [open, updatePosition]);

  if (!content) return <>{children}</>;

  const tooltip =
    open && typeof document !== "undefined"
      ? createPortal(
          <span
            id={tipId}
            role="tooltip"
            style={{ position: "fixed", zIndex: 9999, ...style }}
            className={cn(
              "pointer-events-none rounded-xl border border-white/80 bg-ink px-3 py-2 text-left text-[11px] font-normal normal-case leading-relaxed tracking-normal text-white shadow-lift",
              wide ? "w-64" : "w-56 max-w-[min(16rem,calc(100vw-2rem))]",
            )}
          >
            {content}
          </span>,
          document.body,
        )
      : null;

  return (
    <>
      <span
        ref={triggerRef}
        className={cn("inline-flex max-w-full", className)}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        aria-describedby={open ? tipId : undefined}
      >
        {children}
      </span>
      {tooltip}
    </>
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
