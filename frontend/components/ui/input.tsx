import * as React from "react";

import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          "h-11 w-full rounded-2xl border border-white/80 bg-white/70 px-4 text-sm text-ink outline-none placeholder:text-muted/70 shadow-card backdrop-blur-md transition focus:border-blush/50 focus:ring-2 focus:ring-blush/30",
          className,
        )}
        {...props}
      />
    );
  },
);

Input.displayName = "Input";
