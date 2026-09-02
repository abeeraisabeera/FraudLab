import * as React from "react";

import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.ComponentProps<"input">>(({ className, ...props }, ref) => {
  return (
    <input
      ref={ref}
      className={cn(
        "h-10 w-full border border-ink bg-paper px-3 text-sm outline-none placeholder:text-stone-500 focus:ring-2 focus:ring-olive",
        className,
      )}
      {...props}
    />
  );
});

Input.displayName = "Input";
