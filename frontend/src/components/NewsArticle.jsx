"use client";

import { TrendingUp, TrendingDown } from "react-feather";
import { cn } from "@/lib/utils";

export function NewsArticle({ source, title, time, sentiment }) {
  const isPositive = sentiment > 0;

  return (
    <div className="w-full rounded-lg bg-card p-4 shadow-sm transition-all hover:bg-accent/10">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white">{source}</span>
          <span className="text-xs text-white/70">{time}</span>
        </div>
        <div className="flex items-center gap-1">
          {isPositive ? (
            <TrendingUp className="h-4 w-4 text-green-500" />
          ) : (
            <TrendingDown className="h-4 w-4 text-red-500" />
          )}
          <span
            className={cn(
              "text-sm font-medium",
              isPositive ? "text-green-500" : "text-red-500"
            )}
          >
            {Math.abs(sentiment).toFixed(2)}
          </span>
        </div>
      </div>
      <h3 className="mt-2 text-base font-medium text-white">{title}</h3>
    </div>
  );
}
