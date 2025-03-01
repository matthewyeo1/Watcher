"use client";
import { useState, useEffect } from "react";
import { TrendingUp, TrendingDown } from "react-feather";
import { cn } from "@/lib/utils";
import { useStock } from "../contexts/StockContext";

export function NewsArticle({ source, title, time, sentiment, url }) {
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
      <h3 className="mt-2 text-base font-medium text-white hover:underline">
        <a href={url}>{title}</a>
      </h3>
    </div>
  );
}

export function NewsArticleList() {
  const { selectedStock } = useStock();
  const [newsArticles, setNewsArticles] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchNewsArticles() {
      try {
        setIsLoading(true);
        setError(null);

        const url = new URL("http://localhost:5000/api/news");
        url.searchParams.append("stock", selectedStock);

        const response = await fetch(url);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to fetch news articles");
        }

        const data = await response.json();
        setNewsArticles(data);
        setIsLoading(false);
      } catch (err) {
        setError(err.message);
        setIsLoading(false);
      }
    }

    fetchNewsArticles();
  }, [selectedStock]);

  if (isLoading) {
    return <div className="text-center text-sm text-white/70">Loading...</div>;
  }

  if (error) {
    return (
      <div className="text-center text-sm text-red-500">Error: {error}</div>
    );
  }

  return (
    <div className="mt-4 space-y-4">
      {newsArticles.map((article) => (
        <NewsArticle
          key={article.id}
          source="bloomberg"
          title={article.headline}
          time="time"
          sentiment={article.sentimentScore}
          url={article.url}
        />
      ))}
    </div>
  );
}
