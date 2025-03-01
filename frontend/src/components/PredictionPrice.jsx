"use client";

import React, { useEffect } from "react";
import { useStock } from "../contexts/StockContext";

export const PredictionPrice = () => {
  const { selectedStock, lastStockClose } = useStock();
  const [prediction, setPrediction] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  useEffect(() => {
    async function fetchPrediction() {
      try {
        setLoading(true);
        setError(null);

        const url = new URL("http://localhost:5000/api/predictions");
        url.searchParams.append("stock", selectedStock);

        const response = await fetch(url);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to fetch prediction data");
        }

        const data = await response.json();
        setPrediction(data);
      } catch (error) {
        console.error("Error fetching prediction data:", error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    }

    fetchPrediction();
  }, [selectedStock]);

  if (loading) {
    return <div>Loading prediction...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!prediction || prediction.length === 0) {
    return <div>No prediction data available</div>;
  }

  const latestPrediction = prediction[prediction.length - 1];
  const predictedPrice = latestPrediction.close;

  return (
    <div className="p-6  space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white">
          Closing Stock Price for Today:
        </h2>
        <div className="text-2xl font-bold text-white">
          ${lastStockClose.toFixed(2)}
        </div>
      </div>
      <div>
        <h2 className="text-lg font-semibold text-white">
          Predicted Stock Price for Tomorrow:
        </h2>
        {predictedPrice > lastStockClose ? (
          <div className="text-2xl font-bold text-green-500">
            ${predictedPrice.toFixed(2)}
          </div>
        ) : (
          <div className="text-2xl font-bold text-red-500">
            ${predictedPrice.toFixed(2)}
          </div>
        )}
      </div>
    </div>
  );
};
