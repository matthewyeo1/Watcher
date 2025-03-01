"use client";
import {
  AreaSeries,
  createChart,
  ColorType,
  LineSeries,
} from "lightweight-charts";
import React, { useEffect, useRef } from "react";
import { STOCK_OPTIONS, useStock } from "../contexts/StockContext";

const grey800 = "#1e293b";
const grey700 = "#334155";

export const ChartComponent = (props) => {
  const {
    data,
    colors: {
      backgroundColor = "black",
      lineColor = "#2962FF",
      textColor = "white",
      areaTopColor = "#2962FF",
      areaBottomColor = "rgba(41, 98, 255, 0.28)",
    } = {},
  } = props;

  const chartContainerRef = useRef();

  useEffect(() => {
    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: backgroundColor },
        textColor,
      },
      width: chartContainerRef.current.clientWidth,
      height: 300,
      grid: {
        vertLines: { color: grey800 },
        horzLines: { color: grey800 },
      },
      crosshair: {
        vertLine: {
          labelBackgroundColor: grey800,
        },
        horzLine: {
          labelBackgroundColor: grey800,
        },
      },
    });
    chart.timeScale().fitContent();

    const newSeries = chart.addSeries(LineSeries, {
      color: lineColor,
    });
    newSeries.setData(data);

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [
    data,
    backgroundColor,
    lineColor,
    textColor,
    areaTopColor,
    areaBottomColor,
  ]);

  return <div ref={chartContainerRef} />;
};

export function StockChart(props) {
  const { selectedStock, setSelectedStock, setLastStockClose } = useStock();
  const [chartData, setChartData] = React.useState([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  useEffect(() => {
    async function fetchStockData() {
      try {
        setIsLoading(true);
        setError(null);

        const url = new URL("http://localhost:5000/api/stocks");
        url.searchParams.append("stock", selectedStock);

        const response = await fetch(url);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to fetch stock data");
        }

        const data = await response.json();
        // Sort data by date (oldest to newest)
        const sortedData = data
          .sort((a, b) => new Date(a.time) - new Date(b.time))
          .map((item) => ({
            ...item,
            time: new Date(item.time).toISOString().split("T")[0],
            value: item.close,
          }));
        setChartData(sortedData);
        setLastStockClose(sortedData[sortedData.length - 1].close);
      } catch (error) {
        console.error("Error fetching stock data:", error);
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    }

    fetchStockData();
  }, [selectedStock]);

  return (
    <div className="p-6 rounded-lg bg-slate-900 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white">Stock Price Chart</h2>
        <select
          value={selectedStock}
          onChange={(e) => setSelectedStock(e.target.value)}
          className="bg-slate-800 text-white px-4 py-2 rounded-md border border-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 hover:bg-slate-700 transition-colors"
        >
          {STOCK_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {isLoading ? (
        <div className="h-[300px] flex items-center justify-center text-slate-400">
          Loading chart data...
        </div>
      ) : error ? (
        <div className="h-[300px] flex items-center justify-center text-red-400">
          Error: {error}
        </div>
      ) : chartData.length === 0 ? (
        <div className="h-[300px] flex items-center justify-center text-slate-400">
          No data available
        </div>
      ) : (
        <ChartComponent {...props} data={chartData} />
      )}
    </div>
  );
}
