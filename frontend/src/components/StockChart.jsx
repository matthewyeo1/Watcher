"use client";
import { AreaSeries, createChart, ColorType } from "lightweight-charts";
import React, { useEffect, useRef } from "react";

export const ChartComponent = (props) => {
  const {
    data,
    colors: {
      backgroundColor = "white",
      lineColor = "#2962FF",
      textColor = "black",
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
    });
    chart.timeScale().fitContent();

    const newSeries = chart.addSeries(AreaSeries, {
      lineColor,
      topColor: areaTopColor,
      bottomColor: areaBottomColor,
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
  const [chartData, setChartData] = React.useState([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const { stock } = props;

  useEffect(() => {
    async function fetchStockData() {
      try {
        setIsLoading(true);
        setError(null);

        const url = new URL("http://localhost:5000/api/stocks");
        if (stock) {
          url.searchParams.append("stock", stock);
        }

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
      } catch (error) {
        console.error("Error fetching stock data:", error);
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    }

    fetchStockData();
  }, [stock]);

  if (isLoading) {
    return <div>Loading chart data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (chartData.length === 0) {
    return <div>No data available</div>;
  }

  return <ChartComponent {...props} data={chartData} />;
}
