"use client";

import React, { createContext, useContext, useState } from "react";

export const STOCK_OPTIONS = [
  { value: "ALIBABA", label: "Alibaba" },
  { value: "AMAZON", label: "Amazon" },
  { value: "APPLE", label: "Apple" },
  { value: "HP", label: "HP" },
  { value: "INTEL", label: "Intel" },
  { value: "META", label: "Meta" },
  { value: "MICROSOFT", label: "Microsoft" },
  { value: "NVIDIA", label: "NVIDIA" },
  { value: "TESLA", label: "Tesla" },
];

const StockContext = createContext();

export function StockProvider({ children }) {
  const [selectedStock, setSelectedStock] = useState(STOCK_OPTIONS[0].value);

  return (
    <StockContext.Provider value={{ selectedStock, setSelectedStock }}>
      {children}
    </StockContext.Provider>
  );
}

export function useStock() {
  const context = useContext(StockContext);
  if (!context) {
    throw new Error("useStock must be used within a StockProvider");
  }
  return context;
}
