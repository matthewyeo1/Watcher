const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { parse } = require("csv-parse");

const app = express();
const PORT = process.env.PORT || 5000;

// Enable CORS for frontend requests
app.use(
  cors({
    origin: "http://localhost:3000", // Adjust this to match your frontend URL
  })
);
// Parse query parameters
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Helper function to filter results by stock name
function filterByStock(data, stockName) {
  if (!stockName) return true;
  return data.Stock.toUpperCase() === stockName.toUpperCase();
}

app.get("/api/predictions", (req, res) => {
  const csvFilePath = path.join(__dirname, "predictions.csv");
  const results = [];
  const stockName = req.query.stock; // Get stock from query parameter

  fs.createReadStream(csvFilePath)
    .pipe(
      parse({
        columns: true,
        skip_empty_lines: true,
      })
    )
    .on("data", (data) => {
      // Only push data if it matches the stock filter
      if (filterByStock(data, stockName)) {
        results.push({
          stock: data.Stock,
          time: data.Date,
          close: parseFloat(data.Close),
        });
      }
    })
    .on("end", () => {
      if (results.length === 0) {
        res.status(404).json({
          error: stockName
            ? `No data found for stock: ${stockName}`
            : "No stock data available",
        });
        return;
      }
      res.json(results);
    })
    .on("error", (error) => {
      console.error("Error parsing CSV:", error);
      res.status(500).json({ error: "Failed to process stock data" });
    });
});

app.get("/api/stocks", (req, res) => {
  const csvFilePath = path.join(__dirname, "stocks_data.csv");
  const results = [];
  const stockName = req.query.stock; // Get stock from query parameter

  fs.createReadStream(csvFilePath)
    .pipe(
      parse({
        columns: true,
        skip_empty_lines: true,
      })
    )
    .on("data", (data) => {
      // Only push data if it matches the stock filter
      if (filterByStock(data, stockName)) {
        results.push({
          stock: data.Stock,
          time: data.Date,
          close: parseFloat(data.Close),
        });
      }
    })
    .on("end", () => {
      if (results.length === 0) {
        res.status(404).json({
          error: stockName
            ? `No data found for stock: ${stockName}`
            : "No stock data available",
        });
        return;
      }
      res.json(results);
    })
    .on("error", (error) => {
      console.error("Error parsing CSV:", error);
      res.status(500).json({ error: "Failed to process stock data" });
    });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
