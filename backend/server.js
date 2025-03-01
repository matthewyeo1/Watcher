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

// Helper function to process CSV data
function processStockData(csvFilePath, stockName) {
  return new Promise((resolve, reject) => {
    const results = [];

    fs.createReadStream(csvFilePath)
      .pipe(
        parse({
          columns: true,
          skip_empty_lines: true,
        })
      )
      .on("data", (data) => {
        if (filterByStock(data, stockName)) {
          results.push({
            stock: data.Stock,
            time: data.Date,
            close: parseFloat(data.Close),
          });
        }
      })
      .on("end", () => {
        resolve(results);
      })
      .on("error", (error) => {
        reject(error);
      });
  });
}

// Helper function to handle API response
async function handleStockDataRequest(req, res, filename) {
  try {
    const stockName = req.query.stock;
    const csvFilePath = path.join(__dirname, filename);
    const results = await processStockData(csvFilePath, stockName);

    if (results.length === 0) {
      res.status(404).json({
        error: stockName
          ? `No data found for stock: ${stockName}`
          : "No stock data available",
      });
      return;
    }

    res.json(results);
  } catch (error) {
    console.error(`Error processing ${filename}:`, error);
    res.status(500).json({ error: "Failed to process stock data" });
  }
}

// API endpoints
app.get("/api/predictions", (req, res) => {
  handleStockDataRequest(req, res, "predictions.csv");
});

app.get("/api/stocks", (req, res) => {
  handleStockDataRequest(req, res, "stocks_data.csv");
});

app.get("/api/news", (req, res) => {
  const csvFilePath = path.join(__dirname, "stocks_news_sentiment.csv");
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
          headline: data.Headline,
          sentimentScore: data["Sentiment Score"],
          url: data.URL,
        });
      }
    })
    .on("end", () => {
      if (results.length === 0) {
        res.status(404).json({
          error: stockName
            ? `No data found for stock: ${stockName}`
            : "No news data available",
        });
        return;
      }
      res.json(results);
    })
    .on("error", (error) => {
      console.error("Error parsing CSV:", error);
      res.status(500).json({ error: "Failed to process news data" });
    });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
