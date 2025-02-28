import yfinance as yf
import time
from enum import Enum

# Enum for stock names and tickers
class Stock(Enum):
    APPLE = "AAPL"
    TESLA = "TSLA"
    NVIDIA = "NVDA"
    AMAZON = "AMZN"
    MICROSOFT = "MSFT"
    META = "META"
    ALIBABA = "BABA"
    TENCENT = "0700.HK"
    INTEL = "INTC"
    HP = "HPQ"

# Filter out stocks without tickers
stock_tickers = {stock.name: stock.value for stock in Stock if stock.value}

def fetch_stock_prices():
    """Fetch real-time stock prices from Yahoo Finance and display them."""
    while True:
        print("\n--- Live Stock Prices ---")
        for stock, ticker in stock_tickers.items():
            try:
                stock_data = yf.Ticker(ticker)
                latest_price = stock_data.fast_info["last_price"]  # Get latest real-time price
                print(f"{stock}: ${latest_price:.2f}")
            except Exception as e:
                print(f"Error fetching {stock} ({ticker}): {e}")

        print("\nUpdating in 5 seconds...\n")
        time.sleep(5)  # Wait 5 seconds before fetching again


