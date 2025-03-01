import yfinance as yf
import datetime
import pandas as pd
from stocks import Stock

# Filter out stocks without tickers
stock_tickers = {stock.name: stock.value for stock in Stock if stock.value}

# Define market open time (U.S. stock market: 9:30 AM EST)
MARKET_OPEN_TIME = "09:30:00"
MARKET_CLOSE_TIME = "16:00:00"

def fetch_stock_prices():
    """Fetch 5-minute interval stock prices from market open until now."""
    
    # Get the current time in EST
    now = datetime.datetime.now().astimezone(datetime.timezone.utc)
    
    # Format today's date
    today = now.date().strftime("%Y-%m-%d")
    
    # Convert market open time to datetime
    market_open_dt = pd.Timestamp(f"{today} {MARKET_OPEN_TIME}", tz="America/New_York")
    
    # Fetch stock data only if market is open
    if now < market_open_dt:
        print("Market has not opened yet.")
        return
    
    print("\n--- 5-Minute Interval Stock Prices from Market Open ---")
    
    for stock, ticker in stock_tickers.items():
        try:
            stock_data = yf.Ticker(ticker)
            
            # Fetch historical data with 5-minute intervals from today
            history = stock_data.history(interval="5m", period="1d")
            
            # Ensure there is data
            if history.empty:
                print(f"No data available for {stock} ({ticker})")
                continue
            
            # Filter out data before market open
            history = history[history.index >= market_open_dt]
            
            # Display the stock prices at 5-minute intervals
            print(f"\n{stock} ({ticker}):")
            print(history[["Close"]])
        
        except Exception as e:
            print(f"Error fetching {stock} ({ticker}): {e}")

# Run the function
fetch_stock_prices()
