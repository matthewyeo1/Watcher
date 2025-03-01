import yfinance as yf
import datetime

def fetch_stock_prices_for_date(selected_date):
    """Fetch stock prices at 5-minute intervals for a given past date."""
    
    # Convert to datetime object
    date_obj = datetime.datetime.strptime(selected_date, "%Y-%m-%d")
    
    # If the date falls on a weekend, adjust to the last Friday
    if date_obj.weekday() == 5:  # Saturday
        date_obj -= datetime.timedelta(days=1)
    elif date_obj.weekday() == 6:  # Sunday
        date_obj -= datetime.timedelta(days=2)
    
    selected_date = date_obj.strftime("%Y-%m-%d")  # Update the date
    
    print(f"\n--- 5-Minute Interval Stock Prices for {selected_date} ---")

    stock_tickers = {  # Replace with your actual stock tickers
        "Alibaba": "BABA",
        "Amazon": "AMZN",
        "Apple": "AAPL",
        "HP": "HPQ",
        "Intel": "INTC",
        "Meta": "META",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Tesla": "TSLA"
    }

    for stock, ticker in stock_tickers.items():
        try:
            stock_data = yf.Ticker(ticker)
            df = stock_data.history(interval="5m", start=selected_date, end=selected_date)

            if df.empty:
                print(f"No data available for {stock} ({ticker}) on {selected_date}")
            else:
                print(f"\n{stock} ({ticker}) Prices on {selected_date}:")
                print(df[['Open', 'High', 'Low', 'Close']])
        except Exception as e:
            print(f"Error fetching {stock} ({ticker}): {e}")
