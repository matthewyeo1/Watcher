# Importing functions from transformer.py
from transformer import run_sentiment_analysis
from stock_tracker import fetch_stock_prices

def main():
    # Running sentiment analysis
    run_sentiment_analysis()

    # Fetch stock prices, if needed
    fetch_stock_prices()

if __name__ == "__main__":
    main()
