import requests
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from stocks import Stock
from datetime import datetime

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

API_KEY = 'cv1b9fhr01qhkk81oos0cv1b9fhr01qhkk81oosg'

def hours_ago(timestamp):
    current_time = datetime.now()
    news_time = datetime.utcfromtimestamp(timestamp)
    time_difference = current_time - news_time
    return round(time_difference.total_seconds() / 3600)

def get_stock_news(symbol):
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2025-03-01&to=2025-03-02&token={API_KEY}'
    response = requests.get(url)
    return response.json()[:5] if response.status_code == 200 else []

def analyze_sentiment(text):
    return sia.polarity_scores(text)['compound']

def display_stock_news():
    for stock in Stock:
        print(f"\nFetching news for {stock.name} ({stock.value}):")
        news_articles = get_stock_news(stock.value)

        if news_articles:
            for idx, article in enumerate(news_articles, 1):
                hours_diff = hours_ago(article['datetime'])
                title = article['headline']
                sentiment_score = analyze_sentiment(title)
                sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

                print(f"{idx}. Title: {title}")
                print(f"   Source: {article['source']}")
                print(f"   Published: {article['datetime']} ({hours_diff} hours ago)")
                print(f"   URL: {article['url']}")
                print(f"   Sentiment: {sentiment_label} (Score: {sentiment_score})")
                print('-' * 80)
        else:
            print("No news available.")
        time.sleep(1)

def refresh_news_hourly():
    while True:
        print("\nFetching the latest stock news...")
        display_stock_news()
        print("\nWaiting for the next refresh in 1 hour...\n")
        time.sleep(3600)

refresh_news_hourly()
