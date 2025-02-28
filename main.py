import requests
from bs4 import BeautifulSoup as bs
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from urls import urls
from stocks import Stock
import numpy as np  
import pandas as pd
import yfinance as yf
from stock_tracker import fetch_stock_prices

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

new_words = {
    "bankruptcy": -3.5, "crash": -3.0, "bearish": -2.5, "layoff": -2.0, 
    "downturn": -2.5, "recession": -3.0, "default": -3.5, "sell-off": -3.0,
    "slip": -1.0, "worse": -1.5, "pressure": -1.0
}
sia.lexicon.update(new_words)


def get_article_content(url):
    response = requests.get(url, headers=headers)
    #print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        page_content = bs(response.content, 'html.parser')

        article_text = ''
        
        paragraphs = page_content.find_all('p')
        headlines = page_content.find_all(['h1', 'h2', 'h3'])
        meta_desc = page_content.find("meta", attrs={"name": "description"})
        og_desc = page_content.find("meta", property="og:description")

        for p in paragraphs:
            article_text += p.get_text() + " "
        for h in headlines:
            article_text += h.get_text() + " "
        if meta_desc and 'content' in meta_desc.attrs:
            article_text += meta_desc['content'] + " "
        if og_desc and 'content' in og_desc.attrs:
            article_text += og_desc['content'] + " "

        return article_text.strip()

    else:
        print(f"Failed to retrieve the article from {url}")
        return None


def analyze_sentiment(article_text):
    sentiment = sia.polarity_scores(article_text)

    negative_words = list(new_words.keys())
    found_neg_words = [word for word in negative_words if word in article_text.lower()]

    sentiment['neg'] 
    sentiment['pos']  

    if found_neg_words:
        sentiment['neg'] *= 1.5  
        sentiment['pos'] *= 0.5 

    sentiment['neg'] = round(sentiment['neg'], 3)
    sentiment['pos'] = round(sentiment['pos'], 3) 

    return sentiment
    


def print_sentiment_matrix(Stock, sentiment):
    
    array = np.array = [
        [Stock.value],
        [sentiment['neg']],
        [sentiment['neu']],
        [sentiment['pos']]
    ]
    
   
    #print("Sentiment Matrix (neg, neu, pos):")
    print(array)

def run_sentiment_analysis():
    for Stock, stocks in urls.items(): 
        for url in stocks: 
            #print(f"Scraping article from: {url}")
            article_text = get_article_content(url)
        
            if article_text:
                sentiment = analyze_sentiment(article_text)
                #print(f"Sentiment Analysis for {url}:")
                print_sentiment_matrix(Stock, sentiment)
                print("\n" + "-"*50 + "\n")
        
            time.sleep(2)

def main():
    # run_sentiment_analysis()
    fetch_stock_prices()

if __name__ == "__main__":
    main()
