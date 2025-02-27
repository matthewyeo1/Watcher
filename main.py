import requests
from bs4 import BeautifulSoup as bs
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from urls import urls
from stocks import Stock
import numpy as np  

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_article_content(url):
    response = requests.get(url, headers=headers)
    #print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        page_content = bs(response.content, 'html.parser')

        article_text = ''
        
        paragraphs = page_content.find_all('p')
        for p in paragraphs:
            article_text += p.get_text()

        return article_text
    else:
        print(f"Failed to retrieve the article from {url}")
        return None


def analyze_sentiment(article_text):
    sentiment = sia.polarity_scores(article_text)
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

def main():
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

if __name__ == "__main__":
    main()
