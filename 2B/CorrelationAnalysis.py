import snscrape.modules.twitter as sntwitter
import praw
import ccxt
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
import time

# ======================
# 2A: SOCIAL MEDIA SCRAPING
# ======================

def scrape_twitter(keyword, limit=100):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{datetime.now().date()}').get_items()):
        if i >= limit:
            break
        tweets.append({
            'date': tweet.date,
            'content': tweet.content,
            'username': tweet.user.username,
            'likes': tweet.likeCount
        })
    return pd.DataFrame(tweets)

def scrape_reddit(keyword, limit=50):
    reddit = praw.Reddit(
        client_id='yQPZuCrG3KfavK9vYqSxsg',
        client_secret='Ngqhi-ZwOi8RgkT0_MacvjgyO1vNIw',
        user_agent='"python:ECE473FiveSigma:v1.0 (by /u/Educational-Knee5736)"' 
    )
    posts = []
    for post in reddit.subreddit('all').search(keyword, limit=limit):
        posts.append({
            'date': datetime.fromtimestamp(post.created_utc),
            'content': post.title,
            'username': post.author.name if post.author else None,
            'upvotes': post.score
        })
    return pd.DataFrame(posts)

def analyze_sentiment(df):
    df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

# ======================
# 2B: MARKET/YIELD DATA
# ======================

def get_price_data(symbol='DOGE/USDT'):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=24)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.set_index('timestamp')

def get_defi_yield(token_address='0x6982508145454ce325ddbe47a25d4ec3d2311933'):  # PEPE example
    query = f"""{{
      reserves(where: {{underlyingAsset: "{token_address}"}}) {{
        name
        liquidityRate
        variableBorrowRate
        utilizationRate
      }}
    }}"""
    try:
        response = requests.post('https://api.thegraph.com/subgraphs/name/aave/protocol-v3', 
                               json={'query': query}, timeout=10)
        return response.json()['data']['reserves'][0]
    except Exception as e:
        print(f"Yield data error: {e}")
        return None

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    # 1. Scrape Social Data
    print("Scraping social data...")
    twitter_data = scrape_twitter("PEPE OR memecoin", 200)
    reddit_data = scrape_reddit("PEPE", 100)
    
    # 2. Analyze Sentiment
    twitter_data = analyze_sentiment(twitter_data)
    reddit_data = analyze_sentiment(reddit_data)
    
    # 3. Get Market Data
    print("Fetching market data...")
    price_data = get_price_data()
    yield_data = get_defi_yield()
    
    # 4. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    twitter_data.to_csv(f'twitter_data_{timestamp}.csv')
    reddit_data.to_csv(f'reddit_data_{timestamp}.csv')
    price_data.to_csv(f'price_data_{timestamp}.csv')
    
    if yield_data:
        pd.DataFrame([yield_data]).to_csv(f'yield_data_{timestamp}.csv')
    
    print(f"Data collection complete at {timestamp}")