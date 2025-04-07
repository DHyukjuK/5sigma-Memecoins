import os
import asyncio
import requests
import pandas as pd
import ccxt
import praw
from datetime import datetime, timedelta
from requests_oauthlib import OAuth1
from dotenv import load_dotenv
from textblob import TextBlob

# Load environment variables
load_dotenv()

# ======================
# TWITTER SCRAPER (OAuth 1.0a)
# ======================
class TwitterScraper:
    def __init__(self):
        self.base_url = "https://api.twitter.com/2/tweets/search/recent"
        self.oauth = OAuth1(
            os.getenv('TWITTER_API_KEY'),
            os.getenv('TWITTER_API_SECRET'),
            os.getenv('TWITTER_ACCESS_TOKEN'),
            os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )
    
    async def get_tweets(self, query, max_results=100):
        params = {
            'query': query,
            'max_results': str(max_results),
            'tweet.fields': 'created_at,public_metrics,author_id',
            'expansions': 'author_id',
            'user.fields': 'username'
        }
        
        headers = {
            'User-Agent': 'MemecoinTracker/1.0',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(
                self.base_url,
                auth=self.oauth,
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                return self._process_response(response.json())
            else:
                print(f"Twitter API error: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Twitter request failed: {e}")
            return pd.DataFrame(columns=['date', 'content', 'username', 'likes'])

    def _process_response(self, response):
        tweets = []
        users = {u['id']: u['username'] for u in response.get('includes', {}).get('users', [])}
        
        for tweet in response.get('data', []):
            tweets.append({
                'date': tweet['created_at'],
                'content': tweet['text'],
                'username': users.get(tweet['author_id']),
                'likes': tweet['public_metrics']['like_count'],
                'retweets': tweet['public_metrics']['retweet_count']
            })
        
        return pd.DataFrame(tweets)

# ======================
# REDDIT SCRAPER
# ======================
class RedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
    
    async def get_posts(self, query, subreddit='CryptoCurrency', limit=100):
        try:
            posts = []
            for submission in self.reddit.subreddit(subreddit).search(query, limit=limit):
                posts.append({
                    'date': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'content': submission.title,
                    'username': str(submission.author),
                    'upvotes': submission.score,
                    'subreddit': subreddit,
                    'url': submission.url
                })
            return pd.DataFrame(posts)
        except Exception as e:
            print(f"Reddit scraping failed: {e}")
            return pd.DataFrame(columns=['date', 'content', 'username', 'upvotes', 'subreddit'])

# ======================
# MARKET DATA COLLECTOR
# ======================
class MarketData:
    def __init__(self):
        self.exchanges = {
            'kucoin': ccxt.kucoin(),
            'bybit': ccxt.bybit(),
            'okx': ccxt.okx()
        }
    
    async def get_price_data(self, symbol='PEPE/USDT', timeframe='1h', limit=24):
        for name, exchange in self.exchanges.items():
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['exchange'] = name
                return df.set_index('timestamp')
            except Exception as e:
                print(f"{name} failed: {e}")
                continue
        
        # Fallback to CoinGecko
        try:
            return await self._coingecko_fallback()
        except Exception as e:
            print(f"All price data sources failed: {e}")
            return pd.DataFrame()

    async def _coingecko_fallback(self):
        url = "https://api.coingecko.com/api/v3/coins/pepe/market_chart"
        params = {'vs_currency': 'usd', 'days': '1'}
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')

# ======================
# SENTIMENT ANALYSIS
# ======================
def analyze_sentiment(df):
    if df.empty or 'content' not in df.columns:
        return df
    
    df['sentiment'] = df['content'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    return df

# ======================
# MAIN EXECUTION
# ======================
async def main():
    print(f"\n{'='*40}")
    print(f"{datetime.now():%Y-%m-%d %H:%M} - Starting Data Collection")
    print(f"{'='*40}\n")

    # Initialize scrapers
    twitter = TwitterScraper()
    reddit = RedditScraper()
    market = MarketData()

    # Data collection
    print("🟡 Collecting Twitter data...")
    twitter_data = await twitter.get_tweets("PEPE OR memecoin lang:en -is:retweet")
    twitter_data = analyze_sentiment(twitter_data)

    print("🟡 Collecting Reddit data...")
    reddit_data = await reddit.get_posts("PEPE OR memecoin")
    reddit_data = analyze_sentiment(reddit_data)

    print("🟡 Collecting market data...")
    price_data = await market.get_price_data()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not twitter_data.empty:
        twitter_data.to_csv(f'data/twitter_{timestamp}.csv', index=False)
    if not reddit_data.empty:
        reddit_data.to_csv(f'data/reddit_{timestamp}.csv', index=False)
    if not price_data.empty:
        price_data.to_csv(f'data/market_{timestamp}.csv')

    # Print summary
    print(f"\n{'='*40}")
    print("Data Collection Complete")
    print(f"{'='*40}")
    print(f"🐦 Twitter: {len(twitter_data)} tweets")
    print(f"📌 Reddit: {len(reddit_data)} posts")
    print(f"📈 Market: {len(price_data)} price points")
    print(f"💾 Saved with timestamp: {timestamp}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Run async main function
    asyncio.run(main())