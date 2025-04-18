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

class HypeAnalyzer:
    def __init__(self):
        self.correlation_threshold = 0.7  # Minimum correlation to consider significant
        self.volume_spike_threshold = 2.0  # 200% volume increase
        
    async def analyze(self, twitter_data, reddit_data, price_data):
        """Core analysis comparing social hype vs market activity"""
        results = {
            'strong_signals': [],
            'weak_signals': [],
            'metrics': {}
        }
        
        # 1. Preprocess data for alignment
        aligned_data = self._align_data(twitter_data, reddit_data, price_data)
        
        # 2. Calculate key metrics
        results['metrics'].update(self._calculate_correlations(aligned_data))
        results['metrics'].update(self._detect_volume_spikes(aligned_data))
        results['metrics'].update(self._sentiment_trends(aligned_data))
        
        # 3. Generate signals
        results['strong_signals'] = self._find_strong_signals(aligned_data)
        results['weak_signals'] = self._find_weak_signals(aligned_data)
        
        return results
    
    def _align_data(self, twitter, reddit, price):
        """Combine all data into hourly buckets"""

        required_cols = ['date', 'content', 'sentiment']

        for df_name, df in [('Twitter', twitter), ('Reddit', reddit)]:
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"{df_name} data is missing required column: '{col}'")

        if 'timestamp' in price.columns:
            price['timestamp'] = pd.to_datetime(price['timestamp'])
            price.set_index('timestamp', inplace=True)
        else:
            price.index = pd.to_datetime(price.index)

        twitter['date'] = pd.to_datetime(twitter['date'])
        reddit['date'] = pd.to_datetime(reddit['date'])

        twitter.set_index('date', inplace=True)
        reddit.set_index('date', inplace=True)

        # Resample price data to hourly
        # Change all instances of '1H' to '1h' (lowercase h)
        price_hourly = price.resample('1h').agg({  # Line 65
            'close': 'last',
            'volume': 'sum'
        })

        social_metrics = pd.DataFrame({
            'twitter_mentions': twitter.resample('1h')['content'].count(),  # Line 72
            'reddit_posts': reddit.resample('1h')['content'].count(),      # Line 73
            'avg_sentiment': (
                twitter.resample('1h')['sentiment'].mean() +               # Line 75
                reddit.resample('1h')['sentiment'].mean()                  # Line 76
            ) / 2
        })

        # Merge everything
        return pd.merge(
            price_hourly,
            social_metrics,
            left_index=True,
            right_index=True,
            how='left'
        ).ffill()

    
    def _calculate_correlations(self, df):
        """Calculate Pearson correlations between metrics with NaN handling"""
        return {
            'price_sentiment_corr': df['close'].corr(df['avg_sentiment']),
            'volume_mentions_corr': df['volume'].corr(
                df['twitter_mentions'] + df['reddit_posts']
            )
        }
    
    def _detect_volume_spikes(self, df):
        """Identify sudden volume increases"""
        df['volume_pct_change'] = df['volume'].pct_change()
        spikes = df[df['volume_pct_change'] > self.volume_spike_threshold]
        return {
            'volume_spikes_count': len(spikes),
            'last_volume_spike': spikes.index[-1] if not spikes.empty else None
        }
    
    def _sentiment_trends(self, df):
        """Analyze sentiment movement"""
        return {
            'sentiment_trend': 'up' if df['avg_sentiment'].iloc[-1] > df['avg_sentiment'].mean() else 'down',
            'sentiment_volatility': df['avg_sentiment'].std()
        }
    
    def _find_strong_signals(self, df):
        """Conditions for strong hype validation"""
        signals = []
        
        # Condition 1: Volume spike + social mentions increase
        recent = df.iloc[-4:]  # Last 4 hours
        if (recent['volume_pct_change'].mean() > 1.0 and 
            recent['twitter_mentions'].mean() > df['twitter_mentions'].mean() * 1.5):
            signals.append("STRONG_VOLUME_SOCIAL_CORRELATION")
            
        # Condition 2: Rising price + improving sentiment
        if (df['close'].iloc[-1] > df['close'].iloc[-6] and 
            df['avg_sentiment'].iloc[-1] > df['avg_sentiment'].iloc[-6]):
            signals.append("PRICE_SENTIMENT_UPTREND")
            
        return signals
    
    def _find_weak_signals(self, df):
        """Conditions where hype isn't backed by activity"""
        signals = []
        
        # Condition 1: High mentions but flat volume
        if (df['twitter_mentions'].iloc[-1] > df['twitter_mentions'].mean() * 2 and
            abs(df['volume_pct_change'].iloc[-1]) < 0.3):
            signals.append("HIGH_MENTIONS_LOW_VOLUME")
            
        # Condition 2: Positive sentiment but price dropping
        if (df['avg_sentiment'].iloc[-1] > 0.5 and 
            df['close'].iloc[-1] < df['close'].iloc[-6]):
            signals.append("BULLISH_SENTIMENT_BEARISH_PRICE")
            
        return signals
    
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
                'date': tweet['created_at'],  # Ensure 'created_at' is present
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
    
    async def get_posts(self, query, subreddit='wallstreetbetscrypto+cryptocurrency+memecoins+solana+dogecoin+bitcoin+ethereum+ethtrader+solanamemecoins', limit=100):
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
                return df

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
        return df


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
    print(twitter_data.head())  # Add this after collecting the data to inspect it

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

    print("🔍 Analyzing hype vs market data...")
    analyzer = HypeAnalyzer()
    print("\n📋 Dataframe Column Checks:")
    print(f"Twitter columns: {twitter_data.columns.tolist()}")
    print(f"Reddit columns: {reddit_data.columns.tolist()}")
    print(f"Market columns: {price_data.columns.tolist()}")
    if twitter_data.empty or reddit_data.empty or price_data.empty:
        print("❌ One or more data sources returned empty. Aborting analysis.")
        return
    analysis_results = await analyzer.analyze(twitter_data, reddit_data, price_data)
    
    # Save analysis results
    pd.DataFrame(analysis_results['metrics'], index=[0]).to_csv(f'data/analysis_{timestamp}.csv')
    
    # Print actionable insights
    print(f"\n{'='*40}")
    print("Hype Analysis Results")
    print(f"{'='*40}")
    
    if analysis_results['strong_signals']:
        print("🚀 STRONG SIGNALS (Real Trend Likely)")
        for signal in analysis_results['strong_signals']:
            print(f"- {signal}")
    
    if analysis_results['weak_signals']:
        print("\n WEAK SIGNALS (Possible Fake Hype)")
        for signal in analysis_results['weak_signals']:
            print(f"- {signal}")
    
    print("\n📊 Key Metrics:")
    for metric, value in analysis_results['metrics'].items():
        print(f"- {metric.replace('_', ' ').title()}: {value:.2f}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Run async main function
    asyncio.run(main())