import praw
import pandas as pd
import numpy as np
import torch  
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import warnings
warnings.filterwarnings("ignore")

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize Reddit API
def init_reddit():
    try:
        reddit = praw.Reddit(
            client_id="yQPZuCrG3KfavK9vYqSxsg",          # Replace with your actual client ID
            client_secret="Ngqhi-ZwOi8RgkT0_MacvjgyO1vNIw",  # Replace with your actual client secret
            user_agent="ECE473FiveSigma:v1.0 (by /u/Educational-Knee5736)", 
        )
        # Test authentication
        try:
            reddit.user.me()
            print("Reddit authentication successful")
        except:
            print("Reddit read-only mode")
        return reddit
    except Exception as e:
        print(f"Reddit API initialization failed: {str(e)}")
        return None

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def get_reddit_sentiment(crypto_name, limit=500):
    reddit = init_reddit()
    if not reddit:
        return pd.DataFrame()
    
    search_query = (
        f"{crypto_name} OR SHIB OR #SHIB OR $SHIB OR SHIBA INU OR #SHIBA OR SHIBA OR SHIBAINU"
    )
    posts = []
    now = datetime.utcnow()
    one_day_ago = now - timedelta(days=1)

    for submission in reddit.subreddit("all").search(
        query=search_query,
        limit=500,
        sort="new",
        time_filter="day"  # to reduce server load
    ):
        post_time = datetime.fromtimestamp(submission.created_utc)
        if post_time < one_day_ago:
            continue

        posts.append({
            'date': post_time.date(),
            'title': submission.title,
            'text': submission.selftext,
            'score': submission.score
        })
        time.sleep(0.5)

    if not posts:
        return pd.DataFrame()
    
    df = pd.DataFrame(posts)
    df['content'] = df['title'] + " " + df['text']
    df['polarity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['vader'] = df['content'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Only return today's aggregated sentiment
    sentiment_today = df.groupby('date').agg({
        'polarity': 'mean',
        'vader': 'mean',
        'score': 'sum',
        'subjectivity': 'mean'
    }).reset_index()
    
    return sentiment_today

def get_crypto_data(ticker, start_date, end_date):
    """Get historical cryptocurrency data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")

        # Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Just use Close price
        data = data[['Close']].copy()

        data['Daily_Return'] = data['Close'].pct_change() * 100
        data['Next_Day_Return'] = data['Daily_Return'].shift(-1)
        data = data.reset_index()
        data['date'] = data['Date'].dt.date
        data = data.drop(columns=['Date'])

        return data.dropna()
    except Exception as e:
        print(f"Error fetching crypto data: {str(e)}")
        return pd.DataFrame()

def prepare_dataset(sentiment_df, price_df):
    try:
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        price_df['date'] = pd.to_datetime(price_df['date']).dt.date

        # Shift sentiment by 1 day back to align with price data (e.g., use 4/16 sentiment to predict 4/17 return)
        sentiment_df['date'] = sentiment_df['date'].apply(lambda d: d - timedelta(days=1))

        merged = pd.merge(sentiment_df, price_df, on='date', how='inner')

        print("Merged dataset:", len(merged), "samples")

        if merged.empty:
            raise ValueError("Merge resulted in empty DataFrame - no overlapping dates")

        X = merged[['polarity', 'vader', 'score', 'subjectivity', 'Daily_Return']]
        y = merged['Next_Day_Return']
        return X, y

    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def train_model(X, y):
    """Train a linear regression model"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Model RMSE: {rmse:.2f}%")
        
        return model, X_test, y_test, predictions
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def plot_results(y_test, predictions, crypto_name):
    """Plot actual vs predicted returns"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Returns')
    plt.plot(y_test.index, predictions, label='Predicted Returns')
    plt.title(f"{crypto_name} - Actual vs Predicted Daily Returns")
    plt.xlabel("Date Index")
    plt.ylabel("Daily Return (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_crypto(crypto_name, ticker, days_back=90):
    """Full analysis pipeline"""
    print(f"\nAnalyzing {crypto_name} ({ticker}) for the past {days_back} days...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Get Reddit sentiment data
    print("1. Fetching Reddit sentiment data...")
    sentiment_df = get_reddit_sentiment(crypto_name)
    if sentiment_df.empty:
        print("Warning: No sentiment data found")
        return None, None, None
    
    # Get price data
    print("2. Fetching price data...")
    
    price_df = get_crypto_data(ticker, start_date, end_date)
    if price_df.empty:
        print("Error: No price data available")
        return None, None, None
    
    print(f"Sentiment data points: {len(sentiment_df)}")
    print(f"Price data points: {len(price_df)}")
    
    # Prepare dataset
    print("3. Preparing dataset...")
    X, y = prepare_dataset(sentiment_df, price_df)
    print(f"Merged dataset: {len(X)} samples")
    if len(X) <= 1:
        print("\n⚠️ Not enough data to train a model. Skipping training...")
        
        print("\n🔍 24-Hour Sentiment Summary:")
        print(X.iloc[0])

        weights = np.array([5, 10, 0.01, 1, 0])
        pred_return = X.iloc[0].values @ weights.T

        print(f"\n📈 Estimated Return: {pred_return:.2f}%")
        print("Suggested Action:", "BUY" if pred_return > 0 else "SELL")
        
        return None, X, y

    if len(X) != 1:
        print("Could not find sentiment + price for today. Try again later.")
        return None, X, y

    print("\n🔍 24-Hour Sentiment Summary:")
    print(X.iloc[0])

    # === Rule-based or pretrained weights ===
    weights = np.array([5, 10, 0.01, 1, 0])  # [polarity, vader, score, subjectivity, Daily_Return]
    pred_return = X.iloc[0].values @ weights.T

    print(f"\n📈 Estimated Return: {pred_return:.2f}%")
    print("Suggested Action:", "BUY" if pred_return > 0 else "SELL")

    from train_lstm_model import train_lstm_model

    model, scaler, seq_len = train_lstm_model(X, y)

    # Predict the next day
    latest_seq = scaler.transform(X.iloc[-seq_len:].values)
    latest_seq_tensor = torch.tensor(latest_seq[np.newaxis, :, :], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_return = model(latest_seq_tensor).item()

    print(f"\nPredicted Next Day Return (LSTM): {pred_return:.2f}%")
    print("Suggested Action:", "BUY" if pred_return > 0 else "SELL")

    if X.empty or y.empty:
        print("Error: Could not prepare dataset")
        return None, None, None
    
    # Train model
    print("4. Training model...")
    try:
        model, X_test, y_test, predictions = train_model(X, y)
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        return None, None, None
    
    # Show results
    print("5. Generating results...")
    plot_results(y_test, predictions, crypto_name)
    
    # Show current sentiment
    print("\nCurrent Sentiment Analysis:")
    print(sentiment_df.iloc[-1][['date', 'polarity', 'vader', 'subjectivity']])
    
    # Make prediction
    try:
        latest_features = X.iloc[-1:].values
        pred_return = model.predict(latest_features)[0]
        print(f"\nPredicted Next Day Return: {pred_return:.2f}%")
        print("Suggested Action:", "BUY" if pred_return > 0 else "SELL")
    except Exception as e:
        print(f"\nCould not make prediction: {str(e)}")
    
    return model, X, y

if __name__ == "__main__":
    # Example usage
    crypto_name = "Shiba Inu"
    ticker = "SHIB-USD"
    
    model, X, y = analyze_crypto(crypto_name, ticker, days_back=90)
    
    if model is None:
        print("\nAnalysis failed. Please check the error messages above.")
    else:
        print("\nAnalysis completed successfully!")
