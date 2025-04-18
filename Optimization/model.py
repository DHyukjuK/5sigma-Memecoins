# src/optimisation_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing       import load_raw_reddit, load_price_csv, align_time_series
from src.feature_engineering import compute_sentiment
from train_lstm_model        import train_lstm_model  # your LSTM trainer

class OptimisationModel:
    def __init__(
        self,
        ticker: str,
        sentiment_paths: tuple,      # (posts_json, comments_json)
        price_path: str,             # e.g. 'data/raw/price_TRUMP-USD.csv'
        risk_free_rate: float = 0.0, # daily rf
        risk_aversion: float = 1.0,   # A ∈ {1,…,5}
        vol_window: int = 30,        # days (or hours) of lookback
        allow_short: bool = False
    ):
        self.ticker      = ticker
        self.posts_path  = sentiment_paths[0]
        self.comments_path= sentiment_paths[1]
        self.price_path  = price_path
        self.rf          = risk_free_rate
        self.A           = risk_aversion
        self.vol_window  = vol_window
        self.allow_short = allow_short

    def fit_predictor(self):
        # 1) Load & score sentiment
        sent = load_raw_reddit(self.posts_path, self.comments_path)
        sent = compute_sentiment(sent)

        # 2) Load price & align
        price = load_price_csv(self.price_path)
        df    = align_time_series(sent, price)  # columns: ['close','volume','sentiment']

        # 3) Prepare windowed data & fit LSTM
        #    train_lstm_model should return (model, scaler, seq_len)
        X = df[['sentiment']].values
        y = df['close'].pct_change().shift(-1).dropna().values
        # align shapes
        X, y = X[:-1], y
        model, scaler, seq_len = train_lstm_model(X, y)

        return df, model, scaler, seq_len

    def compute_allocations(self):
        df, model, scaler, seq_len = self.fit_predictor()

        # 4) Build mu_{t+1} series
        mu_vals = []
        for i in range(len(df)-seq_len):
            window = df['sentiment'].values[i : i+seq_len].reshape(1, seq_len, 1)
            window_scaled = scaler.transform(window.reshape(-1, seq_len))\
                                  .reshape(1, seq_len, 1)
            mu_pred = model.predict(window_scaled)[0,0]
            mu_vals.append(mu_pred)

        mu_index  = df.index[seq_len:]
        mu_series = pd.Series(mu_vals, index=mu_index, name='mu')

        # 5) Compute realized returns R_t and rolling variance
        df['return'] = df['close'].pct_change()
        sigma2 = df['return'].rolling(self.vol_window).var()
        sigma2 = sigma2.loc[mu_index]

        # 6) Compute weights
        w = (mu_series - self.rf) / (self.A * sigma2)
        if not self.allow_short:
            w = w.clip(0.0, 1.0)
        w.name = 'allocation'

        return pd.concat([mu_series, sigma2, w], axis=1)

if __name__ == '__main__':
    opt = OptimisationModel(
        ticker='TRUMP-USD',
        sentiment_paths=('data/raw/reddit_posts.json','data/raw/reddit_comments.json'),
        price_path='data/raw/price_TRUMP-USD.csv',
        risk_free_rate=0.0001,
        risk_aversion=2,
        vol_window=24,    # e.g. last 24 hours
        allow_short=False
    )
    df_out = opt.compute_allocations()
    print(df_out.tail(10))
