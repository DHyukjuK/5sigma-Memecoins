# stage2_allocation.py

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# your existing modules
from LSTM import get_reddit_sentiment, get_crypto_data, prepare_dataset
from train_lstm_model import train_lstm_model

class OneAssetOptimizer:
    def __init__(self, returns: pd.Series, window: int = 30, rf: float = 0.0, A: int = 3):
        """
        returns : pd.Series of past realized returns (in decimal, e.g. 0.02 for 2%)
        window  : lookback length for rolling variance
        rf      : per-period risk free rate (decimal)
        A       : risk aversion bucket (1–5)
        """
        self.returns = returns.copy()
        self.window = window
        self.rf = rf
        self.A = A

    def sigma2(self) -> float:
        """Rolling variance over last `window` points."""
        return float(self.returns.rolling(self.window).var().iloc[-1])

    def weight(self, mu: float) -> float:
        """
        mu : E[R_{t+1}] (decimal, e.g. 0.015 for 1.5%)
        returns clipped w in [0,1]
        """
        σ2 = self.sigma2()
        if σ2 <= 0 or np.isnan(σ2):
            return 0.0
        raw_w = (mu - self.rf) / (self.A * σ2)
        return float(np.clip(raw_w, 0.0, 1.0))

def main():
    # — user parameters —
    crypto_name = "Bitcoin"
    ticker      = "BTC-USD"
    days_back   = 60        # how many days of history to fetch
    lookback    = 30        # window for variance
    rf          = 0.0001    # e.g. 0.01% per day
    A_bucket    = 2         # risk aversion

    # 1) fetch sentiment
    sent_df = get_reddit_sentiment(crypto_name)
    if sent_df is None or sent_df.empty:
        raise RuntimeError("No sentiment data")

    # 2) fetch price data
    end = datetime.now()
    start = end - timedelta(days=days_back)
    price_df = get_crypto_data(ticker, start, end)
    if price_df is None or price_df.empty:
        raise RuntimeError("No price data")

    # 3) build feature/target
    X, y = prepare_dataset(sent_df, price_df)
    if X.empty or y.empty:
        raise RuntimeError("Dataset empty after merge")

    # 4) train LSTM
    model, scaler, seq_len = train_lstm_model(X, y)
    model.eval()

    # 5) build realized‐returns series (decimal)
    #    LSTM.py’s Daily_Return is pct_change*100, so divide by 100
    rets = price_df['Daily_Return'].div(100.0).dropna()
    # drop last row since Next_Day_Return is shifted
    rets = rets.iloc[:-1]

    # 6) instantiate optimizer
    opt = OneAssetOptimizer(returns=rets, window=lookback, rf=rf, A=A_bucket)

    # 7) form the latest sequence for prediction
    latest_X = X.iloc[-seq_len:]
    latest_scaled = scaler.transform(latest_X.values)
    latest_tensor = torch.tensor(latest_scaled[np.newaxis, :, :], dtype=torch.float32)

    # 8) predict one‐step‐ahead return (model outputs percent)
    mu_pct = model(latest_tensor).item()        # e.g. 1.23 means 1.23%
    mu_dec = mu_pct / 100.0                     # e.g. 0.0123

    # 9) compute optimal weight
    w = opt.weight(mu_dec)

    # 10) output
    print(f"Predicted return tomorrow: {mu_pct:.2f}%")
    print(f"Estimated σ² over last {lookback} days: {opt.sigma2():.6f}")
    print(f"→ Optimal allocation: {w*100:.2f}% of portfolio to {crypto_name}")

if __name__ == "__main__":
    main()
