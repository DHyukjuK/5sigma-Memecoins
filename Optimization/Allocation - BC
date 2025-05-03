
# stage2_allocation.py

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# your existing modules
from LSTM import get_reddit_sentiment, get_crypto_data, prepare_dataset
from train_lstm_model import train_lstm_model

class OneAssetOptimizer:
    def __init__(self, returns: pd.Series, window: int = 30, rf: float = 0.0, A: float = 3):
        self.returns = returns.copy()
        self.window = window
        self.rf = rf
        self.A = A
        print(f"[Optimizer init] window={window}, rf={rf}, A={A}")
        print(f"[Optimizer init] returns.tail()=\n{self.returns.tail()}")

    def sigma2(self) -> float:
        var = float(self.returns.rolling(self.window).var().iloc[-1])
        print(f"[sigma2] variance over last {self.window} = {var:.6f}")
        return var

    def weight(self, mu: float) -> float:
        σ2 = self.sigma2()
        print(f"[weight] mu={mu:.6f}, rf={self.rf}, σ2={σ2:.6f}")
        if σ2 <= 0 or np.isnan(σ2):
            print("[weight] σ2 invalid, returning 0.0")
            return 0.0
        raw_w = (mu - self.rf) / (self.A * σ2)
        clipped = float(np.clip(raw_w, 0.0, 1.0))
        print(f"[weight] raw_w={raw_w:.6f}, clipped_w={clipped:.6f}")
        return clipped

def main():
    # — user parameters —
    crypto_name = "Shiba Inu"
    ticker      = "SHIB-USD"
    days_back   = 60
    lookback    = 30
    rf          = 0.0001
    A_bucket    = float(input("Enter risk aversion A: "))
    print(f"[params] crypto_name={crypto_name}, ticker={ticker}")
    print(f"[params] days_back={days_back}, lookback={lookback}, rf={rf}, A_bucket={A_bucket}")

    # 1) fetch sentiment
    sent_df = get_reddit_sentiment(crypto_name)
    print(f"[sentiment] head:\n{sent_df.head()}\nshape={sent_df.shape}")
    if sent_df is None or sent_df.empty:
        raise RuntimeError("No sentiment data")

    # 2) fetch price data
    end = datetime.now()
    start = end - timedelta(days=days_back)
    sent_df['date'] = pd.to_datetime(sent_df['date']).dt.date
    all_days = pd.date_range(start=start.date(),
                             end=  end.date(),
                             freq='D').date
    sent_df = (
        sent_df
        .set_index('date')
        .reindex(all_days, fill_value=0)
        .rename_axis('date')
        .reset_index()
    )
    print(f"[sentiment] backfilled to {sent_df.shape[0]} days")
    price_df = get_crypto_data(ticker, start, end)
    print(f"[prices] head:\n{price_df.head()}\nshape={price_df.shape}")
    if price_df is None or price_df.empty:
        raise RuntimeError("No price data")

    # 3) build feature/target
    X, y = prepare_dataset(sent_df, price_df)
    print(f"[dataset] X.head():\n{X.head()}\ny.head():\n{y.head()}")
    print(f"[dataset] X.shape={X.shape}, y.shape={y.shape}")
    if X.empty or y.empty:
        raise RuntimeError("Dataset empty after merge")

    # 4) train LSTM



    if len(X) < 2:
        print("[warning] only", len(X), "merged row(s)—skipping LSTM.")
        # fallback: use zero expected return, or last observed next-day return
        mu_pct = float(y.iloc[-1] if len(y) else 0.0)
        seq_len = 1
        scaler = None
        model  = None
    else:
        model, scaler, seq_len = train_lstm_model(X, y)
        model.eval()


    # 5) build realized‐returns series (decimal)
    rets = price_df['Daily_Return'].div(100.0).dropna().iloc[:-1]
    print(f"[returns] tail:\n{rets.tail()}\ndescribe:\n{rets.describe()}")

    # 6) instantiate optimizer
    opt = OneAssetOptimizer(returns=rets, window=lookback, rf=rf, A=A_bucket)

   # 7) Predict μₜ₊₁ (percent) using LSTM or fallback
    if model is not None and scaler is not None:
        latest_X = X.iloc[-seq_len:]
        print(f"[debug] latest_X.shape = {latest_X.shape}")
        latest_scaled = scaler.transform(latest_X.values)
        latest_tensor = torch.tensor(
            latest_scaled[np.newaxis, :, :],
            dtype=torch.float32
        )
        mu_pct = model(latest_tensor).item()
        print(f"[debug] model-predicted mu_pct = {mu_pct:.6f}%")
    else:
        # fallback: use the last observed Next_Day_Return (or zero)
        mu_pct = float(y.iloc[-1] if len(y) else 0.0)
        print(f"[debug] fallback mu_pct = {mu_pct:.6f}%")

    # convert to decimal
    mu_dec = mu_pct / 100.0

    print(f"[prediction] mu_pct={mu_pct:.6f}%, mu_dec={mu_dec:.6f}")

    # 9) compute optimal weight
    w = opt.weight(mu_dec)

    # 10) output
    print(f"\nPredicted return tomorrow: {mu_pct:.2f}%")
    print(f"Estimated σ² over last {lookback} days: {opt.sigma2():.6f}")
    print(f"→ Optimal allocation: {w*100:.2f}% of portfolio to {crypto_name}")

if __name__ == "__main__":
    main()
