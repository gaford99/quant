import yfinance as yf
import pandas as pd
from .utils_io import save_df

PAIRS = ["BRL=X", "TRY=X", "ZAR=X"]  # USD/BRL, USD/TRY, USD/ZAR

def get_fx_history(period="10y", interval="1d"):
    frames = []
    for p in PAIRS:
        df = yf.download(p, period=period, interval=interval, auto_adjust=True, progress=False)
        df = df[["Close"]].rename(columns={"Close": p})
        frames.append(df)
    out = pd.concat(frames, axis=1).dropna()
    save_df(out, "fx_history_daily")
    return out
