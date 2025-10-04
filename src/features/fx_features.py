import numpy as np
import pandas as pd

def make_fx_features(prices: pd.DataFrame, window: int = 20):
    """
    Build simple FX features from a wide price DataFrame (columns = tickers).
    Returns a dict with:
      - 'returns': log returns (same columns, one fewer row)
      - 'vol_rolling': rolling annualized volatility over `window` days
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    # basic sanity: need at least window+1 rows to compute rolling vol
    if prices.shape[0] < window + 1:
        raise ValueError(f"need at least {window+1} rows, got {prices.shape[0]}")

    rets = np.log(prices).diff().dropna()
    vol = rets.rolling(window).std() * np.sqrt(252)
    return {"returns": rets, "vol_rolling": vol}
