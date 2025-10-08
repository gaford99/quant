import numpy as np
import pandas as pd

def make_fx_features(prices: pd.DataFrame, window: int = 20):
    rets = np.log(prices).diff().dropna()
    vol = rets.rolling(window).std() * np.sqrt(252)
    return {"returns": rets, "vol_rolling": vol}

def tag_regimes(vol_series: pd.Series, threshold="median") -> pd.Series:
    """
    Label each date as 'low' or 'high' volatility.

    threshold:
      - 'median' -> split by the series median
      - float    -> absolute cutoff (e.g., 0.025 for 2.5% daily vol)

    Returns: pd.Series with same index containing 'low'/'high'
    """
    if not isinstance(vol_series, pd.Series):
        raise TypeError("vol_series must be a pandas Series")

    if isinstance(threshold, str) and threshold.lower() == "median":
        cutoff = vol_series.median()
    else:
        cutoff = float(threshold)

    labels = np.where(vol_series > cutoff, "high", "low")
    return pd.Series(labels, index=vol_series.index, name="regime")