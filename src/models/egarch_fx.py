import pandas as pd
from arch import arch_model

def fit_egarch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "normal"):
    """
    Fit an EGARCH(p,q) model to a returns series.
    Captures volatility asymmetry (leverage effect).
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    am = arch_model(returns.dropna(), vol="EGARCH", p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    return res

def forecast_egarch_volatility(res, horizon: int = 5) -> pd.Series:
    f = res.forecast(horizon=horizon)
    sigma = (f.variance.values[-1, :] ** 0.5)
    return pd.Series(sigma, name="forecast_vol")
