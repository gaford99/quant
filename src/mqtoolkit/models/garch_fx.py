import pandas as pd
from arch import arch_model

def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "normal"):
    """
    Fit a simple GARCH(p,q) model to a 1D returns series.
    Returns: ARCHModelResult
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    # Drop NaNs just in case
    r = returns.dropna()

    # Build and fit model
    am = arch_model(r, vol="Garch", p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    return res

def forecast_volatility(res, horizon: int = 3) -> pd.Series:
    """
    Forecast conditional volatility for `horizon` steps using a fitted model result.
    Returns a pd.Series (length=horizon) of positive forecasts.
    """
    f = res.forecast(horizon=horizon)
    # The forecast object stores conditional variances; take sqrt to get sigma
    sigma = (f.variance.values[-1, :] ** 0.5)
    return pd.Series(sigma, name="forecast_vol")
