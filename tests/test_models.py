import numpy as np
import pandas as pd

from src.features.fx_features import tag_regimes
from src.models.garch_fx import fit_garch, forecast_volatility

def test_tag_regimes_threshold_numeric():
    s = pd.Series([0.01, 0.02, 0.03, 0.04],
                  index=pd.date_range("2020-01-01", periods=4))
    regimes = tag_regimes(s, threshold=0.025)
    assert set(regimes.unique()) == {"low", "high"}
    assert (regimes.index == s.index).all()
    assert regimes.iloc[0] == "low" and regimes.iloc[-1] == "high"

def test_tag_regimes_threshold_median():
    s = pd.Series([0.01, 0.02, 0.03, 0.04],
                  index=pd.date_range("2020-01-01", periods=4))
    regimes = tag_regimes(s, threshold="median")
    assert set(regimes.unique()) == {"low", "high"}

def test_garch_and_forecast_shapes():
    np.random.seed(7)
    data = np.random.normal(0, 1, 400)  # synthetic returns
    s = pd.Series(data)
    res = fit_garch(s, p=1, q=1)
    fcast = forecast_volatility(res, horizon=3)
    assert len(fcast) == 3
    assert (fcast > 0).all()
