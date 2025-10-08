import pandas as pd
from src.features.fx_features import make_fx_features

def test_make_fx_features():
    idx = pd.date_range("2023-01-01", periods=30, freq="B")
    df = pd.DataFrame({"BRL=X": range(1,31), "TRY=X": range(2,32)}, index=idx)
    feats = make_fx_features(df)
    assert "returns" in feats and "vol_rolling" in feats
    assert feats["returns"].shape[0] == 29
