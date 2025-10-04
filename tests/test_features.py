import pandas as pd
from src.features.fx_features import make_fx_features

def test_make_fx_features_shapes():
    idx = pd.date_range("2023-01-01", periods=30, freq="B")
    df = pd.DataFrame({"BRL=X": range(1,31), "TRY=X": range(2,32)}, index=idx)
    feats = make_fx_features(df, window=5)
    assert "returns" in feats and "vol_rolling" in feats
    # one diff lost row
    assert feats["returns"].shape[0] == 29
    # rolling vol will have NaNs at the start, but column count matches input
    assert feats["vol_rolling"].shape[1] == df.shape[1]
