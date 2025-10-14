import numpy as np
import pandas as pd

from src.data.fetch_fx import get_fx_history
from src.features.fx_features import make_fx_features
from src.models.garch_fx import fit_garch, forecast_volatility
from src.models.egarch_fx import fit_egarch, forecast_egarch_volatility

# 1) data: EM FX prices -> daily log returns
fx = get_fx_history(period="10y", interval="1d")
rets = make_fx_features(fx)["returns"]
series = rets["TRY=X"].dropna()  # pick one EM pair (TRY, BRL, or ZAR)

# 2) define "big shock" threshold (top decile by |return|)
thr = series.abs().quantile(0.90)
shock_days = series.index[series.abs() >= thr]
print(f"Shock count: {len(shock_days)} (threshold={thr:.4f})")

# 3) rolling-fit: for each shock day t, fit on a lookback window and forecast t+1 vol
lookback = 750  # ~3y of trading days
rows = []
for t in shock_days:
    # ensure we have enough history
    pos = series.index.get_loc(t)
    if pos < lookback + 5:  # need buffer for estimation stability
        continue
    window = series.iloc[pos - lookback : pos]  # history up to but not including t

    try:
        g_res = fit_garch(window, p=1, q=1)
        e_res = fit_egarch(window, p=1, q=1)

        g_f = forecast_volatility(g_res, horizon=1).iloc[0]
        e_f = forecast_egarch_volatility(e_res, horizon=1).iloc[0]

        rows.append({
            "date": t,
            "ret_t": series.loc[t],
            "sign": "neg" if series.loc[t] < 0 else "pos",
            "garch_sigma_t1": float(g_f),
            "egarch_sigma_t1": float(e_f),
            "egarch_gamma": float(e_res.params.get("gamma[1]", np.nan))  # naming per arch
        })
    except Exception as ex:
        # skip edge cases that fail to converge
        continue

df = pd.DataFrame(rows).set_index("date").sort_index()
print("Sample:\n", df.head())

# 4) compare conditional forecasts by sign for SHOCK days only
summary = df.groupby("sign")[["garch_sigma_t1","egarch_sigma_t1"]].mean().rename(columns={
    "garch_sigma_t1":"GARCH mean σ(t+1)",
    "egarch_sigma_t1":"EGARCH mean σ(t+1)"
})
print("\nMean forecasted σ(t+1) on SHOCK days, by sign:\n", summary)

# simple effect sizes
garch_diff = summary.loc["neg","GARCH mean σ(t+1)"] - summary.loc["pos","GARCH mean σ(t+1)"]
egarch_diff = summary.loc["neg","EGARCH mean σ(t+1)"] - summary.loc["pos","EGARCH mean σ(t+1)"]
print(f"\nΔ (neg - pos) GARCH:  {garch_diff:.6f}")
print(f"Δ (neg - pos) EGARCH: {egarch_diff:.6f}")

# optional: quick nonparametric test
from scipy.stats import mannwhitneyu
g_pos = df.query("sign=='pos'")["garch_sigma_t1"]
g_neg = df.query("sign=='neg'")["garch_sigma_t1"]
e_pos = df.query("sign=='pos'")["egarch_sigma_t1"]
e_neg = df.query("sign=='neg'")["egarch_sigma_t1"]

print("\nMann-Whitney U p-values (neg vs pos):")
print("GARCH :", mannwhitneyu(g_neg, g_pos, alternative="greater").pvalue)
print("EGARCH:", mannwhitneyu(e_neg, e_pos, alternative="greater").pvalue)

# 5) interpret EGARCH gamma
print("\nEGARCH gamma (median across fits):", df["egarch_gamma"].median())
