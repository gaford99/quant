import plotly.graph_objects as go
import pandas as pd

def plot_vol_regimes(returns: pd.Series, rolling_vol: pd.Series,
                     garch_forecast: pd.Series, regimes: pd.Series):
    fig = go.Figure()

    # realized rolling vol
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol,
                             name="Realized Rolling Vol", line=dict(color="blue")))
    # GARCH forecast (align end)
    fcast_idx = garch_forecast.index
    fig.add_trace(go.Scatter(x=fcast_idx, y=garch_forecast,
                             name="Forecast Vol (next)", line=dict(color="orange", dash="dot")))

    # Highlight regimes
    colors = regimes.map({"high": "rgba(255,0,0,0.2)", "low": "rgba(0,255,0,0.1)"})
    for i, (date, c) in enumerate(colors.items()):
        fig.add_shape(type="rect",
                      x0=date, x1=date, y0=0, y1=rolling_vol.max(),
                      fillcolor=c, line_width=0)
    fig.update_layout(title="Volatility & Regimes",
                      yaxis_title="Volatility", xaxis_title="Date",
                      template="plotly_white", height=600)
    return fig
