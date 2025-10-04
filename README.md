# quant

# Macro-Quant-Toolkit
Three mini-projects in one: (1) Yield curve modeling, (2) EM FX volatility, (3) Anomaly detection on macro spreads.
Clean code, tests, and a small Dash app that turns analysis into decisions.

## Why this exists
- Reproducible data pipelines (public sources), quantitative models, and visuals for Macro desks.
- Shows data discipline (tests, versioned dependencies), predictive analytics, and BI-style summaries.

## Quickstart
```bash
conda env create -f environment.yml
conda activate macro-quant
pytest -q
python -c "from src.data.fetch_fx import get_fx_history; print(get_fx_history().tail())"
