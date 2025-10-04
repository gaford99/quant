from pathlib import Path
import pandas as pd

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, name: str):
    path = DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=True)
    return path
