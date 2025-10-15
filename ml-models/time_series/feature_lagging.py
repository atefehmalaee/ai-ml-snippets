"""
Lag/rolling features for time series.
"""
import pandas as pd

def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 7, 14)):
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df

def add_rolling_features(df: pd.DataFrame, col: str, windows=(7, 14, 28)):
    for w in windows:
        df[f"{col}_rollmean{w}"] = df[col].rolling(window=w, min_periods=1).mean()
        df[f"{col}_rollstd{w}"] = df[col].rolling(window=w, min_periods=1).std()
    return df
