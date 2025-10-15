"""
Data Cleaning Utilities
-----------------------
Handles missing values, duplicates, and basic preprocessing.
"""

import pandas as pd
import numpy as np


def remove_duplicates(df: pd.DataFrame):
    """Removes duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"✅ Removed {before - len(df)} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean"):
    """
    Fills missing values based on strategy.
    Options: 'mean', 'median', 'mode', 'drop'
    """
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        raise ValueError("Invalid strategy")

    print(f"✅ Missing values handled using '{strategy}' strategy.")
    return df
