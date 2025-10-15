"""
Outlier Detection and Removal
-----------------------------
Provides IQR-based and Z-score-based filtering.
"""

import pandas as pd
import numpy as np


def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5):
    """Removes outliers based on the IQR rule."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[column] >= (Q1 - factor * IQR)) & (df[column] <= (Q3 + factor * IQR))
    removed = len(df) - mask.sum()
    print(f"✅ Removed {removed} outliers from '{column}' using IQR.")
    return df[mask]


def remove_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """Removes outliers based on Z-score threshold."""
    z = (df[column] - df[column].mean()) / df[column].std()
    mask = abs(z) < threshold
    removed = len(df) - mask.sum()
    print(f"✅ Removed {removed} outliers from '{column}' using Z-score.")
    return df[mask]
