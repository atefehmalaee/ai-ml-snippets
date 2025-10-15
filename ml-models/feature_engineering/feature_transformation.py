"""
Feature Transformation Utilities
--------------------------------
Log, Box-Cox, and Power transforms for skewed data.
"""

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


def log_transform(df: pd.DataFrame, columns: list):
    """Applies log(1+x) to specified columns."""
    for col in columns:
        df[col] = np.log1p(df[col].clip(lower=0))
    print(f"✅ Applied log transform to: {columns}")
    return df


def boxcox_transform(df: pd.DataFrame, columns: list):
    """Applies Box-Cox transform to positive-valued columns."""
    for col in columns:
        if (df[col] <= 0).any():
            print(f"⚠️ Skipping {col}: Box-Cox requires positive values.")
            continue
        df[col], _ = boxcox(df[col])
    print(f"✅ Applied Box-Cox transform to: {columns}")
    return df


def yeo_johnson_transform(df: pd.DataFrame, columns: list):
    """Applies Yeo-Johnson transform (works for negatives too)."""
    transformer = PowerTransformer(method="yeo-johnson")
    df[columns] = transformer.fit_transform(df[columns])
    print(f"✅ Applied Yeo-Johnson transform to: {columns}")
    return df
