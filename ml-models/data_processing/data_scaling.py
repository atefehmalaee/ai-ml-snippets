"""
Data Scaling Utilities
----------------------
Normalizes or standardizes numeric features.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd


def scale_features(df: pd.DataFrame, method="standard"):
    """
    Scales numerical columns using chosen method.
    Options: 'standard', 'minmax'
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"✅ Scaled columns {list(numeric_cols)} using {method} scaler.")
    return df
