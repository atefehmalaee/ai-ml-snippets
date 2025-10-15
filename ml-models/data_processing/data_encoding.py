"""
Data Encoding Utilities
-----------------------
Encodes categorical variables for ML models.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def label_encode_columns(df: pd.DataFrame, columns: list):
    """Label encodes categorical columns."""
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    print(f"✅ Label encoded columns: {columns}")
    return df


def one_hot_encode(df: pd.DataFrame, columns: list):
    """Applies one-hot encoding to given columns."""
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    print(f"✅ One-hot encoded columns: {columns}")
    return df
