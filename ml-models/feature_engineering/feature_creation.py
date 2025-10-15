"""
Feature Creation Utilities
--------------------------
Add ratio, polynomial, and domain-specific features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def create_ratio_features(df: pd.DataFrame, num_pairs: list):
    """Creates ratio features for given pairs of numeric columns."""
    for (num, denom) in num_pairs:
        new_col = f"{num}_to_{denom}_ratio"
        df[new_col] = df[num] / (df[denom] + 1e-9)
    print(f"✅ Created ratio features: {[f'{a}_to_{b}_ratio' for a, b in num_pairs]}")
    return df


def add_polynomial_features(df: pd.DataFrame, columns: list, degree=2):
    """Creates polynomial features for numeric columns."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    transformed = poly.fit_transform(df[columns])
    names = poly.get_feature_names_out(columns)
    new_df = pd.DataFrame(transformed, columns=names)
    print(f"✅ Created polynomial features (degree={degree}) for {columns}")
    return pd.concat([df.reset_index(drop=True), new_df], axis=1)
