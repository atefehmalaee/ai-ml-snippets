"""
Feature Interaction Utilities
-----------------------------
Automatically generates interaction (product/sum) terms.
"""

import pandas as pd
import itertools


def create_interaction_terms(df: pd.DataFrame, columns: list):
    """Creates pairwise product and sum interaction features."""
    for col_a, col_b in itertools.combinations(columns, 2):
        df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
        df[f"{col_a}_plus_{col_b}"] = df[col_a] + df[col_b]
    print(f"✅ Created interaction terms for: {columns}")
    return df
