"""
Utility functions for loading, cleaning, splitting,
and scaling datasets for classical ML models.

Usage:
    from utils.data_preprocessing import load_and_preprocess_data
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_and_preprocess_data(test_size: float = 0.2, scale: bool = True):
    """
    Loads the Iris dataset and splits into train/test sets.

    Args:
        test_size (float): proportion of test data (default 0.2)
        scale (bool): whether to apply StandardScaler (default True)

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame/Series)
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, y_train, y_test
