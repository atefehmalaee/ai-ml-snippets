"""
Data Splitting Utilities
------------------------
Split data into train/test sets or train/validation/test.
"""

from sklearn.model_selection import train_test_split


def split_train_test(X, y, test_size=0.2, random_state=42):
    """Performs a basic train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✅ Split data: {len(X_train)} train / {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def split_train_val_test(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Splits data into train, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )

    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )

    print(f"✅ Train/Val/Test: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
