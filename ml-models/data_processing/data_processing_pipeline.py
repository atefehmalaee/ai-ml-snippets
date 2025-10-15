"""
Comprehensive Data Processing Pipeline
--------------------------------------
Automatically handles numerical, categorical, and text columns for
classical ML models. Uses scikit-learn’s ColumnTransformer and Pipeline.

Features:
- Detects column types automatically
- Scales numeric features
- One-hot encodes categorical features
- TF-IDF vectorizes text columns
- Returns a clean preprocessing pipeline ready for model training
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd


def detect_column_types(df: pd.DataFrame, text_threshold: int = 20):
    """
    Automatically detect column types based on dtype and text length.

    Args:
        df (pd.DataFrame): Input dataset
        text_threshold (int): Minimum average word count to classify a column as text

    Returns:
        tuple: (numeric_cols, categorical_cols, text_cols)
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols, text_cols = [], []

    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        avg_len = df[col].astype(str).str.split().apply(len).mean()
        if avg_len > text_threshold:
            text_cols.append(col)
        else:
            categorical_cols.append(col)

    print(f"🔍 Detected columns:")
    print(f" - Numeric: {numeric_cols}")
    print(f" - Categorical: {categorical_cols}")
    print(f" - Text: {text_cols}")

    return numeric_cols, categorical_cols, text_cols


def create_data_processing_pipeline(df: pd.DataFrame):
    """
    Builds a preprocessing pipeline for numeric, categorical, and text columns.

    Args:
        df (pd.DataFrame): Input dataset (used only for type detection)

    Returns:
        ColumnTransformer: Preprocessing transformer ready for use in Pipeline
    """
    numeric_cols, categorical_cols, text_cols = detect_column_types(df)

    # Define transformations
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    text_transformer = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=500, stop_words="english"))
    ])

    # Combine all
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("txt", text_transformer, text_cols)
        ]
    )

    print("✅ Data processing pipeline created successfully.")
    return preprocessor


def build_full_pipeline(df: pd.DataFrame, model):
    """
    Combine preprocessing and model into one unified pipeline.

    Args:
        df (pd.DataFrame): Example dataset (used for column detection)
        model: Any sklearn-compatible model (e.g., LogisticRegression())

    Returns:
        sklearn.Pipeline: Combined pipeline
    """
    preprocessor = create_data_processing_pipeline(df)
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    print("🚀 Full ML pipeline ready (preprocessing + model).")
    return full_pipeline
