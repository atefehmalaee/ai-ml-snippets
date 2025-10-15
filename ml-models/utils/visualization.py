"""
Reusable visualization helpers:
- Feature importance plots (tree-based models)
- Residual and prediction plots (regression)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_importance(model, feature_names, top_n=10, title="Feature Importance"):
    """Displays bar plot of feature importances (for tree-based models)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices],
        palette="viridis"
    )
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """Shows residual distribution for regression models."""
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True, color="steelblue")
    plt.axvline(0, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    """Scatter plot comparing true vs predicted values."""
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()
