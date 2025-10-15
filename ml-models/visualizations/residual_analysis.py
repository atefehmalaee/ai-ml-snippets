"""
Residual and regression diagnostic plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

def plot_residual_distribution(y_true, y_pred):
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color="steelblue")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.show()

def qq_plot(y_true, y_pred):
    from scipy import stats
    residuals = y_true - y_pred
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ Plot of Residuals")
    plt.show()
