"""
Feature importance visualizations (tree, permutation, SHAP).
"""
import pandas as pd
import matplotlib.pyplot as plt
import shap

def plot_tree_importance(model, feature_names, top_n=15):
    importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
    importances.plot(kind="barh", color="teal")
    plt.title("Top Feature Importances (Tree Model)")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.show()

def plot_permutation_importance(perm_df):
    perm_df.head(15).plot(kind="barh", x="feature", y="importance_mean", legend=False, color="coral")
    plt.title("Permutation Importance")
    plt.xlabel("Mean Importance")
    plt.gca().invert_yaxis()
    plt.show()

def plot_shap_summary(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
