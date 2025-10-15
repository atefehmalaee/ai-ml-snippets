"""
Global & local explanations with SHAP (tree & linear models).
pip install shap
"""
import shap
import numpy as np

def shap_summary_plot(trained_model, X_train, max_display=10):
    explainer = shap.Explainer(trained_model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, max_display=max_display, show=True)

def shap_waterfall_single(trained_model, X_train, row_index=0):
    explainer = shap.Explainer(trained_model, X_train)
    sv = explainer(X_train[row_index:row_index+1])
    shap.plots.waterfall(sv[0], show=True)
