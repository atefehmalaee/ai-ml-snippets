"""
Local explanations with LIME for tabular data.
pip install lime
"""
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

def lime_explain_row(trained_model, X_train, X_sample, feature_names, class_names=None, row_index=0):
    explainer = LimeTabularExplainer(
        X_train.astype(float),
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )
    predict_fn = lambda x: trained_model.predict_proba(x).astype(float)
    exp = explainer.explain_instance(X_sample[row_index].astype(float), predict_fn, num_features=10)
    exp.show_in_notebook(show_table=True)
    return exp
