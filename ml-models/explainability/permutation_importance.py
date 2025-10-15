"""
Model-agnostic permutation importance.
"""
from sklearn.inspection import permutation_importance
import pandas as pd

def permutation_importance_df(trained_model, X_val, y_val, n_repeats=10, scoring=None, random_state=42):
    r = permutation_importance(trained_model, X_val, y_val, n_repeats=n_repeats,
                               scoring=scoring, random_state=random_state, n_jobs=-1)
    df = pd.DataFrame({"feature": X_val.columns, "importance_mean": r.importances_mean,
                       "importance_std": r.importances_std}).sort_values("importance_mean", ascending=False)
    print(df.head(10))
    return df
