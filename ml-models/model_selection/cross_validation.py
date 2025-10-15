"""
Reusable cross-validation helpers for sklearn models.
"""
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np

def kfold_cv_score(model, X, y, cv=5, scoring="accuracy", stratified=True, random_state=42):
    if stratified:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring, n_jobs=-1)
    print(f"{model.__class__.__name__} | {scoring} -> mean={scores.mean():.3f} ± {scores.std():.3f}")
    return scores
