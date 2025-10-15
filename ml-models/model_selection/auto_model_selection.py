"""
Quick model comparison across candidates.
"""
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

def compare_classifiers(models, X_train, y_train, X_val, y_val):
    results = []
    for m in models:
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        acc = accuracy_score(y_val, pred)
        results.append((m.__class__.__name__, acc))
        print(f"{m.__class__.__name__}: accuracy={acc:.3f}")
    return sorted(results, key=lambda x: x[1], reverse=True)

def compare_regressors(models, X_train, y_train, X_val, y_val):
    results = []
    for m in models:
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        r2 = r2_score(y_val, pred)
        results.append((m.__class__.__name__, r2))
        print(f"{m.__class__.__name__}: R2={r2:.3f}")
    return sorted(results, key=lambda x: x[1], reverse=True)
