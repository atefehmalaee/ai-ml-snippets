"""
IsolationForest for anomaly detection.
"""
from sklearn.ensemble import IsolationForest
import numpy as np

def fit_isolation_forest(X, contamination=0.05, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    return iso

def predict_anomalies(model, X):
    preds = model.predict(X)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)
    return preds, scores
