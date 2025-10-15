"""
One-Class SVM for anomaly detection.
"""
from sklearn.svm import OneClassSVM

def fit_oneclass_svm(X, nu=0.05, kernel="rbf", gamma="scale"):
    oc = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    oc.fit(X)
    return oc

def predict_anomalies(model, X):
    preds = model.predict(X)          # -1 = anomaly, 1 = normal
    return preds
