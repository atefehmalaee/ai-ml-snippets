"""
Logistic Regression example with sklearn.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils.data_preprocessing import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
