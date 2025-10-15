"""
Random Forest classifier example.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils.data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data()

model = RandomForestClassifier(
    n_estimators=100, max_depth=None, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
