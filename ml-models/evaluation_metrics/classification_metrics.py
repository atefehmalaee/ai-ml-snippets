"""
Classification Metrics Utilities
--------------------------------
Includes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt


def classification_summary(y_true, y_pred, y_proba=None, model_name="Model"):
    """Prints and visualizes standard classification metrics."""
    print(f"\n📊 {model_name} — Classification Report")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.3f}")

    if y_proba is not None:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_proba, multi_class='ovr'):.3f}")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
