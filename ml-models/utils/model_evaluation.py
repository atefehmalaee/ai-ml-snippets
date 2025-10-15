"""
Reusable model evaluation utilities:
- Accuracy, precision, recall, F1
- Confusion matrix
- ROC curve plotting
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification_model(y_true, y_pred, model_name="Model"):
    """Prints standard classification metrics."""
    print(f"\n📊 Evaluation Report for {model_name}")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision : {precision_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"Recall    : {recall_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred, average='weighted'):.3f}")


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Displays confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_score, model_name="Model"):
    """Plots ROC curve given probability scores or decision function."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
