"""
Clustering Evaluation Metrics
-----------------------------
Includes silhouette score, Davies–Bouldin index, and Calinski–Harabasz index.
"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def clustering_summary(X, labels, model_name="Clustering Model"):
    """Prints key unsupervised clustering evaluation metrics."""
    if len(set(labels)) <= 1:
        print("⚠️ Cannot compute metrics — only one cluster found.")
        return

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print(f"\n🤖 {model_name} — Clustering Metrics")
    print(f"Silhouette Score        : {sil:.3f} (higher is better)")
    print(f"Calinski–Harabasz Index : {ch:.3f} (higher is better)")
    print(f"Davies–Bouldin Index    : {db:.3f} (lower is better)")
clustering_metrics.py
