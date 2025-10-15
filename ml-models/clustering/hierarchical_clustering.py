"""
Agglomerative Hierarchical Clustering Example
---------------------------------------------
Clusters are formed by recursively merging smaller clusters.
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

# Train model
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Accent")
plt.title("Agglomerative Hierarchical Clustering")
plt.show()

# Optional: visualize dendrogram
plt.figure(figsize=(6, 3))
Z = linkage(X, method='ward')
dendrogram(Z)
plt.title("Dendrogram (Ward linkage)")
plt.show()
