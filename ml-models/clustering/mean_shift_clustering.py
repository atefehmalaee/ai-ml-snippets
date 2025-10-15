"""
Mean Shift Clustering
---------------------
Automatically detects number of clusters based on data density.
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)

# Train MeanShift
mean_shift = MeanShift()
labels = mean_shift.fit_predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm")
plt.scatter(mean_shift.cluster_centers_[:, 0], mean_shift.cluster_centers_[:, 1],
            color="black", marker="x", s=200, label="Centers")
plt.title("Mean Shift Clustering")
plt.legend()
plt.show()
