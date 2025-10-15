"""
DBSCAN (Density-Based Spatial Clustering)
-----------------------------------------
Groups data based on density and automatically detects outliers.
"""

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Create dataset
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Train DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_pred = dbscan.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
plt.title("DBSCAN Clustering (with Outliers)")
plt.show()
