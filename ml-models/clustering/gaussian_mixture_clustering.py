"""
Gaussian Mixture Model (GMM)
----------------------------
Probabilistic clustering using mixture of Gaussians.
"""

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set2")
plt.title("Gaussian Mixture Clustering")
plt.show()
