"""
Cluster visualization for KMeans, DBSCAN, etc.
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters_2d(X, labels, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=30)
    plt.title("2D Cluster Visualization (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

def elbow_plot(distortions, k_range):
    plt.plot(k_range, distortions, "bo-")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Distortion")
    plt.show()
