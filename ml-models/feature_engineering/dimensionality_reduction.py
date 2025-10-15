"""
Dimensionality Reduction Utilities
----------------------------------
Supports PCA, LDA (for classification), and t-SNE for visualization.
"""

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


def apply_pca(X, n_components=2, plot=False):
    """Reduces dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"✅ PCA reduced to {n_components} components ({explained:.2f}% variance).")

    if plot:
        plt.scatter(reduced[:, 0], reduced[:, 1])
        plt.title("PCA Visualization")
        plt.show()

    return pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])


def apply_lda(X, y, n_components=2):
    """Performs Linear Discriminant Analysis for classification."""
    lda = LDA(n_components=n_components)
    reduced = lda.fit_transform(X, y)
    print(f"✅ LDA reduced to {n_components} components.")
    return pd.DataFrame(reduced, columns=[f"LD{i+1}" for i in range(n_components)])


def apply_tsne(X, n_components=2, perplexity=30):
    """Performs t-SNE for nonlinear dimensionality reduction (visualization only)."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(X)
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("t-SNE Visualization")
    plt.show()
    return pd.DataFrame(reduced, columns=["Dim1", "Dim2"])
