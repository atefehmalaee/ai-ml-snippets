"""
Data distribution visualizations: histograms, density plots, pairplots.
"""
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, columns=None, bins=30):
    cols = columns or df.select_dtypes(include="number").columns
    df[cols].hist(bins=bins, figsize=(14, 8), edgecolor="black")
    plt.suptitle("Feature Distributions", fontsize=16)


def plot_pairwise(df, hue=None):
    sns.pairplot(df, hue=hue, diag_kind="kde", corner=True)
    plt.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=16)
    plt.show()
