"""
Feature Selection Utilities
---------------------------
Includes correlation filter, univariate selection, and tree-based feature importance.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9):
    """Drops features with correlation higher than the given threshold."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(~corr_matrix.isnull(), 0).where(
        pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    print(f"✅ Dropped {len(to_drop)} highly correlated features: {to_drop}")
    return df_reduced


def select_top_k_features(X, y, k=10):
    """Selects top k features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"✅ Top {k} features selected: {list(selected_features)}")
    return pd.DataFrame(X_new, columns=selected_features)


def tree_based_selection(X, y, n_features=10):
    """Selects features based on Random Forest importance."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(n_features).index
    print(f"✅ Top {n_features} tree-based features: {list(top_features)}")
    return X[top_features]
