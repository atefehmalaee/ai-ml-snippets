"""
GridSearchCV / RandomizedSearchCV utilities.
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search(model, param_grid, X, y, cv=5, scoring="accuracy", n_jobs=-1):
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    gs.fit(X, y)
    print(f"Best params: {gs.best_params_} | Best {scoring}: {gs.best_score_:.3f}")
    return gs.best_estimator_, gs.cv_results_

def random_search(model, param_distributions, X, y, n_iter=30, cv=5, scoring="accuracy", n_jobs=-1, random_state=42):
    rs = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=cv,
                            scoring=scoring, n_jobs=n_jobs, random_state=random_state)
    rs.fit(X, y)
    print(f"Best params: {rs.best_params_} | Best {scoring}: {rs.best_score_:.3f}")
    return rs.best_estimator_, rs.cv_results_
