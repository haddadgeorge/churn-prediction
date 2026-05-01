from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10, min_samples_leaf=5):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model