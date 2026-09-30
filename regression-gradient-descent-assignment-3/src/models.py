"""Baseline regression models."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


def build_models(random_state=42):
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
        ),
    }


def fit_models(X_train, y_train, random_state=42):
    models = build_models(random_state)
    predictions = {}
    fitted = {}

    return models, fitted, predictions
