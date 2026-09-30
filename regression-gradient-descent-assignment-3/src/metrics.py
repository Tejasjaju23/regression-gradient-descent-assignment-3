"""Regression evaluation metrics."""

from math import sqrt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def metrics_from_predictions(y_true, predictions):
    return {
        name: regression_metrics(y_true, pred)
        for name, pred in predictions.items()
    }
