import numpy as np
from src.metrics import regression_metrics


def test_perfect_predictions():
    y = np.array([1, 2, 3])
    metrics = regression_metrics(y, y)
    assert metrics["MAE"] == 0
    assert metrics["MSE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1
