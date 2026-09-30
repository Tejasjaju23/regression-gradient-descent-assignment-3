import numpy as np
from src.gradient_descent import GradientDescentLinearRegression


def test_gradient_descent_learns_simple_line():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 3 * X[:, 0] + 2

    model = GradientDescentLinearRegression(
        learning_rate=0.01,
        iterations=10000,
        tolerance=1e-10,
        min_iterations=100,
    )
    model.fit(X, y)
    predictions = model.predict(X)

    assert model.loss_history_[-1] < model.loss_history_[0]
    assert np.mean((predictions - y) ** 2) < 1.0
