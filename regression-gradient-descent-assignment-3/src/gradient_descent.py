"""Batch Gradient Descent implementation for Linear Regression."""

from dataclasses import dataclass
import numpy as np


@dataclass
class GDResult:
    weights: np.ndarray
    bias: float
    loss_history: list
    iterations: int
    converged: bool

    def predict(self, X):
        return np.asarray(X) @ self.weights + self.bias


class GradientDescentLinearRegression:
    def __init__(
        self,
        learning_rate=0.01,
        iterations=5000,
        tolerance=1e-8,
        min_iterations=100,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.min_iterations = min_iterations
        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []
        self.converged_ = False

    @staticmethod
    def _loss(y_true, y_pred):
        error = y_pred - y_true
        return float(np.mean(error ** 2) / 2.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = float(np.mean(y))

        previous_loss = None

        for iteration in range(1, self.iterations + 1):
            predictions = X @ self.weights_ + self.bias_
            error = predictions - y

            grad_w = (X.T @ error) / n_samples
            grad_b = float(np.mean(error))

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            new_predictions = X @ self.weights_ + self.bias_
            loss = self._loss(y, new_predictions)
            self.loss_history_.append(loss)

            if (
                previous_loss is not None
                and iteration >= self.min_iterations
                and abs(previous_loss - loss) < self.tolerance
            ):
                self.converged_ = True
                break

            previous_loss = loss

        return self

    def predict(self, X):
        if self.weights_ is None:
            raise RuntimeError("Call fit before predict")
        return np.asarray(X, dtype=float) @ self.weights_ + self.bias_

    @property
    def n_iter_(self):
        return len(self.loss_history_)
